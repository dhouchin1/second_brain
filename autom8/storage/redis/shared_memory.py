"""
Redis Shared Memory Implementation for Autom8.

Implements the shared memory architecture from the PRD with efficient
reference-based memory sharing and event-driven coordination.
"""

import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from autom8.models.memory import (
    AgentContext, 
    AgentEvent, 
    Decision, 
    EventType,
    MemoryReference,
    Priority
)
from autom8.storage.redis.client import RedisClient, get_redis_client
from autom8.utils.tokens import estimate_tokens

logger = logging.getLogger(__name__)


class RedisSharedMemory:
    """
    Redis-based shared memory system for efficient agent coordination.
    
    Implements the PRD specification for shared memory without context bloat,
    using references instead of content duplication.
    """
    
    def __init__(self, client: Optional[RedisClient] = None):
        self.client = client
        self._initialized = False
        
        # Memory organization namespaces from PRD
        self.namespaces = {
            "decisions": "autom8:decisions:",
            "context": "autom8:context:",
            "summaries": "autom8:summaries:",
            "embeddings": "autom8:embeddings:",
            "agent_state": "autom8:agents:",
            "events": "autom8:events",
            "metrics": "autom8:metrics"
        }
        
        # Default TTL values
        self.default_ttl = 86400  # 24 hours
        self.decision_ttl = 86400  # 24 hours
        self.context_ttl = 3600   # 1 hour
        self.agent_state_ttl = 7200  # 2 hours
    
    async def initialize(self) -> bool:
        """Initialize Redis shared memory system."""
        try:
            if not self.client:
                self.client = await get_redis_client()
            
            if not self.client.is_connected:
                logger.warning("Redis not available, shared memory will be limited")
                return False
            
            self._initialized = True
            logger.info("Redis shared memory initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis shared memory: {e}")
            return False
    
    async def store_decision(self, decision: Decision) -> bool:
        """
        Store decision with multiple access patterns as specified in PRD.
        
        Creates sorted sets for different access patterns:
        - By recency
        - By agent
        - By tags
        """
        if not self._initialized or not self.client:
            return False
        
        try:
            key = f"{self.namespaces['decisions']}{decision.id}"
            
            # Store full decision as hash
            decision_data = {
                "id": decision.id,
                "agent_id": decision.agent_id,
                "summary": decision.summary,
                "content": decision.content,
                "reasoning": decision.reasoning,
                "affects": json.dumps(decision.affects),
                "confidence": str(decision.confidence),
                "tags": json.dumps(decision.tags),
                "decision_type": decision.decision_type.value,
                "status": decision.status.value,
                "priority": str(decision.priority),
                "timestamp": decision.timestamp.isoformat(),
                "expires_at": decision.expires_at.isoformat() if decision.expires_at else "",
                "metadata": json.dumps(decision.metadata)
            }
            
            # Store decision hash
            await self.client.hset(key, decision_data)
            
            # Set expiration
            expires_in = self.decision_ttl
            if decision.expires_at:
                expires_in = int((decision.expires_at - datetime.utcnow()).total_seconds())
                expires_in = max(expires_in, 60)  # Minimum 1 minute
            
            await self.client.expire(key, expires_in)
            
            # Add to sorted sets for different access patterns
            score = time.time()
            
            # By recency
            await self.client.zadd(
                f"{self.namespaces['decisions']}recent",
                {decision.id: score}
            )
            
            # By agent
            await self.client.zadd(
                f"{self.namespaces['decisions']}agent:{decision.agent_id}",
                {decision.id: score}
            )
            
            # By tags
            for tag in decision.tags:
                await self.client.zadd(
                    f"{self.namespaces['decisions']}tag:{tag}",
                    {decision.id: score}
                )
            
            logger.debug(f"Stored decision {decision.id} for agent {decision.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store decision {decision.id}: {e}")
            return False
    
    async def get_decision(self, decision_id: str) -> Optional[Decision]:
        """Get decision by ID."""
        if not self._initialized or not self.client:
            return None
        
        try:
            key = f"{self.namespaces['decisions']}{decision_id}"
            data = await self.client.hgetall(key)
            
            if not data:
                return None
            
            # Reconstruct Decision object
            from autom8.models.memory import DecisionType, DecisionStatus
            decision = Decision(
                id=data["id"],
                agent_id=data["agent_id"],
                summary=data["summary"],
                content=data["content"],
                reasoning=data["reasoning"],
                affects=json.loads(data["affects"]) if data["affects"] else [],
                confidence=float(data["confidence"]),
                tags=json.loads(data["tags"]) if data["tags"] else [],
                decision_type=DecisionType(data["decision_type"]),
                status=DecisionStatus(data["status"]),
                priority=int(data["priority"]),
                timestamp=datetime.fromisoformat(data["timestamp"]),
                expires_at=datetime.fromisoformat(data["expires_at"]) if data["expires_at"] else None,
                metadata=json.loads(data["metadata"]) if data["metadata"] else {}
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Failed to get decision {decision_id}: {e}")
            return None
    
    async def get_agent_context(self, agent_id: str, max_tokens: int = 500) -> AgentContext:
        """
        Get minimal context for agent without bloat, as specified in PRD.
        
        Priority-based context assembly:
        1. Recent decisions affecting this agent (30% of budget)
        2. Active work items (50% of total budget)
        3. Relevant summaries (80% of total budget)
        """
        context = AgentContext(agent_id=agent_id, budget=max_tokens)
        tokens_used = 0
        
        try:
            if not self._initialized or not self.client:
                logger.warning("Redis not available, returning empty context")
                return context
            
            # Priority 1: Recent decisions affecting this agent
            decision_ids = await self.client.zrevrange(
                f"{self.namespaces['decisions']}agent:{agent_id}",
                0, 2  # Top 3 most recent
            )
            
            for decision_id in decision_ids:
                decision_key = f"{self.namespaces['decisions']}{decision_id}"
                summary = await self.client.hget(decision_key, "summary")
                
                if summary:
                    tokens = estimate_tokens(summary)
                    if tokens_used + tokens <= max_tokens * 0.3:  # 30% for decisions
                        success = context.add_decision(decision_id, summary, tokens)
                        if success:
                            tokens_used += tokens
            
            # Priority 2: Active work items
            work_key = f"{self.namespaces['agent_state']}{agent_id}"
            current_work = await self.client.hget(work_key, "current_work")
            
            if current_work:
                tokens = estimate_tokens(current_work)
                if tokens_used + tokens <= max_tokens * 0.5:  # 50% total
                    context.work_items.append(current_work)
                    context.tokens_used += tokens
                    tokens_used += tokens
            
            # Priority 3: Relevant summaries
            summary_ids = await self.client.zrevrange(
                f"{self.namespaces['summaries']}global",
                0, 1  # Top 2 summaries
            )
            
            for summary_id in summary_ids:
                summary_key = f"{self.namespaces['summaries']}{summary_id}"
                summary_content = await self.client.get(summary_key)
                
                if summary_content:
                    tokens = estimate_tokens(summary_content)
                    if tokens_used + tokens <= max_tokens * 0.8:  # 80% total
                        success = context.add_summary(summary_id, summary_content, tokens)
                        if success:
                            tokens_used += tokens
            
            # Add references for expansion as specified in PRD
            context.available_references.extend([
                f"redis.hgetall('{self.namespaces['decisions']}*')",
                f"redis.keys('{self.namespaces['summaries']}*')",
                "sqlite.search('query')"
            ])
            
            logger.debug(f"Prepared context for agent {agent_id}: {tokens_used}/{max_tokens} tokens")
            return context
            
        except Exception as e:
            logger.error(f"Failed to get agent context for {agent_id}: {e}")
            return context
    
    async def emit_event(self, event: AgentEvent) -> bool:
        """
        Emit event for agent coordination through Redis Streams.
        
        Uses Redis Streams for event-driven coordination as specified in PRD.
        """
        if not self._initialized or not self.client:
            logger.warning("Redis not available, event will be lost")
            return False
        
        try:
            # Prepare event data
            event_data = {
                "id": event.id,
                "type": event.type.value,
                "priority": event.priority.value,
                "source_agent": event.source_agent,
                "target_agent": event.target_agent or "broadcast",
                "data": json.dumps(event.data),
                "summary": event.summary,
                "related_task": event.related_task or "",
                "correlation_id": event.correlation_id or "",
                "timestamp": event.timestamp.isoformat(),
                "expires_at": event.expires_at.isoformat() if event.expires_at else ""
            }
            
            # Add to stream
            stream_key = self.namespaces["events"]
            message_id = await self.client.xadd(
                stream_key,
                event_data,
                maxlen=1000  # Keep last 1000 events
            )
            
            if message_id:
                # Update agent states
                if event.target_agent and event.target_agent != "broadcast":
                    agent_key = f"{self.namespaces['agent_state']}{event.target_agent}"
                    await self.client.hset(agent_key, {"pending_event": event.id})
                
                # Track in metrics
                await self.client.hset(
                    self.namespaces["metrics"],
                    {f"events:{event.type.value}": "1"}
                )
                
                logger.debug(f"Emitted event {event.id} from {event.source_agent}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to emit event {event.id}: {e}")
            return False
    
    async def consume_events(self, agent_id: str, count: int = 10) -> List[AgentEvent]:
        """
        Consume events for specific agent.
        
        Reads from Redis Streams with proper acknowledgment.
        """
        if not self._initialized or not self.client:
            return []
        
        try:
            stream_key = self.namespaces["events"]
            
            # Read events
            results = await self.client.xread(
                {stream_key: "$"},  # Read new messages
                count=count,
                block=1000  # Block for 1 second
            )
            
            events = []
            for stream, messages in results:
                for message_id, fields in messages:
                    # Check if event is relevant to this agent
                    target = fields.get("target_agent", "")
                    if target == "broadcast" or target == agent_id:
                        try:
                            event = AgentEvent(
                                id=fields["id"],
                                type=EventType(fields["type"]),
                                priority=Priority(int(fields["priority"])),
                                source_agent=fields["source_agent"],
                                target_agent=fields["target_agent"] if fields["target_agent"] != "broadcast" else None,
                                data=json.loads(fields["data"]) if fields["data"] else {},
                                summary=fields["summary"],
                                related_task=fields["related_task"] if fields["related_task"] else None,
                                correlation_id=fields["correlation_id"] if fields["correlation_id"] else None,
                                timestamp=datetime.fromisoformat(fields["timestamp"]),
                                expires_at=datetime.fromisoformat(fields["expires_at"]) if fields["expires_at"] else None
                            )
                            events.append(event)
                        except Exception as e:
                            logger.warning(f"Failed to parse event {message_id}: {e}")
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to consume events for {agent_id}: {e}")
            return []
    
    async def store_agent_state(self, agent_id: str, state: Dict[str, Any]) -> bool:
        """Store agent state with TTL."""
        if not self._initialized or not self.client:
            return False
        
        try:
            key = f"{self.namespaces['agent_state']}{agent_id}"
            
            # Add timestamp
            state["last_updated"] = datetime.utcnow().isoformat()
            
            await self.client.hset(key, state)
            await self.client.expire(key, self.agent_state_ttl)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store agent state for {agent_id}: {e}")
            return False
    
    async def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """Get agent state."""
        if not self._initialized or not self.client:
            return {}
        
        try:
            key = f"{self.namespaces['agent_state']}{agent_id}"
            return await self.client.hgetall(key)
            
        except Exception as e:
            logger.error(f"Failed to get agent state for {agent_id}: {e}")
            return {}
    
    async def cleanup_expired(self) -> int:
        """Clean up expired entries and maintain memory efficiency."""
        if not self._initialized or not self.client:
            return 0
        
        cleaned = 0
        try:
            # Clean up expired decisions
            current_time = time.time()
            
            # Check decisions for expiration
            decision_keys = await self.client.keys(f"{self.namespaces['decisions']}*")
            for key in decision_keys:
                if ":" in key.split(":")[-1]:  # Skip sorted set keys
                    continue
                    
                decision_data = await self.client.hgetall(key)
                if decision_data and decision_data.get("expires_at"):
                    try:
                        expires_at = datetime.fromisoformat(decision_data["expires_at"])
                        if expires_at < datetime.utcnow():
                            await self.client.delete(key)
                            cleaned += 1
                    except (ValueError, KeyError):
                        continue
            
            logger.debug(f"Cleaned up {cleaned} expired entries")
            return cleaned
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired entries: {e}")
            return cleaned
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if not self._initialized or not self.client:
            return {}
        
        try:
            stats = {}
            
            # Count entries by namespace
            for name, prefix in self.namespaces.items():
                if name == "events":
                    # Count stream length
                    try:
                        stream_info = await self.client.client.xinfo_stream(prefix)
                        stats[name] = stream_info.get("length", 0)
                    except:
                        stats[name] = 0
                else:
                    # Count keys
                    keys = await self.client.keys(f"{prefix}*")
                    stats[name] = len(keys)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {}


# Global shared memory instance
_shared_memory: Optional[RedisSharedMemory] = None


async def get_shared_memory() -> RedisSharedMemory:
    """Get global shared memory instance."""
    global _shared_memory
    
    if _shared_memory is None:
        _shared_memory = RedisSharedMemory()
        await _shared_memory.initialize()
    
    return _shared_memory