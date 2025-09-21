"""
Data Service for Autom8 Dashboard API

Provides data aggregation and transformation services for the dashboard API,
integrating with Redis shared memory, EventBus, and other backend services.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter

from autom8.storage.redis.shared_memory import RedisSharedMemory, get_shared_memory
from autom8.storage.redis.events import EventBus, get_event_bus
from autom8.models.memory import AgentEvent, EventType, Priority, DecisionType
from autom8.config.settings import get_settings

logger = logging.getLogger(__name__)


class DashboardDataService:
    """Service for aggregating and transforming dashboard data."""
    
    def __init__(self, shared_memory: Optional[RedisSharedMemory] = None, event_bus: Optional[EventBus] = None):
        self.shared_memory = shared_memory
        self.event_bus = event_bus
        self._cache = {}
        self._cache_ttl = {}
        self._cache_timeout = 30  # Cache for 30 seconds
    
    async def initialize(self):
        """Initialize the data service."""
        if not self.shared_memory:
            self.shared_memory = await get_shared_memory()
        if not self.event_bus:
            self.event_bus = await get_event_bus()
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is still valid."""
        if key not in self._cache_ttl:
            return False
        return datetime.utcnow() < self._cache_ttl[key]
    
    def _cache_set(self, key: str, value: Any):
        """Set cache entry with TTL."""
        self._cache[key] = value
        self._cache_ttl[key] = datetime.utcnow() + timedelta(seconds=self._cache_timeout)
    
    def _cache_get(self, key: str) -> Optional[Any]:
        """Get cached value if valid."""
        if self._is_cache_valid(key):
            return self._cache[key]
        return None
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        cache_key = "system_metrics"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "redis_status": False,
            "memory_stats": {},
            "event_counts": {},
            "performance": {},
            "errors": []
        }
        
        try:
            if self.shared_memory and self.shared_memory._initialized:
                metrics["redis_status"] = True
                
                # Get memory statistics
                memory_stats = await self.shared_memory.get_memory_stats()
                metrics["memory_stats"] = memory_stats
                
                # Get recent events for analysis
                if self.event_bus:
                    recent_events = await self.event_bus.get_events(
                        start_time=datetime.utcnow() - timedelta(hours=1),
                        limit=1000
                    )
                    
                    # Analyze event types
                    event_counter = Counter(event.type for event in recent_events)
                    metrics["event_counts"] = dict(event_counter)
                    
                    # Calculate performance metrics
                    metrics["performance"] = {
                        "events_per_hour": len(recent_events),
                        "avg_event_size": sum(len(str(event.data)) for event in recent_events) / max(len(recent_events), 1),
                        "priority_distribution": dict(Counter(event.priority for event in recent_events))
                    }
                
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            metrics["errors"].append(str(e))
        
        self._cache_set(cache_key, metrics)
        return metrics
    
    async def get_agent_details(self) -> List[Dict[str, Any]]:
        """Get detailed information about all agents."""
        cache_key = "agent_details"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        
        agents = []
        
        if not self.shared_memory or not self.shared_memory._initialized:
            return agents
        
        try:
            # Get all agent state keys
            agent_keys = await self.shared_memory.client.keys(f"{self.shared_memory.namespaces['agent_state']}*")
            
            for key in agent_keys:
                agent_id = key.replace(self.shared_memory.namespaces['agent_state'], '')
                agent_state = await self.shared_memory.get_agent_state(agent_id)
                
                if agent_state:
                    # Get recent decisions for this agent
                    recent_decisions = await self._get_agent_recent_decisions(agent_id)
                    
                    agent_info = {
                        "id": agent_id,
                        "name": agent_state.get("name", agent_id),
                        "status": agent_state.get("status", "unknown"),
                        "last_active": agent_state.get("last_updated"),
                        "current_work": agent_state.get("current_work"),
                        "performance_score": float(agent_state.get("performance_score", 0.0)),
                        "tasks_completed": int(agent_state.get("tasks_completed", 0)),
                        "error_count": int(agent_state.get("error_count", 0)),
                        "recent_decisions": len(recent_decisions),
                        "decision_types": list(set(d.get("decision_type", "unknown") for d in recent_decisions))
                    }
                    agents.append(agent_info)
        
        except Exception as e:
            logger.error(f"Error getting agent details: {e}")
        
        self._cache_set(cache_key, agents)
        return agents
    
    async def _get_agent_recent_decisions(self, agent_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent decisions for a specific agent."""
        if not self.shared_memory:
            return []
        
        try:
            # Get recent decision IDs for this agent
            decision_ids = await self.shared_memory.client.zrevrange(
                f"{self.shared_memory.namespaces['decisions']}agent:{agent_id}",
                0, 50  # Last 50 decisions
            )
            
            decisions = []
            for decision_id in decision_ids:
                decision = await self.shared_memory.get_decision(decision_id)
                if decision:
                    # Check if decision is within time range
                    if decision.timestamp > datetime.utcnow() - timedelta(hours=hours):
                        decisions.append({
                            "id": decision.id,
                            "summary": decision.summary,
                            "decision_type": decision.decision_type.value,
                            "status": decision.status.value,
                            "confidence": decision.confidence,
                            "timestamp": decision.timestamp.isoformat(),
                            "tags": decision.tags
                        })
            
            return decisions
        
        except Exception as e:
            logger.error(f"Error getting recent decisions for agent {agent_id}: {e}")
            return []
    
    async def get_model_performance_data(self) -> Dict[str, Any]:
        """Get model performance and usage statistics."""
        cache_key = "model_performance"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        
        # Placeholder implementation - in a real system, this would query
        # model performance tracking systems
        performance_data = {
            "models": [
                {
                    "name": "llama3.1:8b",
                    "provider": "ollama",
                    "available": True,
                    "requests_count": 150,
                    "avg_response_time": 2.3,
                    "success_rate": 98.5,
                    "cost_per_request": 0.0,
                    "last_used": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
                    "queue_length": 0,
                    "error_rate": 1.5
                },
                {
                    "name": "claude-3-sonnet",
                    "provider": "anthropic",
                    "available": True,
                    "requests_count": 45,
                    "avg_response_time": 1.8,
                    "success_rate": 99.2,
                    "cost_per_request": 0.003,
                    "last_used": (datetime.utcnow() - timedelta(minutes=15)).isoformat(),
                    "queue_length": 2,
                    "error_rate": 0.8
                },
                {
                    "name": "gpt-4",
                    "provider": "openai",
                    "available": False,
                    "requests_count": 12,
                    "avg_response_time": 3.1,
                    "success_rate": 97.8,
                    "cost_per_request": 0.03,
                    "last_used": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                    "queue_length": 0,
                    "error_rate": 2.2
                }
            ],
            "usage_trends": self._generate_usage_trends(),
            "performance_trends": self._generate_performance_trends()
        }
        
        self._cache_set(cache_key, performance_data)
        return performance_data
    
    def _generate_usage_trends(self) -> List[Dict[str, Any]]:
        """Generate mock usage trend data."""
        trends = []
        now = datetime.utcnow()
        
        for i in range(24):  # Last 24 hours
            timestamp = now - timedelta(hours=23-i)
            trends.append({
                "timestamp": timestamp.isoformat(),
                "total_requests": max(10 + i * 2, 0),
                "local_requests": max(8 + i * 1.5, 0),
                "cloud_requests": max(2 + i * 0.5, 0)
            })
        
        return trends
    
    def _generate_performance_trends(self) -> List[Dict[str, Any]]:
        """Generate mock performance trend data."""
        trends = []
        now = datetime.utcnow()
        
        for i in range(12):  # Last 12 hours
            timestamp = now - timedelta(hours=11-i)
            trends.append({
                "timestamp": timestamp.isoformat(),
                "avg_response_time": 2.0 + (i % 3) * 0.5,
                "success_rate": 98.0 + (i % 2),
                "error_rate": 2.0 - (i % 2)
            })
        
        return trends
    
    async def get_complexity_analysis_data(self) -> Dict[str, Any]:
        """Get complexity analysis statistics and trends."""
        cache_key = "complexity_analysis"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        
        # In a real implementation, this would query complexity analysis results
        # from shared memory or a dedicated analytics store
        complexity_data = {
            "total_analyses": 500,
            "avg_complexity_score": 0.65,
            "distribution": {
                "trivial": 120,
                "simple": 180,
                "moderate": 150,
                "complex": 40,
                "frontier": 10
            },
            "recent_analyses": self._generate_recent_analyses(),
            "trend_data": self._generate_complexity_trends(),
            "last_analysis": (datetime.utcnow() - timedelta(minutes=2)).isoformat()
        }
        
        self._cache_set(cache_key, complexity_data)
        return complexity_data
    
    def _generate_recent_analyses(self) -> List[Dict[str, Any]]:
        """Generate mock recent complexity analyses."""
        analyses = []
        now = datetime.utcnow()
        
        complexity_levels = ["trivial", "simple", "moderate", "complex", "frontier"]
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for i in range(20):
            level_idx = i % len(complexity_levels)
            analyses.append({
                "id": f"analysis_{i}",
                "timestamp": (now - timedelta(minutes=i*5)).isoformat(),
                "complexity_level": complexity_levels[level_idx],
                "score": scores[level_idx] + (i % 3) * 0.05,
                "task_type": f"task_type_{i % 5}",
                "agent_id": f"agent_{i % 3}"
            })
        
        return analyses
    
    def _generate_complexity_trends(self) -> List[Dict[str, Any]]:
        """Generate mock complexity trend data."""
        trends = []
        now = datetime.utcnow()
        
        for i in range(24):  # Last 24 hours
            timestamp = now - timedelta(hours=23-i)
            trends.append({
                "timestamp": timestamp.isoformat(),
                "avg_complexity": 0.6 + (i % 5) * 0.05,
                "count": 20 + i,
                "distribution": {
                    "trivial": 5 + i % 3,
                    "simple": 8 + i % 4,
                    "moderate": 6 + i % 3,
                    "complex": 1 + i % 2,
                    "frontier": i % 2
                }
            })
        
        return trends
    
    async def get_routing_statistics(self) -> Dict[str, Any]:
        """Get model routing statistics and performance."""
        cache_key = "routing_statistics"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        
        routing_data = {
            "total_routes": 500,
            "local_routes": 420,
            "cloud_routes": 80,
            "avg_routing_time": 0.05,
            "routing_accuracy": 94.2,
            "model_distribution": {
                "llama3.1:8b": 380,
                "claude-3-sonnet": 70,
                "gpt-4": 50
            },
            "routing_trends": self._generate_routing_trends(),
            "decision_factors": {
                "complexity_based": 350,
                "availability_based": 100,
                "cost_based": 50
            },
            "fallback_events": 12
        }
        
        self._cache_set(cache_key, routing_data)
        return routing_data
    
    def _generate_routing_trends(self) -> List[Dict[str, Any]]:
        """Generate mock routing trend data."""
        trends = []
        now = datetime.utcnow()
        
        for i in range(24):  # Last 24 hours
            timestamp = now - timedelta(hours=23-i)
            trends.append({
                "timestamp": timestamp.isoformat(),
                "total_routes": 15 + i,
                "local_routes": 12 + i,
                "cloud_routes": 3,
                "avg_routing_time": 0.05 + (i % 3) * 0.01,
                "accuracy": 94.0 + (i % 5)
            })
        
        return trends
    
    async def get_context_optimization_data(self) -> Dict[str, Any]:
        """Get context usage and optimization metrics."""
        cache_key = "context_optimization"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        
        context_data = {
            "total_contexts": 750,
            "avg_context_size": 450.5,
            "optimization_savings": 125000.0,
            "cache_hit_rate": 78.5,
            "memory_efficiency": 85.2,
            "context_trends": self._generate_context_trends(),
            "optimization_techniques": {
                "reference_based": 65,
                "compression": 20,
                "pruning": 15
            },
            "size_distribution": {
                "small": 300,
                "medium": 350,
                "large": 100
            }
        }
        
        self._cache_set(cache_key, context_data)
        return context_data
    
    def _generate_context_trends(self) -> List[Dict[str, Any]]:
        """Generate mock context optimization trend data."""
        trends = []
        now = datetime.utcnow()
        
        for i in range(24):  # Last 24 hours
            timestamp = now - timedelta(hours=23-i)
            trends.append({
                "timestamp": timestamp.isoformat(),
                "contexts_created": 25 + i,
                "avg_size": 450 + (i % 10) * 20,
                "optimization_rate": 80 + (i % 15),
                "cache_hits": 18 + i % 8
            })
        
        return trends
    
    async def get_recent_events(self, limit: int = 50, event_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get recent events for real-time updates."""
        if not self.event_bus:
            return []
        
        try:
            events = await self.event_bus.get_events(
                start_time=datetime.utcnow() - timedelta(hours=1),
                event_types=event_types,
                limit=limit
            )
            
            return [
                {
                    "id": event.id,
                    "type": event.type,
                    "source_agent": event.source_agent,
                    "target_agent": event.target_agent,
                    "summary": event.summary,
                    "priority": event.priority,
                    "timestamp": event.timestamp.isoformat(),
                    "data_size": len(str(event.data))
                }
                for event in events
            ]
        
        except Exception as e:
            logger.error(f"Error getting recent events: {e}")
            return []
    
    async def cleanup_cache(self):
        """Clean up expired cache entries."""
        now = datetime.utcnow()
        expired_keys = [
            key for key, expiry in self._cache_ttl.items()
            if now > expiry
        ]
        
        for key in expired_keys:
            self._cache.pop(key, None)
            self._cache_ttl.pop(key, None)
        
        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")


# Global service instance
_data_service: Optional[DashboardDataService] = None


async def get_data_service() -> DashboardDataService:
    """Get global data service instance."""
    global _data_service
    
    if _data_service is None:
        _data_service = DashboardDataService()
        await _data_service.initialize()
    
    return _data_service