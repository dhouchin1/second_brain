"""
Context Broker for Autom8 Shared Memory Architecture.

Manages context slices with full transparency, using Redis for efficient
shared memory and SQLite for durable storage as specified in the PRD.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from autom8.core.memory.embeddings import LocalEmbedder
from autom8.models.context import ContextPackage, ContextPreview, ContextSource, ContextSourceType
from autom8.models.memory import AgentContext, Decision
from autom8.storage.redis.shared_memory import RedisSharedMemory, get_shared_memory
from autom8.storage.sqlite.manager import SQLiteManager, get_sqlite_manager
from autom8.utils.tokens import estimate_tokens

logger = logging.getLogger(__name__)


class ContextBroker:
    """
    Context Broker managing efficient context preparation with Redis + SQLite.
    
    Implements the PRD specification for hybrid retrieval with reference-based
    memory that prevents context bloat.
    """
    
    def __init__(
        self,
        redis_memory: Optional[RedisSharedMemory] = None,
        sqlite_storage: Optional[SQLiteManager] = None,
        embedder: Optional[LocalEmbedder] = None
    ):
        self.redis_memory = redis_memory
        self.sqlite_storage = sqlite_storage
        self.embedder = embedder or LocalEmbedder()
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the context broker with storage backends."""
        try:
            # Initialize Redis shared memory
            if not self.redis_memory:
                self.redis_memory = await get_shared_memory()
            
            # Initialize SQLite storage
            if not self.sqlite_storage:
                self.sqlite_storage = await get_sqlite_manager()
            
            # Initialize embedder
            await self.embedder.initialize()
            
            self._initialized = True
            logger.info("Context broker initialized with Redis and SQLite backends")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize context broker: {e}")
            return False
    
    async def prepare_context(
        self, 
        query: str, 
        agent_id: str,
        max_tokens: int = 500
    ) -> ContextPackage:
        """
        Prepare minimal, relevant context as specified in PRD.
        
        Implementation follows PRD steps:
        1. Get query embedding for semantic search
        2. Semantic retrieval from sqlite-vec
        3. Get agent's working context from Redis
        4. Add critical pinned items
        5. Generate summary if over budget
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        package = ContextPackage(max_tokens=max_tokens)
        
        try:
            # Step 1: Get query embedding for semantic search
            query_embedding = await self.embedder.embed(query)
            
            # Step 2: Semantic retrieval from SQLite
            if query_embedding is not None:
                semantic_ids = await self._semantic_search(query_embedding, k=5)
                
                for doc_id in semantic_ids:
                    content = await self._get_content_by_id(doc_id)
                    if content and self._can_fit_content(package, content):
                        source = ContextSource(
                            type=ContextSourceType.RETRIEVED,
                            content=content,
                            tokens=estimate_tokens(content),
                            source=f"semantic_search:{doc_id}",
                            priority=1
                        )
                        package.add_source(source)
            
            # Step 3: Get agent's working context from Redis
            if self.redis_memory:
                agent_context = await self.redis_memory.get_agent_context(
                    agent_id,
                    max_tokens=package.remaining_tokens()
                )
                
                # Add agent decisions as context sources
                for decision_ref in agent_context.decisions:
                    decision = await self.redis_memory.get_decision(decision_ref.key.split(":")[-1])
                    if decision:
                        source = ContextSource(
                            type=ContextSourceType.MEMORY,
                            content=decision.summary,
                            tokens=estimate_tokens(decision.summary),
                            source=f"decision:{decision.id}",
                            priority=2,
                            expandable=True
                        )
                        if self._can_fit_source(package, source):
                            package.add_source(source)
            
            # Step 4: Add critical pinned items
            pinned_items = await self._get_pinned_content(
                limit_tokens=package.remaining_tokens()
            )
            
            for item in pinned_items:
                source = ContextSource(
                    type=ContextSourceType.PINNED,
                    content=item["content"],
                    tokens=item["tokens"],
                    source=f"pinned:{item['id']}",
                    priority=3
                )
                if self._can_fit_source(package, source):
                    package.add_source(source)
            
            # Step 5: Generate summary if over budget
            if package.is_over_budget():
                summary = await self._generate_summary(package.get_all_content())
                if summary:
                    # Replace content with summary
                    summary_source = ContextSource(
                        type=ContextSourceType.SUMMARY,
                        content=summary,
                        tokens=estimate_tokens(summary),
                        source="auto_generated_summary",
                        priority=0
                    )
                    package.clear_sources()
                    package.add_source(summary_source)
            
            # Add query as primary source
            query_source = ContextSource(
                type=ContextSourceType.QUERY,
                content=query,
                tokens=estimate_tokens(query),
                source="user_query",
                priority=0
            )
            package.add_source(query_source)
            
            # Performance tracking
            processing_time = time.time() - start_time
            logger.debug(f"Context prepared for {agent_id}: {package.total_tokens}/{max_tokens} tokens in {processing_time:.3f}s")
            
            return package
            
        except Exception as e:
            logger.error(f"Failed to prepare context for agent {agent_id}: {e}")
            # Return minimal context with just the query
            fallback_package = ContextPackage(max_tokens=max_tokens)
            query_source = ContextSource(
                type=ContextSourceType.QUERY,
                content=query,
                tokens=estimate_tokens(query),
                source="user_query",
                priority=0
            )
            fallback_package.add_source(query_source)
            return fallback_package
    
    async def store_context_with_embedding(
        self,
        context_id: str,
        content: str,
        summary: Optional[str] = None,
        topic: Optional[str] = None,
        priority: int = 0,
        pinned: bool = False,
        expires_at: Optional[datetime] = None,
        source_type: str = "user_provided",
        metadata: Optional[Dict] = None
    ) -> bool:
        """Store context content with automatic embedding generation."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Generate embedding for the content
            embedding = await self.embedder.embed(content)
            
            if embedding is not None and self.sqlite_storage:
                # Store context with embedding
                success = await self.sqlite_storage.store_context_with_embedding(
                    context_id=context_id,
                    content=content,
                    embedding=embedding,
                    summary=summary,
                    topic=topic,
                    priority=priority,
                    pinned=pinned,
                    expires_at=expires_at,
                    source_type=source_type,
                    metadata=metadata,
                    model_name=self.embedder.model_name
                )
                
                if success:
                    logger.debug(f"Stored context with embedding: {context_id}")
                    return True
            
            # Fallback: store without embedding
            elif self.sqlite_storage:
                return await self.sqlite_storage.store_context(
                    context_id, content, summary, topic, priority,
                    pinned, expires_at, source_type, metadata
                )
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to store context with embedding {context_id}: {e}")
            return False
    
    async def find_similar_content(
        self,
        query_text: str,
        k: int = 5,
        threshold: float = 0.7,
        topic: Optional[str] = None
    ) -> List[Dict]:
        """Find content similar to the given query text."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Generate embedding for query
            query_embedding = await self.embedder.embed(query_text)
            
            if query_embedding is not None and self.sqlite_storage:
                # Perform semantic search
                results = await self.sqlite_storage.semantic_search(
                    query_embedding=query_embedding,
                    k=k,
                    threshold=threshold,
                    topic=topic
                )
                
                return results
            
            # Fallback to text search
            elif self.sqlite_storage:
                text_results = await self.sqlite_storage.search_context(
                    query=query_text,
                    topic=topic,
                    limit=k
                )
                return text_results
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to find similar content: {e}")
            return []
    
    async def rebuild_all_embeddings(self) -> Dict[str, Any]:
        """Rebuild embeddings for all stored content."""
        if not self._initialized:
            await self.initialize()
        
        try:
            if not self.sqlite_storage:
                return {"error": "SQLite storage not available"}
            
            # Get current stats
            vector_stats_before = await self.sqlite_storage.get_vector_stats()
            
            # Rebuild embeddings
            rebuilt_count = await self.sqlite_storage.rebuild_embeddings(
                model_name=self.embedder.model_name
            )
            
            # Get updated stats
            vector_stats_after = await self.sqlite_storage.get_vector_stats()
            
            return {
                "rebuilt_count": rebuilt_count,
                "before": vector_stats_before,
                "after": vector_stats_after,
                "success": rebuilt_count > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to rebuild embeddings: {e}")
            return {"error": str(e), "success": False}
    
    async def store_context_result(
        self,
        agent_id: str,
        query: str,
        context_package: ContextPackage,
        response: str,
        success: bool = True
    ) -> bool:
        """Store context usage result for learning and optimization."""
        try:
            if self.sqlite_storage:
                await self.sqlite_storage.record_context_usage(
                    agent_id=agent_id,
                    query=query,
                    context_tokens=context_package.total_tokens,
                    response_tokens=estimate_tokens(response),
                    success=success,
                    sources_used=len(context_package.sources)
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store context result: {e}")
            return False
    
    async def _semantic_search(self, query_embedding, k: int = 5) -> List[str]:
        """Perform semantic search using embeddings."""
        try:
            if self.sqlite_storage:
                # Use the enhanced semantic search with sqlite-vec
                results = await self.sqlite_storage.semantic_search(
                    query_embedding=query_embedding,
                    k=k,
                    threshold=0.5  # Minimum similarity threshold
                )
                return [result['id'] for result in results]
            return []
            
        except Exception as e:
            logger.warning(f"Semantic search failed, using fallback: {e}")
            # Fallback to text search
            if self.sqlite_storage:
                return await self.sqlite_storage.search_context_ids(limit=k)
            return []
    
    async def _get_content_by_id(self, doc_id: str) -> Optional[str]:
        """Get content by document ID from SQLite."""
        try:
            if self.sqlite_storage:
                content = await self.sqlite_storage.get_context_by_id(doc_id)
                return content.get("content") if content else None
            return None
            
        except Exception as e:
            logger.error(f"Failed to get content {doc_id}: {e}")
            return None
    
    async def _get_pinned_content(self, limit_tokens: int) -> List[Dict]:
        """Get pinned/priority content that should always be included."""
        try:
            if self.sqlite_storage:
                return await self.sqlite_storage.get_pinned_context(token_limit=limit_tokens)
            return []
            
        except Exception as e:
            logger.error(f"Failed to get pinned content: {e}")
            return []
    
    async def _generate_summary(self, content: str) -> Optional[str]:
        """
        Generate compressed summary using local model.
        
        Uses fast local model for summarization as specified in PRD.
        """
        try:
            # In a full implementation, this would use a local summarization model
            # For now, return a simple truncated version
            if len(content) > 500:
                return content[:400] + "... [content summarized for token efficiency]"
            return content
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return None
    
    def _can_fit_content(self, package: ContextPackage, content: str) -> bool:
        """Check if content fits in remaining token budget."""
        tokens = estimate_tokens(content)
        return package.remaining_tokens() >= tokens
    
    def _can_fit_source(self, package: ContextPackage, source: ContextSource) -> bool:
        """Check if source fits in remaining token budget."""
        return package.remaining_tokens() >= source.tokens
    
    async def get_context_stats(self) -> Dict:
        """Get context broker usage statistics."""
        try:
            stats = {
                "redis_available": self.redis_memory and self.redis_memory._initialized,
                "sqlite_available": self.sqlite_storage and self.sqlite_storage._initialized,
                "embedder_available": self.embedder.is_available(),
            }
            
            # Get memory stats from Redis
            if self.redis_memory and self.redis_memory._initialized:
                memory_stats = await self.redis_memory.get_memory_stats()
                stats["memory_usage"] = memory_stats
            
            # Get storage stats from SQLite
            if self.sqlite_storage and self.sqlite_storage._initialized:
                storage_stats = await self.sqlite_storage.get_storage_stats()
                stats["storage_usage"] = storage_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get context stats: {e}")
            return {}
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.redis_memory:
                await self.redis_memory.cleanup_expired()
            
            if self.sqlite_storage:
                await self.sqlite_storage.cleanup_old_records()
            
        except Exception as e:
            logger.error(f"Context broker cleanup failed: {e}")


class ContextPackage:
    """
    Context package for managing token budgets and source optimization.
    
    Internal helper class for the Context Broker.
    """
    
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.sources: List[ContextSource] = []
        self._total_tokens = 0
    
    @property
    def total_tokens(self) -> int:
        """Calculate total tokens from all sources."""
        return sum(source.tokens for source in self.sources)
    
    def remaining_tokens(self) -> int:
        """Get remaining token budget."""
        return max(0, self.max_tokens - self.total_tokens)
    
    def is_over_budget(self) -> bool:
        """Check if package exceeds token budget."""
        return self.total_tokens > self.max_tokens
    
    def add_source(self, source: ContextSource) -> bool:
        """Add source if budget allows."""
        if self.remaining_tokens() >= source.tokens:
            self.sources.append(source)
            return True
        return False
    
    def clear_sources(self) -> None:
        """Clear all sources."""
        self.sources.clear()
    
    def get_all_content(self) -> str:
        """Get concatenated content from all sources."""
        return "\n".join(source.content for source in self.sources)


# Global context broker instance
_context_broker: Optional[ContextBroker] = None


async def get_context_broker() -> ContextBroker:
    """Get global context broker instance."""
    global _context_broker
    
    if _context_broker is None:
        _context_broker = ContextBroker()
        await _context_broker.initialize()
    
    return _context_broker