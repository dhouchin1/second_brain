"""
Advanced Performance Optimization System for Autom8

Provides multi-layer caching, async processing pipelines, intelligent resource
management, and performance analytics for optimal system performance.
"""

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, Set, Tuple, TypeVar, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
import redis.asyncio as redis

from autom8.core.complexity.analyzer import ComplexityAnalyzer, ComplexityScore
from autom8.core.context.inspector import ContextInspector, ContextPreview
from autom8.core.routing.router import ModelRouter, ModelSelection
from autom8.models.routing import RoutingPreferences
from autom8.storage.redis.client import RedisClient
from autom8.storage.sqlite.manager import SQLiteManager
from autom8.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class CacheKey(BaseModel):
    """Structured cache key with metadata."""
    namespace: str = Field(description="Cache namespace (e.g., complexity, context, routing)")
    identifier: str = Field(description="Unique identifier within namespace")
    version: str = Field(default="v1", description="Cache version for invalidation")
    ttl_seconds: Optional[int] = Field(default=None, description="Time to live in seconds")
    
    def __str__(self) -> str:
        return f"{self.namespace}:{self.version}:{self.identifier}"


class CacheMetrics(BaseModel):
    """Cache performance metrics."""
    namespace: str
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    l1_hits: int = 0  # Redis hits
    l2_hits: int = 0  # SQLite hits
    avg_fetch_time_ms: float = 0.0
    total_size_bytes: int = 0
    evictions: int = 0
    
    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    @property
    def l1_hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.l1_hits / self.total_requests


class CacheBackend(Protocol[T]):
    """Protocol for cache backend implementations."""
    
    async def get(self, key: str) -> Optional[T]:
        """Get value by key."""
        ...
    
    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> bool:
        """Set value with optional TTL."""
        ...
    
    async def delete(self, key: str) -> bool:
        """Delete value by key."""
        ...
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        ...
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        ...
    
    async def get_size(self) -> int:
        """Get total cache size in bytes."""
        ...


class RedisCache(Generic[T]):
    """Redis-based cache backend for fast L1 caching."""
    
    def __init__(self, redis_client: RedisClient, serializer: Optional[Callable] = None):
        self.redis_client = redis_client
        self.serializer = serializer or self._default_serializer
        self.deserializer = self._default_deserializer
    
    def _default_serializer(self, value: T) -> bytes:
        """Default JSON serialization."""
        import json
        if hasattr(value, 'json'):
            return value.json().encode()
        return json.dumps(value, default=str).encode()
    
    def _default_deserializer(self, data: bytes) -> T:
        """Default JSON deserialization."""
        import json
        return json.loads(data.decode())
    
    async def get(self, key: str) -> Optional[T]:
        try:
            data = await self.redis_client.get(key)
            if data:
                return self.deserializer(data)
            return None
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> bool:
        try:
            data = self.serializer(value)
            if ttl:
                return await self.redis_client.setex(key, ttl, data)
            else:
                return await self.redis_client.set(key, data)
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        try:
            result = await self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        try:
            return await self.redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis exists error for key {key}: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                return await self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Redis clear pattern error for {pattern}: {e}")
            return 0
    
    async def get_size(self) -> int:
        try:
            info = await self.redis_client.info('memory')
            return int(info.get('used_memory', 0))
        except Exception as e:
            logger.error(f"Redis size error: {e}")
            return 0


class SQLiteCache(Generic[T]):
    """SQLite-based cache backend for persistent L2 caching."""
    
    def __init__(self, sqlite_manager: SQLiteManager, table_name: str = "cache_data"):
        self.sqlite_manager = sqlite_manager
        self.table_name = table_name
        self._initialized = False
    
    async def initialize(self):
        """Initialize cache table."""
        if not self._initialized:
            await self._create_cache_table()
            self._initialized = True
    
    async def _create_cache_table(self):
        """Create cache table if it doesn't exist."""
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            key TEXT PRIMARY KEY,
            value BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            access_count INTEGER DEFAULT 1,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            size_bytes INTEGER DEFAULT 0
        )
        """
        await self.sqlite_manager.execute_query(create_sql)
        
        # Create indexes
        await self.sqlite_manager.execute_query(
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_expires ON {self.table_name}(expires_at)"
        )
    
    def _serialize(self, value: T) -> bytes:
        """Serialize value for storage."""
        import pickle
        return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> T:
        """Deserialize value from storage."""
        import pickle
        return pickle.loads(data)
    
    async def get(self, key: str) -> Optional[T]:
        try:
            now = datetime.utcnow()
            
            # Get value and check expiration
            query = f"""
            SELECT value FROM {self.table_name} 
            WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)
            """
            result = await self.sqlite_manager.fetch_one(query, (key, now))
            
            if result:
                # Update access statistics
                update_sql = f"""
                UPDATE {self.table_name} 
                SET access_count = access_count + 1, last_accessed = ?
                WHERE key = ?
                """
                await self.sqlite_manager.execute_query(update_sql, (now, key))
                
                return self._deserialize(result[0])
            
            return None
            
        except Exception as e:
            logger.error(f"SQLite get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> bool:
        try:
            data = self._serialize(value)
            size_bytes = len(data)
            now = datetime.utcnow()
            expires_at = now + timedelta(seconds=ttl) if ttl else None
            
            # Upsert value
            upsert_sql = f"""
            INSERT OR REPLACE INTO {self.table_name} 
            (key, value, created_at, expires_at, size_bytes) 
            VALUES (?, ?, ?, ?, ?)
            """
            await self.sqlite_manager.execute_query(
                upsert_sql, (key, data, now, expires_at, size_bytes)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"SQLite set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        try:
            delete_sql = f"DELETE FROM {self.table_name} WHERE key = ?"
            result = await self.sqlite_manager.execute_query(delete_sql, (key,))
            return result.rowcount > 0 if result else False
        except Exception as e:
            logger.error(f"SQLite delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        try:
            now = datetime.utcnow()
            query = f"""
            SELECT 1 FROM {self.table_name} 
            WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)
            """
            result = await self.sqlite_manager.fetch_one(query, (key, now))
            return result is not None
        except Exception as e:
            logger.error(f"SQLite exists error for key {key}: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        try:
            # Convert glob pattern to SQL LIKE pattern
            like_pattern = pattern.replace('*', '%').replace('?', '_')
            delete_sql = f"DELETE FROM {self.table_name} WHERE key LIKE ?"
            result = await self.sqlite_manager.execute_query(delete_sql, (like_pattern,))
            return result.rowcount if result else 0
        except Exception as e:
            logger.error(f"SQLite clear pattern error for {pattern}: {e}")
            return 0
    
    async def get_size(self) -> int:
        try:
            query = f"SELECT SUM(size_bytes) FROM {self.table_name}"
            result = await self.sqlite_manager.fetch_one(query)
            return int(result[0]) if result and result[0] else 0
        except Exception as e:
            logger.error(f"SQLite size error: {e}")
            return 0
    
    async def cleanup_expired(self) -> int:
        """Clean up expired cache entries."""
        try:
            now = datetime.utcnow()
            delete_sql = f"DELETE FROM {self.table_name} WHERE expires_at IS NOT NULL AND expires_at <= ?"
            result = await self.sqlite_manager.execute_query(delete_sql, (now,))
            cleaned = result.rowcount if result else 0
            
            if cleaned > 0:
                logger.info(f"Cleaned {cleaned} expired cache entries")
            
            return cleaned
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
            return 0


class MultiLayerCache(Generic[T]):
    """
    Sophisticated multi-layer caching system with L1 (Redis) and L2 (SQLite) tiers.
    
    Provides intelligent cache population, eviction policies, and performance analytics.
    """
    
    def __init__(
        self,
        namespace: str,
        l1_backend: CacheBackend[T],
        l2_backend: CacheBackend[T],
        default_ttl: int = 3600,
        max_l1_size: int = 100_000_000,  # 100MB
        max_l2_size: int = 1_000_000_000,  # 1GB
    ):
        self.namespace = namespace
        self.l1_backend = l1_backend
        self.l2_backend = l2_backend
        self.default_ttl = default_ttl
        self.max_l1_size = max_l1_size
        self.max_l2_size = max_l2_size
        
        # Metrics tracking
        self.metrics = CacheMetrics(namespace=namespace)
        self.access_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
    
    def start_background_cleanup(self):
        """Start background cleanup task."""
        if not self._cleanup_task or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._background_cleanup())
    
    async def _background_cleanup(self):
        """Background task for cache maintenance."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Clean up expired L2 entries
                if hasattr(self.l2_backend, 'cleanup_expired'):
                    await self.l2_backend.cleanup_expired()
                
                # Check size limits and evict if necessary
                await self._enforce_size_limits()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def get(self, key: CacheKey) -> Optional[T]:
        """
        Get value from cache with intelligent L1/L2 hierarchy.
        
        Args:
            key: Cache key with metadata
            
        Returns:
            Cached value or None if not found
        """
        start_time = time.perf_counter()
        key_str = str(key)
        
        self.metrics.total_requests += 1
        
        try:
            # Try L1 cache first (Redis - fastest)
            value = await self.l1_backend.get(key_str)
            if value is not None:
                self.metrics.cache_hits += 1
                self.metrics.l1_hits += 1
                self._record_access_pattern(key_str, 'l1_hit')
                
                fetch_time = (time.perf_counter() - start_time) * 1000
                self._update_avg_fetch_time(fetch_time)
                
                logger.debug(f"L1 cache hit for {key.namespace}:{key.identifier}")
                return value
            
            # Try L2 cache (SQLite - persistent)
            value = await self.l2_backend.get(key_str)
            if value is not None:
                self.metrics.cache_hits += 1
                self.metrics.l2_hits += 1
                self._record_access_pattern(key_str, 'l2_hit')
                
                # Promote to L1 cache for future access
                asyncio.create_task(
                    self.l1_backend.set(key_str, value, key.ttl_seconds or self.default_ttl)
                )
                
                fetch_time = (time.perf_counter() - start_time) * 1000
                self._update_avg_fetch_time(fetch_time)
                
                logger.debug(f"L2 cache hit for {key.namespace}:{key.identifier}")
                return value
            
            # Cache miss
            self.metrics.cache_misses += 1
            self._record_access_pattern(key_str, 'miss')
            
            logger.debug(f"Cache miss for {key.namespace}:{key.identifier}")
            return None
            
        except Exception as e:
            logger.error(f"Cache get error for {key_str}: {e}")
            self.metrics.cache_misses += 1
            return None
    
    async def set(
        self,
        key: CacheKey,
        value: T,
        ttl: Optional[int] = None,
        populate_l1: bool = True
    ) -> bool:
        """
        Set value in cache with intelligent tier population.
        
        Args:
            key: Cache key with metadata
            value: Value to cache
            ttl: Time to live in seconds (overrides key.ttl_seconds)
            populate_l1: Whether to populate L1 cache immediately
            
        Returns:
            True if successful
        """
        key_str = str(key)
        cache_ttl = ttl or key.ttl_seconds or self.default_ttl
        
        try:
            # Always store in L2 for persistence
            l2_success = await self.l2_backend.set(key_str, value, cache_ttl)
            
            # Optionally store in L1 for speed
            l1_success = True
            if populate_l1:
                l1_success = await self.l1_backend.set(key_str, value, cache_ttl)
            
            if l2_success:
                self._record_access_pattern(key_str, 'set')
                logger.debug(f"Cache set successful for {key.namespace}:{key.identifier}")
            
            return l2_success and l1_success
            
        except Exception as e:
            logger.error(f"Cache set error for {key_str}: {e}")
            return False
    
    async def delete(self, key: CacheKey) -> bool:
        """Delete value from both cache tiers."""
        key_str = str(key)
        
        try:
            l1_result = await self.l1_backend.delete(key_str)
            l2_result = await self.l2_backend.delete(key_str)
            
            success = l1_result or l2_result
            if success:
                self._record_access_pattern(key_str, 'delete')
                logger.debug(f"Cache delete for {key.namespace}:{key.identifier}")
            
            return success
            
        except Exception as e:
            logger.error(f"Cache delete error for {key_str}: {e}")
            return False
    
    async def clear_namespace(self, namespace: Optional[str] = None) -> int:
        """Clear all entries in namespace."""
        target_namespace = namespace or self.namespace
        pattern = f"{target_namespace}:*"
        
        try:
            l1_cleared = await self.l1_backend.clear_pattern(pattern)
            l2_cleared = await self.l2_backend.clear_pattern(pattern)
            
            total_cleared = l1_cleared + l2_cleared
            logger.info(f"Cleared {total_cleared} entries from namespace {target_namespace}")
            
            return total_cleared
            
        except Exception as e:
            logger.error(f"Cache clear error for namespace {target_namespace}: {e}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            l1_size = await self.l1_backend.get_size()
            l2_size = await self.l2_backend.get_size()
            
            # Recent access patterns
            recent_patterns = {}
            for key, accesses in self.access_patterns.items():
                if accesses:
                    recent_patterns[key] = {
                        'access_count': len(accesses),
                        'last_access_type': accesses[-1][0] if accesses else None,
                        'last_access_time': accesses[-1][1] if accesses else None
                    }
            
            return {
                'namespace': self.namespace,
                'metrics': self.metrics.dict(),
                'l1_size_bytes': l1_size,
                'l2_size_bytes': l2_size,
                'total_size_bytes': l1_size + l2_size,
                'recent_access_patterns': recent_patterns,
                'size_limits': {
                    'l1_max': self.max_l1_size,
                    'l2_max': self.max_l2_size
                }
            }
            
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {}
    
    def _record_access_pattern(self, key: str, access_type: str):
        """Record access pattern for cache optimization."""
        self.access_patterns[key].append((access_type, datetime.utcnow()))
    
    def _update_avg_fetch_time(self, fetch_time: float):
        """Update average fetch time with exponential moving average."""
        alpha = 0.1  # Learning rate
        self.metrics.avg_fetch_time_ms = (
            (1 - alpha) * self.metrics.avg_fetch_time_ms + alpha * fetch_time
        )
    
    async def _enforce_size_limits(self):
        """Enforce cache size limits with intelligent eviction."""
        try:
            l1_size = await self.l1_backend.get_size()
            l2_size = await self.l2_backend.get_size()
            
            # Evict from L1 if over limit
            if l1_size > self.max_l1_size:
                # This would implement LRU or other eviction strategy
                logger.info(f"L1 cache size limit exceeded ({l1_size} > {self.max_l1_size})")
                # Implementation would evict least recently used items
            
            # Evict from L2 if over limit
            if l2_size > self.max_l2_size:
                logger.info(f"L2 cache size limit exceeded ({l2_size} > {self.max_l2_size})")
                # Implementation would evict oldest or least accessed items
            
        except Exception as e:
            logger.error(f"Size enforcement error: {e}")


class CachedComplexityAnalyzer:
    """Complexity analyzer with intelligent caching."""
    
    def __init__(
        self,
        analyzer: ComplexityAnalyzer,
        cache: MultiLayerCache[ComplexityScore]
    ):
        self.analyzer = analyzer
        self.cache = cache
    
    def _generate_cache_key(self, query: str, context: Optional[Dict] = None) -> CacheKey:
        """Generate cache key from query content and context."""
        # Create content hash for cache key
        content = query + str(context or {})
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        return CacheKey(
            namespace="complexity",
            identifier=content_hash,
            ttl_seconds=self._calculate_cache_ttl_from_query(query)
        )
    
    def _calculate_cache_ttl_from_query(self, query: str) -> int:
        """Calculate cache TTL based on query characteristics."""
        base_ttl = 3600  # 1 hour
        
        # Longer cache for simple queries
        if len(query) < 100:
            return base_ttl * 2
        
        # Shorter cache for complex or code-heavy queries
        if 'def ' in query or 'class ' in query or '```' in query:
            return base_ttl // 2
        
        return base_ttl
    
    async def analyze_with_cache(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> ComplexityScore:
        """Analyze query with intelligent caching."""
        
        cache_key = self._generate_cache_key(query, context)
        
        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for complexity analysis: {cache_key.identifier[:8]}...")
            return cached_result
        
        # Perform analysis
        result = await self.analyzer.analyze(query, context)
        
        # Cache result with confidence-based TTL
        confidence_multiplier = 1 + result.confidence  # 1-2x multiplier
        ttl = int(cache_key.ttl_seconds * confidence_multiplier)
        cache_key.ttl_seconds = ttl
        
        # Cache the result
        await self.cache.set(cache_key, result, ttl)
        
        logger.debug(f"Complexity analysis cached with {ttl}s TTL")
        return result


class CachedContextInspector:
    """Context inspector with intelligent caching and optimization."""
    
    def __init__(
        self,
        inspector: ContextInspector,
        cache: MultiLayerCache[ContextPreview]
    ):
        self.inspector = inspector
        self.cache = cache
    
    def _generate_cache_key(
        self,
        query: str,
        agent_id: str,
        context_sources_hash: str,
        model_target: Optional[str]
    ) -> CacheKey:
        """Generate cache key for context preview."""
        content = f"{query}:{agent_id}:{context_sources_hash}:{model_target or ''}"
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        return CacheKey(
            namespace="context",
            identifier=content_hash,
            ttl_seconds=1800  # 30 minutes
        )
    
    def _hash_context_sources(self, context_sources: Optional[List] = None) -> str:
        """Generate hash for context sources."""
        if not context_sources:
            return "none"
        
        # Create hash from source content and metadata
        content = ""
        for source in context_sources:
            if hasattr(source, 'content'):
                content += source.content + str(getattr(source, 'priority', 0))
        
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    async def preview_with_cache(
        self,
        query: str,
        agent_id: str,
        context_sources: Optional[List] = None,
        model_target: Optional[str] = None,
        complexity_score: Optional[float] = None
    ) -> ContextPreview:
        """Create context preview with caching."""
        
        sources_hash = self._hash_context_sources(context_sources)
        cache_key = self._generate_cache_key(query, agent_id, sources_hash, model_target)
        
        # Try cache first
        cached_preview = await self.cache.get(cache_key)
        if cached_preview is not None:
            logger.debug(f"Cache hit for context preview: {cache_key.identifier[:8]}...")
            return cached_preview
        
        # Generate preview
        preview = await self.inspector.preview(
            query=query,
            agent_id=agent_id,
            context_sources=context_sources,
            model_target=model_target,
            complexity_score=complexity_score
        )
        
        # Cache the preview
        await self.cache.set(cache_key, preview)
        
        logger.debug(f"Context preview cached")
        return preview


class AsyncProcessingPipeline:
    """
    Asynchronous processing pipeline for parallel operations.
    
    Optimizes query processing by running independent operations concurrently
    and managing resource utilization.
    """
    
    def __init__(
        self,
        max_concurrent_operations: int = 10,
        timeout_seconds: float = 30.0
    ):
        self.max_concurrent_operations = max_concurrent_operations
        self.timeout_seconds = timeout_seconds
        self.semaphore = asyncio.Semaphore(max_concurrent_operations)
        
        # Performance tracking
        self.operation_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.success_counts: Dict[str, int] = defaultdict(int)
    
    async def process_query_pipeline(
        self,
        query: str,
        cached_complexity_analyzer: CachedComplexityAnalyzer,
        cached_context_inspector: CachedContextInspector,
        model_router: ModelRouter,
        agent_id: str = "pipeline",
        context: Optional[Dict] = None,
        preferences: Optional[RoutingPreferences] = None
    ) -> Dict[str, Any]:
        """
        Process query with optimized parallel operations.
        
        Args:
            query: The query to process
            cached_complexity_analyzer: Cached complexity analyzer
            cached_context_inspector: Cached context inspector
            model_router: Model router for routing decisions
            agent_id: Agent ID for context
            context: Additional context
            preferences: Routing preferences
            
        Returns:
            Complete processing result with timing information
        """
        start_time = time.perf_counter()
        
        async with self.semaphore:
            try:
                # Start independent operations concurrently
                tasks = {}
                
                # Complexity analysis
                tasks['complexity'] = asyncio.create_task(
                    self._timed_operation(
                        'complexity_analysis',
                        cached_complexity_analyzer.analyze_with_cache(query, context)
                    )
                )
                
                # Context preparation (simplified - would prepare context sources)
                tasks['context'] = asyncio.create_task(
                    self._timed_operation(
                        'context_preparation',
                        self._prepare_context_sources(query, context)
                    )
                )
                
                # Model availability check
                tasks['model_availability'] = asyncio.create_task(
                    self._timed_operation(
                        'model_availability',
                        self._check_model_availability(model_router)
                    )
                )
                
                # Wait for all independent operations with timeout
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks.values(), return_exceptions=True),
                        timeout=self.timeout_seconds
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Pipeline timeout after {self.timeout_seconds}s")
                    # Cancel remaining tasks
                    for task in tasks.values():
                        if not task.done():
                            task.cancel()
                    raise
                
                # Process results
                complexity_result, context_result, availability_result = results
                
                # Handle any exceptions
                if isinstance(complexity_result, Exception):
                    logger.error(f"Complexity analysis failed: {complexity_result}")
                    raise complexity_result
                
                if isinstance(context_result, Exception):
                    logger.error(f"Context preparation failed: {context_result}")
                    context_result = []  # Use empty context as fallback
                
                if isinstance(availability_result, Exception):
                    logger.error(f"Model availability check failed: {availability_result}")
                    availability_result = []  # Use empty list as fallback
                
                # Now perform dependent operation (model routing)
                routing_task = asyncio.create_task(
                    self._timed_operation(
                        'model_routing',
                        model_router.route(
                            query=query,
                            complexity=complexity_result,
                            context_tokens=self._estimate_context_tokens(query, context_result),
                            preferences=preferences
                        )
                    )
                )
                
                routing_result = await routing_task
                if isinstance(routing_result, Exception):
                    logger.error(f"Model routing failed: {routing_result}")
                    raise routing_result
                
                # Calculate total processing time
                total_time = time.perf_counter() - start_time
                
                return {
                    'complexity': complexity_result,
                    'context_sources': context_result,
                    'available_models': availability_result,
                    'routing': routing_result,
                    'processing_time_ms': total_time * 1000,
                    'pipeline_stats': self.get_performance_stats()
                }
                
            except Exception as e:
                logger.error(f"Pipeline processing failed: {e}")
                raise
    
    async def _timed_operation(self, operation_name: str, operation_coro):
        """Execute operation with timing and error tracking."""
        start_time = time.perf_counter()
        
        try:
            result = await operation_coro
            
            # Record success
            execution_time = time.perf_counter() - start_time
            self.operation_times[operation_name].append(execution_time)
            self.success_counts[operation_name] += 1
            
            return result
            
        except Exception as e:
            # Record error
            self.error_counts[operation_name] += 1
            logger.error(f"Operation {operation_name} failed: {e}")
            raise
    
    async def _prepare_context_sources(self, query: str, context: Optional[Dict]) -> List:
        """Prepare context sources for the query."""
        # This would implement actual context source preparation
        # For now, return empty list
        await asyncio.sleep(0.01)  # Simulate work
        return []
    
    async def _check_model_availability(self, model_router: ModelRouter) -> List:
        """Check available models."""
        # Get available models from router
        try:
            available_models = model_router.model_registry.get_available_models()
            return available_models
        except Exception as e:
            logger.error(f"Model availability check failed: {e}")
            return []
    
    def _estimate_context_tokens(self, query: str, context_sources: List) -> int:
        """Estimate context tokens for routing."""
        # Simple estimation based on query length and context sources
        base_tokens = len(query.split()) * 1.3  # Rough token estimation
        context_tokens = len(context_sources) * 100  # Assume 100 tokens per source
        return int(base_tokens + context_tokens)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the pipeline."""
        stats = {}
        
        for operation, times in self.operation_times.items():
            if times:
                stats[operation] = {
                    'avg_time_ms': (sum(times) / len(times)) * 1000,
                    'min_time_ms': min(times) * 1000,
                    'max_time_ms': max(times) * 1000,
                    'success_count': self.success_counts[operation],
                    'error_count': self.error_counts[operation],
                    'success_rate': self.success_counts[operation] / (
                        self.success_counts[operation] + self.error_counts[operation]
                    ) if (self.success_counts[operation] + self.error_counts[operation]) > 0 else 0
                }
        
        return stats


class PerformanceOptimizer:
    """
    Main performance optimization system integrating all optimization components.
    
    Provides a unified interface for caching, async processing, and performance
    monitoring across the Autom8 system.
    """
    
    def __init__(
        self,
        redis_client: RedisClient,
        sqlite_manager: SQLiteManager,
        complexity_analyzer: ComplexityAnalyzer,
        context_inspector: ContextInspector,
        model_router: ModelRouter
    ):
        self.redis_client = redis_client
        self.sqlite_manager = sqlite_manager
        
        # Initialize cache backends
        self.redis_cache = RedisCache[Any](redis_client)
        self.sqlite_cache = SQLiteCache[Any](sqlite_manager, "performance_cache")
        
        # Initialize multi-layer caches for different components
        self.complexity_cache = MultiLayerCache[ComplexityScore](
            namespace="complexity",
            l1_backend=self.redis_cache,
            l2_backend=self.sqlite_cache,
            default_ttl=3600
        )
        
        self.context_cache = MultiLayerCache[ContextPreview](
            namespace="context",
            l1_backend=self.redis_cache,
            l2_backend=self.sqlite_cache,
            default_ttl=1800
        )
        
        # Initialize cached components
        self.cached_complexity_analyzer = CachedComplexityAnalyzer(
            complexity_analyzer, self.complexity_cache
        )
        
        self.cached_context_inspector = CachedContextInspector(
            context_inspector, self.context_cache
        )
        
        # Initialize async processing pipeline
        self.processing_pipeline = AsyncProcessingPipeline(
            max_concurrent_operations=10,
            timeout_seconds=30.0
        )
        
        self.model_router = model_router
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the performance optimization system."""
        try:
            # Initialize SQLite cache tables
            await self.sqlite_cache.initialize()
            
            # Start background cache maintenance
            self.complexity_cache.start_background_cleanup()
            self.context_cache.start_background_cleanup()
            
            self._initialized = True
            logger.info("Performance optimizer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize performance optimizer: {e}")
            return False
    
    async def process_query_optimized(
        self,
        query: str,
        agent_id: str = "optimized_pipeline",
        context: Optional[Dict] = None,
        preferences: Optional[RoutingPreferences] = None
    ) -> Dict[str, Any]:
        """
        Process query with full optimization pipeline.
        
        This is the main entry point for optimized query processing that
        leverages caching, async processing, and intelligent resource management.
        """
        if not self._initialized:
            raise RuntimeError("Performance optimizer not initialized")
        
        return await self.processing_pipeline.process_query_pipeline(
            query=query,
            cached_complexity_analyzer=self.cached_complexity_analyzer,
            cached_context_inspector=self.cached_context_inspector,
            model_router=self.model_router,
            agent_id=agent_id,
            context=context,
            preferences=preferences
        )
    
    async def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        try:
            # Get cache statistics
            complexity_stats = await self.complexity_cache.get_cache_stats()
            context_stats = await self.context_cache.get_cache_stats()
            
            # Get pipeline statistics
            pipeline_stats = self.processing_pipeline.get_performance_stats()
            
            # Calculate overall optimization metrics
            total_requests = (
                complexity_stats.get('metrics', {}).get('total_requests', 0) +
                context_stats.get('metrics', {}).get('total_requests', 0)
            )
            
            total_cache_hits = (
                complexity_stats.get('metrics', {}).get('cache_hits', 0) +
                context_stats.get('metrics', {}).get('cache_hits', 0)
            )
            
            overall_hit_rate = total_cache_hits / total_requests if total_requests > 0 else 0.0
            
            return {
                'overall_metrics': {
                    'total_cache_requests': total_requests,
                    'total_cache_hits': total_cache_hits,
                    'overall_hit_rate': overall_hit_rate,
                    'optimization_enabled': True
                },
                'complexity_cache': complexity_stats,
                'context_cache': context_stats,
                'pipeline_performance': pipeline_stats,
                'recommendations': self._generate_optimization_recommendations(
                    complexity_stats, context_stats, pipeline_stats
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to get optimization stats: {e}")
            return {'error': str(e)}
    
    def _generate_optimization_recommendations(
        self,
        complexity_stats: Dict,
        context_stats: Dict,
        pipeline_stats: Dict
    ) -> List[str]:
        """Generate optimization recommendations based on performance data."""
        recommendations = []
        
        # Cache hit rate recommendations
        complexity_hit_rate = complexity_stats.get('metrics', {}).get('hit_rate', 0)
        context_hit_rate = context_stats.get('metrics', {}).get('hit_rate', 0)
        
        if complexity_hit_rate < 0.5:
            recommendations.append(
                f"Low complexity cache hit rate ({complexity_hit_rate:.1%}). "
                "Consider increasing cache TTL or reviewing query patterns."
            )
        
        if context_hit_rate < 0.3:
            recommendations.append(
                f"Low context cache hit rate ({context_hit_rate:.1%}). "
                "Context changes frequently - consider optimizing context sources."
            )
        
        # Pipeline performance recommendations
        for operation, stats in pipeline_stats.items():
            error_rate = stats.get('error_count', 0) / (
                stats.get('success_count', 0) + stats.get('error_count', 0)
            ) if (stats.get('success_count', 0) + stats.get('error_count', 0)) > 0 else 0
            
            if error_rate > 0.1:  # >10% error rate
                recommendations.append(
                    f"High error rate for {operation} ({error_rate:.1%}). "
                    "Review operation reliability and error handling."
                )
            
            avg_time = stats.get('avg_time_ms', 0)
            if avg_time > 1000:  # >1 second
                recommendations.append(
                    f"Slow {operation} performance ({avg_time:.0f}ms average). "
                    "Consider optimization or increased parallelism."
                )
        
        return recommendations
    
    async def clear_all_caches(self) -> Dict[str, int]:
        """Clear all caches and return counts of cleared items."""
        try:
            complexity_cleared = await self.complexity_cache.clear_namespace()
            context_cleared = await self.context_cache.clear_namespace()
            
            return {
                'complexity_cache_cleared': complexity_cleared,
                'context_cache_cleared': context_cleared,
                'total_cleared': complexity_cleared + context_cleared
            }
            
        except Exception as e:
            logger.error(f"Failed to clear caches: {e}")
            return {'error': str(e)}
    
    async def optimize_cache_configuration(self) -> Dict[str, Any]:
        """
        Analyze cache performance and suggest configuration optimizations.
        
        Returns optimization suggestions based on access patterns and performance.
        """
        try:
            stats = await self.get_optimization_stats()
            
            # Analyze access patterns for each cache
            optimizations = {
                'complexity_cache': self._analyze_cache_optimization(
                    stats.get('complexity_cache', {})
                ),
                'context_cache': self._analyze_cache_optimization(
                    stats.get('context_cache', {})
                ),
                'overall': []
            }
            
            # Overall system optimizations
            overall_hit_rate = stats.get('overall_metrics', {}).get('overall_hit_rate', 0)
            if overall_hit_rate < 0.6:
                optimizations['overall'].append(
                    "Consider increasing cache sizes or TTL values to improve hit rates"
                )
            
            return optimizations
            
        except Exception as e:
            logger.error(f"Cache optimization analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_cache_optimization(self, cache_stats: Dict) -> List[str]:
        """Analyze individual cache performance for optimization suggestions."""
        suggestions = []
        
        metrics = cache_stats.get('metrics', {})
        hit_rate = metrics.get('hit_rate', 0)
        l1_hit_rate = metrics.get('l1_hit_rate', 0)
        
        # L1 vs L2 hit rate analysis
        if l1_hit_rate < 0.3 and hit_rate > 0.6:
            suggestions.append(
                "High L2 hit rate but low L1 hit rate suggests L1 cache is too small "
                "or TTL is too short"
            )
        
        # Size analysis
        l1_size = cache_stats.get('l1_size_bytes', 0)
        l2_size = cache_stats.get('l2_size_bytes', 0)
        
        if l1_size > cache_stats.get('size_limits', {}).get('l1_max', 0) * 0.9:
            suggestions.append("L1 cache approaching size limit - consider increasing limit")
        
        if l2_size > cache_stats.get('size_limits', {}).get('l2_max', 0) * 0.9:
            suggestions.append("L2 cache approaching size limit - consider cleanup or limit increase")
        
        return suggestions


# Integration utilities

async def create_performance_optimizer(
    redis_client: RedisClient,
    sqlite_manager: SQLiteManager,
    complexity_analyzer: ComplexityAnalyzer,
    context_inspector: ContextInspector,
    model_router: ModelRouter
) -> PerformanceOptimizer:
    """Factory function to create and initialize performance optimizer."""
    
    optimizer = PerformanceOptimizer(
        redis_client=redis_client,
        sqlite_manager=sqlite_manager,
        complexity_analyzer=complexity_analyzer,
        context_inspector=context_inspector,
        model_router=model_router
    )
    
    success = await optimizer.initialize()
    if not success:
        raise RuntimeError("Failed to initialize performance optimizer")
    
    return optimizer