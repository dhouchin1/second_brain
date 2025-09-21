"""
Cached Token Counting with Performance Optimization.

Provides intelligent caching for token counts to reduce computational overhead,
especially important for context management and routing decisions.
"""

import asyncio
import hashlib
import time
import pickle
from collections import OrderedDict, defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from autom8.utils.tokens import TokenCounter, get_token_counter
from autom8.utils.logging import get_logger
from autom8.storage.redis.client import get_redis_client
from autom8.storage.sqlite.manager import get_sqlite_manager

logger = get_logger(__name__)


class TokenCacheEntry(BaseModel):
    """Cache entry for token counts."""
    content_hash: str = Field(description="Hash of the content")
    model: Optional[str] = Field(default=None, description="Model used for counting")
    token_count: int = Field(description="Cached token count")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    accessed_at: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = Field(default=1)
    content_length: int = Field(description="Length of original content")
    cost_estimate: Optional[float] = Field(default=None, description="Estimated cost")


class TokenCacheMetrics(BaseModel):
    """Metrics for token cache performance."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_writes: int = 0
    avg_computation_time_ms: float = 0.0
    avg_cache_retrieval_ms: float = 0.0
    memory_cache_size: int = 0
    redis_cache_size: int = 0
    sqlite_cache_size: int = 0
    total_tokens_cached: int = 0
    total_cost_saved: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    @property
    def performance_improvement(self) -> float:
        """Calculate performance improvement ratio."""
        if self.avg_computation_time_ms == 0:
            return 0.0
        return self.avg_computation_time_ms / max(0.1, self.avg_cache_retrieval_ms)


class CachedTokenCounter:
    """
    High-performance token counter with multi-level caching.
    
    Implements L1 (memory) -> L2 (Redis) -> L3 (SQLite) caching hierarchy
    with intelligent cache warming and performance optimization.
    """
    
    def __init__(
        self,
        memory_cache_size: int = 10000,
        cache_ttl_seconds: int = 3600,
        enable_redis: bool = True,
        enable_sqlite: bool = True
    ):
        self.token_counter = get_token_counter()
        self.memory_cache_size = memory_cache_size
        self.cache_ttl_seconds = cache_ttl_seconds
        self.enable_redis = enable_redis
        self.enable_sqlite = enable_sqlite
        
        # L1 Memory cache (LRU)
        self._memory_cache: OrderedDict[str, TokenCacheEntry] = OrderedDict()
        
        # Performance metrics
        self.metrics = TokenCacheMetrics()
        
        # Performance tracking
        self._computation_times: deque = deque(maxlen=100)
        self._cache_retrieval_times: deque = deque(maxlen=100)
        
        # Model-specific cache statistics
        self._model_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Initialize backends
        self._redis_client = None
        self._sqlite_manager = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize cache backends."""
        if self._initialized:
            return
        
        try:
            if self.enable_redis:
                self._redis_client = await get_redis_client()
                if not self._redis_client.is_connected:
                    logger.warning("Redis not available, disabling Redis cache")
                    self.enable_redis = False
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
            self.enable_redis = False
        
        try:
            if self.enable_sqlite:
                self._sqlite_manager = await get_sqlite_manager()
                await self._create_cache_table()
        except Exception as e:
            logger.warning(f"SQLite initialization failed: {e}")
            self.enable_sqlite = False
        
        self._initialized = True
        logger.info(f"Cached token counter initialized (Redis: {self.enable_redis}, SQLite: {self.enable_sqlite})")
    
    async def _create_cache_table(self):
        """Create SQLite cache table."""
        if not self._sqlite_manager:
            return
        
        conn = await self._sqlite_manager._get_connection()
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS token_cache (
                content_hash TEXT PRIMARY KEY,
                model TEXT,
                token_count INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1,
                content_length INTEGER,
                cost_estimate REAL,
                expires_at TIMESTAMP
            )
        """)
        
        # Create indexes for performance
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_token_cache_model ON token_cache(model)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_token_cache_accessed ON token_cache(accessed_at DESC)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_token_cache_expires ON token_cache(expires_at)
        """)
        
        await conn.commit()
        logger.debug("Token cache table created/verified")
    
    def _generate_cache_key(
        self,
        content: str,
        model: Optional[str] = None,
        encoding: Optional[str] = None
    ) -> str:
        """Generate cache key for content and model."""
        # Include model and encoding in the hash for model-specific caching
        key_data = f"{content}:{model or 'default'}:{encoding or 'default'}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content only."""
        return hashlib.md5(content.encode()).hexdigest()
    
    async def count_tokens(
        self,
        content: str,
        model: Optional[str] = None,
        encoding: Optional[str] = None,
        force_recompute: bool = False
    ) -> int:
        """
        Count tokens with intelligent caching.
        
        Args:
            content: Text to count tokens for
            model: Model name for accurate counting
            encoding: Specific encoding to use
            force_recompute: Skip cache and recompute
            
        Returns:
            Token count
        """
        if not content:
            return 0
        
        await self.initialize()
        
        start_time = time.perf_counter()
        self.metrics.total_requests += 1
        
        cache_key = self._generate_cache_key(content, model, encoding)
        content_hash = self._generate_content_hash(content)
        
        # Check cache hierarchy if not forcing recompute
        if not force_recompute:
            # Try L1 memory cache
            cached_entry = await self._get_from_memory_cache(cache_key)
            if cached_entry:
                await self._record_cache_hit(cached_entry, model, start_time, 'memory')
                return cached_entry.token_count
            
            # Try L2 Redis cache
            if self.enable_redis:
                cached_entry = await self._get_from_redis_cache(cache_key)
                if cached_entry:
                    # Promote to L1 cache
                    await self._store_in_memory_cache(cache_key, cached_entry)
                    await self._record_cache_hit(cached_entry, model, start_time, 'redis')
                    return cached_entry.token_count
            
            # Try L3 SQLite cache
            if self.enable_sqlite:
                cached_entry = await self._get_from_sqlite_cache(cache_key)
                if cached_entry:
                    # Promote to higher levels
                    await self._store_in_memory_cache(cache_key, cached_entry)
                    if self.enable_redis:
                        await self._store_in_redis_cache(cache_key, cached_entry)
                    await self._record_cache_hit(cached_entry, model, start_time, 'sqlite')
                    return cached_entry.token_count
        
        # Cache miss - compute token count
        computation_start = time.perf_counter()
        token_count = self.token_counter.count_tokens(content, model, encoding)
        computation_time = (time.perf_counter() - computation_start) * 1000
        
        # Create cache entry
        cache_entry = TokenCacheEntry(
            content_hash=content_hash,
            model=model,
            token_count=token_count,
            content_length=len(content),
            cost_estimate=self.token_counter.estimate_cost(content, model or 'default', True) if model else None
        )
        
        # Store in all cache levels
        await self._store_in_memory_cache(cache_key, cache_entry)
        if self.enable_redis:
            await self._store_in_redis_cache(cache_key, cache_entry)
        if self.enable_sqlite:
            await self._store_in_sqlite_cache(cache_key, cache_entry)
        
        # Update metrics
        self.metrics.cache_misses += 1
        self.metrics.cache_writes += 1
        self._computation_times.append(computation_time)
        self._update_computation_metrics()
        
        # Update model statistics
        if model:
            self._model_stats[model]['requests'] += 1
            self._model_stats[model]['tokens'] += token_count
        
        total_time = (time.perf_counter() - start_time) * 1000
        logger.debug(f"Token count computed for {model or 'default'}: {token_count} tokens in {total_time:.2f}ms")
        
        return token_count
    
    async def _get_from_memory_cache(self, cache_key: str) -> Optional[TokenCacheEntry]:
        """Get entry from L1 memory cache."""
        if cache_key not in self._memory_cache:
            return None
        
        entry = self._memory_cache[cache_key]
        
        # Check TTL
        if self._is_expired(entry):
            del self._memory_cache[cache_key]
            return None
        
        # Move to end (LRU)
        self._memory_cache.move_to_end(cache_key)
        entry.accessed_at = datetime.utcnow()
        entry.access_count += 1
        
        return entry
    
    async def _get_from_redis_cache(self, cache_key: str) -> Optional[TokenCacheEntry]:
        """Get entry from L2 Redis cache."""
        if not self._redis_client:
            return None
        
        try:
            cached_data = await self._redis_client.get(f"token_cache:{cache_key}")
            if not cached_data:
                return None
            
            entry_dict = pickle.loads(cached_data.encode('latin1'))
            entry = TokenCacheEntry(**entry_dict)
            
            # Check TTL
            if self._is_expired(entry):
                await self._redis_client.delete(f"token_cache:{cache_key}")
                return None
            
            entry.accessed_at = datetime.utcnow()
            entry.access_count += 1
            
            return entry
            
        except Exception as e:
            logger.warning(f"Redis cache retrieval failed: {e}")
            return None
    
    async def _get_from_sqlite_cache(self, cache_key: str) -> Optional[TokenCacheEntry]:
        """Get entry from L3 SQLite cache."""
        if not self._sqlite_manager:
            return None
        
        try:
            conn = await self._sqlite_manager._get_connection()
            
            async with conn.execute("""
                SELECT content_hash, model, token_count, created_at, accessed_at,
                       access_count, content_length, cost_estimate, expires_at
                FROM token_cache 
                WHERE content_hash = ?
                AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """, (cache_key,)) as cursor:
                row = await cursor.fetchone()
                
                if not row:
                    return None
                
                entry = TokenCacheEntry(
                    content_hash=row[0],
                    model=row[1],
                    token_count=row[2],
                    created_at=datetime.fromisoformat(row[3]),
                    accessed_at=datetime.fromisoformat(row[4]),
                    access_count=row[5],
                    content_length=row[6],
                    cost_estimate=row[7]
                )
                
                # Update access statistics
                await conn.execute("""
                    UPDATE token_cache 
                    SET accessed_at = CURRENT_TIMESTAMP, access_count = access_count + 1
                    WHERE content_hash = ?
                """, (cache_key,))
                await conn.commit()
                
                return entry
                
        except Exception as e:
            logger.warning(f"SQLite cache retrieval failed: {e}")
            return None
    
    async def _store_in_memory_cache(self, cache_key: str, entry: TokenCacheEntry):
        """Store entry in L1 memory cache with LRU eviction."""
        # Evict if at capacity
        while len(self._memory_cache) >= self.memory_cache_size:
            oldest_key, _ = self._memory_cache.popitem(last=False)
            logger.debug(f"Evicted cache entry: {oldest_key[:8]}...")
        
        self._memory_cache[cache_key] = entry
        self.metrics.memory_cache_size = len(self._memory_cache)
    
    async def _store_in_redis_cache(self, cache_key: str, entry: TokenCacheEntry):
        """Store entry in L2 Redis cache."""
        if not self._redis_client:
            return
        
        try:
            entry_dict = entry.dict()
            cached_data = pickle.dumps(entry_dict).decode('latin1')
            
            await self._redis_client.set(
                f"token_cache:{cache_key}",
                cached_data,
                ex=self.cache_ttl_seconds
            )
            
        except Exception as e:
            logger.warning(f"Redis cache storage failed: {e}")
    
    async def _store_in_sqlite_cache(self, cache_key: str, entry: TokenCacheEntry):
        """Store entry in L3 SQLite cache."""
        if not self._sqlite_manager:
            return
        
        try:
            conn = await self._sqlite_manager._get_connection()
            
            expires_at = entry.created_at + timedelta(seconds=self.cache_ttl_seconds)
            
            await conn.execute("""
                INSERT OR REPLACE INTO token_cache 
                (content_hash, model, token_count, created_at, accessed_at,
                 access_count, content_length, cost_estimate, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cache_key,
                entry.model,
                entry.token_count,
                entry.created_at.isoformat(),
                entry.accessed_at.isoformat(),
                entry.access_count,
                entry.content_length,
                entry.cost_estimate,
                expires_at.isoformat()
            ))
            
            await conn.commit()
            
        except Exception as e:
            logger.warning(f"SQLite cache storage failed: {e}")
    
    async def _record_cache_hit(
        self,
        entry: TokenCacheEntry,
        model: Optional[str],
        start_time: float,
        cache_level: str
    ):
        """Record cache hit metrics."""
        retrieval_time = (time.perf_counter() - start_time) * 1000
        
        self.metrics.cache_hits += 1
        self._cache_retrieval_times.append(retrieval_time)
        self._update_retrieval_metrics()
        
        # Update model statistics
        if model:
            self._model_stats[model]['cache_hits'] += 1
            self._model_stats[model]['tokens'] += entry.token_count
        
        logger.debug(f"Cache hit ({cache_level}) for {model or 'default'}: {entry.token_count} tokens in {retrieval_time:.2f}ms")
    
    def _is_expired(self, entry: TokenCacheEntry) -> bool:
        """Check if cache entry is expired."""
        age = datetime.utcnow() - entry.created_at
        return age.total_seconds() > self.cache_ttl_seconds
    
    def _update_computation_metrics(self):
        """Update average computation time metrics."""
        if self._computation_times:
            self.metrics.avg_computation_time_ms = sum(self._computation_times) / len(self._computation_times)
    
    def _update_retrieval_metrics(self):
        """Update average cache retrieval time metrics."""
        if self._cache_retrieval_times:
            self.metrics.avg_cache_retrieval_ms = sum(self._cache_retrieval_times) / len(self._cache_retrieval_times)
    
    async def count_tokens_batch(
        self,
        contents: List[str],
        model: Optional[str] = None,
        encoding: Optional[str] = None
    ) -> List[int]:
        """
        Count tokens for multiple content items efficiently.
        
        Args:
            contents: List of text content to count
            model: Model name for accurate counting
            encoding: Specific encoding to use
            
        Returns:
            List of token counts corresponding to input contents
        """
        # Process in parallel for better performance
        tasks = [
            self.count_tokens(content, model, encoding)
            for content in contents
        ]
        
        return await asyncio.gather(*tasks)
    
    async def estimate_cost_batch(
        self,
        contents: List[str],
        model: str,
        is_input: bool = True
    ) -> List[float]:
        """
        Estimate costs for multiple content items.
        
        Args:
            contents: List of text content
            model: Model name
            is_input: True for input tokens, False for output tokens
            
        Returns:
            List of cost estimates
        """
        token_counts = await self.count_tokens_batch(contents, model)
        
        costs = []
        for token_count in token_counts:
            cost_per_token = self.token_counter._get_model_cost_per_token(model, is_input)
            costs.append(token_count * cost_per_token)
        
        return costs
    
    async def get_cache_metrics(self) -> TokenCacheMetrics:
        """Get comprehensive cache metrics."""
        # Update cache sizes
        self.metrics.memory_cache_size = len(self._memory_cache)
        
        if self.enable_redis and self._redis_client:
            try:
                redis_keys = await self._redis_client.keys("token_cache:*")
                self.metrics.redis_cache_size = len(redis_keys)
            except Exception:
                pass
        
        if self.enable_sqlite and self._sqlite_manager:
            try:
                conn = await self._sqlite_manager._get_connection()
                async with conn.execute("SELECT COUNT(*) FROM token_cache WHERE expires_at > CURRENT_TIMESTAMP") as cursor:
                    row = await cursor.fetchone()
                    self.metrics.sqlite_cache_size = row[0] if row else 0
            except Exception:
                pass
        
        # Calculate total tokens cached
        self.metrics.total_tokens_cached = sum(
            entry.token_count for entry in self._memory_cache.values()
        )
        
        return self.metrics
    
    async def get_model_statistics(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """Get per-model cache statistics."""
        stats = {}
        
        for model, model_stats in self._model_stats.items():
            requests = model_stats['requests']
            cache_hits = model_stats['cache_hits']
            tokens = model_stats['tokens']
            
            stats[model] = {
                'total_requests': requests,
                'cache_hits': cache_hits,
                'cache_misses': requests - cache_hits,
                'hit_rate': (cache_hits / requests * 100) if requests > 0 else 0,
                'total_tokens': tokens,
                'avg_tokens_per_request': tokens / requests if requests > 0 else 0
            }
        
        return stats
    
    async def warm_cache(self, contents: List[str], model: Optional[str] = None):
        """
        Warm cache with frequently used content.
        
        Args:
            contents: List of content to pre-cache
            model: Model to use for token counting
        """
        logger.info(f"Warming cache with {len(contents)} items for model {model or 'default'}")
        
        # Process in batches to avoid overwhelming the system
        batch_size = 50
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            await self.count_tokens_batch(batch, model)
            
            # Brief pause between batches
            await asyncio.sleep(0.1)
        
        logger.info("Cache warming completed")
    
    async def clear_cache(self, model: Optional[str] = None):
        """
        Clear cache entries.
        
        Args:
            model: Clear only entries for specific model, or None for all
        """
        if model:
            # Clear specific model entries
            keys_to_remove = []
            for key, entry in self._memory_cache.items():
                if entry.model == model:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._memory_cache[key]
            
            logger.info(f"Cleared cache entries for model: {model}")
        else:
            # Clear all entries
            self._memory_cache.clear()
            self.metrics = TokenCacheMetrics()  # Reset metrics
            self._model_stats.clear()
            
            logger.info("Cleared all cache entries")
    
    async def cleanup_expired(self) -> int:
        """Clean up expired cache entries."""
        cleaned = 0
        
        # Clean memory cache
        expired_keys = []
        for key, entry in self._memory_cache.items():
            if self._is_expired(entry):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._memory_cache[key]
            cleaned += 1
        
        # Clean SQLite cache
        if self.enable_sqlite and self._sqlite_manager:
            try:
                conn = await self._sqlite_manager._get_connection()
                cursor = await conn.execute("""
                    DELETE FROM token_cache 
                    WHERE expires_at IS NOT NULL AND expires_at <= CURRENT_TIMESTAMP
                """)
                sqlite_cleaned = cursor.rowcount
                await conn.commit()
                cleaned += sqlite_cleaned
            except Exception as e:
                logger.error(f"SQLite cleanup failed: {e}")
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} expired cache entries")
        
        return cleaned


# Global cached token counter instance
_cached_token_counter: Optional[CachedTokenCounter] = None


async def get_cached_token_counter() -> CachedTokenCounter:
    """Get global cached token counter instance."""
    global _cached_token_counter
    
    if _cached_token_counter is None:
        _cached_token_counter = CachedTokenCounter()
        await _cached_token_counter.initialize()
    
    return _cached_token_counter


# Convenience functions

async def count_tokens_cached(
    content: str,
    model: Optional[str] = None,
    encoding: Optional[str] = None
) -> int:
    """
    Convenience function to count tokens with caching.
    
    Args:
        content: Text to count tokens for
        model: Model name for accurate counting
        encoding: Specific encoding to use
        
    Returns:
        Token count
    """
    counter = await get_cached_token_counter()
    return await counter.count_tokens(content, model, encoding)


async def estimate_cost_cached(
    content: str,
    model: str,
    is_input: bool = True
) -> float:
    """
    Convenience function to estimate cost with caching.
    
    Args:
        content: Text to process
        model: Model name
        is_input: True for input tokens, False for output tokens
        
    Returns:
        Estimated cost in USD
    """
    counter = await get_cached_token_counter()
    token_count = await counter.count_tokens(content, model)
    
    cost_per_token = counter.token_counter._get_model_cost_per_token(model, is_input)
    return token_count * cost_per_token