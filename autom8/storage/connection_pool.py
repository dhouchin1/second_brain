"""
Advanced Connection Pooling for Autom8 Storage Systems.

Provides enterprise-grade connection pooling for SQLite and Redis connections
with concurrent access management, health monitoring, and automatic recovery.
"""

import asyncio
import aiosqlite
import redis.asyncio as redis
import time
import logging
from abc import ABC, abstractmethod
from asyncio import Queue, Semaphore, Event
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Any, AsyncContextManager, Dict, List, Optional, Protocol, Union
from collections import deque

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pools."""
    min_connections: int = 2
    max_connections: int = 20
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0  # 5 minutes
    max_retries: int = 3
    retry_delay: float = 1.0
    health_check_interval: float = 60.0  # 1 minute
    pool_recycle: Optional[float] = 3600.0  # 1 hour


@dataclass
class ConnectionMetrics:
    """Connection pool metrics."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    connections_created: int = 0
    connections_destroyed: int = 0
    connection_errors: int = 0
    pool_hits: int = 0
    pool_misses: int = 0
    avg_wait_time_ms: float = 0.0
    max_wait_time_ms: float = 0.0


class PooledConnection:
    """Wrapper for pooled connections with metadata."""
    
    def __init__(self, connection: Any, pool: "BaseConnectionPool"):
        self.connection = connection
        self.pool = pool
        self.created_at = time.time()
        self.last_used = time.time()
        self.use_count = 0
        self.is_healthy = True
        
    def mark_used(self):
        """Mark connection as recently used."""
        self.last_used = time.time()
        self.use_count += 1
        
    def is_expired(self, pool_recycle: Optional[float] = None) -> bool:
        """Check if connection is expired."""
        if pool_recycle is None:
            return False
        return (time.time() - self.created_at) > pool_recycle
        
    def is_idle_too_long(self, idle_timeout: float) -> bool:
        """Check if connection has been idle too long."""
        return (time.time() - self.last_used) > idle_timeout


class BaseConnectionPool(ABC):
    """Base class for connection pools."""
    
    def __init__(self, config: ConnectionPoolConfig):
        self.config = config
        self.metrics = ConnectionMetrics()
        
        # Connection management
        self._pool: Queue[PooledConnection] = Queue(maxsize=config.max_connections)
        self._semaphore = Semaphore(config.max_connections)
        self._active_connections: Dict[int, PooledConnection] = {}
        
        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = Event()
        
        # Performance tracking
        self._wait_times: deque = deque(maxlen=100)
        
    @abstractmethod
    async def _create_connection(self) -> Any:
        """Create a new database connection."""
        pass
        
    @abstractmethod
    async def _close_connection(self, connection: Any) -> None:
        """Close a database connection."""
        pass
        
    @abstractmethod
    async def _validate_connection(self, connection: Any) -> bool:
        """Check if connection is still valid."""
        pass
        
    async def initialize(self) -> None:
        """Initialize the connection pool."""
        logger.info(f"Initializing connection pool with {self.config.min_connections}-{self.config.max_connections} connections")
        
        # Create minimum connections
        for _ in range(self.config.min_connections):
            try:
                conn = await self._create_connection()
                pooled_conn = PooledConnection(conn, self)
                await self._pool.put(pooled_conn)
                self.metrics.total_connections += 1
                self.metrics.idle_connections += 1
                self.metrics.connections_created += 1
            except Exception as e:
                logger.error(f"Failed to create initial connection: {e}")
                self.metrics.connection_errors += 1
        
        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info(f"Connection pool initialized with {self.metrics.total_connections} connections")
    
    async def _health_check_loop(self):
        """Background task for connection health checking."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _perform_health_check(self):
        """Perform health check on idle connections."""
        connections_to_remove = []
        
        # Check idle connections in pool
        temp_connections = []
        
        # Drain pool to check connections
        while not self._pool.empty():
            try:
                pooled_conn = self._pool.get_nowait()
                temp_connections.append(pooled_conn)
            except asyncio.QueueEmpty:
                break
        
        # Validate and filter connections
        for pooled_conn in temp_connections:
            try:
                # Check if connection is expired or idle too long
                if (pooled_conn.is_expired(self.config.pool_recycle) or 
                    pooled_conn.is_idle_too_long(self.config.idle_timeout)):
                    connections_to_remove.append(pooled_conn)
                    continue
                
                # Validate connection health
                is_valid = await self._validate_connection(pooled_conn.connection)
                if not is_valid:
                    pooled_conn.is_healthy = False
                    connections_to_remove.append(pooled_conn)
                    continue
                
                # Return healthy connection to pool
                await self._pool.put(pooled_conn)
                
            except Exception as e:
                logger.error(f"Health check failed for connection: {e}")
                connections_to_remove.append(pooled_conn)
        
        # Remove unhealthy connections
        for pooled_conn in connections_to_remove:
            await self._remove_connection(pooled_conn)
        
        # Ensure minimum connections
        await self._ensure_minimum_connections()
    
    async def _ensure_minimum_connections(self):
        """Ensure minimum number of connections in pool."""
        current_total = self.metrics.total_connections
        
        if current_total < self.config.min_connections:
            needed = self.config.min_connections - current_total
            
            for _ in range(needed):
                try:
                    conn = await self._create_connection()
                    pooled_conn = PooledConnection(conn, self)
                    await self._pool.put(pooled_conn)
                    self.metrics.total_connections += 1
                    self.metrics.idle_connections += 1
                    self.metrics.connections_created += 1
                except Exception as e:
                    logger.error(f"Failed to create connection during health check: {e}")
                    self.metrics.connection_errors += 1
                    break
    
    async def _remove_connection(self, pooled_conn: PooledConnection):
        """Remove a connection from the pool."""
        try:
            await self._close_connection(pooled_conn.connection)
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")
        finally:
            self.metrics.total_connections -= 1
            self.metrics.idle_connections -= 1
            self.metrics.connections_destroyed += 1
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncContextManager[Any]:
        """Get a connection from the pool."""
        start_time = time.time()
        pooled_conn = None
        
        try:
            # Wait for available connection slot
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.config.connection_timeout
            )
            
            # Try to get existing connection from pool
            try:
                pooled_conn = self._pool.get_nowait()
                self.metrics.pool_hits += 1
                
                # Validate connection
                if not await self._validate_connection(pooled_conn.connection):
                    await self._remove_connection(pooled_conn)
                    pooled_conn = None
                    
            except asyncio.QueueEmpty:
                self.metrics.pool_misses += 1
                
            # Create new connection if needed
            if pooled_conn is None:
                if self.metrics.total_connections < self.config.max_connections:
                    conn = await self._create_connection()
                    pooled_conn = PooledConnection(conn, self)
                    self.metrics.total_connections += 1
                    self.metrics.connections_created += 1
                else:
                    # Wait for a connection to become available
                    pooled_conn = await asyncio.wait_for(
                        self._pool.get(),
                        timeout=self.config.connection_timeout
                    )
            
            # Track timing
            wait_time = (time.time() - start_time) * 1000
            self._wait_times.append(wait_time)
            self._update_wait_time_metrics(wait_time)
            
            # Mark as active
            pooled_conn.mark_used()
            self.metrics.active_connections += 1
            self.metrics.idle_connections -= 1
            
            conn_id = id(pooled_conn)
            self._active_connections[conn_id] = pooled_conn
            
            try:
                yield pooled_conn.connection
            finally:
                # Return connection to pool
                await self._return_connection(pooled_conn)
                
        except asyncio.TimeoutError:
            logger.error(f"Connection timeout after {self.config.connection_timeout}s")
            raise
        except Exception as e:
            logger.error(f"Failed to get connection: {e}")
            self.metrics.connection_errors += 1
            raise
        finally:
            self._semaphore.release()
    
    async def _return_connection(self, pooled_conn: PooledConnection):
        """Return connection to pool."""
        conn_id = id(pooled_conn)
        self._active_connections.pop(conn_id, None)
        
        self.metrics.active_connections -= 1
        self.metrics.idle_connections += 1
        
        try:
            # Check if connection should be recycled
            if pooled_conn.is_expired(self.config.pool_recycle):
                await self._remove_connection(pooled_conn)
            else:
                await self._pool.put(pooled_conn)
        except Exception as e:
            logger.error(f"Failed to return connection: {e}")
            await self._remove_connection(pooled_conn)
    
    def _update_wait_time_metrics(self, wait_time: float):
        """Update wait time metrics."""
        if self._wait_times:
            self.metrics.avg_wait_time_ms = sum(self._wait_times) / len(self._wait_times)
            self.metrics.max_wait_time_ms = max(self._wait_times)
    
    async def get_metrics(self) -> ConnectionMetrics:
        """Get current pool metrics."""
        self.metrics.idle_connections = self._pool.qsize()
        return self.metrics
    
    async def close(self):
        """Close all connections and shut down pool."""
        logger.info("Shutting down connection pool...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all idle connections
        while not self._pool.empty():
            try:
                pooled_conn = self._pool.get_nowait()
                await self._close_connection(pooled_conn.connection)
                self.metrics.connections_destroyed += 1
            except Exception as e:
                logger.error(f"Error closing pooled connection: {e}")
        
        # Close active connections
        for pooled_conn in list(self._active_connections.values()):
            try:
                await self._close_connection(pooled_conn.connection)
                self.metrics.connections_destroyed += 1
            except Exception as e:
                logger.error(f"Error closing active connection: {e}")
        
        self._active_connections.clear()
        self.metrics.total_connections = 0
        self.metrics.active_connections = 0
        self.metrics.idle_connections = 0
        
        logger.info("Connection pool shut down complete")


class SQLiteConnectionPool(BaseConnectionPool):
    """SQLite connection pool with WAL mode and optimizations."""
    
    def __init__(self, db_path: str, config: Optional[ConnectionPoolConfig] = None):
        super().__init__(config or ConnectionPoolConfig())
        self.db_path = db_path
    
    async def _create_connection(self) -> aiosqlite.Connection:
        """Create optimized SQLite connection."""
        conn = await aiosqlite.connect(
            self.db_path,
            timeout=self.config.connection_timeout
        )
        
        # Apply SQLite optimizations
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA cache_size=10000")
        await conn.execute("PRAGMA temp_store=memory")
        await conn.execute("PRAGMA mmap_size=268435456")  # 256MB
        await conn.execute("PRAGMA page_size=4096")
        
        # Enable foreign keys
        await conn.execute("PRAGMA foreign_keys=ON")
        
        return conn
    
    async def _close_connection(self, connection: aiosqlite.Connection) -> None:
        """Close SQLite connection."""
        try:
            await connection.close()
        except Exception as e:
            logger.error(f"Error closing SQLite connection: {e}")
    
    async def _validate_connection(self, connection: aiosqlite.Connection) -> bool:
        """Validate SQLite connection."""
        try:
            await connection.execute("SELECT 1")
            return True
        except Exception:
            return False


class RedisConnectionPool(BaseConnectionPool):
    """Redis connection pool with cluster support."""
    
    def __init__(self, redis_url: str, config: Optional[ConnectionPoolConfig] = None):
        super().__init__(config or ConnectionPoolConfig())
        self.redis_url = redis_url
    
    async def _create_connection(self) -> redis.Redis:
        """Create Redis connection."""
        return redis.from_url(
            self.redis_url,
            socket_timeout=self.config.connection_timeout,
            decode_responses=True
        )
    
    async def _close_connection(self, connection: redis.Redis) -> None:
        """Close Redis connection."""
        try:
            await connection.aclose()
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")
    
    async def _validate_connection(self, connection: redis.Redis) -> bool:
        """Validate Redis connection."""
        try:
            await connection.ping()
            return True
        except Exception:
            return False


class ConnectionPoolManager:
    """Manages multiple connection pools for different databases."""
    
    def __init__(self):
        self._pools: Dict[str, BaseConnectionPool] = {}
        self._initialized = False
    
    async def initialize_sqlite_pool(
        self, 
        name: str, 
        db_path: str, 
        config: Optional[ConnectionPoolConfig] = None
    ) -> SQLiteConnectionPool:
        """Initialize SQLite connection pool."""
        if name in self._pools:
            raise ValueError(f"Pool {name} already exists")
        
        pool = SQLiteConnectionPool(db_path, config)
        await pool.initialize()
        self._pools[name] = pool
        
        logger.info(f"SQLite connection pool '{name}' initialized for {db_path}")
        return pool
    
    async def initialize_redis_pool(
        self, 
        name: str, 
        redis_url: str, 
        config: Optional[ConnectionPoolConfig] = None
    ) -> RedisConnectionPool:
        """Initialize Redis connection pool."""
        if name in self._pools:
            raise ValueError(f"Pool {name} already exists")
        
        pool = RedisConnectionPool(redis_url, config)
        await pool.initialize()
        self._pools[name] = pool
        
        logger.info(f"Redis connection pool '{name}' initialized for {redis_url}")
        return pool
    
    def get_pool(self, name: str) -> Optional[BaseConnectionPool]:
        """Get connection pool by name."""
        return self._pools.get(name)
    
    async def get_connection(self, pool_name: str):
        """Get connection from named pool."""
        pool = self.get_pool(pool_name)
        if not pool:
            raise ValueError(f"Pool {pool_name} not found")
        
        return pool.get_connection()
    
    async def get_all_metrics(self) -> Dict[str, ConnectionMetrics]:
        """Get metrics for all pools."""
        metrics = {}
        for name, pool in self._pools.items():
            metrics[name] = await pool.get_metrics()
        return metrics
    
    async def close_all(self):
        """Close all connection pools."""
        logger.info("Closing all connection pools...")
        
        for name, pool in self._pools.items():
            try:
                await pool.close()
                logger.info(f"Closed connection pool: {name}")
            except Exception as e:
                logger.error(f"Error closing pool {name}: {e}")
        
        self._pools.clear()
        self._initialized = False
        
        logger.info("All connection pools closed")


# Global pool manager instance
_pool_manager: Optional[ConnectionPoolManager] = None


async def get_pool_manager() -> ConnectionPoolManager:
    """Get global connection pool manager."""
    global _pool_manager
    
    if _pool_manager is None:
        _pool_manager = ConnectionPoolManager()
    
    return _pool_manager


async def initialize_default_pools():
    """Initialize default connection pools for Autom8."""
    manager = await get_pool_manager()
    
    # SQLite pool for main database
    sqlite_config = ConnectionPoolConfig(
        min_connections=2,
        max_connections=10,
        connection_timeout=30.0,
        idle_timeout=300.0,
        pool_recycle=3600.0
    )
    
    try:
        await manager.initialize_sqlite_pool(
            "main_db", 
            "./autom8.db", 
            sqlite_config
        )
    except Exception as e:
        logger.error(f"Failed to initialize SQLite pool: {e}")
    
    # Redis pool for caching
    redis_config = ConnectionPoolConfig(
        min_connections=3,
        max_connections=20,
        connection_timeout=5.0,
        idle_timeout=180.0,
        pool_recycle=1800.0
    )
    
    try:
        await manager.initialize_redis_pool(
            "cache", 
            "redis://localhost:6379", 
            redis_config
        )
    except Exception as e:
        logger.error(f"Failed to initialize Redis pool: {e}")


# Usage utilities

@asynccontextmanager
async def get_sqlite_connection(pool_name: str = "main_db"):
    """Get SQLite connection from pool."""
    manager = await get_pool_manager()
    async with manager.get_connection(pool_name) as conn:
        yield conn


@asynccontextmanager  
async def get_redis_connection(pool_name: str = "cache"):
    """Get Redis connection from pool."""
    manager = await get_pool_manager()
    async with manager.get_connection(pool_name) as conn:
        yield conn


async def close_all_pools():
    """Close all connection pools."""
    global _pool_manager
    
    if _pool_manager:
        await _pool_manager.close_all()
        _pool_manager = None