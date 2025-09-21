"""
Redis Client for Autom8 Shared Memory System.

Provides Redis connection management, health checking, and basic operations
for the shared memory architecture specified in the PRD.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import redis.asyncio as redis
from pydantic import BaseModel

from autom8.config.settings import get_settings

logger = logging.getLogger(__name__)


class RedisConfig(BaseModel):
    """Redis configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 50
    connection_timeout: int = 5
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[int, int] = {}
    retry_on_timeout: bool = True
    decode_responses: bool = True


class RedisClient:
    """
    Redis client for Autom8 shared memory operations.
    
    Provides connection management, health checking, and core Redis operations
    needed for the shared memory architecture.
    """
    
    def __init__(self, config: Optional[RedisConfig] = None):
        self.config = config or self._load_config()
        self._pool: Optional[redis.ConnectionPool] = None
        self._client: Optional[redis.Redis] = None
        self._connected = False
        self._last_health_check: Optional[datetime] = None
        
    def _load_config(self) -> RedisConfig:
        """Load Redis configuration from settings."""
        settings = get_settings()
        
        return RedisConfig(
            host=getattr(settings, 'redis_host', 'localhost'),
            port=getattr(settings, 'redis_port', 6379),
            db=0,
            max_connections=50,
            connection_timeout=5
        )
    
    async def connect(self) -> bool:
        """Establish Redis connection."""
        try:
            self._pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.connection_timeout,
                socket_keepalive=self.config.socket_keepalive,
                socket_keepalive_options=self.config.socket_keepalive_options,
                retry_on_timeout=self.config.retry_on_timeout,
                decode_responses=self.config.decode_responses
            )
            
            self._client = redis.Redis(connection_pool=self._pool)
            
            # Test connection
            await self._client.ping()
            self._connected = True
            self._last_health_check = datetime.utcnow()
            
            logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            return False
    
    async def disconnect(self) -> None:
        """Close Redis connections."""
        if self._client:
            await self._client.aclose()
        if self._pool:
            await self._pool.aclose()
        
        self._connected = False
        logger.info("Disconnected from Redis")
    
    async def health_check(self) -> bool:
        """Check Redis connection health."""
        try:
            if not self._client:
                return False
                
            await self._client.ping()
            self._last_health_check = datetime.utcnow()
            return True
            
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            self._connected = False
            return False
    
    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self._connected
    
    @property
    def client(self) -> Optional[redis.Redis]:
        """Get Redis client instance."""
        return self._client
    
    # Core Redis operations
    
    async def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        if not self._client:
            return None
        try:
            return await self._client.get(key)
        except Exception as e:
            logger.error(f"Redis GET failed for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Union[str, Dict, List], ex: Optional[int] = None) -> bool:
        """Set key-value with optional expiration."""
        if not self._client:
            return False
        
        try:
            # Serialize complex types
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            result = await self._client.set(key, value, ex=ex)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Redis SET failed for key {key}: {e}")
            return False
    
    async def delete(self, *keys: str) -> int:
        """Delete one or more keys."""
        if not self._client:
            return 0
        try:
            return await self._client.delete(*keys)
        except Exception as e:
            logger.error(f"Redis DELETE failed for keys {keys}: {e}")
            return 0
    
    async def exists(self, *keys: str) -> int:
        """Check if keys exist."""
        if not self._client:
            return 0
        try:
            return await self._client.exists(*keys)
        except Exception as e:
            logger.error(f"Redis EXISTS failed for keys {keys}: {e}")
            return 0
    
    async def expire(self, key: str, time: int) -> bool:
        """Set key expiration time."""
        if not self._client:
            return False
        try:
            return await self._client.expire(key, time)
        except Exception as e:
            logger.error(f"Redis EXPIRE failed for key {key}: {e}")
            return False
    
    async def hget(self, name: str, key: str) -> Optional[str]:
        """Get hash field value."""
        if not self._client:
            return None
        try:
            return await self._client.hget(name, key)
        except Exception as e:
            logger.error(f"Redis HGET failed for {name}.{key}: {e}")
            return None
    
    async def hgetall(self, name: str) -> Dict[str, str]:
        """Get all hash fields and values."""
        if not self._client:
            return {}
        try:
            return await self._client.hgetall(name)
        except Exception as e:
            logger.error(f"Redis HGETALL failed for {name}: {e}")
            return {}
    
    async def hset(self, name: str, mapping: Dict[str, Any]) -> int:
        """Set hash fields."""
        if not self._client:
            return 0
        
        try:
            # Serialize complex values
            serialized_mapping = {}
            for key, value in mapping.items():
                if isinstance(value, (dict, list)):
                    serialized_mapping[key] = json.dumps(value)
                else:
                    serialized_mapping[key] = str(value)
            
            return await self._client.hset(name, mapping=serialized_mapping)
            
        except Exception as e:
            logger.error(f"Redis HSET failed for {name}: {e}")
            return 0
    
    async def zadd(self, name: str, mapping: Dict[str, float]) -> int:
        """Add members to sorted set."""
        if not self._client:
            return 0
        try:
            return await self._client.zadd(name, mapping)
        except Exception as e:
            logger.error(f"Redis ZADD failed for {name}: {e}")
            return 0
    
    async def zrevrange(self, name: str, start: int, end: int) -> List[str]:
        """Get sorted set members in reverse order."""
        if not self._client:
            return []
        try:
            return await self._client.zrevrange(name, start, end)
        except Exception as e:
            logger.error(f"Redis ZREVRANGE failed for {name}: {e}")
            return []
    
    async def xadd(self, name: str, fields: Dict[str, Any], 
                   message_id: str = "*", maxlen: Optional[int] = None) -> Optional[str]:
        """Add entry to stream."""
        if not self._client:
            return None
        
        try:
            # Serialize complex fields
            serialized_fields = {}
            for key, value in fields.items():
                if isinstance(value, (dict, list)):
                    serialized_fields[key] = json.dumps(value)
                else:
                    serialized_fields[key] = str(value)
            
            return await self._client.xadd(name, serialized_fields, id=message_id, maxlen=maxlen)
            
        except Exception as e:
            logger.error(f"Redis XADD failed for stream {name}: {e}")
            return None
    
    async def xread(self, streams: Dict[str, str], count: Optional[int] = None, 
                    block: Optional[int] = None) -> List:
        """Read from streams."""
        if not self._client:
            return []
        try:
            return await self._client.xread(streams, count=count, block=block)
        except Exception as e:
            logger.error(f"Redis XREAD failed: {e}")
            return []
    
    async def xreadgroup(self, group: str, consumer: str, streams: Dict[str, str], 
                        count: Optional[int] = None, block: Optional[int] = None,
                        noack: bool = False) -> List:
        """Read from streams using consumer group."""
        if not self._client:
            return []
        try:
            return await self._client.xreadgroup(group, consumer, streams, 
                                               count=count, block=block, noack=noack)
        except Exception as e:
            logger.error(f"Redis XREADGROUP failed: {e}")
            return []
    
    async def xgroup_create(self, name: str, group: str, id: str = "0", mkstream: bool = False) -> bool:
        """
        Create consumer group.

        This operation is idempotent - if the group already exists, it will return True.
        Only actual errors will return False.
        """
        if not self._client:
            return False
        try:
            await self._client.xgroup_create(name, group, id=id, mkstream=mkstream)
            logger.debug(f"Created consumer group '{group}' for stream '{name}'")
            return True
        except Exception as e:
            error_msg = str(e).lower()

            # Check if this is the expected "group already exists" error
            if "busygroup" in error_msg or "already exists" in error_msg:
                logger.debug(f"Consumer group '{group}' already exists for stream '{name}' - this is expected")
                return True

            # This is an actual error
            logger.error(f"Redis XGROUP CREATE failed for stream '{name}' group '{group}': {e}")
            return False
    
    async def xack(self, name: str, group: str, *message_ids: str) -> int:
        """Acknowledge messages."""
        if not self._client:
            return 0
        try:
            return await self._client.xack(name, group, *message_ids)
        except Exception as e:
            logger.error(f"Redis XACK failed: {e}")
            return 0
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        if not self._client:
            return []
        try:
            return await self._client.keys(pattern)
        except Exception as e:
            logger.error(f"Redis KEYS failed for pattern {pattern}: {e}")
            return []
    
    async def xrange(self, name: str, min: str = "-", max: str = "+", count: Optional[int] = None) -> List:
        """Get range of entries from stream."""
        if not self._client:
            return []
        try:
            return await self._client.xrange(name, min=min, max=max, count=count)
        except Exception as e:
            logger.error(f"Redis XRANGE failed for stream {name}: {e}")
            return []
    
    async def xtrim(self, name: str, maxlen: Optional[int] = None, minid: Optional[str] = None) -> int:
        """Trim stream to specified length or minimum ID."""
        if not self._client:
            return 0
        try:
            if minid:
                return await self._client.xtrim(name, minid=minid)
            elif maxlen:
                return await self._client.xtrim(name, maxlen=maxlen)
            return 0
        except Exception as e:
            logger.error(f"Redis XTRIM failed for stream {name}: {e}")
            return 0
    
    async def flushdb(self) -> bool:
        """Clear current database."""
        if not self._client:
            return False
        try:
            await self._client.flushdb()
            return True
        except Exception as e:
            logger.error(f"Redis FLUSHDB failed: {e}")
            return False


# Global Redis client instance
_redis_client: Optional[RedisClient] = None


async def get_redis_client() -> RedisClient:
    """Get global Redis client instance."""
    global _redis_client
    
    if _redis_client is None:
        _redis_client = RedisClient()
        if not await _redis_client.connect():
            logger.warning("Redis connection failed, operations will be limited")
    
    return _redis_client


async def close_redis_client() -> None:
    """Close global Redis client."""
    global _redis_client
    
    if _redis_client:
        await _redis_client.disconnect()
        _redis_client = None