"""
Redis Shared Memory Implementation

Fast, efficient memory sharing between agents
without context bloat.
"""

from autom8.storage.redis.client import RedisClient, get_redis_client
from autom8.storage.redis.shared_memory import RedisSharedMemory, get_shared_memory

__all__ = [
    "RedisClient",
    "get_redis_client", 
    "RedisSharedMemory",
    "get_shared_memory",
]