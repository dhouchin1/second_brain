"""
Advanced API Authentication and Rate Limiting System.

Provides enterprise-grade authentication with JWT tokens, API keys,
rate limiting, and comprehensive security middleware.
"""

import asyncio
import hashlib
import hmac
from jose import jwt
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque
from functools import wraps

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from autom8.utils.logging import get_logger
from autom8.storage.redis.client import get_redis_client

logger = get_logger(__name__)


class AuthConfig(BaseModel):
    """Authentication configuration."""
    jwt_secret: str = Field(description="JWT signing secret")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiry_hours: int = Field(default=24, description="JWT token expiry in hours")
    api_key_length: int = Field(default=32, description="API key length")
    require_auth: bool = Field(default=True, description="Whether authentication is required")
    rate_limit_enabled: bool = Field(default=True, description="Whether rate limiting is enabled")


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    requests_per_minute: int = Field(default=60, description="Requests per minute limit")
    requests_per_hour: int = Field(default=1000, description="Requests per hour limit")
    requests_per_day: int = Field(default=10000, description="Requests per day limit")
    burst_size: int = Field(default=10, description="Burst size for token bucket")
    window_size_seconds: int = Field(default=60, description="Rate limit window size")


class UserPermissions(BaseModel):
    """User permissions model."""
    can_read: bool = Field(default=True, description="Can read data")
    can_write: bool = Field(default=False, description="Can write/modify data")
    can_admin: bool = Field(default=False, description="Admin privileges")
    allowed_endpoints: Set[str] = Field(default_factory=set, description="Allowed endpoints")
    rate_limit_multiplier: float = Field(default=1.0, description="Rate limit multiplier")


class AuthToken(BaseModel):
    """Authentication token model."""
    user_id: str = Field(description="User identifier")
    permissions: UserPermissions = Field(description="User permissions")
    issued_at: datetime = Field(description="Token issue time")
    expires_at: datetime = Field(description="Token expiry time")
    token_type: str = Field(default="bearer", description="Token type")


class APIKey(BaseModel):
    """API key model."""
    key_id: str = Field(description="API key identifier")
    key_hash: str = Field(description="Hashed API key")
    user_id: str = Field(description="Associated user ID")
    permissions: UserPermissions = Field(description="Key permissions")
    created_at: datetime = Field(description="Creation time")
    last_used: Optional[datetime] = Field(default=None, description="Last usage time")
    usage_count: int = Field(default=0, description="Usage count")
    is_active: bool = Field(default=True, description="Whether key is active")


class RateLimitEntry(BaseModel):
    """Rate limit tracking entry."""
    user_id: str = Field(description="User identifier")
    endpoint: str = Field(description="API endpoint")
    requests: deque = Field(default_factory=lambda: deque(maxlen=1000), description="Request timestamps")
    total_requests: int = Field(default=0, description="Total requests")
    last_request: datetime = Field(description="Last request time")
    blocked_until: Optional[datetime] = Field(default=None, description="Blocked until time")


class AuthenticationManager:
    """Advanced authentication and authorization manager."""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.rate_config = RateLimitConfig()
        self.security = HTTPBearer(auto_error=False)
        
        # In-memory storage (in production, use Redis/database)
        self.api_keys: Dict[str, APIKey] = {}
        self.rate_limits: Dict[str, RateLimitEntry] = defaultdict(lambda: RateLimitEntry(
            user_id="", endpoint="", last_request=datetime.now(timezone.utc)
        ))
        
        # Redis client for distributed rate limiting
        self._redis_client = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize authentication manager."""
        if self._initialized:
            return
        
        try:
            self._redis_client = await get_redis_client()
            if not self._redis_client.is_connected:
                logger.warning("Redis not available, using in-memory rate limiting")
                self._redis_client = None
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}, using in-memory rate limiting")
            self._redis_client = None
        
        # Load API keys (in production, load from database)
        await self._load_default_keys()
        
        self._initialized = True
        logger.info("Authentication manager initialized")
    
    async def _load_default_keys(self):
        """Load default API keys for demonstration."""
        # Create demo API keys
        demo_keys = [
            {
                "key_id": "demo_admin",
                "raw_key": "autom8_admin_key_demo_2024",
                "user_id": "admin_user",
                "permissions": UserPermissions(
                    can_read=True,
                    can_write=True,
                    can_admin=True,
                    allowed_endpoints={"*"},
                    rate_limit_multiplier=5.0
                )
            },
            {
                "key_id": "demo_user",
                "raw_key": "autom8_user_key_demo_2024",
                "user_id": "regular_user",
                "permissions": UserPermissions(
                    can_read=True,
                    can_write=False,
                    can_admin=False,
                    allowed_endpoints={"/api/system/status", "/api/models/status", "/api/complexity/stats"},
                    rate_limit_multiplier=1.0
                )
            },
            {
                "key_id": "demo_poweruser",
                "raw_key": "autom8_power_key_demo_2024",
                "user_id": "power_user",
                "permissions": UserPermissions(
                    can_read=True,
                    can_write=True,
                    can_admin=False,
                    allowed_endpoints={"*"},
                    rate_limit_multiplier=2.0
                )
            }
        ]
        
        for key_data in demo_keys:
            key_hash = self._hash_api_key(key_data["raw_key"])
            api_key = APIKey(
                key_id=key_data["key_id"],
                key_hash=key_hash,
                user_id=key_data["user_id"],
                permissions=key_data["permissions"],
                created_at=datetime.now(timezone.utc)
            )
            self.api_keys[key_data["raw_key"]] = api_key
        
        logger.info(f"Loaded {len(demo_keys)} demo API keys")
    
    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key for secure storage."""
        return hashlib.sha256(f"{api_key}{self.config.jwt_secret}".encode()).hexdigest()
    
    def generate_jwt_token(self, user_id: str, permissions: UserPermissions) -> str:
        """Generate JWT token for user."""
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(hours=self.config.jwt_expiry_hours)
        
        payload = {
            "user_id": user_id,
            "permissions": permissions.dict(),
            "iat": int(now.timestamp()),
            "exp": int(expiry.timestamp()),
            "iss": "autom8-api"
        }
        
        return jwt.encode(payload, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)
    
    def verify_jwt_token(self, token: str) -> Optional[AuthToken]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.config.jwt_secret, algorithms=[self.config.jwt_algorithm])
            
            return AuthToken(
                user_id=payload["user_id"],
                permissions=UserPermissions(**payload["permissions"]),
                issued_at=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
                expires_at=datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
            )
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
    
    async def verify_api_key(self, api_key: str) -> Optional[APIKey]:
        """Verify API key and return associated data."""
        if api_key not in self.api_keys:
            return None
        
        key_data = self.api_keys[api_key]
        
        if not key_data.is_active:
            return None
        
        # Update usage statistics
        key_data.last_used = datetime.now(timezone.utc)
        key_data.usage_count += 1
        
        return key_data
    
    async def authenticate_request(self, credentials: Optional[HTTPAuthorizationCredentials]) -> Optional[AuthToken]:
        """Authenticate request using Bearer token or API key."""
        if not self.config.require_auth:
            # Return default permissions when auth is disabled
            return AuthToken(
                user_id="anonymous",
                permissions=UserPermissions(can_read=True, can_write=True, can_admin=True),
                issued_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1)
            )
        
        if not credentials:
            return None
        
        token = credentials.credentials
        
        # Try JWT token first
        jwt_auth = self.verify_jwt_token(token)
        if jwt_auth:
            return jwt_auth
        
        # Try API key
        api_key_data = await self.verify_api_key(token)
        if api_key_data:
            return AuthToken(
                user_id=api_key_data.user_id,
                permissions=api_key_data.permissions,
                issued_at=api_key_data.created_at,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=self.config.jwt_expiry_hours)
            )
        
        return None
    
    async def check_rate_limit(self, user_id: str, endpoint: str, permissions: UserPermissions) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limits."""
        if not self.config.rate_limit_enabled:
            return True, {"allowed": True, "reason": "Rate limiting disabled"}
        
        now = datetime.now(timezone.utc)
        rate_key = f"{user_id}:{endpoint}"
        
        # Get or create rate limit entry
        if rate_key not in self.rate_limits:
            self.rate_limits[rate_key] = RateLimitEntry(
                user_id=user_id,
                endpoint=endpoint,
                last_request=now
            )
        
        entry = self.rate_limits[rate_key]
        
        # Check if currently blocked
        if entry.blocked_until and now < entry.blocked_until:
            remaining = (entry.blocked_until - now).total_seconds()
            return False, {
                "allowed": False,
                "reason": "Rate limit exceeded",
                "retry_after": int(remaining),
                "blocked_until": entry.blocked_until.isoformat()
            }
        
        # Apply rate limit multiplier based on permissions
        multiplier = permissions.rate_limit_multiplier
        effective_limit = int(self.rate_config.requests_per_minute * multiplier)
        
        # Clean old requests (older than window)
        window_start = now - timedelta(seconds=self.rate_config.window_size_seconds)
        entry.requests = deque([ts for ts in entry.requests if ts > window_start], maxlen=1000)
        
        # Check current rate
        current_requests = len(entry.requests)
        
        if current_requests >= effective_limit:
            # Block for remaining window time
            entry.blocked_until = now + timedelta(seconds=self.rate_config.window_size_seconds)
            return False, {
                "allowed": False,
                "reason": "Rate limit exceeded",
                "limit": effective_limit,
                "current": current_requests,
                "window_seconds": self.rate_config.window_size_seconds,
                "retry_after": self.rate_config.window_size_seconds
            }
        
        # Add current request
        entry.requests.append(now)
        entry.total_requests += 1
        entry.last_request = now
        
        return True, {
            "allowed": True,
            "limit": effective_limit,
            "remaining": effective_limit - current_requests - 1,
            "reset_time": (window_start + timedelta(seconds=self.rate_config.window_size_seconds)).isoformat()
        }
    
    def check_endpoint_permission(self, endpoint: str, permissions: UserPermissions) -> bool:
        """Check if user has permission to access endpoint."""
        if "*" in permissions.allowed_endpoints:
            return True
        
        # Check exact match
        if endpoint in permissions.allowed_endpoints:
            return True
        
        # Check prefix matches
        for allowed in permissions.allowed_endpoints:
            if allowed.endswith("*") and endpoint.startswith(allowed[:-1]):
                return True
        
        return False
    
    async def get_auth_stats(self) -> Dict[str, Any]:
        """Get authentication and rate limiting statistics."""
        total_keys = len(self.api_keys)
        active_keys = sum(1 for key in self.api_keys.values() if key.is_active)
        total_requests = sum(entry.total_requests for entry in self.rate_limits.values())
        
        # Rate limit statistics
        blocked_users = sum(1 for entry in self.rate_limits.values() 
                           if entry.blocked_until and entry.blocked_until > datetime.now(timezone.utc))
        
        # Recent activity
        recent_threshold = datetime.now(timezone.utc) - timedelta(minutes=5)
        recent_requests = sum(1 for entry in self.rate_limits.values() 
                             if entry.last_request > recent_threshold)
        
        return {
            "total_api_keys": total_keys,
            "active_api_keys": active_keys,
            "total_requests": total_requests,
            "blocked_users": blocked_users,
            "recent_requests_5min": recent_requests,
            "rate_limiting_enabled": self.config.rate_limit_enabled,
            "authentication_required": self.config.require_auth,
            "rate_limits": {
                "requests_per_minute": self.rate_config.requests_per_minute,
                "requests_per_hour": self.rate_config.requests_per_hour,
                "window_size_seconds": self.rate_config.window_size_seconds
            }
        }


# Global authentication manager
_auth_manager: Optional[AuthenticationManager] = None


async def get_auth_manager() -> AuthenticationManager:
    """Get global authentication manager."""
    global _auth_manager
    
    if _auth_manager is None:
        import os
        config = AuthConfig(
            jwt_secret=os.getenv("AUTOM8_JWT_SECRET", "autom8_super_secret_key_2024_change_in_production"),
            require_auth=os.getenv("AUTOM8_REQUIRE_AUTH", "true").lower() == "true",
            rate_limit_enabled=os.getenv("AUTOM8_RATE_LIMIT", "true").lower() == "true"
        )
        _auth_manager = AuthenticationManager(config)
        await _auth_manager.initialize()
    
    return _auth_manager


# Dependency injection functions
async def get_current_user(request: Request, credentials: Optional[HTTPAuthorizationCredentials] = None) -> AuthToken:
    """Get current authenticated user."""
    auth_manager = await get_auth_manager()
    
    # Extract credentials from Authorization header if not provided
    if not credentials:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    
    auth_token = await auth_manager.authenticate_request(credentials)
    
    if not auth_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return auth_token


async def check_rate_limit_dependency(request: Request, auth_token: AuthToken) -> bool:
    """Check rate limiting for current request."""
    auth_manager = await get_auth_manager()
    
    endpoint = request.url.path
    allowed, info = await auth_manager.check_rate_limit(
        auth_token.user_id, 
        endpoint, 
        auth_token.permissions
    )
    
    if not allowed:
        # Add rate limit headers
        headers = {
            "X-RateLimit-Limit": str(info.get("limit", 0)),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(info.get("retry_after", 60))
        }
        
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {info.get('reason', 'Too many requests')}",
            headers=headers
        )
    
    return True


def require_permissions(read: bool = False, write: bool = False, admin: bool = False):
    """Decorator to require specific permissions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract auth_token from kwargs or get from current context
            auth_token = kwargs.get('auth_token')
            if not auth_token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            permissions = auth_token.permissions
            
            if read and not permissions.can_read:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Read permission required"
                )
            
            if write and not permissions.can_write:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Write permission required"
                )
            
            if admin and not permissions.can_admin:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin permission required"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Utility functions
def generate_api_key() -> str:
    """Generate a new API key."""
    import secrets
    return secrets.token_urlsafe(32)


async def revoke_api_key(api_key: str) -> bool:
    """Revoke an API key."""
    auth_manager = await get_auth_manager()
    
    if api_key in auth_manager.api_keys:
        auth_manager.api_keys[api_key].is_active = False
        return True
    
    return False