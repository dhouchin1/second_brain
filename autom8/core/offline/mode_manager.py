"""
Offline Mode Manager for Autom8

Provides comprehensive offline mode support with graceful degradation,
ensuring continuous operation when cloud services are unavailable.
Prioritizes local models and cached responses while maintaining functionality.
"""

import asyncio
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field

from autom8.core.error_management import EnhancedErrorManager, ErrorCategory, ErrorSeverity
from autom8.models.complexity import ComplexityTier, ComplexityScore  
from autom8.models.routing import Model, ModelProvider, ModelType, PrivacyLevel
from autom8.storage.redis.events import EventBus
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class ConnectivityStatus(str, Enum):
    """System connectivity status levels."""
    ONLINE = "online"           # Full connectivity to all services
    DEGRADED = "degraded"       # Partial connectivity, some services unavailable
    LOCAL_ONLY = "local_only"   # Only local services available
    OFFLINE = "offline"         # No connectivity, cached responses only


class ServiceType(str, Enum):
    """Types of external services that can be offline."""
    CLOUD_MODELS = "cloud_models"       # External LLM APIs (OpenAI, Anthropic, etc.)
    REDIS = "redis"                     # Redis cache and event bus
    INTERNET = "internet"               # General internet connectivity
    LOCAL_MODELS = "local_models"       # Local model inference (Ollama)
    VECTOR_DB = "vector_db"            # Vector database for embeddings
    CONFIG_SYNC = "config_sync"        # Configuration synchronization
    TELEMETRY = "telemetry"            # Usage analytics and telemetry


class ServiceStatus(BaseModel):
    """Status of an external service."""
    service_type: ServiceType
    is_available: bool = True
    last_check: datetime = Field(default_factory=datetime.utcnow)
    last_success: Optional[datetime] = None
    consecutive_failures: int = 0
    error_message: Optional[str] = None
    response_time_ms: Optional[float] = None
    check_interval_seconds: int = 30


class OfflineCapability(BaseModel):
    """Defines what functionality is available in offline mode."""
    can_route_queries: bool = True
    can_use_cached_responses: bool = True
    can_analyze_complexity: bool = True
    available_models: List[str] = Field(default_factory=list)
    degraded_features: List[str] = Field(default_factory=list)
    unavailable_features: List[str] = Field(default_factory=list)
    offline_storage_available: bool = True
    
    @computed_field
    @property
    def functionality_score(self) -> float:
        """Calculate overall functionality score in current mode (0-1)."""
        base_score = 0.5  # Base functionality
        
        if self.can_route_queries:
            base_score += 0.3
        if self.can_use_cached_responses:
            base_score += 0.1
        if len(self.available_models) > 0:
            base_score += 0.1
        
        # Penalty for degraded features
        degradation_penalty = len(self.degraded_features) * 0.02
        unavailable_penalty = len(self.unavailable_features) * 0.05
        
        return max(0.0, min(1.0, base_score - degradation_penalty - unavailable_penalty))


class CachedResponse(BaseModel):
    """Cached response for offline use."""
    id: UUID = Field(default_factory=uuid4)
    query_hash: str = Field(description="Hash of the original query")
    query_text: str = Field(description="Original query text")
    response_text: str = Field(description="Cached response")
    model_used: str = Field(description="Model that generated this response")
    complexity_tier: ComplexityTier = Field(description="Query complexity")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = 0
    quality_score: float = Field(default=0.8, ge=0.0, le=1.0)
    
    # Context
    context_size: int = 0
    estimated_tokens: int = 0
    original_cost: float = 0.0
    
    @computed_field
    @property
    def age_hours(self) -> float:
        """Age of cached response in hours."""
        return (datetime.utcnow() - self.created_at).total_seconds() / 3600
    
    @computed_field
    @property
    def freshness_score(self) -> float:
        """Freshness score based on age and access patterns."""
        # Decay over time
        age_penalty = min(0.5, self.age_hours / 168)  # Week half-life
        
        # Boost for frequent access
        access_boost = min(0.2, self.access_count / 10)
        
        return max(0.0, min(1.0, 1.0 - age_penalty + access_boost))


class OfflineFallbackConfig(BaseModel):
    """Configuration for offline fallback behavior."""
    # Service monitoring
    health_check_interval: int = 30  # seconds
    service_timeout: float = 5.0  # seconds
    max_consecutive_failures: int = 3
    
    # Model preferences for offline mode
    preferred_local_models: List[str] = Field(default_factory=lambda: [
        "llama3.2:3b", "llama3.2:7b", "phi-3", "mistral:7b"
    ])
    minimum_local_model: str = "llama3.2:3b"
    
    # Cache management
    max_cached_responses: int = 1000
    cache_expiry_hours: int = 168  # 1 week
    enable_response_caching: bool = True
    
    # Degradation behavior
    auto_degrade_threshold: int = 2  # failures before auto-degradation
    auto_recover_threshold: int = 1  # successes before auto-recovery
    fallback_to_simple_routing: bool = True
    
    # User experience
    notify_offline_mode: bool = True
    show_degraded_features: bool = True
    offline_mode_timeout: int = 300  # seconds before user notification


class OfflineModeManager:
    """
    Manages offline mode operations and graceful degradation.
    
    Provides comprehensive offline support by:
    - Monitoring external service availability
    - Maintaining cached responses for reuse
    - Falling back to local models when cloud services are unavailable
    - Providing degraded but functional service during outages
    """
    
    def __init__(
        self,
        error_manager: EnhancedErrorManager,
        event_bus: EventBus,
        config: Optional[OfflineFallbackConfig] = None
    ):
        """
        Initialize offline mode manager.
        
        Args:
            error_manager: Error management system for handling service failures
            event_bus: Event bus for offline mode notifications
            config: Offline fallback configuration
        """
        self.error_manager = error_manager
        self.event_bus = event_bus
        self.config = config or OfflineFallbackConfig()
        
        # Service monitoring
        self.service_statuses: Dict[ServiceType, ServiceStatus] = {
            service_type: ServiceStatus(service_type=service_type)
            for service_type in ServiceType
        }
        
        # Offline capabilities and state
        self.current_status = ConnectivityStatus.ONLINE
        self.offline_capabilities = OfflineCapability()
        self.offline_since: Optional[datetime] = None
        self.last_connectivity_check = datetime.utcnow()
        
        # Response caching
        self.cached_responses: Dict[str, CachedResponse] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Local model management
        self.available_local_models: List[Model] = []
        self.local_model_health: Dict[str, float] = {}
        
        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
        
        logger.info("Offline mode manager initialized")
    
    async def initialize(self) -> None:
        """Initialize offline mode manager and start monitoring."""
        
        # Perform initial service health checks
        await self._perform_initial_health_checks()
        
        # Update offline capabilities
        await self._update_offline_capabilities()
        
        # Start monitoring task
        await self.start_monitoring()
        
        logger.info(f"Offline mode manager initialized - Status: {self.current_status}")
    
    async def start_monitoring(self) -> None:
        """Start continuous service monitoring."""
        
        if self._monitoring_task and not self._monitoring_task.done():
            logger.warning("Monitoring already started")
            return
        
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started offline mode monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop service monitoring."""
        
        self._is_monitoring = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped offline mode monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Continuous monitoring loop for service health."""
        
        while self._is_monitoring:
            try:
                # Check service health
                await self._check_all_services()
                
                # Update connectivity status
                old_status = self.current_status
                new_status = self._determine_connectivity_status()
                
                if old_status != new_status:
                    await self._handle_status_change(old_status, new_status)
                
                # Clean up expired cache entries
                await self._cleanup_cache()
                
                # Wait for next check
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                await self.error_manager.handle_error(
                    error=e,
                    category=ErrorCategory.NETWORK,
                    severity=ErrorSeverity.MEDIUM,
                    component="offline_mode_manager",
                    context={"operation": "monitoring_loop"}
                )
                
                # Continue monitoring despite errors
                await asyncio.sleep(self.config.health_check_interval)
    
    async def _perform_initial_health_checks(self) -> None:
        """Perform initial health checks for all services."""
        
        logger.info("Performing initial service health checks")
        
        # Check each service
        check_tasks = [
            self._check_service_health(service_type)
            for service_type in ServiceType
        ]
        
        await asyncio.gather(*check_tasks, return_exceptions=True)
        
        # Log initial status
        online_services = [
            st.service_type.value for st in self.service_statuses.values()
            if st.is_available
        ]
        offline_services = [
            st.service_type.value for st in self.service_statuses.values()
            if not st.is_available
        ]
        
        logger.info(f"Initial health check complete. Online: {online_services}, Offline: {offline_services}")
    
    async def _check_all_services(self) -> None:
        """Check health of all monitored services."""
        
        # Run health checks concurrently
        check_tasks = [
            self._check_service_health(service_type)
            for service_type in ServiceType
        ]
        
        results = await asyncio.gather(*check_tasks, return_exceptions=True)
        
        # Log any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                service_type = list(ServiceType)[i]
                logger.error(f"Health check failed for {service_type}: {result}")
    
    async def _check_service_health(self, service_type: ServiceType) -> None:
        """Check health of a specific service."""
        
        status = self.service_statuses[service_type]
        start_time = time.perf_counter()
        
        try:
            # Service-specific health checks
            is_healthy = await self._perform_service_check(service_type)
            response_time = (time.perf_counter() - start_time) * 1000  # ms
            
            if is_healthy:
                status.is_available = True
                status.last_success = datetime.utcnow()
                status.consecutive_failures = 0
                status.error_message = None
                status.response_time_ms = response_time
            else:
                status.is_available = False
                status.consecutive_failures += 1
                status.response_time_ms = None
            
        except Exception as e:
            status.is_available = False
            status.consecutive_failures += 1
            status.error_message = str(e)
            status.response_time_ms = None
            
            # Log service failures
            if status.consecutive_failures <= 3:  # Avoid spam
                logger.warning(f"Service {service_type} health check failed: {e}")
        
        finally:
            status.last_check = datetime.utcnow()
    
    async def _perform_service_check(self, service_type: ServiceType) -> bool:
        """Perform actual health check for a service type."""
        
        if service_type == ServiceType.CLOUD_MODELS:
            # Check if we can reach external APIs
            return await self._check_cloud_model_connectivity()
        
        elif service_type == ServiceType.REDIS:
            # Check Redis connectivity
            return await self._check_redis_connectivity()
        
        elif service_type == ServiceType.INTERNET:
            # Check general internet connectivity
            return await self._check_internet_connectivity()
        
        elif service_type == ServiceType.LOCAL_MODELS:
            # Check local model availability (Ollama, etc.)
            return await self._check_local_models_connectivity()
        
        elif service_type == ServiceType.VECTOR_DB:
            # Check vector database connectivity
            return await self._check_vector_db_connectivity()
        
        elif service_type == ServiceType.CONFIG_SYNC:
            # Check configuration sync service
            return await self._check_config_sync_connectivity()
        
        elif service_type == ServiceType.TELEMETRY:
            # Check telemetry service
            return await self._check_telemetry_connectivity()
        
        return True
    
    async def _check_cloud_model_connectivity(self) -> bool:
        """Check connectivity to cloud model APIs."""
        try:
            # Simple connectivity check - would integrate with actual API clients
            await asyncio.wait_for(
                asyncio.sleep(0.01),  # Placeholder for actual API check
                timeout=self.config.service_timeout
            )
            return True
        except asyncio.TimeoutError:
            return False
    
    async def _check_redis_connectivity(self) -> bool:
        """Check Redis connectivity."""
        try:
            # Would integrate with actual Redis client
            await asyncio.wait_for(
                asyncio.sleep(0.01),  # Placeholder for Redis ping
                timeout=self.config.service_timeout
            )
            return True
        except asyncio.TimeoutError:
            return False
    
    async def _check_internet_connectivity(self) -> bool:
        """Check general internet connectivity."""
        try:
            # Would use actual connectivity check (DNS lookup, HTTP request, etc.)
            await asyncio.wait_for(
                asyncio.sleep(0.01),  # Placeholder
                timeout=self.config.service_timeout
            )
            return True
        except asyncio.TimeoutError:
            return False
    
    async def _check_local_models_connectivity(self) -> bool:
        """Check local model service connectivity."""
        try:
            # Would check Ollama or other local inference services
            await asyncio.wait_for(
                asyncio.sleep(0.01),  # Placeholder for Ollama health check
                timeout=self.config.service_timeout
            )
            return True
        except asyncio.TimeoutError:
            return False
    
    async def _check_vector_db_connectivity(self) -> bool:
        """Check vector database connectivity."""
        try:
            # Would check sqlite-vec or other vector database
            await asyncio.wait_for(
                asyncio.sleep(0.01),  # Placeholder
                timeout=self.config.service_timeout
            )
            return True
        except asyncio.TimeoutError:
            return False
    
    async def _check_config_sync_connectivity(self) -> bool:
        """Check configuration synchronization service."""
        # Config sync is typically local, so usually available
        return True
    
    async def _check_telemetry_connectivity(self) -> bool:
        """Check telemetry service connectivity."""
        try:
            # Would check actual telemetry endpoint
            await asyncio.wait_for(
                asyncio.sleep(0.01),  # Placeholder
                timeout=self.config.service_timeout
            )
            return True
        except asyncio.TimeoutError:
            return False
    
    def _determine_connectivity_status(self) -> ConnectivityStatus:
        """Determine overall connectivity status based on service health."""
        
        # Check critical services
        cloud_models_ok = self.service_statuses[ServiceType.CLOUD_MODELS].is_available
        local_models_ok = self.service_statuses[ServiceType.LOCAL_MODELS].is_available
        internet_ok = self.service_statuses[ServiceType.INTERNET].is_available
        redis_ok = self.service_statuses[ServiceType.REDIS].is_available
        
        # Determine status
        if cloud_models_ok and local_models_ok and internet_ok and redis_ok:
            return ConnectivityStatus.ONLINE
        
        elif local_models_ok and (cloud_models_ok or internet_ok):
            return ConnectivityStatus.DEGRADED
        
        elif local_models_ok:
            return ConnectivityStatus.LOCAL_ONLY
        
        else:
            return ConnectivityStatus.OFFLINE
    
    async def _handle_status_change(
        self,
        old_status: ConnectivityStatus,
        new_status: ConnectivityStatus
    ) -> None:
        """Handle connectivity status changes."""
        
        self.current_status = new_status
        
        # Update offline timestamp
        if new_status != ConnectivityStatus.ONLINE and old_status == ConnectivityStatus.ONLINE:
            self.offline_since = datetime.utcnow()
        elif new_status == ConnectivityStatus.ONLINE:
            self.offline_since = None
        
        # Update offline capabilities
        await self._update_offline_capabilities()
        
        # Emit status change event
        await self._emit_status_change_event(old_status, new_status)
        
        # Log status change
        logger.info(f"Connectivity status changed: {old_status} -> {new_status}")
        
        if new_status != ConnectivityStatus.ONLINE:
            logger.warning(f"Operating in {new_status} mode - some features may be limited")
    
    async def _update_offline_capabilities(self) -> None:
        """Update available offline capabilities based on current status."""
        
        capabilities = OfflineCapability()
        
        if self.current_status == ConnectivityStatus.ONLINE:
            # Full functionality available
            capabilities.can_route_queries = True
            capabilities.can_use_cached_responses = True
            capabilities.can_analyze_complexity = True
            capabilities.available_models = self._get_all_available_models()
            capabilities.offline_storage_available = True
        
        elif self.current_status == ConnectivityStatus.DEGRADED:
            # Most functionality available, some cloud services down
            capabilities.can_route_queries = True
            capabilities.can_use_cached_responses = True
            capabilities.can_analyze_complexity = True
            capabilities.available_models = self._get_available_models_degraded()
            capabilities.degraded_features = ["cloud_model_routing", "real_time_telemetry"]
            capabilities.offline_storage_available = True
        
        elif self.current_status == ConnectivityStatus.LOCAL_ONLY:
            # Local functionality only
            capabilities.can_route_queries = True
            capabilities.can_use_cached_responses = True
            capabilities.can_analyze_complexity = True
            capabilities.available_models = self._get_local_models_only()
            capabilities.degraded_features = ["cloud_models", "remote_sync", "telemetry"]
            capabilities.unavailable_features = ["cloud_model_routing", "remote_configuration"]
            capabilities.offline_storage_available = True
        
        elif self.current_status == ConnectivityStatus.OFFLINE:
            # Minimal functionality, cached responses only
            capabilities.can_route_queries = False
            capabilities.can_use_cached_responses = True
            capabilities.can_analyze_complexity = False
            capabilities.available_models = []
            capabilities.degraded_features = ["query_routing", "complexity_analysis"]
            capabilities.unavailable_features = [
                "model_inference", "cloud_models", "local_models", 
                "real_time_routing", "telemetry", "remote_sync"
            ]
            capabilities.offline_storage_available = True
        
        self.offline_capabilities = capabilities
    
    def _get_all_available_models(self) -> List[str]:
        """Get all available models when online."""
        # Would integrate with actual model registry
        return [
            "claude-sonnet", "gpt-4", "llama3.2:7b", "llama3.2:3b", "mistral:7b"
        ]
    
    def _get_available_models_degraded(self) -> List[str]:
        """Get available models in degraded mode."""
        models = []
        
        # Include local models if available
        if self.service_statuses[ServiceType.LOCAL_MODELS].is_available:
            models.extend(["llama3.2:7b", "llama3.2:3b", "mistral:7b"])
        
        # Include some cloud models if internet is available
        if self.service_statuses[ServiceType.INTERNET].is_available:
            models.extend(["claude-sonnet"])  # Conservative selection
        
        return models
    
    def _get_local_models_only(self) -> List[str]:
        """Get only local models."""
        if self.service_statuses[ServiceType.LOCAL_MODELS].is_available:
            return ["llama3.2:7b", "llama3.2:3b", "mistral:7b"]
        return []
    
    async def _emit_status_change_event(
        self,
        old_status: ConnectivityStatus,
        new_status: ConnectivityStatus
    ) -> None:
        """Emit connectivity status change event."""
        
        try:
            from autom8.models.events import EventType, EventPriority
            
            await self.event_bus.emit_event(
                event_type=EventType.SYSTEM_STATUS_CHANGED,
                priority=EventPriority.HIGH,
                data={
                    'old_status': old_status.value,
                    'new_status': new_status.value,
                    'offline_since': self.offline_since.isoformat() if self.offline_since else None,
                    'functionality_score': self.offline_capabilities.functionality_score,
                    'available_models': self.offline_capabilities.available_models,
                    'degraded_features': self.offline_capabilities.degraded_features,
                    'unavailable_features': self.offline_capabilities.unavailable_features
                }
            )
        except Exception as e:
            logger.error(f"Failed to emit status change event: {e}")
    
    async def _cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        
        current_time = datetime.utcnow()
        expired_keys = []
        
        for key, cached_response in self.cached_responses.items():
            # Check if response is expired
            age_hours = (current_time - cached_response.created_at).total_seconds() / 3600
            if age_hours > self.config.cache_expiry_hours:
                expired_keys.append(key)
        
        # Remove expired entries
        for key in expired_keys:
            del self.cached_responses[key]
        
        # Limit cache size
        if len(self.cached_responses) > self.config.max_cached_responses:
            # Remove oldest entries
            sorted_responses = sorted(
                self.cached_responses.items(),
                key=lambda x: x[1].last_accessed
            )
            
            excess_count = len(self.cached_responses) - self.config.max_cached_responses
            for i in range(excess_count):
                del self.cached_responses[sorted_responses[i][0]]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    # Public API methods
    
    async def get_offline_routing_recommendation(
        self,
        complexity: ComplexityScore,
        preferred_models: List[str]
    ) -> Optional[str]:
        """Get routing recommendation for offline/degraded mode."""
        
        available_models = set(self.offline_capabilities.available_models)
        
        # Try preferred models first
        for model in preferred_models:
            if model in available_models:
                return model
        
        # Fall back to any available model suitable for complexity
        if complexity.recommended_tier in [ComplexityTier.TRIVIAL, ComplexityTier.SIMPLE]:
            # Simple queries can use lightweight models
            for model in ["llama3.2:3b", "phi-3"]:
                if model in available_models:
                    return model
        
        elif complexity.recommended_tier == ComplexityTier.MODERATE:
            # Moderate queries need more capable models
            for model in ["llama3.2:7b", "mistral:7b"]:
                if model in available_models:
                    return model
        
        # Return any available model as last resort
        if available_models:
            return list(available_models)[0]
        
        return None
    
    async def cache_response(
        self,
        query: str,
        response: str,
        model_used: str,
        complexity: ComplexityScore,
        context_size: int = 0,
        estimated_cost: float = 0.0
    ) -> None:
        """Cache a response for offline use."""
        
        if not self.config.enable_response_caching:
            return
        
        # Generate query hash
        import hashlib
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        
        # Create cached response
        cached_response = CachedResponse(
            query_hash=query_hash,
            query_text=query,
            response_text=response,
            model_used=model_used,
            complexity_tier=complexity.recommended_tier,
            context_size=context_size,
            estimated_tokens=len(response.split()),  # Rough estimate
            original_cost=estimated_cost
        )
        
        # Store in cache
        self.cached_responses[query_hash] = cached_response
        
        logger.debug(f"Cached response for query hash {query_hash}")
    
    async def get_cached_response(self, query: str) -> Optional[CachedResponse]:
        """Get cached response for a query."""
        
        # Generate query hash
        import hashlib
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        
        cached_response = self.cached_responses.get(query_hash)
        if cached_response:
            # Update access tracking
            cached_response.last_accessed = datetime.utcnow()
            cached_response.access_count += 1
            self.cache_hits += 1
            
            logger.debug(f"Cache hit for query hash {query_hash}")
            return cached_response
        
        self.cache_misses += 1
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive offline mode status."""
        
        return {
            'connectivity_status': self.current_status.value,
            'offline_since': self.offline_since.isoformat() if self.offline_since else None,
            'functionality_score': self.offline_capabilities.functionality_score,
            'capabilities': self.offline_capabilities.dict(),
            'service_statuses': {
                service_type.value: {
                    'available': status.is_available,
                    'last_check': status.last_check.isoformat(),
                    'consecutive_failures': status.consecutive_failures,
                    'response_time_ms': status.response_time_ms
                }
                for service_type, status in self.service_statuses.items()
            },
            'cache_stats': {
                'cached_responses': len(self.cached_responses),
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0
            }
        }
    
    def is_feature_available(self, feature: str) -> bool:
        """Check if a specific feature is available in current mode."""
        
        return (
            feature not in self.offline_capabilities.unavailable_features and
            feature not in self.offline_capabilities.degraded_features
        )
    
    def get_degradation_message(self) -> Optional[str]:
        """Get user-friendly message about current degradation."""
        
        if self.current_status == ConnectivityStatus.ONLINE:
            return None
        
        elif self.current_status == ConnectivityStatus.DEGRADED:
            return (
                "Some cloud services are unavailable. Using local models and cached responses. "
                "Functionality may be limited."
            )
        
        elif self.current_status == ConnectivityStatus.LOCAL_ONLY:
            return (
                "Operating in local-only mode. Only local models are available. "
                "Cloud features are temporarily unavailable."
            )
        
        elif self.current_status == ConnectivityStatus.OFFLINE:
            return (
                "Operating in offline mode. Only cached responses are available. "
                "New queries cannot be processed until connectivity is restored."
            )
        
        return "Unknown connectivity status."


# Utility functions for integration

async def create_offline_mode_manager(
    error_manager: EnhancedErrorManager,
    event_bus: EventBus,
    config: Optional[OfflineFallbackConfig] = None
) -> OfflineModeManager:
    """Create and initialize an offline mode manager."""
    
    manager = OfflineModeManager(error_manager, event_bus, config)
    await manager.initialize()
    return manager


def get_local_first_fallback_models() -> List[str]:
    """Get prioritized list of local fallback models."""
    return [
        "llama3.2:7b",    # Best balance of capability and speed
        "llama3.2:3b",    # Faster, good for simple tasks
        "mistral:7b",     # Alternative architecture
        "phi-3",          # Smallest, fastest fallback
    ]