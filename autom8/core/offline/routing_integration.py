"""
Offline Mode Routing Integration

Integrates offline mode manager with the existing model routing system,
providing seamless fallback behavior and graceful degradation.
"""

import asyncio
from typing import Any, Dict, List, Optional

from autom8.core.offline.mode_manager import (
    OfflineModeManager, ConnectivityStatus, ServiceType, get_local_first_fallback_models
)
from autom8.core.routing.router import ModelRouter
from autom8.core.routing.adaptive_router import AdaptiveModelRouter
from autom8.core.error_management import ErrorCategory, ErrorSeverity
from autom8.models.complexity import ComplexityScore, ComplexityTier
from autom8.models.routing import Model, ModelSelection, RoutingPreferences, ModelProvider, ModelType, PrivacyLevel
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class OfflineAwareRouter:
    """
    Router wrapper that provides offline-aware model selection.
    
    Enhances existing routers with offline mode capabilities:
    - Automatic fallback to local models when cloud services are down
    - Cached response serving during outages
    - Graceful degradation with user notifications
    - Intelligent model selection based on connectivity status
    """
    
    def __init__(
        self,
        base_router: ModelRouter,
        offline_manager: OfflineModeManager
    ):
        """
        Initialize offline-aware router.
        
        Args:
            base_router: Base router (ModelRouter or AdaptiveModelRouter)
            offline_manager: Offline mode manager for connectivity monitoring
        """
        self.base_router = base_router
        self.offline_manager = offline_manager
        
        # Integration state
        self.fallback_active = False
        self.last_offline_notification: Optional[str] = None
        
        logger.info("Offline-aware router initialized")
    
    async def route_query(
        self,
        query: str,
        complexity: ComplexityScore,
        context_tokens: int = 0,
        preferences: Optional[RoutingPreferences] = None
    ) -> ModelSelection:
        """
        Route query with offline mode awareness.
        
        Provides intelligent fallback behavior based on connectivity status:
        - Online: Use normal routing with full model selection
        - Degraded: Prefer local models, use cached responses when possible
        - Local Only: Use only local models
        - Offline: Serve cached responses only
        """
        
        # Check for cached response first if offline or degraded
        status = self.offline_manager.current_status
        if status in [ConnectivityStatus.DEGRADED, ConnectivityStatus.LOCAL_ONLY, ConnectivityStatus.OFFLINE]:
            cached_response = await self.offline_manager.get_cached_response(query)
            if cached_response:
                return await self._create_cached_response_selection(cached_response, complexity)
        
        # Handle fully offline mode
        if status == ConnectivityStatus.OFFLINE:
            if cached_response is None:
                raise OfflineUnavailableError(
                    "Service is offline and no cached response available",
                    self.offline_manager.get_degradation_message()
                )
        
        # Adjust preferences for offline/degraded mode
        adjusted_preferences = self._adjust_preferences_for_offline_mode(preferences, status)
        
        try:
            # Attempt normal routing with adjusted preferences
            selection = await self.base_router.route_query(
                query=query,
                complexity=complexity,
                context_tokens=context_tokens,
                preferences=adjusted_preferences
            )
            
            # Validate model availability in current mode
            available_models = set(self.offline_manager.offline_capabilities.available_models)
            if selection.primary_model.name not in available_models:
                # Primary model not available, try fallback
                if selection.fallback_model and selection.fallback_model.name in available_models:
                    selection = self._switch_to_fallback(selection)
                else:
                    # Neither primary nor fallback available, find alternative
                    selection = await self._find_offline_alternative(selection, complexity, status)
            
            # Cache the selection for future offline use
            await self._cache_selection_info(query, selection, complexity)
            
            return selection
            
        except Exception as e:
            # Handle routing failures during degraded connectivity
            logger.warning(f"Primary routing failed during {status} mode: {e}")
            
            # Try cached response as emergency fallback
            if cached_response:
                logger.info("Using cached response due to routing failure")
                return await self._create_cached_response_selection(cached_response, complexity)
            
            # Try emergency local routing
            emergency_selection = await self._emergency_local_routing(query, complexity, preferences)
            if emergency_selection:
                return emergency_selection
            
            # Ultimate fallback: re-raise original error with context
            raise OfflineRoutingError(
                f"Routing failed in {status} mode: {str(e)}",
                status,
                self.offline_manager.get_degradation_message()
            ) from e
    
    def _adjust_preferences_for_offline_mode(
        self,
        preferences: Optional[RoutingPreferences],
        status: ConnectivityStatus
    ) -> RoutingPreferences:
        """Adjust routing preferences based on connectivity status."""
        
        # Use default preferences if none provided
        if preferences is None:
            preferences = RoutingPreferences()
        else:
            # Create copy to avoid modifying original
            preferences = RoutingPreferences(**preferences.model_dump())
        
        # Adjust based on connectivity status
        if status == ConnectivityStatus.DEGRADED:
            # Prefer local models in degraded mode
            preferences.prefer_local = True
            preferences.max_latency_ms = min(preferences.max_latency_ms, 10000)  # More lenient timing
            preferences.preferred_models = self._get_degraded_model_preferences(preferences.preferred_models)
        
        elif status == ConnectivityStatus.LOCAL_ONLY:
            # Force local-only operation
            preferences.prefer_local = True
            preferences.max_cost_per_query = 0.0  # No cost for local models
            preferences.preferred_models = self._get_local_only_model_preferences()
            # Block cloud models
            cloud_models = ["claude-sonnet", "claude-haiku", "gpt-4", "gpt-3.5-turbo"]
            preferences.blocked_models.extend(cloud_models)
        
        elif status == ConnectivityStatus.OFFLINE:
            # In offline mode, preferences don't matter as we can't route
            pass
        
        return preferences
    
    def _get_degraded_model_preferences(self, current_preferences: List[str]) -> List[str]:
        """Get model preferences optimized for degraded connectivity."""
        
        # Start with local fallback models
        degraded_preferences = get_local_first_fallback_models()
        
        # Add current preferences that are local
        local_models = {"llama3.2:7b", "llama3.2:3b", "mistral:7b", "phi-3"}
        for model in current_preferences:
            if model in local_models and model not in degraded_preferences:
                degraded_preferences.append(model)
        
        # Add one reliable cloud model if internet is available
        if self.offline_manager.service_statuses[ServiceType.INTERNET].is_available:
            degraded_preferences.append("claude-haiku")  # Fast, reliable cloud model
        
        return degraded_preferences
    
    def _get_local_only_model_preferences(self) -> List[str]:
        """Get model preferences for local-only mode."""
        return get_local_first_fallback_models()
    
    def _switch_to_fallback(self, selection: ModelSelection) -> ModelSelection:
        """Switch a selection to use its fallback model."""
        
        if not selection.fallback_model:
            raise ValueError("No fallback model available in selection")
        
        # Create new selection with fallback as primary
        fallback_selection = ModelSelection(
            primary_model=selection.fallback_model,
            alternatives=selection.alternatives,
            fallback_model=None,  # No fallback for the fallback
            selection_reasoning=f"Switched to fallback due to connectivity: {selection.selection_reasoning}",
            confidence=max(0.0, selection.confidence - 0.2),  # Reduced confidence
            routing_factors=selection.routing_factors,
            estimated_quality=max(0.0, selection.estimated_quality - 0.1),
            estimated_latency_ms=selection.estimated_latency_ms * 1.2,  # Assume slightly slower
            estimated_cost=selection.estimated_cost * 0.5,  # Local models typically cheaper
            complexity_tier=selection.complexity_tier,
            user_preferences=selection.user_preferences,
            constraints=selection.constraints + ["offline_fallback_active"]
        )
        
        logger.info(f"Switched to fallback model: {selection.fallback_model.name}")
        return fallback_selection
    
    async def _find_offline_alternative(
        self,
        original_selection: ModelSelection,
        complexity: ComplexityScore,
        status: ConnectivityStatus
    ) -> ModelSelection:
        """Find alternative model when original selection is unavailable offline."""
        
        # Get offline routing recommendation
        preferred_models = [original_selection.primary_model.name]
        if original_selection.fallback_model:
            preferred_models.append(original_selection.fallback_model.name)
        
        recommended_model = await self.offline_manager.get_offline_routing_recommendation(
            complexity, preferred_models
        )
        
        if not recommended_model:
            raise OfflineUnavailableError(
                f"No suitable models available in {status} mode",
                self.offline_manager.get_degradation_message()
            )
        
        # Find the model object
        # This would integrate with the actual model registry
        alternative_model = self._get_model_by_name(recommended_model)
        if not alternative_model:
            raise OfflineUnavailableError(
                f"Recommended model '{recommended_model}' not found",
                self.offline_manager.get_degradation_message()
            )
        
        # Create alternative selection
        alternative_selection = ModelSelection(
            primary_model=alternative_model,
            alternatives=[],
            fallback_model=None,
            selection_reasoning=f"Offline alternative for {original_selection.primary_model.name} in {status} mode",
            confidence=0.6,  # Lower confidence for emergency routing
            routing_factors={"offline_emergency": 1.0},
            estimated_quality=0.7,  # Conservative estimate
            estimated_latency_ms=2000,  # Conservative estimate
            estimated_cost=0.0,  # Assume local model
            complexity_tier=complexity.recommended_tier,
            user_preferences={},
            constraints=["offline_mode", f"status_{status.value}"]
        )
        
        logger.warning(f"Using offline alternative: {recommended_model}")
        return alternative_selection
    
    def _get_model_by_name(self, model_name: str) -> Optional[Model]:
        """Get model object by name from registry."""
        # This would integrate with the actual model registry
        # For now, create a basic model object
        from autom8.models.routing import ModelProvider, ModelType, PrivacyLevel
        
        # Basic model configurations
        model_configs = {
            "llama3.2:7b": {
                "display_name": "Llama 3.2 7B",
                "provider": ModelProvider.OLLAMA,
                "model_type": ModelType.LOCAL,
                "capability_score": 0.7,
                "avg_latency_ms": 1500,
                "max_context_tokens": 8192,
                "is_local": True,
                "energy_usage": 0.3,
                "privacy_level": PrivacyLevel.PRIVATE
            },
            "llama3.2:3b": {
                "display_name": "Llama 3.2 3B",
                "provider": ModelProvider.OLLAMA,
                "model_type": ModelType.LOCAL,
                "capability_score": 0.5,
                "avg_latency_ms": 800,
                "max_context_tokens": 8192,
                "is_local": True,
                "energy_usage": 0.2,
                "privacy_level": PrivacyLevel.PRIVATE
            },
            "mistral:7b": {
                "display_name": "Mistral 7B",
                "provider": ModelProvider.OLLAMA,
                "model_type": ModelType.LOCAL,
                "capability_score": 0.6,
                "avg_latency_ms": 1200,
                "max_context_tokens": 8192,
                "is_local": True,
                "energy_usage": 0.25,
                "privacy_level": PrivacyLevel.PRIVATE
            }
        }
        
        config = model_configs.get(model_name)
        if config:
            return Model(name=model_name, **config)
        
        return None
    
    async def _emergency_local_routing(
        self,
        query: str,
        complexity: ComplexityScore,
        preferences: Optional[RoutingPreferences]
    ) -> Optional[ModelSelection]:
        """Emergency routing using only local models."""
        
        # Try the most basic local model routing
        fallback_models = get_local_first_fallback_models()
        available_models = set(self.offline_manager.offline_capabilities.available_models)
        
        for model_name in fallback_models:
            if model_name in available_models:
                model = self._get_model_by_name(model_name)
                if model:
                    emergency_selection = ModelSelection(
                        primary_model=model,
                        alternatives=[],
                        fallback_model=None,
                        selection_reasoning=f"Emergency local routing due to connectivity issues",
                        confidence=0.5,  # Low confidence
                        routing_factors={"emergency_local": 1.0},
                        estimated_quality=0.6,
                        estimated_latency_ms=3000,
                        estimated_cost=0.0,
                        complexity_tier=complexity.recommended_tier,
                        user_preferences={},
                        constraints=["emergency_routing", "local_only"]
                    )
                    
                    logger.warning(f"Emergency local routing to: {model_name}")
                    return emergency_selection
        
        return None
    
    async def _create_cached_response_selection(
        self,
        cached_response,
        complexity: ComplexityScore
    ) -> ModelSelection:
        """Create a ModelSelection for a cached response."""
        
        # Create a placeholder model for the cached response
        cached_model = Model(
            name=f"cached_{cached_response.model_used}",
            display_name=f"Cached {cached_response.model_used}",
            provider=ModelProvider.OLLAMA,  # Use valid enum value
            model_type=ModelType.LOCAL,     # Use valid enum value
            capability_score=cached_response.quality_score,
            avg_latency_ms=50,  # Cache access is fast
            max_context_tokens=8192,
            is_local=True,
            energy_usage=0.0,  # No energy for cached responses
            privacy_level=PrivacyLevel.PRIVATE
        )
        
        cached_selection = ModelSelection(
            primary_model=cached_model,
            alternatives=[],
            fallback_model=None,
            selection_reasoning=f"Serving cached response from {cached_response.model_used} "
                              f"(age: {cached_response.age_hours:.1f}h, quality: {cached_response.quality_score:.2f})",
            confidence=cached_response.freshness_score,
            routing_factors={
                "cache_hit": 1.0,
                "freshness": cached_response.freshness_score,
                "quality": cached_response.quality_score
            },
            estimated_quality=cached_response.quality_score,
            estimated_latency_ms=50,
            estimated_cost=0.0,
            complexity_tier=complexity.recommended_tier,
            user_preferences={},
            constraints=["cached_response", f"original_model_{cached_response.model_used}"]
        )
        
        logger.info(f"Serving cached response (age: {cached_response.age_hours:.1f}h)")
        return cached_selection
    
    async def _cache_selection_info(
        self,
        query: str,
        selection: ModelSelection,
        complexity: ComplexityScore
    ) -> None:
        """Cache information about a successful routing selection."""
        
        # This would be called after successful inference to cache the actual response
        # For now, we just log the selection for potential future caching
        logger.debug(f"Selection completed: {selection.primary_model.name} for complexity {complexity.recommended_tier}")
    
    async def cache_response(
        self,
        query: str,
        response: str,
        selection: ModelSelection,
        complexity: ComplexityScore
    ) -> None:
        """Cache a successful response for offline use."""
        
        await self.offline_manager.cache_response(
            query=query,
            response=response,
            model_used=selection.primary_model.name,
            complexity=complexity,
            context_size=len(query.split()),  # Rough estimate
            estimated_cost=selection.estimated_cost
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status including offline mode information."""
        
        offline_status = self.offline_manager.get_status()
        
        return {
            "connectivity": offline_status['connectivity_status'],
            "offline_since": offline_status['offline_since'],
            "functionality_score": offline_status['functionality_score'],
            "fallback_active": self.fallback_active,
            "available_models": offline_status['capabilities']['available_models'],
            "degraded_features": offline_status['capabilities']['degraded_features'],
            "unavailable_features": offline_status['capabilities']['unavailable_features'],
            "cache_stats": offline_status['cache_stats'],
            "degradation_message": self.offline_manager.get_degradation_message()
        }
    
    def is_feature_available(self, feature: str) -> bool:
        """Check if a feature is available in current connectivity mode."""
        return self.offline_manager.is_feature_available(feature)


# Custom exceptions for offline mode

class OfflineRoutingError(Exception):
    """Raised when routing fails due to offline conditions."""
    
    def __init__(self, message: str, status: ConnectivityStatus, degradation_message: Optional[str] = None):
        super().__init__(message)
        self.status = status
        self.degradation_message = degradation_message


class OfflineUnavailableError(OfflineRoutingError):
    """Raised when a service is completely unavailable offline."""
    
    def __init__(self, message: str, degradation_message: Optional[str] = None):
        super().__init__(message, ConnectivityStatus.OFFLINE, degradation_message)


# Integration utilities

async def create_offline_aware_router(
    base_router: ModelRouter,
    offline_manager: OfflineModeManager
) -> OfflineAwareRouter:
    """Create an offline-aware router wrapper."""
    
    return OfflineAwareRouter(base_router, offline_manager)


def add_offline_support_to_router(router: ModelRouter, offline_manager: OfflineModeManager) -> OfflineAwareRouter:
    """Add offline support to an existing router."""
    
    return OfflineAwareRouter(router, offline_manager)