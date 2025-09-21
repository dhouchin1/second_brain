"""
Enhanced Routing Module with Dynamic Preference Learning

This module provides intelligent model routing with machine learning capabilities
for continuous improvement based on user behavior and outcomes.
"""

from .router import ModelRouter, PerformanceTracker
from .advanced_scoring import AdvancedModelScorer, ScoringMethod, ModelScore, get_advanced_scorer
from .preference_learning import (
    PreferenceLearningEngine, 
    LearningAlgorithm, 
    UserProfile, 
    UserInteraction,
    FeedbackType,
    OutcomeType,
    LearningPattern
)
from .interaction_tracker import InteractionTracker, get_interaction_tracker
from .adaptive_router import AdaptiveModelRouter, AdaptiveRoutingConfig, ConfidenceScorer
from .preference_storage import (
    PreferenceStorageManager, 
    SQLiteStorageBackend, 
    FileStorageBackend,
    get_preference_storage
)
from autom8.models.routing import ModelSelection, Model, ModelStats

__all__ = [
    # Base routing
    "ModelRouter",
    "PerformanceTracker",
    "ModelSelection",
    "Model", 
    "ModelStats",
    
    # Advanced scoring
    "AdvancedModelScorer", 
    "ScoringMethod",
    "ModelScore",
    "get_advanced_scorer",
    
    # Preference learning
    "PreferenceLearningEngine",
    "LearningAlgorithm", 
    "UserProfile",
    "UserInteraction",
    "FeedbackType",
    "OutcomeType", 
    "LearningPattern",
    
    # Interaction tracking
    "InteractionTracker",
    "get_interaction_tracker",
    
    # Adaptive routing
    "AdaptiveModelRouter",
    "AdaptiveRoutingConfig",
    "ConfidenceScorer",
    
    # Storage
    "PreferenceStorageManager",
    "SQLiteStorageBackend", 
    "FileStorageBackend",
    "get_preference_storage",
]


async def get_adaptive_router(**kwargs):
    """
    Get a fully configured adaptive router instance.
    
    This is the main entry point for creating an adaptive router with
    all learning capabilities enabled.
    """
    from .adaptive_router import AdaptiveModelRouter, AdaptiveRoutingConfig
    from .preference_storage import get_preference_storage
    
    # Initialize storage
    storage_manager = await get_preference_storage()
    
    # Configure adaptive router
    config = AdaptiveRoutingConfig()
    if "config" in kwargs:
        config = kwargs.pop("config")
    
    router = AdaptiveModelRouter(config=config, **kwargs)
    
    # Initialize with storage integration
    if await router.initialize():
        # Connect storage to learning engine
        router.learning_engine._storage_manager = storage_manager
        return router
    else:
        raise RuntimeError("Failed to initialize adaptive router")


async def create_basic_router(**kwargs):
    """Create a basic ModelRouter without learning capabilities"""
    router = ModelRouter(**kwargs)
    await router.initialize()
    return router