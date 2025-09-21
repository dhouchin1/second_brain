"""
Offline Mode Support for Autom8

Provides comprehensive offline mode capabilities with graceful degradation,
ensuring continuous operation when cloud services are unavailable.

Key Features:
- Automatic service health monitoring
- Intelligent fallback to local models
- Response caching for offline use
- Graceful degradation with user notifications
- Local-first operation when connectivity is limited

Usage:
    ```python
    from autom8.core.offline import (
        OfflineModeManager, OfflineAwareRouter,
        create_offline_mode_manager, create_offline_aware_router
    )
    
    # Create offline mode manager
    offline_manager = await create_offline_mode_manager(error_manager, event_bus)
    
    # Enhance existing router with offline capabilities
    offline_router = await create_offline_aware_router(base_router, offline_manager)
    
    # Use as normal - offline mode is handled automatically
    selection = await offline_router.route_query(query, complexity)
    ```
"""

from .mode_manager import (
    OfflineModeManager,
    ConnectivityStatus,
    ServiceType,
    ServiceStatus,
    OfflineCapability,
    CachedResponse,
    OfflineFallbackConfig,
    create_offline_mode_manager,
    get_local_first_fallback_models
)

from .routing_integration import (
    OfflineAwareRouter,
    OfflineRoutingError,
    OfflineUnavailableError,
    create_offline_aware_router,
    add_offline_support_to_router
)

__all__ = [
    # Core offline mode management
    'OfflineModeManager',
    'ConnectivityStatus',
    'ServiceType',
    'ServiceStatus',
    'OfflineCapability',
    'CachedResponse',
    'OfflineFallbackConfig',
    
    # Router integration
    'OfflineAwareRouter',
    'OfflineRoutingError',
    'OfflineUnavailableError',
    
    # Utility functions
    'create_offline_mode_manager',
    'create_offline_aware_router',
    'add_offline_support_to_router',
    'get_local_first_fallback_models'
]