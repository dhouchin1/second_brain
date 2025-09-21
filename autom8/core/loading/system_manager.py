"""
Autom8 System Manager with Progressive Loading.

Provides centralized system initialization with progressive loading,
component health monitoring, and graceful degradation capabilities.
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field

from autom8.core.loading.progressive_loader import (
    ProgressiveLoader, LoadingProgress, LoadingState, create_standard_loader
)
from autom8.core.loading.components import create_standard_components
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class SystemHealth(BaseModel):
    """Overall system health status"""
    status: str = "initializing"  # initializing, healthy, degraded, critical
    ready_components: int = 0
    total_components: int = 0
    failed_components: List[str] = Field(default_factory=list)
    degraded_components: List[str] = Field(default_factory=list)
    startup_time: Optional[float] = None
    last_health_check: Optional[datetime] = None

    @property
    def is_operational(self) -> bool:
        """Check if system is operational (critical components ready)"""
        return self.status in ["healthy", "degraded"]

    @property
    def availability_percentage(self) -> float:
        """Calculate component availability percentage"""
        if self.total_components == 0:
            return 100.0
        return (self.ready_components / self.total_components) * 100


class SystemManager:
    """
    Central system manager with progressive loading capabilities.

    Manages system initialization, component health monitoring,
    and provides graceful degradation when components fail.
    """

    def __init__(self, max_concurrent_loads: int = 5):
        self.loader = create_standard_loader(max_concurrent_loads)
        self.health = SystemHealth()
        self._components: Dict[str, Any] = {}
        self._initialized = False
        self._startup_start_time: Optional[float] = None

        # Component availability tracking
        self._critical_components = {"model_router", "complexity_analyzer"}
        self._important_components = {"budget_manager", "template_system", "connection_pool"}
        self._optional_components = {"local_embedder", "vector_manager"}

        # Health monitoring
        self._health_check_interval = 300  # 5 minutes
        self._health_check_task: Optional[asyncio.Task] = None

    async def initialize_system(self,
                              system_config: Optional[Dict[str, Any]] = None,
                              progress_callback: Optional[callable] = None,
                              timeout_seconds: Optional[float] = 120.0) -> bool:
        """
        Initialize the Autom8 system with progressive loading.

        Args:
            system_config: Configuration for components
            progress_callback: Callback for progress updates
            timeout_seconds: Overall timeout for initialization

        Returns:
            True if system is operational (critical components loaded)
        """
        self._startup_start_time = time.time()
        logger.info("Starting Autom8 system initialization with progressive loading")

        try:
            # Create components with configuration
            config = system_config or {}
            components = create_standard_components(**config)

            # Register components with loader
            for component in components.values():
                self.loader.register_component(component)

            # Register callbacks
            if progress_callback:
                self.loader.register_progress_callback(progress_callback)

            self.loader.register_progress_callback(self._update_health_from_progress)
            self.loader.register_completion_callback(self._on_loading_complete)

            # Start progressive loading
            success = await asyncio.wait_for(
                self.loader.load_all_components(),
                timeout=timeout_seconds
            ) if timeout_seconds else await self.loader.load_all_components()

            # Extract loaded component instances
            self._extract_component_instances()

            # Determine system health
            self._assess_system_health()

            # Start health monitoring
            if self._health_check_task is None:
                self._health_check_task = asyncio.create_task(self._periodic_health_check())

            startup_time = time.time() - self._startup_start_time
            self.health.startup_time = startup_time

            logger.info(f"System initialization completed in {startup_time:.2f}s: "
                       f"status={self.health.status}, "
                       f"ready={self.health.ready_components}/{self.health.total_components}")

            self._initialized = True
            return self.health.is_operational

        except asyncio.TimeoutError:
            logger.error(f"System initialization timed out after {timeout_seconds}s")
            self._assess_system_health()
            return self.health.is_operational

        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self.health.status = "critical"
            return False

    def get_component(self, name: str) -> Optional[Any]:
        """Get a loaded component instance."""
        return self._components.get(name)

    def is_component_ready(self, name: str) -> bool:
        """Check if a specific component is ready."""
        return self.loader.is_component_ready(name)

    def get_system_health(self) -> SystemHealth:
        """Get current system health status."""
        return self.health

    def get_component_health(self, name: str) -> Optional[Dict[str, Any]]:
        """Get health details for a specific component."""
        health = self.loader.get_component_health(name)
        if not health:
            return None

        return {
            "state": health.state,
            "load_time": health.load_time,
            "last_attempt": health.last_attempt,
            "attempt_count": health.attempt_count,
            "error_message": health.error_message,
            "is_available": health.is_available,
            "can_retry": health.can_retry
        }

    def get_all_component_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health details for all components."""
        return {
            name: self.get_component_health(name)
            for name in self.loader.components.keys()
        }

    async def retry_failed_components(self) -> int:
        """Retry loading failed components."""
        logger.info("Retrying failed components")
        success_count = await self.loader.retry_failed_components()

        # Update component instances and health
        self._extract_component_instances()
        self._assess_system_health()

        return success_count

    def can_handle_query(self, query_type: str = "general") -> bool:
        """
        Check if system can handle a query based on component availability.

        Args:
            query_type: Type of query (general, semantic_search, budget_check, etc.)
        """
        # Check critical components
        if not self.is_component_ready("model_router"):
            logger.warning("Cannot handle query: model router not available")
            return False

        if not self.is_component_ready("complexity_analyzer"):
            logger.warning("Cannot handle query: complexity analyzer not available")
            return False

        # Check query-specific requirements
        if query_type == "semantic_search":
            if not self.is_component_ready("local_embedder"):
                logger.info("Semantic search degraded: embedder not available")
                # Can still handle with degraded functionality

        elif query_type == "budget_check":
            if not self.is_component_ready("budget_manager"):
                logger.info("Budget checking degraded: budget manager not available")
                # Can still route without budget constraints

        return True

    def get_available_features(self) -> List[str]:
        """Get list of available features based on component readiness."""
        features = []

        if self.is_component_ready("model_router"):
            features.append("model_routing")

        if self.is_component_ready("complexity_analyzer"):
            features.append("complexity_analysis")

        if self.is_component_ready("budget_manager"):
            features.append("budget_control")

        if self.is_component_ready("template_system"):
            features.append("template_processing")

        if self.is_component_ready("local_embedder"):
            features.append("local_embeddings")

        if self.is_component_ready("vector_manager"):
            features.append("semantic_search")

        if self.is_component_ready("connection_pool"):
            features.append("optimized_database_access")

        return features

    def _extract_component_instances(self) -> None:
        """Extract loaded component instances from the loader."""
        self._components.clear()

        for name, component in self.loader.components.items():
            instance = component.get_instance()
            if instance:
                self._components[name] = instance

    def _assess_system_health(self) -> None:
        """Assess overall system health based on component states."""
        all_health = self.loader.get_all_health()
        self.health.total_components = len(all_health)

        ready_components = []
        failed_components = []
        degraded_components = []

        for name, health in all_health.items():
            if health.state == LoadingState.READY:
                ready_components.append(name)
            elif health.state == LoadingState.DEGRADED:
                ready_components.append(name)
                degraded_components.append(name)
            elif health.state in [LoadingState.FAILED, LoadingState.TIMEOUT]:
                failed_components.append(name)

        self.health.ready_components = len(ready_components)
        self.health.failed_components = failed_components
        self.health.degraded_components = degraded_components
        self.health.last_health_check = datetime.utcnow()

        # Determine overall status
        critical_ready = all(
            name in ready_components for name in self._critical_components
        )

        if not critical_ready:
            self.health.status = "critical"
        elif any(name in failed_components for name in self._important_components):
            self.health.status = "degraded"
        elif degraded_components:
            self.health.status = "degraded"
        else:
            self.health.status = "healthy"

    def _update_health_from_progress(self, progress: LoadingProgress) -> None:
        """Update health status from loading progress."""
        self.health.ready_components = progress.loaded_components
        self.health.total_components = progress.total_components

    def _on_loading_complete(self, success: bool) -> None:
        """Called when loading is complete."""
        self._assess_system_health()
        logger.info(f"Loading complete: success={success}, "
                   f"system_status={self.health.status}")

    async def _periodic_health_check(self) -> None:
        """Periodic health check background task."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)

                # Update health assessment
                self._assess_system_health()

                # Log health summary
                logger.debug(f"Health check: {self.health.status} "
                           f"({self.health.ready_components}/{self.health.total_components} ready)")

                # Auto-retry failed components if needed
                failed_count = len(self.health.failed_components)
                if failed_count > 0 and self.health.status == "critical":
                    logger.info(f"Attempting to recover {failed_count} failed components")
                    await self.retry_failed_components()

            except Exception as e:
                logger.error(f"Health check failed: {e}")

    async def shutdown(self) -> None:
        """Gracefully shutdown the system."""
        logger.info("Shutting down Autom8 system")

        # Stop health monitoring
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Shutdown components that support it
        for name, component in self._components.items():
            try:
                if hasattr(component, 'shutdown'):
                    await component.shutdown()
                elif hasattr(component, 'close'):
                    await component.close()
            except Exception as e:
                logger.error(f"Error shutting down component {name}: {e}")

        self._initialized = False
        logger.info("System shutdown complete")


# Convenience functions

async def initialize_autom8_system(
    config: Optional[Dict[str, Any]] = None,
    timeout_seconds: Optional[float] = 120.0,
    show_progress: bool = False
) -> SystemManager:
    """
    Initialize the Autom8 system with progressive loading.

    Args:
        config: System configuration
        timeout_seconds: Initialization timeout
        show_progress: Show progress output

    Returns:
        Initialized SystemManager
    """
    manager = SystemManager()

    # Simple progress callback if requested
    def progress_callback(progress: LoadingProgress):
        if show_progress:
            print(f"\rLoading: {progress.completion_percentage:.1f}% "
                  f"({progress.loaded_components}/{progress.total_components})", end="")

    try:
        success = await manager.initialize_system(
            system_config=config,
            progress_callback=progress_callback if show_progress else None,
            timeout_seconds=timeout_seconds
        )

        if show_progress:
            print()  # New line after progress

        if not success:
            logger.warning("System initialization completed with issues - check component health")

        return manager

    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        if show_progress:
            print()
        raise


def get_system_status_summary(manager: SystemManager) -> Dict[str, Any]:
    """Get a summary of system status for monitoring/debugging."""
    health = manager.get_system_health()

    return {
        "status": health.status,
        "operational": health.is_operational,
        "availability_percentage": health.availability_percentage,
        "components": {
            "ready": health.ready_components,
            "total": health.total_components,
            "failed": health.failed_components,
            "degraded": health.degraded_components
        },
        "startup_time": health.startup_time,
        "last_health_check": health.last_health_check,
        "available_features": manager.get_available_features()
    }