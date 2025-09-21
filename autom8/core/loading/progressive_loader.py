"""
Progressive Loading Manager for Autom8 - Improves startup performance and reduces timeouts.

Provides priority-based loading with parallel execution, graceful degradation,
and component availability tracking for heavy system components.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Generic

from pydantic import BaseModel, Field

from autom8.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class LoadingState(str, Enum):
    """Component loading states"""
    PENDING = "pending"         # Not started
    LOADING = "loading"         # Currently loading
    READY = "ready"            # Successfully loaded
    FAILED = "failed"          # Failed to load
    DEGRADED = "degraded"      # Partially loaded/limited functionality
    TIMEOUT = "timeout"        # Timed out during loading
    RETRYING = "retrying"      # Retrying after failure


class LoadingPriority(str, Enum):
    """Loading priority levels"""
    IMMEDIATE = "immediate"     # Critical components (router, basic context)
    HIGH = "high"              # Important but not blocking (basic DB connections)
    BACKGROUND = "background"   # Heavy components (embeddings, full vector search)
    LAZY = "lazy"              # On-demand components (analytics, forecasting)


class ComponentHealth(BaseModel):
    """Component health status"""
    state: LoadingState = LoadingState.PENDING
    load_time: Optional[float] = None
    last_attempt: Optional[datetime] = None
    attempt_count: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_available(self) -> bool:
        """Check if component is available for use"""
        return self.state in [LoadingState.READY, LoadingState.DEGRADED]

    @property
    def can_retry(self) -> bool:
        """Check if component can be retried"""
        return self.state in [LoadingState.FAILED, LoadingState.TIMEOUT] and self.attempt_count < 3


class LoadableComponent(ABC, Generic[T]):
    """
    Abstract base class for components that can be progressively loaded.

    Provides standardized interface for loading, health checking,
    and fallback behavior.
    """

    def __init__(self, name: str, priority: LoadingPriority = LoadingPriority.BACKGROUND,
                 timeout_seconds: float = 30.0, dependencies: Optional[List[str]] = None):
        self.name = name
        self.priority = priority
        self.timeout_seconds = timeout_seconds
        self.dependencies = dependencies or []
        self.health = ComponentHealth()
        self._instance: Optional[T] = None
        self._fallback_instance: Optional[T] = None

    @abstractmethod
    async def _load_component(self) -> T:
        """Load the actual component. Must be implemented by subclasses."""
        pass

    async def _create_fallback(self) -> Optional[T]:
        """Create a fallback/degraded version of the component."""
        return None

    async def _health_check(self, instance: T) -> bool:
        """Perform health check on loaded component."""
        return instance is not None

    async def load(self) -> bool:
        """Load the component with timeout and error handling."""
        self.health.state = LoadingState.LOADING
        self.health.last_attempt = datetime.utcnow()
        self.health.attempt_count += 1

        start_time = time.time()

        try:
            # Load with timeout
            self._instance = await asyncio.wait_for(
                self._load_component(),
                timeout=self.timeout_seconds
            )

            # Health check
            if await self._health_check(self._instance):
                self.health.state = LoadingState.READY
                self.health.load_time = time.time() - start_time
                logger.info(f"Component '{self.name}' loaded successfully in {self.health.load_time:.2f}s")
                return True
            else:
                raise Exception("Health check failed")

        except asyncio.TimeoutError:
            self.health.state = LoadingState.TIMEOUT
            self.health.error_message = f"Loading timed out after {self.timeout_seconds}s"
            logger.warning(f"Component '{self.name}' timed out during loading")

        except Exception as e:
            self.health.state = LoadingState.FAILED
            self.health.error_message = str(e)
            logger.error(f"Component '{self.name}' failed to load: {e}")

        # Try fallback if main loading failed
        try:
            self._fallback_instance = await self._create_fallback()
            if self._fallback_instance and await self._health_check(self._fallback_instance):
                self.health.state = LoadingState.DEGRADED
                self.health.load_time = time.time() - start_time
                logger.info(f"Component '{self.name}' loaded in degraded mode")
                return True
        except Exception as e:
            logger.error(f"Fallback creation failed for '{self.name}': {e}")

        return False

    def get_instance(self) -> Optional[T]:
        """Get the loaded component instance."""
        if self._instance is not None:
            return self._instance
        return self._fallback_instance

    def is_ready(self) -> bool:
        """Check if component is ready for use."""
        return self.health.is_available

    def get_state(self) -> LoadingState:
        """Get current loading state."""
        return self.health.state


class LoadingProgress(BaseModel):
    """Progress tracking for loading operations"""
    total_components: int = 0
    loaded_components: int = 0
    failed_components: int = 0
    current_loading: List[str] = Field(default_factory=list)

    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage"""
        if self.total_components == 0:
            return 100.0
        return (self.loaded_components / self.total_components) * 100

    @property
    def is_complete(self) -> bool:
        """Check if all components are processed"""
        return (self.loaded_components + self.failed_components) >= self.total_components


class ProgressiveLoader:
    """
    Progressive loading manager that handles component loading in priority order
    with parallel execution where possible.
    """

    def __init__(self, max_concurrent_loads: int = 5):
        self.components: Dict[str, LoadableComponent] = {}
        self.progress = LoadingProgress()
        self.max_concurrent_loads = max_concurrent_loads
        self._loading_semaphore = asyncio.Semaphore(max_concurrent_loads)
        self._progress_callbacks: List[Callable[[LoadingProgress], None]] = []
        self._completion_callbacks: List[Callable[[bool], None]] = []
        self._dependency_graph: Dict[str, Set[str]] = {}

    def register_component(self, component: LoadableComponent) -> None:
        """Register a component for progressive loading."""
        self.components[component.name] = component
        self.progress.total_components = len(self.components)

        # Build dependency graph
        self._dependency_graph[component.name] = set(component.dependencies)

        logger.debug(f"Registered component '{component.name}' with priority {component.priority}")

    def register_progress_callback(self, callback: Callable[[LoadingProgress], None]) -> None:
        """Register callback for progress updates."""
        self._progress_callbacks.append(callback)

    def register_completion_callback(self, callback: Callable[[bool], None]) -> None:
        """Register callback for completion."""
        self._completion_callbacks.append(callback)

    async def load_all_components(self, timeout_seconds: Optional[float] = None) -> bool:
        """
        Load all registered components progressively.

        Returns True if all critical components loaded successfully.
        """
        if not self.components:
            logger.info("No components to load")
            return True

        logger.info(f"Starting progressive loading of {len(self.components)} components")
        start_time = time.time()

        try:
            # Group components by priority
            priority_groups = self._group_by_priority()

            # Load components by priority groups
            success = True
            for priority in [LoadingPriority.IMMEDIATE, LoadingPriority.HIGH,
                           LoadingPriority.BACKGROUND, LoadingPriority.LAZY]:

                if priority not in priority_groups:
                    continue

                group_success = await self._load_priority_group(
                    priority_groups[priority],
                    priority,
                    timeout_seconds
                )

                # Only fail overall loading if immediate/high priority components fail
                if not group_success and priority in [LoadingPriority.IMMEDIATE, LoadingPriority.HIGH]:
                    success = False

            total_time = time.time() - start_time
            ready_count = sum(1 for c in self.components.values() if c.is_ready())

            logger.info(f"Progressive loading completed in {total_time:.2f}s: "
                       f"{ready_count}/{len(self.components)} components ready")

            # Notify completion callbacks
            for callback in self._completion_callbacks:
                try:
                    callback(success)
                except Exception as e:
                    logger.error(f"Completion callback failed: {e}")

            return success

        except Exception as e:
            logger.error(f"Progressive loading failed: {e}")
            return False

    async def _load_priority_group(self, components: List[LoadableComponent],
                                 priority: LoadingPriority,
                                 timeout_seconds: Optional[float]) -> bool:
        """Load a group of components with the same priority."""
        if not components:
            return True

        logger.info(f"Loading {len(components)} {priority.value} priority components")

        # Determine loading strategy based on priority
        if priority == LoadingPriority.IMMEDIATE:
            # Load immediately, one by one with dependencies
            return await self._load_sequential_with_dependencies(components, timeout_seconds)
        elif priority == LoadingPriority.HIGH:
            # Load in parallel but wait for completion
            return await self._load_parallel_with_wait(components, timeout_seconds)
        else:
            # Load in background, don't wait
            asyncio.create_task(self._load_parallel_background(components))
            return True  # Don't fail for background components

    async def _load_sequential_with_dependencies(self, components: List[LoadableComponent],
                                               timeout_seconds: Optional[float]) -> bool:
        """Load components sequentially respecting dependencies."""
        remaining = set(c.name for c in components)
        loaded = set()
        success_count = 0

        while remaining:
            # Find components with satisfied dependencies
            ready_to_load = []
            for comp_name in remaining:
                component = self.components[comp_name]
                deps = self._dependency_graph.get(comp_name, set())
                if deps.issubset(loaded):
                    ready_to_load.append(component)

            if not ready_to_load:
                # Circular dependency or unsatisfied dependencies
                logger.error(f"Cannot resolve dependencies for: {remaining}")
                break

            # Load ready components
            for component in ready_to_load:
                self.progress.current_loading = [component.name]
                self._notify_progress()

                success = await self._load_component_with_semaphore(component)
                if success:
                    loaded.add(component.name)
                    success_count += 1

                remaining.discard(component.name)
                self.progress.current_loading = []
                self._update_progress()

        return success_count > 0  # Success if at least one component loaded

    async def _load_parallel_with_wait(self, components: List[LoadableComponent],
                                     timeout_seconds: Optional[float]) -> bool:
        """Load components in parallel and wait for completion."""
        self.progress.current_loading = [c.name for c in components]
        self._notify_progress()

        # Create loading tasks
        tasks = [
            self._load_component_with_semaphore(component)
            for component in components
        ]

        # Wait for all to complete
        try:
            if timeout_seconds:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout_seconds
                )
            else:
                results = await asyncio.gather(*tasks, return_exceptions=True)

            success_count = sum(1 for r in results if r is True)

            self.progress.current_loading = []
            self._update_progress()

            return success_count > 0

        except asyncio.TimeoutError:
            logger.warning(f"Parallel loading group timed out after {timeout_seconds}s")
            return False

    async def _load_parallel_background(self, components: List[LoadableComponent]) -> None:
        """Load components in background without blocking."""
        tasks = [
            self._load_component_with_semaphore(component)
            for component in components
        ]

        # Fire and forget
        for task in tasks:
            asyncio.create_task(task)

    async def _load_component_with_semaphore(self, component: LoadableComponent) -> bool:
        """Load a component with concurrency control."""
        async with self._loading_semaphore:
            success = await component.load()
            if success:
                self.progress.loaded_components += 1
            else:
                self.progress.failed_components += 1
            self._notify_progress()
            return success

    def _group_by_priority(self) -> Dict[LoadingPriority, List[LoadableComponent]]:
        """Group components by their loading priority."""
        groups = {}
        for component in self.components.values():
            if component.priority not in groups:
                groups[component.priority] = []
            groups[component.priority].append(component)
        return groups

    def _update_progress(self) -> None:
        """Update progress counters and notify callbacks."""
        # Progress is updated in _load_component_with_semaphore
        self._notify_progress()

    def _notify_progress(self) -> None:
        """Notify all progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(self.progress)
            except Exception as e:
                logger.error(f"Progress callback failed: {e}")

    def get_component_health(self, name: str) -> Optional[ComponentHealth]:
        """Get health status of a component."""
        component = self.components.get(name)
        return component.health if component else None

    def get_all_health(self) -> Dict[str, ComponentHealth]:
        """Get health status of all components."""
        return {name: comp.health for name, comp in self.components.items()}

    def is_component_ready(self, name: str) -> bool:
        """Check if a specific component is ready."""
        component = self.components.get(name)
        return component.is_ready() if component else False

    def get_ready_components(self) -> List[str]:
        """Get list of components that are ready."""
        return [name for name, comp in self.components.items() if comp.is_ready()]

    def get_failed_components(self) -> List[str]:
        """Get list of components that failed to load."""
        return [name for name, comp in self.components.items()
                if comp.get_state() == LoadingState.FAILED]

    async def retry_failed_components(self) -> int:
        """Retry loading failed components. Returns number of successful retries."""
        failed_components = [
            comp for comp in self.components.values()
            if comp.health.can_retry
        ]

        if not failed_components:
            return 0

        logger.info(f"Retrying {len(failed_components)} failed components")

        tasks = [
            self._retry_component(comp) for comp in failed_components
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        success_count = sum(1 for r in results if r is True)

        logger.info(f"Retry completed: {success_count}/{len(failed_components)} succeeded")
        return success_count

    async def _retry_component(self, component: LoadableComponent) -> bool:
        """Retry loading a single component."""
        component.health.state = LoadingState.RETRYING
        return await component.load()


# Convenience functions for common loading patterns

async def load_with_progress_bar(loader: ProgressiveLoader,
                                description: str = "Loading components") -> bool:
    """Load components with a simple progress display."""

    def progress_callback(progress: LoadingProgress):
        if progress.current_loading:
            print(f"\r{description}: {progress.completion_percentage:.1f}% "
                  f"({progress.loaded_components}/{progress.total_components}) "
                  f"Loading: {', '.join(progress.current_loading)}", end="")
        else:
            print(f"\r{description}: {progress.completion_percentage:.1f}% "
                  f"({progress.loaded_components}/{progress.total_components})", end="")

    loader.register_progress_callback(progress_callback)

    try:
        success = await loader.load_all_components()
        print()  # New line after progress
        return success
    finally:
        print()  # Ensure new line


def create_standard_loader(max_concurrent: int = 5) -> ProgressiveLoader:
    """Create a standard progressive loader with sensible defaults."""
    return ProgressiveLoader(max_concurrent_loads=max_concurrent)