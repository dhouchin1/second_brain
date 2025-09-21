"""
Concrete implementations of loadable components for Autom8's heavy systems.

Provides progressive loading implementations for BudgetManager, LocalEmbedder,
Vector managers, Connection pools, and Template system.
"""

import asyncio
import time
from typing import Any, Dict, Optional

from autom8.core.loading.progressive_loader import (
    LoadableComponent, LoadingPriority, LoadingState
)
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class BudgetManagerComponent(LoadableComponent):
    """Progressive loading wrapper for BudgetManager with Redis/SQLite dependencies."""

    def __init__(self, redis_client=None, sqlite_manager=None, event_bus=None,
                 timeout_seconds: float = 30.0):
        super().__init__(
            name="budget_manager",
            priority=LoadingPriority.HIGH,  # Important but not critical for basic routing
            timeout_seconds=timeout_seconds
        )
        self.redis_client = redis_client
        self.sqlite_manager = sqlite_manager
        self.event_bus = event_bus

    async def _load_component(self):
        """Load the BudgetManager with all dependencies."""
        from autom8.services.budget import BudgetManager

        budget_manager = BudgetManager(
            redis_client=self.redis_client,
            sqlite_manager=self.sqlite_manager,
            event_bus=self.event_bus
        )

        # Initialize with timeout handling
        success = await budget_manager.initialize()
        if not success:
            raise Exception("BudgetManager initialization failed")

        return budget_manager

    async def _create_fallback(self):
        """Create a fallback budget manager with limited functionality."""
        from autom8.services.budget import BudgetManager

        # Create budget manager without storage backends
        budget_manager = BudgetManager()
        # Set a flag to indicate degraded mode
        budget_manager._degraded_mode = True
        return budget_manager

    async def _health_check(self, instance) -> bool:
        """Check if budget manager is healthy."""
        if not instance:
            return False

        # Check if it has basic functionality
        return hasattr(instance, 'can_afford_query')


class LocalEmbedderComponent(LoadableComponent):
    """Progressive loading wrapper for LocalEmbedder with large model files."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", timeout_seconds: float = 60.0):
        super().__init__(
            name="local_embedder",
            priority=LoadingPriority.BACKGROUND,  # Heavy component, load in background
            timeout_seconds=timeout_seconds
        )
        self.model_name = model_name

    async def _load_component(self):
        """Load the LocalEmbedder with model download."""
        from autom8.core.memory.embeddings import LocalEmbedder

        embedder = LocalEmbedder(model_name=self.model_name)

        # Initialize - this may download large model files
        success = await embedder.initialize()
        if not success:
            raise Exception("LocalEmbedder initialization failed")

        return embedder

    async def _create_fallback(self):
        """Create a fallback embedder that returns None embeddings."""
        class FallbackEmbedder:
            def __init__(self):
                self._initialized = True

            async def embed(self, text: str):
                return None

            async def embed_batch(self, texts):
                return None

            def is_available(self):
                return False

            @property
            def dimension(self):
                return 384

        return FallbackEmbedder()

    async def _health_check(self, instance) -> bool:
        """Check if embedder is healthy."""
        if not instance:
            return False

        # Try a simple embedding to verify functionality
        try:
            if hasattr(instance, 'is_available'):
                return instance.is_available()
            return True
        except Exception:
            return False


class VectorManagerComponent(LoadableComponent):
    """Progressive loading wrapper for Vector managers with database setup."""

    def __init__(self, storage_path: str = "./data", timeout_seconds: float = 45.0):
        super().__init__(
            name="vector_manager",
            priority=LoadingPriority.BACKGROUND,  # Heavy database operations
            timeout_seconds=timeout_seconds,
            dependencies=["local_embedder"]  # Depends on embedder
        )
        self.storage_path = storage_path

    async def _load_component(self):
        """Load the Vector manager with database initialization."""
        try:
            from autom8.storage.sqlite.vector_manager import VectorManager

            vector_manager = VectorManager(storage_path=self.storage_path)
            await vector_manager.initialize()

            return vector_manager
        except ImportError:
            # Fallback if vector manager not available
            raise Exception("VectorManager not available")

    async def _create_fallback(self):
        """Create a fallback vector manager with no persistence."""
        class FallbackVectorManager:
            def __init__(self):
                self._vectors = {}

            async def store_vector(self, vector_id: str, vector, metadata=None):
                self._vectors[vector_id] = {"vector": vector, "metadata": metadata}

            async def search_similar(self, query_vector, top_k=5):
                return []  # No search capability

            async def get_vector(self, vector_id: str):
                return self._vectors.get(vector_id)

        return FallbackVectorManager()

    async def _health_check(self, instance) -> bool:
        """Check if vector manager is healthy."""
        if not instance:
            return False

        # Check if it has basic vector operations
        return hasattr(instance, 'store_vector') and hasattr(instance, 'search_similar')


class ConnectionPoolComponent(LoadableComponent):
    """Progressive loading wrapper for database connection pools."""

    def __init__(self, pool_size: int = 10, timeout_seconds: float = 20.0):
        super().__init__(
            name="connection_pool",
            priority=LoadingPriority.HIGH,  # Important for database access
            timeout_seconds=timeout_seconds
        )
        self.pool_size = pool_size

    async def _load_component(self):
        """Load the connection pool."""
        try:
            from autom8.storage.connection_pool import ConnectionPool

            pool = ConnectionPool(max_connections=self.pool_size)
            await pool.initialize()

            return pool
        except ImportError:
            # Fallback if connection pool not available
            raise Exception("ConnectionPool not available")

    async def _create_fallback(self):
        """Create a fallback connection manager with single connections."""
        class FallbackConnectionManager:
            async def get_connection(self):
                from autom8.storage.sqlite.manager import SQLiteManager
                manager = SQLiteManager()
                await manager.initialize()
                return manager

            async def release_connection(self, conn):
                pass  # No pooling in fallback

        return FallbackConnectionManager()

    async def _health_check(self, instance) -> bool:
        """Check if connection pool is healthy."""
        if not instance:
            return False

        return hasattr(instance, 'get_connection')


class TemplateSystemComponent(LoadableComponent):
    """Progressive loading wrapper for the Template system with complex initialization."""

    def __init__(self, timeout_seconds: float = 25.0):
        super().__init__(
            name="template_system",
            priority=LoadingPriority.HIGH,  # Important for query processing
            timeout_seconds=timeout_seconds
        )

    async def _load_component(self):
        """Load the template system with all validators."""
        from autom8.core.templates.composer import TemplateComposer
        from autom8.core.templates.validator import TemplateValidator
        from autom8.core.templates.enhanced_validator import EnhancedTemplateValidator

        # Initialize template composer
        composer = TemplateComposer()
        await composer.initialize()

        # Initialize validators
        validator = TemplateValidator()
        enhanced_validator = EnhancedTemplateValidator()

        # Create combined template system
        class TemplateSystem:
            def __init__(self):
                self.composer = composer
                self.validator = validator
                self.enhanced_validator = enhanced_validator

            async def compose_template(self, template_id: str, context: Dict[str, Any]):
                return await self.composer.compose(template_id, context)

            async def validate_template(self, template: str):
                return await self.validator.validate(template)

        return TemplateSystem()

    async def _create_fallback(self):
        """Create a fallback template system with basic functionality."""
        class FallbackTemplateSystem:
            async def compose_template(self, template_id: str, context: Dict[str, Any]):
                # Simple string substitution
                template = "Context: {context}"
                return template.format(context=str(context))

            async def validate_template(self, template: str):
                return len(template) > 0  # Basic validation

        return FallbackTemplateSystem()

    async def _health_check(self, instance) -> bool:
        """Check if template system is healthy."""
        if not instance:
            return False

        # Check if it has basic template operations
        return (hasattr(instance, 'compose_template') and
                hasattr(instance, 'validate_template'))


class ModelRouterComponent(LoadableComponent):
    """Progressive loading wrapper for ModelRouter - critical component."""

    def __init__(self, budget_manager=None, timeout_seconds: float = 15.0):
        super().__init__(
            name="model_router",
            priority=LoadingPriority.IMMEDIATE,  # Critical for query routing
            timeout_seconds=timeout_seconds,
            dependencies=[]  # No hard dependencies to avoid blocking
        )
        self.budget_manager = budget_manager

    async def _load_component(self):
        """Load the ModelRouter."""
        from autom8.core.routing.router import ModelRouter

        router = ModelRouter(
            budget_manager=self.budget_manager,
            testing_mode=False
        )

        # Initialize with shorter timeout for critical component
        success = await router.initialize(timeout_seconds=10.0)
        if not success:
            raise Exception("ModelRouter initialization failed")

        return router

    async def _create_fallback(self):
        """Create a fallback router with basic functionality."""
        from autom8.core.routing.router import ModelRouter
        from autom8.models.routing import Model, ModelProvider, ModelType

        # Create router in testing mode for faster initialization
        router = ModelRouter(testing_mode=True)
        await router.initialize()

        return router

    async def _health_check(self, instance) -> bool:
        """Check if router is healthy."""
        if not instance:
            return False

        return hasattr(instance, 'select_model') and instance._initialized


class ComplexityAnalyzerComponent(LoadableComponent):
    """Progressive loading wrapper for ComplexityAnalyzer."""

    def __init__(self, timeout_seconds: float = 10.0):
        super().__init__(
            name="complexity_analyzer",
            priority=LoadingPriority.IMMEDIATE,  # Critical for routing decisions
            timeout_seconds=timeout_seconds
        )

    async def _load_component(self):
        """Load the ComplexityAnalyzer."""
        from autom8.core.complexity.analyzer import ComplexityAnalyzer

        analyzer = ComplexityAnalyzer()
        await analyzer.initialize()

        return analyzer

    async def _create_fallback(self):
        """Create a fallback analyzer with basic heuristics."""
        from autom8.models.complexity import ComplexityScore, ComplexityTier

        class FallbackComplexityAnalyzer:
            async def analyze_query(self, query: str):
                # Simple heuristic based on length
                if len(query) > 1000:
                    tier = ComplexityTier.HIGH
                elif len(query) > 500:
                    tier = ComplexityTier.MEDIUM
                else:
                    tier = ComplexityTier.LOW

                return ComplexityScore(
                    tier=tier,
                    score=len(query) / 1000.0,
                    reasoning=["Fallback analysis based on query length"]
                )

        return FallbackComplexityAnalyzer()

    async def _health_check(self, instance) -> bool:
        """Check if analyzer is healthy."""
        if not instance:
            return False

        return hasattr(instance, 'analyze_query')


# Factory function to create all standard components
def create_standard_components(**kwargs) -> Dict[str, LoadableComponent]:
    """Create all standard Autom8 components for progressive loading."""

    components = {
        'model_router': ModelRouterComponent(
            budget_manager=kwargs.get('budget_manager'),
            timeout_seconds=kwargs.get('router_timeout', 15.0)
        ),
        'complexity_analyzer': ComplexityAnalyzerComponent(
            timeout_seconds=kwargs.get('analyzer_timeout', 10.0)
        ),
        'budget_manager': BudgetManagerComponent(
            redis_client=kwargs.get('redis_client'),
            sqlite_manager=kwargs.get('sqlite_manager'),
            event_bus=kwargs.get('event_bus'),
            timeout_seconds=kwargs.get('budget_timeout', 30.0)
        ),
        'template_system': TemplateSystemComponent(
            timeout_seconds=kwargs.get('template_timeout', 25.0)
        ),
        'connection_pool': ConnectionPoolComponent(
            pool_size=kwargs.get('pool_size', 10),
            timeout_seconds=kwargs.get('pool_timeout', 20.0)
        ),
        'local_embedder': LocalEmbedderComponent(
            model_name=kwargs.get('embedding_model', "all-MiniLM-L6-v2"),
            timeout_seconds=kwargs.get('embedder_timeout', 60.0)
        ),
        'vector_manager': VectorManagerComponent(
            storage_path=kwargs.get('storage_path', "./data"),
            timeout_seconds=kwargs.get('vector_timeout', 45.0)
        ),
    }

    return components