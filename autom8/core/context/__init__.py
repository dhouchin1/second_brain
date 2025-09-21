"""
Context Management Module

Provides complete visibility into what's being sent to LLMs,
enables context editing, and manages context preparation.
"""

from autom8.core.context.inspector import ContextInspector
from autom8.core.context.optimizer import (
    ContextOptimizer,
    OptimizationProfile,
    OptimizationStrategy,
    OptimizationResult,
    get_context_optimizer
)
from autom8.models.context import (
    ContextSource,
    ContextPreview, 
    ContextPackage,
)

__all__ = [
    "ContextInspector",
    "ContextOptimizer",
    "OptimizationProfile",
    "OptimizationStrategy", 
    "OptimizationResult",
    "get_context_optimizer",
    "ContextSource",
    "ContextPreview",
    "ContextPackage",
]