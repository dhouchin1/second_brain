"""
Autom8 v3.0
Context-Aware Multi-Agent Runtime with Intelligent Model Routing

Core Mission: "Democratize the entire LLM ecosystem by matching every query 
with its optimal model, while giving users complete context transparency 
and control."
"""

__version__ = "3.0.0"
__author__ = "Autom8 Team"
__email__ = "team@autom8.dev"

from autom8.core.context.inspector import ContextInspector
from autom8.core.complexity.analyzer import ComplexityAnalyzer
from autom8.core.routing.router import ModelRouter
from autom8.core.memory.broker import ContextBroker
from autom8.agents.base import BaseAgent

__all__ = [
    "ContextInspector",
    "ComplexityAnalyzer", 
    "ModelRouter",
    "ContextBroker",
    "BaseAgent",
    "__version__",
]