"""
Template Management System for Autom8

This module provides comprehensive template management capabilities including:
- Template creation, storage, and versioning
- Variable substitution and template composition  
- Template validation and testing
- CLI integration and workflow automation
- Template sharing and marketplace functionality

The template system integrates deeply with the existing context transparency
and editing systems to provide a seamless user experience.
"""

from .manager import TemplateManager, get_template_manager
from .renderer import TemplateRenderer, VariableSubstitutionEngine
from .composer import TemplateComposer, CompositionEngine
from .validator import TemplateValidator, ValidationEngine
from .storage import TemplateStorage, get_template_storage
from .analytics import TemplateAnalytics, get_template_analytics

__all__ = [
    "TemplateManager",
    "get_template_manager", 
    "TemplateRenderer",
    "VariableSubstitutionEngine",
    "TemplateComposer",
    "CompositionEngine",
    "TemplateValidator",
    "ValidationEngine",
    "TemplateStorage",
    "get_template_storage",
    "TemplateAnalytics",
    "get_template_analytics",
]