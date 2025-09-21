"""
Event System Core Module

Comprehensive event handling system with validation, serialization,
and specialized event types for the Autom8 system.
"""

from autom8.core.events.validation import (
    EnhancedEventValidator, ValidationContext, ValidationLevel, ValidationRule,
    EventValidationResult, validate_event, validate_events, create_custom_validation_rule,
    register_custom_rule, get_event_validator
)

__all__ = [
    # Validation
    'EnhancedEventValidator',
    'ValidationContext',
    'ValidationLevel',
    'ValidationRule',
    'EventValidationResult',
    'validate_event',
    'validate_events',
    'create_custom_validation_rule',
    'register_custom_rule',
    'get_event_validator',
]