"""
Event Validation and Error Handling

Comprehensive validation framework for events with business rules,
constraints validation, error handling, and reporting capabilities.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Set, Callable, Pattern, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum

from pydantic import ValidationError

from autom8.models.events import (
    AgentEvent, EventType, EventPriority, EventCategory, EventSeverity,
    BaseEventPayload, TaskEventPayload, AgentEventPayload, DecisionEventPayload,
    CoordinationEventPayload, ErrorEventPayload, PerformanceEventPayload,
    EventValidationError
)

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Validation levels for different use cases."""
    BASIC = "basic"      # Basic Pydantic validation only
    STANDARD = "standard"  # Standard business rules
    STRICT = "strict"    # Strict validation with all rules
    CUSTOM = "custom"    # Custom validation rules only


class ValidationContext:
    """Context information for event validation."""
    
    def __init__(
        self,
        level: ValidationLevel = ValidationLevel.STANDARD,
        environment: str = "production",
        max_payload_size: int = 1024 * 1024,  # 1MB
        allowed_agents: Optional[Set[str]] = None,
        custom_rules: Optional[List[Callable]] = None
    ):
        self.level = level
        self.environment = environment
        self.max_payload_size = max_payload_size
        self.allowed_agents = allowed_agents or set()
        self.custom_rules = custom_rules or []


class ValidationRule:
    """Base class for validation rules."""
    
    def __init__(self, name: str, description: str, severity: EventSeverity = EventSeverity.ERROR):
        self.name = name
        self.description = description
        self.severity = severity
    
    def validate(self, event: AgentEvent, context: ValidationContext) -> Optional[str]:
        """
        Validate an event against this rule.
        
        Args:
            event: Event to validate
            context: Validation context
            
        Returns:
            Error message if validation fails, None if passes
        """
        raise NotImplementedError("Subclasses must implement validate method")


class EventValidationResult:
    """Result of event validation."""
    
    def __init__(self):
        self.is_valid: bool = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.rule_results: Dict[str, bool] = {}
        self.validation_time_ms: float = 0.0
        self.context_info: Dict[str, Any] = {}
    
    def add_error(self, rule_name: str, message: str) -> None:
        """Add a validation error."""
        self.is_valid = False
        self.errors.append(f"{rule_name}: {message}")
        self.rule_results[rule_name] = False
    
    def add_warning(self, rule_name: str, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(f"{rule_name}: {message}")
    
    def add_success(self, rule_name: str) -> None:
        """Mark a rule as passed."""
        self.rule_results[rule_name] = True
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            'is_valid': self.is_valid,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'rules_passed': sum(1 for passed in self.rule_results.values() if passed),
            'rules_failed': sum(1 for passed in self.rule_results.values() if not passed),
            'validation_time_ms': self.validation_time_ms
        }


class BasicValidationRule(ValidationRule):
    """Basic validation rules for all events."""
    
    def __init__(self):
        super().__init__("basic_validation", "Basic event structure validation")
    
    def validate(self, event: AgentEvent, context: ValidationContext) -> Optional[str]:
        # Check required fields
        if not event.id:
            return "Event ID is required"
        
        if not event.source_agent:
            return "Source agent is required"
        
        if not event.type:
            return "Event type is required"
        
        # Check timestamp
        if event.timestamp > datetime.utcnow() + timedelta(minutes=5):
            return "Event timestamp cannot be more than 5 minutes in the future"
        
        # Check expiration
        if event.expires_at and event.expires_at <= event.timestamp:
            return "Event expiration time must be after timestamp"
        
        return None


class RoutingValidationRule(ValidationRule):
    """Validation rules for event routing."""
    
    def __init__(self):
        super().__init__("routing_validation", "Event routing validation")
    
    def validate(self, event: AgentEvent, context: ValidationContext) -> Optional[str]:
        # Can't have both target_agent and target_agents
        if event.target_agent and event.target_agents:
            return "Event cannot have both target_agent and target_agents"
        
        # Target agents list should not be empty if provided
        if event.target_agents is not None and len(event.target_agents) == 0:
            return "target_agents list cannot be empty if provided"
        
        # Check agent whitelist if configured
        if context.allowed_agents:
            if event.source_agent not in context.allowed_agents:
                return f"Source agent '{event.source_agent}' not in allowed agents"
            
            if event.target_agent and event.target_agent not in context.allowed_agents:
                return f"Target agent '{event.target_agent}' not in allowed agents"
            
            if event.target_agents:
                invalid_targets = [agent for agent in event.target_agents if agent not in context.allowed_agents]
                if invalid_targets:
                    return f"Target agents not in allowed list: {invalid_targets}"
        
        return None


class PayloadValidationRule(ValidationRule):
    """Validation rules for event payloads."""
    
    def __init__(self):
        super().__init__("payload_validation", "Event payload validation")
        
        # Payload type requirements for specific event types
        self.required_payload_types = {
            EventType.TASK_START: TaskEventPayload,
            EventType.TASK_COMPLETE: TaskEventPayload,
            EventType.TASK_FAILED: TaskEventPayload,
            EventType.AGENT_START: AgentEventPayload,
            EventType.AGENT_READY: AgentEventPayload,
            EventType.AGENT_SHUTDOWN: AgentEventPayload,
            EventType.DECISION_MADE: DecisionEventPayload,
            EventType.COORDINATION_REQUEST: CoordinationEventPayload,
            EventType.ERROR_OCCURRED: ErrorEventPayload,
            EventType.PERFORMANCE_METRIC: PerformanceEventPayload,
        }
    
    def validate(self, event: AgentEvent, context: ValidationContext) -> Optional[str]:
        # Check payload size
        if hasattr(event.payload, 'model_dump'):
            payload_size = len(str(event.payload.model_dump_json()))
            if payload_size > context.max_payload_size:
                return f"Payload size ({payload_size} bytes) exceeds maximum ({context.max_payload_size} bytes)"
        
        # Check payload type consistency
        expected_type = self.required_payload_types.get(event.type)
        if expected_type and not isinstance(event.payload, expected_type):
            return f"Event type {event.type} expects payload type {expected_type.__name__}, got {type(event.payload).__name__}"
        
        # Validate specific payload types
        if isinstance(event.payload, TaskEventPayload):
            return self._validate_task_payload(event.payload)
        elif isinstance(event.payload, AgentEventPayload):
            return self._validate_agent_payload(event.payload)
        elif isinstance(event.payload, DecisionEventPayload):
            return self._validate_decision_payload(event.payload)
        elif isinstance(event.payload, ErrorEventPayload):
            return self._validate_error_payload(event.payload)
        
        return None
    
    def _validate_task_payload(self, payload: TaskEventPayload) -> Optional[str]:
        """Validate task event payload."""
        if not payload.task_id:
            return "Task ID is required in task events"
        
        if payload.progress is not None and not (0.0 <= payload.progress <= 1.0):
            return "Task progress must be between 0.0 and 1.0"
        
        return None
    
    def _validate_agent_payload(self, payload: AgentEventPayload) -> Optional[str]:
        """Validate agent event payload."""
        if not payload.agent_id:
            return "Agent ID is required in agent events"
        
        if not payload.status:
            return "Agent status is required in agent events"
        
        return None
    
    def _validate_decision_payload(self, payload: DecisionEventPayload) -> Optional[str]:
        """Validate decision event payload."""
        if not payload.decision_id:
            return "Decision ID is required in decision events"
        
        if not payload.decision_type:
            return "Decision type is required in decision events"
        
        if not payload.decision_summary:
            return "Decision summary is required in decision events"
        
        if payload.confidence is not None and not (0.0 <= payload.confidence <= 1.0):
            return "Decision confidence must be between 0.0 and 1.0"
        
        return None
    
    def _validate_error_payload(self, payload: ErrorEventPayload) -> Optional[str]:
        """Validate error event payload."""
        if not payload.error_id:
            return "Error ID is required in error events"
        
        if not payload.error_type:
            return "Error type is required in error events"
        
        if not payload.error_message:
            return "Error message is required in error events"
        
        return None


class BusinessLogicValidationRule(ValidationRule):
    """Business logic validation rules."""
    
    def __init__(self):
        super().__init__("business_logic", "Business logic validation")
    
    def validate(self, event: AgentEvent, context: ValidationContext) -> Optional[str]:
        # Priority consistency with severity
        if event.severity == EventSeverity.CRITICAL and event.priority < EventPriority.HIGH:
            return "Critical severity events should have high or critical priority"
        
        # Error events should have appropriate severity
        if event.type == EventType.ERROR_OCCURRED and event.severity == EventSeverity.DEBUG:
            return "Error events should not have debug severity"
        
        # Task completion events should have progress information
        if (event.type == EventType.TASK_COMPLETE and 
            isinstance(event.payload, TaskEventPayload) and
            event.payload.progress != 1.0):
            return "Completed tasks should have progress = 1.0"
        
        # Failed tasks should have error information
        if (event.type == EventType.TASK_FAILED and 
            isinstance(event.payload, TaskEventPayload) and
            not event.payload.error):
            return "Failed tasks should include error information"
        
        return None


class SecurityValidationRule(ValidationRule):
    """Security-related validation rules."""
    
    def __init__(self):
        super().__init__("security", "Security validation")
        
        # Patterns for potentially dangerous content
        self.dangerous_patterns = [
            re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'data:.*base64', re.IGNORECASE),
            re.compile(r'eval\s*\(', re.IGNORECASE),
            re.compile(r'exec\s*\(', re.IGNORECASE),
        ]
        
        # Agent ID pattern (alphanumeric, underscore, hyphen only)
        self.agent_id_pattern = re.compile(r'^[a-zA-Z0-9_-]+$')
    
    def validate(self, event: AgentEvent, context: ValidationContext) -> Optional[str]:
        # Validate agent ID format
        if not self.agent_id_pattern.match(event.source_agent):
            return f"Source agent ID contains invalid characters: {event.source_agent}"
        
        if event.target_agent and not self.agent_id_pattern.match(event.target_agent):
            return f"Target agent ID contains invalid characters: {event.target_agent}"
        
        # Check for potentially dangerous content in payload
        payload_json = event.payload.model_dump_json() if hasattr(event.payload, 'model_dump_json') else str(event.payload)
        
        for pattern in self.dangerous_patterns:
            if pattern.search(payload_json):
                return f"Payload contains potentially dangerous content matching pattern: {pattern.pattern}"
        
        # Check metadata for dangerous content
        metadata_str = str(event.metadata)
        for pattern in self.dangerous_patterns:
            if pattern.search(metadata_str):
                return f"Metadata contains potentially dangerous content matching pattern: {pattern.pattern}"
        
        return None


class PerformanceValidationRule(ValidationRule):
    """Performance-related validation rules."""
    
    def __init__(self):
        super().__init__("performance", "Performance validation", EventSeverity.WARNING)
    
    def validate(self, event: AgentEvent, context: ValidationContext) -> Optional[str]:
        # Check event age (warn if too old)
        age_seconds = event.age_seconds
        if age_seconds > 300:  # 5 minutes
            return f"Event is {age_seconds:.1f} seconds old, may indicate processing delays"
        
        # Check payload complexity
        if hasattr(event.payload, 'model_dump'):
            payload_dict = event.payload.model_dump()
            if self._is_deeply_nested(payload_dict, max_depth=5):
                return "Payload is deeply nested, may impact performance"
        
        # Check tags count
        if len(event.tags) > 20:
            return f"Event has {len(event.tags)} tags, consider reducing for better performance"
        
        # Check metadata size
        metadata_size = len(str(event.metadata))
        if metadata_size > 10240:  # 10KB
            return f"Metadata size ({metadata_size} bytes) is large, may impact performance"
        
        return None
    
    def _is_deeply_nested(self, obj: Any, current_depth: int = 0, max_depth: int = 5) -> bool:
        """Check if object is deeply nested."""
        if current_depth >= max_depth:
            return True
        
        if isinstance(obj, dict):
            return any(self._is_deeply_nested(v, current_depth + 1, max_depth) for v in obj.values())
        elif isinstance(obj, list):
            return any(self._is_deeply_nested(item, current_depth + 1, max_depth) for item in obj)
        
        return False


class EnhancedEventValidator:
    """
    Enhanced event validator with comprehensive rule-based validation.
    
    Provides configurable validation with different levels and custom rules.
    """
    
    def __init__(self):
        self.rules: List[ValidationRule] = []
        self._setup_default_rules()
    
    def _setup_default_rules(self) -> None:
        """Setup default validation rules."""
        self.rules = [
            BasicValidationRule(),
            RoutingValidationRule(),
            PayloadValidationRule(),
            BusinessLogicValidationRule(),
            SecurityValidationRule(),
            PerformanceValidationRule(),
        ]
    
    def add_rule(self, rule: ValidationRule) -> None:
        """Add a custom validation rule."""
        self.rules.append(rule)
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a validation rule by name."""
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                del self.rules[i]
                return True
        return False
    
    def validate_event(
        self,
        event: AgentEvent,
        context: Optional[ValidationContext] = None,
        rules_to_run: Optional[List[str]] = None
    ) -> EventValidationResult:
        """
        Validate an event against all configured rules.
        
        Args:
            event: Event to validate
            context: Validation context
            rules_to_run: Specific rules to run (None for all)
            
        Returns:
            Validation result
        """
        start_time = datetime.utcnow()
        context = context or ValidationContext()
        result = EventValidationResult()
        result.context_info = {
            'validation_level': context.level.value,
            'environment': context.environment,
            'rules_count': len(self.rules)
        }
        
        try:
            # Run Pydantic validation first
            try:
                event.model_validate(event.model_dump())
                result.add_success("pydantic_validation")
            except ValidationError as e:
                result.add_error("pydantic_validation", f"Pydantic validation failed: {str(e)}")
                if context.level == ValidationLevel.BASIC:
                    return result  # Stop here for basic validation
            
            # Run custom rules
            for rule in self.rules:
                if rules_to_run and rule.name not in rules_to_run:
                    continue
                
                if context.level == ValidationLevel.BASIC and rule.name != "basic_validation":
                    continue
                
                try:
                    error_message = rule.validate(event, context)
                    if error_message:
                        if rule.severity == EventSeverity.WARNING:
                            result.add_warning(rule.name, error_message)
                        else:
                            result.add_error(rule.name, error_message)
                    else:
                        result.add_success(rule.name)
                        
                except Exception as e:
                    logger.error(f"Rule {rule.name} failed with exception: {e}")
                    result.add_error(rule.name, f"Rule execution failed: {str(e)}")
            
            # Run custom rules from context
            for custom_rule in context.custom_rules:
                try:
                    custom_result = custom_rule(event, context)
                    if custom_result:
                        result.add_error("custom_rule", custom_result)
                    else:
                        result.add_success("custom_rule")
                except Exception as e:
                    result.add_error("custom_rule", f"Custom rule failed: {str(e)}")
            
        except Exception as e:
            logger.error(f"Event validation failed with exception: {e}")
            result.add_error("validation_system", f"Validation system error: {str(e)}")
        
        finally:
            end_time = datetime.utcnow()
            result.validation_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return result
    
    def validate_events_batch(
        self,
        events: List[AgentEvent],
        context: Optional[ValidationContext] = None,
        stop_on_first_error: bool = False
    ) -> List[EventValidationResult]:
        """
        Validate a batch of events.
        
        Args:
            events: Events to validate
            context: Validation context
            stop_on_first_error: Stop validation on first error
            
        Returns:
            List of validation results
        """
        results = []
        
        for event in events:
            result = self.validate_event(event, context)
            results.append(result)
            
            if stop_on_first_error and not result.is_valid:
                break
        
        return results
    
    def get_validation_summary(self, results: List[EventValidationResult]) -> Dict[str, Any]:
        """Get summary of batch validation results."""
        total_events = len(results)
        valid_events = sum(1 for r in results if r.is_valid)
        total_errors = sum(len(r.errors) for r in results)
        total_warnings = sum(len(r.warnings) for r in results)
        avg_time_ms = sum(r.validation_time_ms for r in results) / total_events if total_events > 0 else 0
        
        rule_failures = {}
        for result in results:
            for rule_name, passed in result.rule_results.items():
                if not passed:
                    rule_failures[rule_name] = rule_failures.get(rule_name, 0) + 1
        
        return {
            'total_events': total_events,
            'valid_events': valid_events,
            'invalid_events': total_events - valid_events,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'validation_rate': valid_events / total_events if total_events > 0 else 0,
            'average_validation_time_ms': avg_time_ms,
            'rule_failure_counts': rule_failures
        }


# Global validator instance
_event_validator: Optional[EnhancedEventValidator] = None


def get_event_validator() -> EnhancedEventValidator:
    """Get the global event validator instance."""
    global _event_validator
    if _event_validator is None:
        _event_validator = EnhancedEventValidator()
    return _event_validator


def validate_event(
    event: AgentEvent,
    level: ValidationLevel = ValidationLevel.STANDARD,
    context: Optional[ValidationContext] = None
) -> EventValidationResult:
    """Validate a single event."""
    validator = get_event_validator()
    context = context or ValidationContext(level=level)
    return validator.validate_event(event, context)


def validate_events(
    events: List[AgentEvent],
    level: ValidationLevel = ValidationLevel.STANDARD,
    context: Optional[ValidationContext] = None,
    stop_on_first_error: bool = False
) -> List[EventValidationResult]:
    """Validate multiple events."""
    validator = get_event_validator()
    context = context or ValidationContext(level=level)
    return validator.validate_events_batch(events, context, stop_on_first_error)


def create_custom_validation_rule(
    name: str,
    description: str,
    validation_function: Callable[[AgentEvent, ValidationContext], Optional[str]],
    severity: EventSeverity = EventSeverity.ERROR
) -> ValidationRule:
    """Create a custom validation rule from a function."""
    
    class CustomRule(ValidationRule):
        def __init__(self):
            super().__init__(name, description, severity)
            self.validation_function = validation_function
        
        def validate(self, event: AgentEvent, context: ValidationContext) -> Optional[str]:
            return self.validation_function(event, context)
    
    return CustomRule()


def register_custom_rule(rule: ValidationRule) -> None:
    """Register a custom validation rule globally."""
    validator = get_event_validator()
    validator.add_rule(rule)