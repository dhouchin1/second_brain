"""
Error Handling Classes

Custom exception hierarchy for Autom8 with structured error information,
recovery suggestions, and proper error propagation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class ErrorSeverity(str, Enum):
    """Error severity levels"""
    LOW = "low"           # Minor issues that don't block operation
    MEDIUM = "medium"     # Issues that may affect quality but allow continuation
    HIGH = "high"         # Serious issues that block current operation
    CRITICAL = "critical" # System-level issues requiring immediate attention


class ErrorCategory(str, Enum):
    """Categories of errors for better classification"""
    CONFIGURATION = "configuration"   # Configuration/settings errors
    VALIDATION = "validation"         # Input validation errors
    MODEL = "model"                   # Model-related errors
    CONTEXT = "context"               # Context management errors
    MEMORY = "memory"                 # Shared memory errors
    ROUTING = "routing"               # Model routing errors
    COMPLEXITY = "complexity"         # Complexity analysis errors
    NETWORK = "network"               # Network/API errors
    STORAGE = "storage"               # Database/storage errors
    PERMISSION = "permission"         # Permission/auth errors
    RESOURCE = "resource"             # Resource limitation errors


class ErrorContext(BaseModel):
    """
    Structured error context information
    """
    operation: str = "unknown"
    component: str = "autom8"
    timestamp: datetime = datetime.utcnow()
    
    # Context details
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    model_name: Optional[str] = None
    request_id: Optional[str] = None
    
    # State information
    system_state: Dict[str, Any] = {}
    operation_params: Dict[str, Any] = {}
    
    # Error chain
    caused_by: Optional[str] = None
    error_chain: List[str] = []


class RecoveryAction(BaseModel):
    """
    Suggested recovery action for an error
    """
    action: str
    description: str
    automatic: bool = False  # Can be performed automatically
    user_required: bool = False  # Requires user intervention
    confidence: float = 1.0  # Confidence this will resolve the issue
    
    # Parameters for automatic recovery
    retry_delay: Optional[float] = None
    max_retries: Optional[int] = None


class Autom8Error(Exception):
    """
    Base exception class for all Autom8 errors
    
    Provides structured error information, recovery suggestions,
    and proper error propagation.
    """
    
    def __init__(
        self,
        message: str,
        *,
        category: ErrorCategory = ErrorCategory.CONFIGURATION,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        error_code: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        recovery_actions: Optional[List[RecoveryAction]] = None,
        original_error: Optional[Exception] = None,
        user_message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        
        self.message = message
        self.category = category
        self.severity = severity
        self.error_code = error_code or self._generate_error_code()
        self.context = context or ErrorContext()
        self.recovery_actions = recovery_actions or []
        self.original_error = original_error
        self.user_message = user_message or self._generate_user_message()
        self.details = details or {}
        
        # Add to error chain if there's an original error
        if original_error:
            self.context.caused_by = str(original_error)
            self.context.error_chain.append(str(original_error))
    
    def _generate_error_code(self) -> str:
        """Generate a unique error code"""
        import hashlib
        import time
        
        # Create deterministic but unique error code
        content = f"{self.category.value}:{self.__class__.__name__}:{time.time()}"
        hash_obj = hashlib.md5(content.encode())
        return f"A8-{hash_obj.hexdigest()[:8].upper()}"
    
    def _generate_user_message(self) -> str:
        """Generate user-friendly error message"""
        return f"An error occurred in {self.category.value}: {self.message}"
    
    def add_recovery_action(self, action: RecoveryAction) -> None:
        """Add a recovery action to this error"""
        self.recovery_actions.append(action)
    
    def get_automatic_recovery_actions(self) -> List[RecoveryAction]:
        """Get recovery actions that can be performed automatically"""
        return [action for action in self.recovery_actions if action.automatic]
    
    def get_user_recovery_actions(self) -> List[RecoveryAction]:
        """Get recovery actions that require user intervention"""
        return [action for action in self.recovery_actions if action.user_required]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "user_message": self.user_message,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context.dict() if self.context else None,
            "recovery_actions": [action.dict() for action in self.recovery_actions],
            "original_error": str(self.original_error) if self.original_error else None,
            "details": self.details,
        }
    
    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.error_code}, message='{self.message}')"


class ConfigurationError(Autom8Error):
    """Configuration-related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.CONFIGURATION)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class ValidationError(Autom8Error):
    """Input validation errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.VALIDATION)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class ContextError(Autom8Error):
    """Context management errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.CONTEXT)
        super().__init__(message, **kwargs)


class ComplexityError(Autom8Error):
    """Complexity analysis errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.COMPLEXITY)
        super().__init__(message, **kwargs)


class RoutingError(Autom8Error):
    """Model routing errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.ROUTING)
        super().__init__(message, **kwargs)


class MemoryError(Autom8Error):
    """Shared memory errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.MEMORY)
        super().__init__(message, **kwargs)


class ModelError(Autom8Error):
    """Model-related errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.MODEL)
        super().__init__(message, **kwargs)


class NetworkError(Autom8Error):
    """Network and API errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.NETWORK)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class StorageError(Autom8Error):
    """Database and storage errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.STORAGE)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class PermissionError(Autom8Error):
    """Permission and authentication errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.PERMISSION)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class ResourceError(Autom8Error):
    """Resource limitation errors (memory, disk, etc.)"""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.RESOURCE)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


# Error factory functions for common scenarios

def create_configuration_error(
    message: str,
    config_key: Optional[str] = None,
    expected_type: Optional[str] = None,
    actual_value: Optional[Any] = None
) -> ConfigurationError:
    """Create a configuration error with recovery suggestions"""
    
    details = {}
    if config_key:
        details["config_key"] = config_key
    if expected_type:
        details["expected_type"] = expected_type
    if actual_value is not None:
        details["actual_value"] = str(actual_value)
    
    recovery_actions = []
    
    if config_key:
        recovery_actions.append(RecoveryAction(
            action="check_configuration",
            description=f"Verify the configuration value for '{config_key}'",
            user_required=True,
            confidence=0.9
        ))
    
    recovery_actions.append(RecoveryAction(
        action="use_default_configuration",
        description="Use default configuration values",
        automatic=True,
        confidence=0.7
    ))
    
    return ConfigurationError(
        message=message,
        details=details,
        recovery_actions=recovery_actions
    )


def create_model_unavailable_error(
    model_name: str,
    reason: str = "unknown",
    available_alternatives: Optional[List[str]] = None
) -> ModelError:
    """Create a model unavailable error with fallback suggestions"""
    
    message = f"Model '{model_name}' is not available: {reason}"
    
    details = {
        "model_name": model_name,
        "reason": reason,
        "available_alternatives": available_alternatives or []
    }
    
    recovery_actions = []
    
    if available_alternatives:
        recovery_actions.append(RecoveryAction(
            action="use_alternative_model",
            description=f"Use alternative model: {', '.join(available_alternatives)}",
            automatic=True,
            confidence=0.8
        ))
    
    recovery_actions.append(RecoveryAction(
        action="check_model_installation",
        description=f"Verify that '{model_name}' is properly installed and accessible",
        user_required=True,
        confidence=0.9
    ))
    
    return ModelError(
        message=message,
        details=details,
        recovery_actions=recovery_actions,
        severity=ErrorSeverity.HIGH
    )


def create_context_budget_exceeded_error(
    current_tokens: int,
    budget: int,
    context_sources: Optional[List[str]] = None
) -> ContextError:
    """Create a context budget exceeded error with optimization suggestions"""
    
    message = f"Context size ({current_tokens} tokens) exceeds budget ({budget} tokens)"
    
    details = {
        "current_tokens": current_tokens,
        "budget": budget,
        "overage": current_tokens - budget,
        "context_sources": context_sources or []
    }
    
    recovery_actions = [
        RecoveryAction(
            action="auto_summarize",
            description="Automatically summarize context to fit budget",
            automatic=True,
            confidence=0.8
        ),
        RecoveryAction(
            action="remove_low_priority_context",
            description="Remove low-priority context items",
            automatic=True,
            confidence=0.7
        ),
        RecoveryAction(
            action="increase_budget",
            description="Increase the context token budget",
            user_required=True,
            confidence=0.9
        )
    ]
    
    return ContextError(
        message=message,
        details=details,
        recovery_actions=recovery_actions,
        severity=ErrorSeverity.MEDIUM
    )


def create_network_error(
    operation: str,
    url: Optional[str] = None,
    status_code: Optional[int] = None,
    original_error: Optional[Exception] = None
) -> NetworkError:
    """Create a network error with retry suggestions"""
    
    message = f"Network error during {operation}"
    if status_code:
        message += f" (HTTP {status_code})"
    
    details = {
        "operation": operation,
        "url": url,
        "status_code": status_code
    }
    
    recovery_actions = []
    
    # Suggest retry for transient errors
    if status_code in [429, 500, 502, 503, 504] or not status_code:
        recovery_actions.append(RecoveryAction(
            action="retry_with_backoff",
            description="Retry the operation with exponential backoff",
            automatic=True,
            confidence=0.7,
            retry_delay=1.0,
            max_retries=3
        ))
    
    # Suggest checking connection
    recovery_actions.append(RecoveryAction(
        action="check_network_connection",
        description="Verify network connectivity and service availability",
        user_required=True,
        confidence=0.8
    ))
    
    return NetworkError(
        message=message,
        details=details,
        recovery_actions=recovery_actions,
        original_error=original_error
    )


# Error handling utilities

class ErrorHandler:
    """
    Centralized error handling with automatic recovery
    """
    
    def __init__(self):
        self.error_history: List[Autom8Error] = []
        self.max_history = 100
    
    def handle_error(
        self,
        error: Union[Autom8Error, Exception],
        context: Optional[ErrorContext] = None,
        attempt_recovery: bool = True
    ) -> Optional[Any]:
        """
        Handle an error with optional automatic recovery
        
        Args:
            error: The error to handle
            context: Additional context information
            attempt_recovery: Whether to attempt automatic recovery
            
        Returns:
            Recovery result if successful, None otherwise
        """
        # Convert to Autom8Error if needed
        if not isinstance(error, Autom8Error):
            error = Autom8Error(
                message=str(error),
                original_error=error,
                context=context
            )
        
        # Add to error history
        self.error_history.append(error)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Log the error
        from autom8.utils.logging import get_logger
        logger = get_logger(__name__)
        
        logger.error(
            f"Error occurred: {error.message}",
            extra={
                "error_code": error.error_code,
                "category": error.category.value,
                "severity": error.severity.value,
                "context": error.context.dict() if error.context else None,
                "recovery_actions_available": len(error.recovery_actions)
            }
        )
        
        # Attempt automatic recovery if enabled
        if attempt_recovery:
            for action in error.get_automatic_recovery_actions():
                logger.info(f"Attempting automatic recovery: {action.action}")
                try:
                    result = self._execute_recovery_action(action, error)
                    if result is not None:
                        logger.info(f"Automatic recovery successful: {action.action}")
                        return result
                except Exception as recovery_error:
                    logger.warning(f"Recovery action failed: {recovery_error}")
        
        # Re-raise the error if no recovery was successful
        raise error
    
    def _execute_recovery_action(self, action: RecoveryAction, error: Autom8Error) -> Optional[Any]:
        """Execute a recovery action (placeholder for implementation)"""
        # This would contain the actual recovery logic
        # For now, just return None to indicate no recovery
        return None
    
    def get_recent_errors(self, limit: int = 10) -> List[Autom8Error]:
        """Get recent errors"""
        return self.error_history[-limit:]
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[Autom8Error]:
        """Get errors by category"""
        return [error for error in self.error_history if error.category == category]


# Global error handler instance
_error_handler = None


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def handle_error(
    error: Union[Autom8Error, Exception],
    context: Optional[ErrorContext] = None,
    attempt_recovery: bool = True
) -> Optional[Any]:
    """Convenience function to handle errors"""
    return get_error_handler().handle_error(error, context, attempt_recovery)