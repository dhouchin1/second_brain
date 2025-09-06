# ──────────────────────────────────────────────────────────────────────────────
# File: services/capture_error_handler.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Unified Error Handling & User Feedback System

Provides centralized error classification, handling, and user-friendly feedback
for all capture operations across the Second Brain system.
"""

import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from contextlib import asynccontextmanager
import time

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for appropriate handling and user feedback."""
    LOW = "low"          # Informational warnings, non-blocking issues
    MEDIUM = "medium"    # Degraded functionality but operation continues
    HIGH = "high"        # Operation failed but retry likely to succeed
    CRITICAL = "critical"  # System-level failure requiring immediate attention


class ErrorCategory(Enum):
    """Categories of errors for appropriate handling strategies."""
    VALIDATION = "validation"         # Input validation errors
    PROCESSING = "processing"         # Content processing errors
    NETWORK = "network"              # Network connectivity issues
    STORAGE = "storage"              # Database/filesystem errors
    EXTERNAL_SERVICE = "external_service"  # Third-party service errors
    AUTHENTICATION = "authentication"  # Auth/permission errors
    RESOURCE = "resource"            # Memory/disk/CPU constraints
    CONFIGURATION = "configuration"   # System configuration issues
    UNKNOWN = "unknown"              # Unclassified errors


class RetryStrategy(Enum):
    """Retry strategies for different types of failures."""
    NO_RETRY = "no_retry"
    IMMEDIATE = "immediate"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    CUSTOM = "custom"


@dataclass
class ErrorContext:
    """Context information for error classification and handling."""
    operation: str
    source_type: str
    content_type: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    strategy: RetryStrategy
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    custom_delay_func: Optional[Callable[[int], float]] = None


@dataclass 
class ErrorClassification:
    """Classification and handling instructions for an error."""
    category: ErrorCategory
    severity: ErrorSeverity
    user_message: str
    technical_message: str
    retry_config: Optional[RetryConfig] = None
    suggested_actions: List[str] = field(default_factory=list)
    should_log: bool = True
    should_alert: bool = False
    is_transient: bool = False
    requires_user_action: bool = False


@dataclass
class ProcessingProgress:
    """Progress tracking for long-running operations."""
    operation_id: str
    operation_name: str
    current_step: str
    steps_completed: int
    total_steps: int
    progress_percent: float
    estimated_time_remaining: Optional[float] = None
    can_cancel: bool = True
    detailed_status: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class UserFeedback:
    """User-friendly feedback for capture operations."""
    success: bool
    title: str
    message: str
    severity: ErrorSeverity
    suggested_actions: List[str] = field(default_factory=list)
    retry_available: bool = False
    progress: Optional[ProcessingProgress] = None
    technical_details: Optional[str] = None
    correlation_id: Optional[str] = None


class CaptureErrorHandler:
    """Centralized error handling and user feedback system."""
    
    def __init__(self):
        """Initialize the error handler with classification rules."""
        self.error_patterns: Dict[str, ErrorClassification] = {}
        self.retry_statistics: Dict[str, Dict[str, Any]] = {}
        self.progress_tracking: Dict[str, ProcessingProgress] = {}
        self._setup_default_classifications()
        
    def _setup_default_classifications(self):
        """Setup default error classifications and handling rules."""
        # Validation errors
        self.add_error_pattern(
            r"invalid.*content.*type|unsupported.*format|file.*too.*large",
            ErrorClassification(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                user_message="The content you're trying to capture isn't supported or is too large.",
                technical_message="Content validation failed",
                suggested_actions=[
                    "Check that your file is in a supported format (PDF, images, text, audio)",
                    "Ensure the file is under the size limit (25MB for most content)",
                    "Try uploading a smaller or different file"
                ],
                requires_user_action=True
            )
        )
        
        # Network errors
        self.add_error_pattern(
            r"connection.*refused|timeout|network.*error|dns.*resolution",
            ErrorClassification(
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.HIGH,
                user_message="We're having trouble connecting to external services. This is usually temporary.",
                technical_message="Network connectivity issue",
                retry_config=RetryConfig(
                    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                    max_attempts=3,
                    base_delay=2.0
                ),
                suggested_actions=[
                    "Wait a moment and try again",
                    "Check your internet connection",
                    "The service may be temporarily unavailable"
                ],
                is_transient=True
            )
        )
        
        # Storage errors
        self.add_error_pattern(
            r"disk.*full|storage.*error|database.*locked|permission.*denied",
            ErrorClassification(
                category=ErrorCategory.STORAGE,
                severity=ErrorSeverity.CRITICAL,
                user_message="We're experiencing storage issues. Your content couldn't be saved.",
                technical_message="Storage system error",
                suggested_actions=[
                    "Try again in a few minutes",
                    "Contact support if the problem persists",
                    "Check available disk space"
                ],
                should_alert=True
            )
        )
        
        # External service errors  
        self.add_error_pattern(
            r"ollama.*error|whisper.*error|ocr.*failed|ai.*processing.*failed",
            ErrorClassification(
                category=ErrorCategory.EXTERNAL_SERVICE,
                severity=ErrorSeverity.HIGH,
                user_message="AI processing is temporarily unavailable. Your content was saved but may lack smart features.",
                technical_message="External AI service error",
                retry_config=RetryConfig(
                    strategy=RetryStrategy.LINEAR_BACKOFF,
                    max_attempts=2,
                    base_delay=5.0
                ),
                suggested_actions=[
                    "Content was captured successfully without AI enhancements",
                    "Smart features (summarization, tagging) can be added later",
                    "Try again to enable AI features"
                ],
                is_transient=True
            )
        )
        
        # Resource constraints
        self.add_error_pattern(
            r"memory.*error|out.*of.*memory|resource.*exhausted|processing.*timeout",
            ErrorClassification(
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.HIGH,
                user_message="The content is too complex to process right now. Try breaking it into smaller pieces.",
                technical_message="Resource constraint encountered",
                suggested_actions=[
                    "Try uploading smaller files or shorter content",
                    "Wait for system resources to free up",
                    "Consider processing in batches"
                ],
                is_transient=True
            )
        )
        
        # Authentication errors
        self.add_error_pattern(
            r"unauthorized|authentication.*failed|access.*denied|invalid.*token",
            ErrorClassification(
                category=ErrorCategory.AUTHENTICATION,
                severity=ErrorSeverity.MEDIUM,
                user_message="Authentication failed. Please log in again.",
                technical_message="Authentication/authorization error",
                suggested_actions=[
                    "Log out and log back in",
                    "Check your credentials",
                    "Contact support if you continue to have access issues"
                ],
                requires_user_action=True
            )
        )
    
    def add_error_pattern(self, pattern: str, classification: ErrorClassification):
        """Add a new error pattern and its classification."""
        self.error_patterns[pattern] = classification
    
    def classify_error(self, error: Exception, context: Optional[ErrorContext] = None) -> ErrorClassification:
        """Classify an error and determine appropriate handling strategy."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Check against known patterns
        import re
        for pattern, classification in self.error_patterns.items():
            if re.search(pattern, error_str, re.IGNORECASE) or re.search(pattern, error_type, re.IGNORECASE):
                return classification
        
        # Default classification for unknown errors
        severity = ErrorSeverity.HIGH
        if "warning" in error_str or isinstance(error, (UserWarning,)):
            severity = ErrorSeverity.LOW
        elif "critical" in error_str or isinstance(error, (SystemError, MemoryError)):
            severity = ErrorSeverity.CRITICAL
            
        return ErrorClassification(
            category=ErrorCategory.UNKNOWN,
            severity=severity,
            user_message="An unexpected error occurred. We've logged the details and will investigate.",
            technical_message=f"{type(error).__name__}: {str(error)}",
            retry_config=RetryConfig(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=2
            ),
            suggested_actions=[
                "Try the operation again",
                "Contact support if the problem persists",
                "Include details about what you were trying to capture"
            ],
            is_transient=True
        )
    
    async def handle_error(
        self, 
        error: Exception, 
        context: Optional[ErrorContext] = None,
        operation_id: Optional[str] = None
    ) -> UserFeedback:
        """
        Handle an error and provide appropriate user feedback.
        
        Args:
            error: The exception that occurred
            context: Additional context about the operation
            operation_id: ID for tracking progress/retry operations
            
        Returns:
            UserFeedback: Formatted feedback for the user
        """
        classification = self.classify_error(error, context)
        
        # Log the error appropriately
        if classification.should_log:
            log_level = logging.ERROR if classification.severity in (ErrorSeverity.HIGH, ErrorSeverity.CRITICAL) else logging.WARNING
            logger.log(
                log_level,
                f"Capture error in {context.operation if context else 'unknown'}: {classification.technical_message}",
                extra={
                    "error_category": classification.category.value,
                    "error_severity": classification.severity.value,
                    "context": context.__dict__ if context else None,
                    "operation_id": operation_id
                },
                exc_info=classification.severity == ErrorSeverity.CRITICAL
            )
        
        # Update retry statistics
        if context:
            operation_key = f"{context.operation}:{context.source_type}:{context.content_type}"
            if operation_key not in self.retry_statistics:
                self.retry_statistics[operation_key] = {
                    "total_errors": 0,
                    "by_category": {},
                    "success_rate_24h": 0.0,
                    "last_error_time": None
                }
            
            self.retry_statistics[operation_key]["total_errors"] += 1
            self.retry_statistics[operation_key]["by_category"][classification.category.value] = \
                self.retry_statistics[operation_key]["by_category"].get(classification.category.value, 0) + 1
            self.retry_statistics[operation_key]["last_error_time"] = datetime.now()
        
        # Create user feedback
        feedback = UserFeedback(
            success=False,
            title=f"{classification.severity.value.title()} Error",
            message=classification.user_message,
            severity=classification.severity,
            suggested_actions=classification.suggested_actions,
            retry_available=classification.retry_config is not None,
            technical_details=classification.technical_message if classification.severity == ErrorSeverity.CRITICAL else None,
            correlation_id=operation_id
        )
        
        # Add progress info if available
        if operation_id and operation_id in self.progress_tracking:
            progress = self.progress_tracking[operation_id]
            progress.errors.append(classification.user_message)
            feedback.progress = progress
        
        return feedback
    
    @asynccontextmanager
    async def capture_operation(
        self,
        operation_name: str,
        context: ErrorContext,
        total_steps: int = 1,
        enable_progress: bool = True
    ):
        """
        Context manager for capture operations with automatic error handling and progress tracking.
        
        Usage:
            async with error_handler.capture_operation("pdf_processing", context, 3) as progress:
                await progress.update_step("Extracting text", 1)
                # ... do work ...
                await progress.update_step("Processing with AI", 2)
                # ... do more work ...
                await progress.complete("Processing completed")
        """
        operation_id = f"{operation_name}_{context.source_type}_{int(time.time() * 1000)}"
        
        progress_tracker = None
        if enable_progress:
            progress_tracker = ProcessingProgress(
                operation_id=operation_id,
                operation_name=operation_name,
                current_step="Starting",
                steps_completed=0,
                total_steps=total_steps,
                progress_percent=0.0
            )
            self.progress_tracking[operation_id] = progress_tracker
        
        start_time = time.time()
        
        class ProgressUpdater:
            def __init__(self, handler, tracker):
                self.handler = handler
                self.tracker = tracker
                
            async def update_step(self, step_name: str, step_number: int, detailed_status: str = None):
                if self.tracker:
                    self.tracker.current_step = step_name
                    self.tracker.steps_completed = step_number
                    self.tracker.progress_percent = (step_number / self.tracker.total_steps) * 100
                    self.tracker.detailed_status = detailed_status
                    
                    # Estimate remaining time
                    elapsed = time.time() - start_time
                    if step_number > 0:
                        avg_time_per_step = elapsed / step_number
                        remaining_steps = self.tracker.total_steps - step_number
                        self.tracker.estimated_time_remaining = avg_time_per_step * remaining_steps
            
            async def add_warning(self, message: str):
                if self.tracker:
                    self.tracker.warnings.append(message)
            
            async def complete(self, final_status: str = "Completed"):
                if self.tracker:
                    self.tracker.current_step = final_status
                    self.tracker.steps_completed = self.tracker.total_steps
                    self.tracker.progress_percent = 100.0
                    self.tracker.estimated_time_remaining = 0.0
        
        try:
            yield ProgressUpdater(self, progress_tracker)
            
            # Update success statistics
            if context:
                operation_key = f"{context.operation}:{context.source_type}:{context.content_type}"
                if operation_key not in self.retry_statistics:
                    self.retry_statistics[operation_key] = {
                        "total_errors": 0,
                        "total_successes": 0,
                        "by_category": {},
                        "success_rate_24h": 0.0,
                        "last_success_time": None
                    }
                self.retry_statistics[operation_key]["total_successes"] = \
                    self.retry_statistics[operation_key].get("total_successes", 0) + 1
                self.retry_statistics[operation_key]["last_success_time"] = datetime.now()
                
        except Exception as e:
            # Handle the error and re-raise with context
            await self.handle_error(e, context, operation_id)
            raise
        finally:
            # Clean up progress tracking
            if operation_id in self.progress_tracking:
                # Keep for a short time to allow final status queries
                asyncio.create_task(self._cleanup_progress(operation_id, delay=300))  # 5 minutes
    
    async def _cleanup_progress(self, operation_id: str, delay: int = 300):
        """Clean up progress tracking after a delay."""
        await asyncio.sleep(delay)
        self.progress_tracking.pop(operation_id, None)
    
    def get_progress(self, operation_id: str) -> Optional[ProcessingProgress]:
        """Get current progress for an operation."""
        return self.progress_tracking.get(operation_id)
    
    def get_retry_statistics(self, operation_type: Optional[str] = None) -> Dict[str, Any]:
        """Get retry and error statistics."""
        if operation_type:
            return self.retry_statistics.get(operation_type, {})
        
        # Return aggregated statistics
        total_errors = sum(stats.get("total_errors", 0) for stats in self.retry_statistics.values())
        total_successes = sum(stats.get("total_successes", 0) for stats in self.retry_statistics.values())
        
        return {
            "total_operations": total_errors + total_successes,
            "total_errors": total_errors,
            "total_successes": total_successes,
            "overall_success_rate": (total_successes / max(1, total_errors + total_successes)) * 100,
            "by_operation": self.retry_statistics
        }
    
    async def should_retry(
        self, 
        error: Exception, 
        attempt_number: int,
        context: Optional[ErrorContext] = None
    ) -> Tuple[bool, float]:
        """
        Determine if an operation should be retried and calculate delay.
        
        Returns:
            Tuple of (should_retry, delay_seconds)
        """
        classification = self.classify_error(error, context)
        
        if not classification.retry_config:
            return False, 0.0
        
        retry_config = classification.retry_config
        
        if attempt_number >= retry_config.max_attempts:
            return False, 0.0
        
        if not classification.is_transient:
            return False, 0.0
        
        # Calculate delay based on strategy
        delay = 0.0
        
        if retry_config.strategy == RetryStrategy.IMMEDIATE:
            delay = 0.0
        elif retry_config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = retry_config.base_delay * attempt_number
        elif retry_config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = retry_config.base_delay * (retry_config.backoff_multiplier ** (attempt_number - 1))
        elif retry_config.strategy == RetryStrategy.CUSTOM and retry_config.custom_delay_func:
            delay = retry_config.custom_delay_func(attempt_number)
        
        # Apply jitter if enabled
        if retry_config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # 50-100% of calculated delay
        
        # Cap at maximum delay
        delay = min(delay, retry_config.max_delay)
        
        return True, delay


# Global error handler instance
_error_handler: Optional[CaptureErrorHandler] = None


def get_capture_error_handler() -> CaptureErrorHandler:
    """Get the global capture error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = CaptureErrorHandler()
    return _error_handler


# Convenience decorators and utilities
def handle_capture_errors(operation_name: str = None):
    """Decorator to automatically handle errors in capture operations."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            error_handler = get_capture_error_handler()
            
            # Try to extract context from function arguments
            context = ErrorContext(
                operation=operation_name or func.__name__,
                source_type="unknown",
                content_type="unknown"
            )
            
            # Look for context in kwargs
            if "context" in kwargs:
                context = kwargs["context"]
            elif hasattr(args[0], "__class__") and "capture" in args[0].__class__.__name__.lower():
                # Try to infer from service class
                context.operation = operation_name or func.__name__
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                feedback = await error_handler.handle_error(e, context)
                # Re-raise with user-friendly context
                raise CaptureError(feedback) from e
        
        return wrapper
    return decorator


class CaptureError(Exception):
    """Exception that includes user feedback information."""
    
    def __init__(self, feedback: UserFeedback):
        super().__init__(feedback.message)
        self.feedback = feedback