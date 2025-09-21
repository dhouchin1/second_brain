"""
Enhanced Observability Logger

Provides comprehensive structured logging with correlation tracking,
distributed tracing integration, and performance monitoring.
"""

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import structlog
from pydantic import BaseModel, Field

from autom8.utils.logging import get_logger

# Context variables for correlation tracking
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
span_id_var: ContextVar[Optional[str]] = ContextVar("span_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


class LogLevel(str, Enum):
    """Enhanced log levels for observability."""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SECURITY = "SECURITY"
    AUDIT = "AUDIT"
    BUSINESS = "BUSINESS"
    PERFORMANCE = "PERFORMANCE"


class LogCategory(str, Enum):
    """Log categories for better organization."""
    SYSTEM = "system"
    BUSINESS = "business"  
    SECURITY = "security"
    PERFORMANCE = "performance"
    ERROR = "error"
    AUDIT = "audit"
    MODEL = "model"
    CONTEXT = "context"
    ROUTING = "routing"
    INTEGRATION = "integration"


@dataclass
class CorrelationContext:
    """Context for correlating logs across distributed operations."""
    
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "user_id": self.user_id,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "operation": self.operation,
            "component": self.component,
            "metadata": self.metadata
        }


@dataclass
class TraceContext:
    """Context for distributed tracing integration."""
    
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    start_time: float = field(default_factory=time.time)
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_log(self, level: str, message: str, **kwargs):
        """Add a log entry to this trace context."""
        self.logs.append({
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            **kwargs
        })
    
    def add_tag(self, key: str, value: Any):
        """Add a tag to this trace context."""
        self.tags[key] = value
    
    def get_duration(self) -> float:
        """Get the duration of this trace context."""
        return time.time() - self.start_time


class PerformanceMetric(BaseModel):
    """Performance metric for monitoring."""
    
    operation: str = Field(description="Operation name")
    duration_ms: float = Field(description="Duration in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool = Field(default=True, description="Whether operation succeeded")
    error_type: Optional[str] = Field(default=None, description="Error type if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ObservabilityLogger:
    """
    Enhanced logger with observability features including correlation tracking,
    distributed tracing, and comprehensive performance monitoring.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(name)
        self.structured_logger = structlog.get_logger(name)
        self._performance_metrics: List[PerformanceMetric] = []
        self._alert_thresholds = {
            "error_rate": 0.05,  # 5% error rate threshold
            "response_time_p95": 5000,  # 5 second P95 threshold
            "throughput_min": 10  # 10 requests/minute minimum
        }
    
    def _get_correlation_context(self) -> Dict[str, Any]:
        """Get current correlation context."""
        return {
            "correlation_id": correlation_id_var.get(),
            "trace_id": trace_id_var.get(),
            "span_id": span_id_var.get(),
            "user_id": user_id_var.get(),
            "request_id": request_id_var.get()
        }
    
    def _enhance_log_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance log data with correlation and system context."""
        enhanced = {
            **data,
            **self._get_correlation_context(),
            "logger": self.name,
            "timestamp": datetime.utcnow().isoformat(),
            "hostname": "localhost",  # Could be dynamic
            "pid": "autom8"  # Process identifier
        }
        
        # Remove None values
        return {k: v for k, v in enhanced.items() if v is not None}
    
    def trace(self, message: str, **kwargs):
        """Log at TRACE level."""
        data = self._enhance_log_data({"message": message, "level": "TRACE", **kwargs})
        self.structured_logger.debug(message, **data)
    
    def debug(self, message: str, **kwargs):
        """Log at DEBUG level."""
        data = self._enhance_log_data({"message": message, "level": "DEBUG", **kwargs})
        self.structured_logger.debug(message, **data)
    
    def info(self, message: str, **kwargs):
        """Log at INFO level."""
        data = self._enhance_log_data({"message": message, "level": "INFO", **kwargs})
        self.structured_logger.info(message, **data)
    
    def warning(self, message: str, **kwargs):
        """Log at WARNING level."""
        data = self._enhance_log_data({"message": message, "level": "WARNING", **kwargs})
        self.structured_logger.warning(message, **data)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log at ERROR level."""
        data = self._enhance_log_data({
            "message": message, 
            "level": "ERROR",
            "error_type": type(error).__name__ if error else None,
            "error_message": str(error) if error else None,
            **kwargs
        })
        self.structured_logger.error(message, **data, exc_info=error)
    
    def critical(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log at CRITICAL level."""
        data = self._enhance_log_data({
            "message": message,
            "level": "CRITICAL", 
            "error_type": type(error).__name__ if error else None,
            "error_message": str(error) if error else None,
            **kwargs
        })
        self.structured_logger.critical(message, **data, exc_info=error)
    
    def security(self, message: str, event_type: str = "security", **kwargs):
        """Log security events."""
        data = self._enhance_log_data({
            "message": message,
            "level": "SECURITY",
            "category": LogCategory.SECURITY.value,
            "event_type": event_type,
            **kwargs
        })
        self.structured_logger.warning(message, **data)
    
    def audit(self, message: str, action: str, resource: str = "", **kwargs):
        """Log audit events."""
        data = self._enhance_log_data({
            "message": message,
            "level": "AUDIT",
            "category": LogCategory.AUDIT.value,
            "action": action,
            "resource": resource,
            **kwargs
        })
        self.structured_logger.info(message, **data)
    
    def business(self, message: str, event_type: str = "business_event", **kwargs):
        """Log business events."""
        data = self._enhance_log_data({
            "message": message,
            "level": "BUSINESS",
            "category": LogCategory.BUSINESS.value,
            "event_type": event_type,
            **kwargs
        })
        self.structured_logger.info(message, **data)
    
    def performance(self, operation: str, duration_ms: float, success: bool = True, **kwargs):
        """Log performance metrics."""
        metric = PerformanceMetric(
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            metadata=kwargs
        )
        
        self._performance_metrics.append(metric)
        
        data = self._enhance_log_data({
            "message": f"Performance: {operation}",
            "level": "PERFORMANCE",
            "category": LogCategory.PERFORMANCE.value,
            "operation": operation,
            "duration_ms": duration_ms,
            "success": success,
            **kwargs
        })
        self.structured_logger.info(f"Performance: {operation}", **data)
        
        # Check for performance alerts
        self._check_performance_alerts(metric)
    
    def _check_performance_alerts(self, metric: PerformanceMetric):
        """Check if performance metric triggers any alerts."""
        if metric.duration_ms > self._alert_thresholds["response_time_p95"]:
            self.warning(
                f"High response time detected",
                operation=metric.operation,
                duration_ms=metric.duration_ms,
                threshold=self._alert_thresholds["response_time_p95"],
                alert_type="performance_degradation"
            )
    
    def get_performance_metrics(self, last_n: int = 100) -> List[PerformanceMetric]:
        """Get recent performance metrics."""
        return self._performance_metrics[-last_n:]
    
    def clear_performance_metrics(self):
        """Clear stored performance metrics."""
        self._performance_metrics.clear()


@contextmanager
def trace_context(
    operation: str,
    component: str = "",
    user_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    **kwargs
):
    """Context manager for adding trace context to logs."""
    
    # Generate IDs if not provided
    correlation_id = correlation_id or str(uuid.uuid4())
    trace_id = str(uuid.uuid4())
    span_id = str(uuid.uuid4())
    
    # Set context variables
    correlation_token = correlation_id_var.set(correlation_id)
    trace_token = trace_id_var.set(trace_id)
    span_token = span_id_var.set(span_id)
    
    if user_id:
        user_token = user_id_var.set(user_id)
    
    start_time = time.time()
    
    try:
        logger = get_observability_logger(component or "autom8")
        logger.trace(
            f"Starting operation: {operation}",
            operation=operation,
            component=component,
            **kwargs
        )
        
        yield TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            operation_name=operation,
            start_time=start_time,
            tags=kwargs
        )
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger = get_observability_logger(component or "autom8")
        logger.error(
            f"Operation failed: {operation}",
            operation=operation,
            component=component,
            duration_ms=duration_ms,
            error=e,
            **kwargs
        )
        raise
    
    finally:
        duration_ms = (time.time() - start_time) * 1000
        logger = get_observability_logger(component or "autom8")
        logger.performance(
            operation=operation,
            duration_ms=duration_ms,
            component=component,
            **kwargs
        )
        
        # Reset context variables
        correlation_id_var.reset(correlation_token)
        trace_id_var.reset(trace_token) 
        span_id_var.reset(span_token)
        
        if user_id:
            user_id_var.reset(user_token)


# Global logger instances
_loggers: Dict[str, ObservabilityLogger] = {}


def get_observability_logger(name: str) -> ObservabilityLogger:
    """Get or create an observability logger instance."""
    if name not in _loggers:
        _loggers[name] = ObservabilityLogger(name)
    return _loggers[name]