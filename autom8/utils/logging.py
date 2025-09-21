"""
Logging Configuration

Structured logging setup with performance monitoring,
context inclusion, and configurable output formats.
"""

import json
import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import structlog
from structlog.types import EventDict, Processor

from autom8.config.settings import get_settings


class ContextualFilter(logging.Filter):
    """
    Filter that adds contextual information to log records
    """
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add context to the record
        for key, value in self.context.items():
            setattr(record, key, value)
        
        # Add performance info if available
        if hasattr(record, 'duration'):
            record.performance = True
        
        return True


class PerformanceLogger:
    """
    Context manager for performance logging
    """
    
    def __init__(
        self,
        logger: Union[logging.Logger, structlog.BoundLogger],
        operation: str,
        threshold: float = 1.0,
        context: Optional[Dict[str, Any]] = None
    ):
        self.logger = logger
        self.operation = operation
        self.threshold = threshold
        self.context = context or {}
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            
            log_data = {
                "operation": self.operation,
                "duration": duration,
                "slow": duration > self.threshold,
                **self.context
            }
            
            if exc_type is not None:
                log_data["error"] = str(exc_val)
                log_data["error_type"] = exc_type.__name__
                self.logger.error("Operation failed", **log_data)
            elif duration > self.threshold:
                self.logger.warning("Slow operation detected", **log_data)
            else:
                self.logger.debug("Operation completed", **log_data)


def add_context_processor(logger: logging.Logger, event_dict: EventDict) -> EventDict:
    """
    Processor that adds context to structured logs
    """
    # Add standard context
    event_dict["service"] = "autom8"
    event_dict["version"] = "3.0.0"
    
    # Add logger name
    if logger:
        event_dict["logger"] = logger.name
    
    return event_dict


def add_timestamp_processor(logger: logging.Logger, event_dict: EventDict) -> EventDict:
    """
    Processor that adds ISO timestamp
    """
    import datetime
    event_dict["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    return event_dict


def filter_sensitive_data(logger: logging.Logger, event_dict: EventDict) -> EventDict:
    """
    Processor that filters sensitive data from logs
    """
    sensitive_keys = {
        "api_key", "password", "token", "secret", "auth", "credential",
        "anthropic_api_key", "openai_api_key"
    }
    
    def _filter_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        filtered = {}
        for key, value in d.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                filtered[key] = "***REDACTED***"
            elif isinstance(value, dict):
                filtered[key] = _filter_dict(value)
            elif isinstance(value, str) and len(value) > 20 and any(sensitive in key.lower() for sensitive in ["key", "token"]):
                # Potential API key or token
                filtered[key] = f"{value[:4]}...{value[-4:]}"
            else:
                filtered[key] = value
        return filtered
    
    return _filter_dict(event_dict)


def setup_logging(
    level: str = "INFO",
    structured: bool = True,
    include_context: bool = True,
    file_path: Optional[str] = None,
    max_file_size: str = "10MB",
    backup_count: int = 5,
    log_performance: bool = True,
) -> logging.Logger:
    """
    Set up comprehensive logging for Autom8
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        structured: Use structured logging (JSON format)
        include_context: Include contextual information
        file_path: Path to log file (None for console only)
        max_file_size: Maximum size before rotation
        backup_count: Number of backup files to keep
        log_performance: Enable performance logging
        
    Returns:
        Configured logger instance
    """
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers = []
    
    # Set log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(log_level)
    
    if structured:
        # Configure structlog
        processors: list[Processor] = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            add_context_processor,
            add_timestamp_processor,
            filter_sensitive_data,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]
        
        if include_context:
            processors.append(structlog.contextvars.merge_contextvars)
        
        # Configure output format
        if sys.stderr.isatty():
            # Pretty console output for development
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
        else:
            # JSON output for production
            processors.append(structlog.processors.JSONRenderer())
        
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            context_class=dict,
            cache_logger_on_first_use=True,
        )
        
        # Set up stdlib integration
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
        )
    else:
        # Standard logging format
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Add contextual filter if enabled
    if include_context:
        console_handler.addFilter(ContextualFilter())
    
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if file_path:
        # Ensure log directory exists
        log_path = Path(file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Parse max file size
        size_bytes = _parse_size(max_file_size)
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=file_path,
            maxBytes=size_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        
        if include_context:
            file_handler.addFilter(ContextualFilter())
        
        root_logger.addHandler(file_handler)
    
    # Create main application logger
    logger = logging.getLogger("autom8")
    
    # Add performance logging capability
    if log_performance:
        logger.performance = lambda operation, threshold=1.0, **context: PerformanceLogger(
            logger, operation, threshold, context
        )
    
    return logger


def get_logger(name: str = "autom8") -> logging.Logger:
    """
    Get a logger instance with the specified name
    
    Args:
        name: Logger name (defaults to 'autom8')
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def get_structured_logger(name: str = "autom8") -> structlog.BoundLogger:
    """
    Get a structured logger instance
    
    Args:
        name: Logger name (defaults to 'autom8')
        
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


def _parse_size(size_str: str) -> int:
    """
    Parse size string (e.g., '10MB') to bytes
    
    Args:
        size_str: Size string like '10MB', '1GB', etc.
        
    Returns:
        Size in bytes
    """
    size_str = size_str.upper().strip()
    
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        # Assume bytes
        return int(size_str)


def configure_from_settings():
    """
    Configure logging from application settings
    """
    settings = get_settings()
    
    setup_logging(
        level=settings.log_level,
        structured=settings.log_structured,
        include_context=True,
        file_path=settings.log_file_path if settings.log_file_enabled else None,
        log_performance=settings.log_performance,
    )


# Performance monitoring decorators and context managers
class LoggedOperation:
    """
    Decorator for automatic operation logging
    """
    
    def __init__(
        self,
        operation_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        threshold: float = 1.0,
        log_args: bool = False,
        log_result: bool = False
    ):
        self.operation_name = operation_name
        self.logger = logger or get_logger()
        self.threshold = threshold
        self.log_args = log_args
        self.log_result = log_result
    
    def __call__(self, func):
        operation_name = self.operation_name or f"{func.__module__}.{func.__name__}"
        
        def wrapper(*args, **kwargs):
            context = {"function": func.__name__}
            
            if self.log_args:
                context["args"] = str(args)[:100]  # Truncate long args
                context["kwargs"] = {k: str(v)[:100] for k, v in kwargs.items()}
            
            with PerformanceLogger(self.logger, operation_name, self.threshold, context):
                result = func(*args, **kwargs)
                
                if self.log_result and result is not None:
                    context["result_type"] = type(result).__name__
                    if hasattr(result, '__len__'):
                        context["result_length"] = len(result)
                
                return result
        
        return wrapper


# Convenience function for common logging patterns
def log_context(**context):
    """
    Add context to structured logs
    
    Usage:
        with log_context(user_id="123", operation="data_processing"):
            # All logs in this block will include the context
            logger.info("Processing started")
    """
    return structlog.contextvars.bound_contextvars(**context)


# Initialize logging on module import
def init():
    """Initialize logging with default settings"""
    try:
        configure_from_settings()
    except Exception:
        # Fallback to basic logging if settings unavailable
        setup_logging()