"""
Enhanced Error Monitoring and Handling for Second Brain
"""

import logging
import traceback
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from functools import wraps

@dataclass
class ErrorEvent:
    timestamp: datetime
    error_type: str
    message: str
    component: str
    user_id: Optional[int] = None
    traceback: Optional[str] = None
    context: Dict[str, Any] = None
    severity: str = "error"  # debug, info, warning, error, critical
    resolved: bool = False

class ErrorMonitor:
    """Centralized error monitoring and alerting system"""

    def __init__(self, log_file: Optional[Path] = None):
        self.events = deque(maxlen=1000)  # Keep last 1000 errors
        self.error_counts = defaultdict(int)
        self.error_rates = defaultdict(list)
        self.suppressed_errors = set()
        self.log_file = log_file or Path("logs/errors.jsonl")

        # Thresholds for alerting
        self.rate_thresholds = {
            "critical": 1,  # 1 critical error triggers alert
            "error": 5,     # 5 errors in 5 minutes
            "warning": 20   # 20 warnings in 5 minutes
        }

        self.setup_logging()

    def setup_logging(self):
        """Configure structured logging"""
        self.log_file.parent.mkdir(exist_ok=True)

        # Create formatter for structured logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # File handler for error logs
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(formatter)

        # Console handler for development
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # Configure root logger
        logger = logging.getLogger("second_brain")
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        self.logger = logger

    def capture_error(self,
                     error: Exception,
                     component: str,
                     user_id: Optional[int] = None,
                     context: Optional[Dict[str, Any]] = None,
                     severity: str = "error"):
        """Capture and log an error event"""

        event = ErrorEvent(
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            message=str(error),
            component=component,
            user_id=user_id,
            traceback=traceback.format_exc(),
            context=context or {},
            severity=severity
        )

        self.events.append(event)
        self.error_counts[f"{component}:{event.error_type}"] += 1

        # Track error rates
        rate_key = f"{severity}:{component}"
        now = datetime.now()
        self.error_rates[rate_key].append(now)

        # Clean old rate data (keep last 5 minutes)
        cutoff = now - timedelta(minutes=5)
        self.error_rates[rate_key] = [
            ts for ts in self.error_rates[rate_key] if ts > cutoff
        ]

        # Log the error
        log_data = {
            "error_type": event.error_type,
            "message": event.message,
            "component": component,
            "user_id": user_id,
            "context": context,
            "severity": severity
        }

        if severity == "critical":
            self.logger.critical(json.dumps(log_data))
        elif severity == "error":
            self.logger.error(json.dumps(log_data))
        elif severity == "warning":
            self.logger.warning(json.dumps(log_data))
        else:
            self.logger.info(json.dumps(log_data))

        # Check if we should alert
        self.check_alert_thresholds(rate_key, severity)

        return event

    def check_alert_thresholds(self, rate_key: str, severity: str):
        """Check if error rates exceed thresholds"""
        if severity not in self.rate_thresholds:
            return

        threshold = self.rate_thresholds[severity]
        current_rate = len(self.error_rates[rate_key])

        if current_rate >= threshold:
            self.trigger_alert(rate_key, current_rate, threshold)

    def trigger_alert(self, rate_key: str, current_rate: int, threshold: int):
        """Trigger an alert for high error rates"""
        alert_message = f"High error rate detected: {rate_key} - {current_rate} errors (threshold: {threshold})"
        self.logger.critical(f"ALERT: {alert_message}")

        # Could integrate with external alerting here:
        # - Send to Slack/Discord
        # - Send email alerts
        # - Trigger PagerDuty/Opsgenie
        # - Write to monitoring dashboard

    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_events = [e for e in self.events if e.timestamp > cutoff]

        summary = {
            "total_errors": len(recent_events),
            "by_severity": defaultdict(int),
            "by_component": defaultdict(int),
            "by_error_type": defaultdict(int),
            "top_errors": [],
            "error_rate": len(recent_events) / max(hours, 1),
            "period_hours": hours
        }

        for event in recent_events:
            summary["by_severity"][event.severity] += 1
            summary["by_component"][event.component] += 1
            summary["by_error_type"][event.error_type] += 1

        # Get top 10 most common errors
        error_counts = defaultdict(int)
        for event in recent_events:
            key = f"{event.component}:{event.error_type}"
            error_counts[key] += 1

        summary["top_errors"] = sorted(
            error_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return dict(summary)

    def get_recent_errors(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent error events"""
        recent = list(self.events)[-limit:]
        return [asdict(event) for event in reversed(recent)]

    def suppress_error(self, error_pattern: str):
        """Suppress errors matching a pattern"""
        self.suppressed_errors.add(error_pattern)

    def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        now = datetime.now()
        last_hour = now - timedelta(hours=1)

        recent_critical = len([
            e for e in self.events
            if e.timestamp > last_hour and e.severity == "critical"
        ])

        recent_errors = len([
            e for e in self.events
            if e.timestamp > last_hour and e.severity == "error"
        ])

        status = "healthy"
        if recent_critical > 0:
            status = "critical"
        elif recent_errors > 10:
            status = "unhealthy"
        elif recent_errors > 3:
            status = "degraded"

        return {
            "status": status,
            "critical_errors_last_hour": recent_critical,
            "errors_last_hour": recent_errors,
            "total_events": len(self.events),
            "timestamp": now.isoformat()
        }

# Global error monitor instance
error_monitor = ErrorMonitor()

def monitor_errors(component: str):
    """Decorator to automatically monitor errors in functions"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_monitor.capture_error(e, component, context={
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                })
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_monitor.capture_error(e, component, context={
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                })
                raise

        if 'async' in func.__code__.co_flags or hasattr(func, '__aenter__'):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

@contextmanager
def error_context(component: str, user_id: Optional[int] = None, **context):
    """Context manager for error monitoring"""
    try:
        yield
    except Exception as e:
        error_monitor.capture_error(
            e,
            component,
            user_id=user_id,
            context=context
        )
        raise

# Example usage patterns:
#
# @monitor_errors("database")
# async def get_notes(user_id: int):
#     # Function implementation
#     pass
#
# with error_context("api", user_id=123, endpoint="/api/notes"):
#     # Code that might fail
#     pass