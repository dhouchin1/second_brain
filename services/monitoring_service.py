"""
Comprehensive monitoring service for Second Brain application.
Provides application-level monitoring, health checks, metrics collection,
and integration with production monitoring systems.
"""

import asyncio
import json
import logging
import psutil
import sqlite3
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from functools import wraps

import httpx
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from config import settings


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Individual metric data point."""
    timestamp: float
    value: float
    labels: Optional[Dict[str, str]] = None


@dataclass
class HealthCheck:
    """Health check result."""
    service: str
    status: str  # healthy, degraded, unhealthy
    response_time_ms: float
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_used_gb: float
    disk_free_gb: float
    load_average: List[float]
    active_connections: int


@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    timestamp: float
    active_users: int
    total_requests: int
    request_rate_per_minute: float
    error_rate_percent: float
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    notes_created_today: int
    searches_performed_today: int
    audio_queue_length: int
    failed_authentications_today: int


class MetricsCollector:
    """Collects and stores application metrics."""
    
    def __init__(self, max_points: int = 1000):
        self.max_points = max_points
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.request_times = deque(maxlen=max_points)
        self.error_counts = defaultdict(int)
        self.endpoint_stats = defaultdict(lambda: {"count": 0, "total_time": 0.0, "errors": 0})
        self.security_events = deque(maxlen=100)
        self.daily_counters = defaultdict(int)
        self.last_daily_reset = datetime.now().date()
    
    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric point."""
        point = MetricPoint(timestamp=time.time(), value=value, labels=labels)
        self.metrics[name].append(point)
    
    def record_request(self, endpoint: str, method: str, status_code: int, response_time: float):
        """Record request metrics."""
        self.request_times.append(response_time)
        self.endpoint_stats[f"{method} {endpoint}"]["count"] += 1
        self.endpoint_stats[f"{method} {endpoint}"]["total_time"] += response_time
        
        if status_code >= 400:
            self.endpoint_stats[f"{method} {endpoint}"]["errors"] += 1
            self.error_counts[f"{status_code}"] += 1
        
        self.record_metric("request_duration_ms", response_time * 1000, {
            "endpoint": endpoint,
            "method": method,
            "status": str(status_code)
        })
    
    def record_security_event(self, event_type: str, details: Dict[str, Any]):
        """Record security-related events."""
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "details": details
        }
        self.security_events.append(event)
        
        # Daily counter for security events
        self._ensure_daily_reset()
        if event_type == "failed_authentication":
            self.daily_counters["failed_authentications"] += 1
    
    def increment_daily_counter(self, counter_name: str, amount: int = 1):
        """Increment daily counters."""
        self._ensure_daily_reset()
        self.daily_counters[counter_name] += amount
    
    def _ensure_daily_reset(self):
        """Reset daily counters if new day."""
        today = datetime.now().date()
        if today > self.last_daily_reset:
            self.daily_counters.clear()
            self.last_daily_reset = today
    
    def get_current_metrics(self) -> ApplicationMetrics:
        """Get current application metrics snapshot."""
        now = time.time()
        
        # Calculate request rate (last minute)
        recent_requests = [t for t in self.request_times if now - t < 60]
        request_rate = len(recent_requests)
        
        # Calculate error rate
        total_requests = sum(stats["count"] for stats in self.endpoint_stats.values())
        total_errors = sum(stats["errors"] for stats in self.endpoint_stats.values())
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
        
        # Calculate response time percentiles
        times = list(self.request_times)
        if times:
            times.sort()
            avg_time = sum(times) / len(times) * 1000  # Convert to ms
            p95_time = times[int(len(times) * 0.95)] * 1000 if times else 0
            p99_time = times[int(len(times) * 0.99)] * 1000 if times else 0
        else:
            avg_time = p95_time = p99_time = 0
        
        return ApplicationMetrics(
            timestamp=now,
            active_users=self._get_active_users(),
            total_requests=total_requests,
            request_rate_per_minute=request_rate,
            error_rate_percent=error_rate,
            avg_response_time_ms=avg_time,
            p95_response_time_ms=p95_time,
            p99_response_time_ms=p99_time,
            notes_created_today=self.daily_counters.get("notes_created", 0),
            searches_performed_today=self.daily_counters.get("searches_performed", 0),
            audio_queue_length=self._get_audio_queue_length(),
            failed_authentications_today=self.daily_counters.get("failed_authentications", 0)
        )
    
    def _get_active_users(self) -> int:
        """Get count of active users (simplified - based on recent requests)."""
        # In a real implementation, this would track authenticated sessions
        # For now, estimate based on recent request activity
        return min(len(self.request_times), 10)  # Cap at 10 for demo
    
    def _get_audio_queue_length(self) -> int:
        """Get current audio processing queue length."""
        try:
            from services.audio_queue import audio_queue
            return len(audio_queue.queue)
        except:
            return 0


class HealthChecker:
    """Performs health checks on application dependencies."""
    
    def __init__(self):
        self.timeout = 5.0  # seconds
    
    async def check_database(self) -> HealthCheck:
        """Check database connectivity and status."""
        start_time = time.time()
        try:
            with sqlite3.connect(settings.db_path, timeout=self.timeout) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                table_count = cursor.fetchone()[0]
                
                # Check if main tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('notes', 'users')")
                essential_tables = cursor.fetchall()
                
                response_time = (time.time() - start_time) * 1000
                
                if len(essential_tables) >= 2:
                    return HealthCheck(
                        service="database",
                        status="healthy",
                        response_time_ms=response_time,
                        details={"table_count": table_count, "essential_tables": len(essential_tables)}
                    )
                else:
                    return HealthCheck(
                        service="database",
                        status="degraded",
                        response_time_ms=response_time,
                        error="Missing essential tables"
                    )
                    
        except Exception as e:
            return HealthCheck(
                service="database",
                status="unhealthy",
                response_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    async def check_ollama(self) -> HealthCheck:
        """Check Ollama service availability."""
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Use Ollama's API endpoint
                api_url = settings.ollama_api_url.replace("/api/generate", "/api/tags")
                response = await client.get(api_url)
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])
                    return HealthCheck(
                        service="ollama",
                        status="healthy",
                        response_time_ms=response_time,
                        details={"model_count": len(models)}
                    )
                else:
                    return HealthCheck(
                        service="ollama",
                        status="degraded",
                        response_time_ms=response_time,
                        error=f"HTTP {response.status_code}"
                    )
                    
        except Exception as e:
            return HealthCheck(
                service="ollama",
                status="unhealthy",
                response_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    async def check_file_system(self) -> HealthCheck:
        """Check file system health and accessibility."""
        start_time = time.time()
        try:
            # Check essential directories
            essential_paths = [
                settings.vault_path,
                settings.audio_dir,
                settings.uploads_dir
            ]
            
            issues = []
            for path in essential_paths:
                if not Path(path).exists():
                    issues.append(f"{path} does not exist")
                elif not os.access(path, os.R_OK | os.W_OK):
                    issues.append(f"{path} is not readable/writable")
            
            response_time = (time.time() - start_time) * 1000
            
            if not issues:
                return HealthCheck(
                    service="filesystem",
                    status="healthy",
                    response_time_ms=response_time
                )
            else:
                return HealthCheck(
                    service="filesystem",
                    status="degraded" if len(issues) < len(essential_paths) else "unhealthy",
                    response_time_ms=response_time,
                    error="; ".join(issues)
                )
                
        except Exception as e:
            return HealthCheck(
                service="filesystem",
                status="unhealthy",
                response_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    async def check_whisper(self) -> HealthCheck:
        """Check Whisper transcription service availability."""
        start_time = time.time()
        try:
            whisper_path = settings.whisper_cpp_path
            model_path = settings.whisper_model_path
            
            issues = []
            if not Path(whisper_path).exists():
                issues.append("Whisper binary not found")
            elif not os.access(whisper_path, os.X_OK):
                issues.append("Whisper binary not executable")
            
            if not Path(model_path).exists():
                issues.append("Whisper model not found")
            
            response_time = (time.time() - start_time) * 1000
            
            if not issues:
                return HealthCheck(
                    service="whisper",
                    status="healthy",
                    response_time_ms=response_time
                )
            else:
                return HealthCheck(
                    service="whisper",
                    status="degraded" if settings.transcriber != "whisper" else "unhealthy",
                    response_time_ms=response_time,
                    error="; ".join(issues)
                )
                
        except Exception as e:
            return HealthCheck(
                service="whisper",
                status="unhealthy",
                response_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )


class SystemMonitor:
    """Monitors system resources and performance."""
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system resource metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_used_gb = disk.used / (1024 * 1024 * 1024)
        disk_free_gb = disk.free / (1024 * 1024 * 1024)
        
        # Load average (Unix-like systems)
        try:
            load_avg = list(psutil.getloadavg())
        except AttributeError:
            # Windows doesn't have load average
            load_avg = [0.0, 0.0, 0.0]
        
        # Network connections
        try:
            connections = len(psutil.net_connections())
        except (psutil.PermissionError, psutil.AccessDenied):
            connections = 0
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_usage_percent=disk.percent,
            disk_used_gb=disk_used_gb,
            disk_free_gb=disk_free_gb,
            load_average=load_avg,
            active_connections=connections
        )


class MonitoringService:
    """Main monitoring service orchestrating all monitoring components."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker()
        self.system_monitor = SystemMonitor()
        self.start_time = time.time()
        self.alert_thresholds = {
            "error_rate_percent": 5.0,
            "response_time_p95_ms": 2000.0,
            "memory_percent": 80.0,
            "disk_usage_percent": 90.0,
            "cpu_percent": 90.0,
            "failed_auth_per_minute": 10
        }
        self.alerts_triggered = defaultdict(bool)
        
        logger.info("MonitoringService initialized")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        health_checks = await asyncio.gather(
            self.health_checker.check_database(),
            self.health_checker.check_ollama(),
            self.health_checker.check_file_system(),
            self.health_checker.check_whisper(),
            return_exceptions=True
        )
        
        # Filter out any exceptions and convert to dict
        healthy_checks = []
        for check in health_checks:
            if isinstance(check, HealthCheck):
                healthy_checks.append(asdict(check))
            else:
                # Log exception but continue
                logger.error(f"Health check failed: {check}")
        
        # Determine overall status
        statuses = [check.get("status", "unhealthy") for check in healthy_checks]
        if "unhealthy" in statuses:
            overall_status = "unhealthy"
        elif "degraded" in statuses:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self.start_time,
            "checks": healthy_checks
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        app_metrics = self.metrics_collector.get_current_metrics()
        system_metrics = self.system_monitor.get_system_metrics()
        
        return {
            "timestamp": time.time(),
            "application": asdict(app_metrics),
            "system": asdict(system_metrics),
            "uptime_seconds": time.time() - self.start_time
        }
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        app_metrics = self.metrics_collector.get_current_metrics()
        system_metrics = self.system_monitor.get_system_metrics()
        
        prometheus_lines = []
        
        # Application metrics
        prometheus_lines.extend([
            f"# HELP second_brain_requests_total Total number of requests",
            f"# TYPE second_brain_requests_total counter",
            f"second_brain_requests_total {app_metrics.total_requests}",
            f"",
            f"# HELP second_brain_request_duration_ms Request duration in milliseconds",
            f"# TYPE second_brain_request_duration_ms histogram",
            f"second_brain_request_duration_ms_mean {app_metrics.avg_response_time_ms}",
            f"second_brain_request_duration_ms_p95 {app_metrics.p95_response_time_ms}",
            f"second_brain_request_duration_ms_p99 {app_metrics.p99_response_time_ms}",
            f"",
            f"# HELP second_brain_error_rate_percent Current error rate percentage",
            f"# TYPE second_brain_error_rate_percent gauge",
            f"second_brain_error_rate_percent {app_metrics.error_rate_percent}",
            f"",
            f"# HELP second_brain_active_users Current active users",
            f"# TYPE second_brain_active_users gauge",
            f"second_brain_active_users {app_metrics.active_users}",
            f"",
            f"# HELP second_brain_notes_created_today Notes created today",
            f"# TYPE second_brain_notes_created_today gauge",
            f"second_brain_notes_created_today {app_metrics.notes_created_today}",
            f"",
        ])
        
        # System metrics
        prometheus_lines.extend([
            f"# HELP second_brain_cpu_percent CPU usage percentage",
            f"# TYPE second_brain_cpu_percent gauge",
            f"second_brain_cpu_percent {system_metrics.cpu_percent}",
            f"",
            f"# HELP second_brain_memory_percent Memory usage percentage",
            f"# TYPE second_brain_memory_percent gauge", 
            f"second_brain_memory_percent {system_metrics.memory_percent}",
            f"",
            f"# HELP second_brain_disk_percent Disk usage percentage",
            f"# TYPE second_brain_disk_percent gauge",
            f"second_brain_disk_percent {system_metrics.disk_usage_percent}",
            f"",
            f"# HELP second_brain_uptime_seconds Application uptime in seconds",
            f"# TYPE second_brain_uptime_seconds gauge",
            f"second_brain_uptime_seconds {time.time() - self.start_time}",
        ])
        
        return "\n".join(prometheus_lines)
    
    def check_alert_conditions(self) -> List[Dict[str, Any]]:
        """Check for alert conditions and return triggered alerts."""
        alerts = []
        app_metrics = self.metrics_collector.get_current_metrics()
        system_metrics = self.system_monitor.get_system_metrics()
        
        # Error rate alert
        if app_metrics.error_rate_percent > self.alert_thresholds["error_rate_percent"]:
            if not self.alerts_triggered["high_error_rate"]:
                alerts.append({
                    "type": "high_error_rate",
                    "severity": "warning",
                    "message": f"Error rate is {app_metrics.error_rate_percent:.1f}%",
                    "threshold": self.alert_thresholds["error_rate_percent"],
                    "current_value": app_metrics.error_rate_percent
                })
                self.alerts_triggered["high_error_rate"] = True
        else:
            self.alerts_triggered["high_error_rate"] = False
        
        # Response time alert
        if app_metrics.p95_response_time_ms > self.alert_thresholds["response_time_p95_ms"]:
            if not self.alerts_triggered["slow_response"]:
                alerts.append({
                    "type": "slow_response",
                    "severity": "warning",
                    "message": f"P95 response time is {app_metrics.p95_response_time_ms:.1f}ms",
                    "threshold": self.alert_thresholds["response_time_p95_ms"],
                    "current_value": app_metrics.p95_response_time_ms
                })
                self.alerts_triggered["slow_response"] = True
        else:
            self.alerts_triggered["slow_response"] = False
        
        # Memory usage alert
        if system_metrics.memory_percent > self.alert_thresholds["memory_percent"]:
            if not self.alerts_triggered["high_memory"]:
                alerts.append({
                    "type": "high_memory",
                    "severity": "warning",
                    "message": f"Memory usage is {system_metrics.memory_percent:.1f}%",
                    "threshold": self.alert_thresholds["memory_percent"],
                    "current_value": system_metrics.memory_percent
                })
                self.alerts_triggered["high_memory"] = True
        else:
            self.alerts_triggered["high_memory"] = False
        
        # Disk usage alert
        if system_metrics.disk_usage_percent > self.alert_thresholds["disk_usage_percent"]:
            if not self.alerts_triggered["high_disk"]:
                alerts.append({
                    "type": "high_disk",
                    "severity": "critical",
                    "message": f"Disk usage is {system_metrics.disk_usage_percent:.1f}%",
                    "threshold": self.alert_thresholds["disk_usage_percent"],
                    "current_value": system_metrics.disk_usage_percent
                })
                self.alerts_triggered["high_disk"] = True
        else:
            self.alerts_triggered["high_disk"] = False
        
        return alerts


class MonitoringMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware to collect request/response metrics."""
    
    def __init__(self, app, monitoring_service: MonitoringService):
        super().__init__(app)
        self.monitoring_service = monitoring_service
    
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()
        
        # Extract request info
        method = request.method
        path = request.url.path
        
        # Skip monitoring for monitoring endpoints to avoid recursion
        if path.startswith('/health') or path.startswith('/metrics') or path.startswith('/ready'):
            return await call_next(request)
        
        response = await call_next(request)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Record metrics
        self.monitoring_service.metrics_collector.record_request(
            endpoint=path,
            method=method,
            status_code=response.status_code,
            response_time=response_time
        )
        
        # Add response time header for debugging
        response.headers["X-Response-Time"] = f"{response_time:.3f}s"
        
        return response


# Global monitoring service instance
monitoring_service = MonitoringService()


def get_monitoring_service() -> MonitoringService:
    """Dependency to get monitoring service instance."""
    return monitoring_service


# Decorator for monitoring function execution time
def monitor_execution_time(metric_name: str):
    """Decorator to monitor function execution time."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                monitoring_service.metrics_collector.record_metric(
                    f"{metric_name}_duration_seconds", execution_time
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                monitoring_service.metrics_collector.record_metric(
                    f"{metric_name}_duration_seconds", execution_time, {"status": "error"}
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                monitoring_service.metrics_collector.record_metric(
                    f"{metric_name}_duration_seconds", execution_time
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                monitoring_service.metrics_collector.record_metric(
                    f"{metric_name}_duration_seconds", execution_time, {"status": "error"}
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# Import os module (was missing)
import os