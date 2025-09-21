"""
Comprehensive Metrics Collection and SLI/SLO Monitoring

Provides enterprise-grade metrics collection, aggregation, and monitoring
with support for business metrics, technical metrics, and SLO tracking.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

from autom8.monitoring.logger import get_observability_logger

logger = get_observability_logger(__name__)


class MetricType(str, Enum):
    """Types of metrics supported."""
    COUNTER = "counter"          # Monotonically increasing values
    GAUGE = "gauge"              # Point-in-time values
    HISTOGRAM = "histogram"      # Distribution of values
    SUMMARY = "summary"          # Summary statistics
    TIMER = "timer"              # Timing measurements


class SLOStatus(str, Enum):
    """SLO compliance status."""
    HEALTHY = "healthy"          # Meeting SLO targets
    WARNING = "warning"          # Approaching SLO violation
    VIOLATED = "violated"        # SLO violation detected
    UNKNOWN = "unknown"          # Insufficient data


@dataclass
class MetricValue:
    """Individual metric measurement."""
    
    value: Union[int, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series of metric values."""
    
    name: str
    metric_type: MetricType
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    labels: Dict[str, str] = field(default_factory=dict)
    help_text: str = ""
    
    def add_value(self, value: Union[int, float], labels: Optional[Dict[str, str]] = None, **metadata):
        """Add a new value to the time series."""
        metric_labels = {**self.labels}
        if labels:
            metric_labels.update(labels)
        
        self.values.append(MetricValue(
            value=value,
            labels=metric_labels,
            metadata=metadata
        ))
    
    def get_latest(self) -> Optional[MetricValue]:
        """Get the most recent value."""
        return self.values[-1] if self.values else None
    
    def get_values_in_window(self, duration_minutes: int) -> List[MetricValue]:
        """Get values within a time window."""
        cutoff = datetime.utcnow() - timedelta(minutes=duration_minutes)
        return [v for v in self.values if v.timestamp >= cutoff]
    
    def calculate_rate(self, duration_minutes: int = 5) -> float:
        """Calculate rate of change over time window."""
        values = self.get_values_in_window(duration_minutes)
        if len(values) < 2:
            return 0.0
        
        if self.metric_type == MetricType.COUNTER:
            # For counters, calculate rate as change per minute
            start_value = values[0].value
            end_value = values[-1].value
            time_diff = (values[-1].timestamp - values[0].timestamp).total_seconds() / 60.0
            
            if time_diff > 0:
                return (end_value - start_value) / time_diff
        
        return 0.0
    
    def calculate_percentile(self, percentile: float, duration_minutes: int = 5) -> float:
        """Calculate percentile over time window."""
        values = self.get_values_in_window(duration_minutes)
        if not values:
            return 0.0
        
        sorted_values = sorted([v.value for v in values])
        index = int((percentile / 100.0) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        
        return sorted_values[index]
    
    def calculate_average(self, duration_minutes: int = 5) -> float:
        """Calculate average over time window."""
        values = self.get_values_in_window(duration_minutes)
        if not values:
            return 0.0
        
        return sum(v.value for v in values) / len(values)


@dataclass
class SLOTarget:
    """Service Level Objective target definition."""
    
    name: str
    description: str
    target_percentage: float  # e.g., 99.9 for 99.9% availability
    success_metric: str  # Metric name for successful operations
    total_metric: str    # Metric name for total operations
    measurement_window_minutes: int = 60  # Time window for measurement
    alert_threshold_percentage: float = 95.0  # Alert when below this % of target
    
    # Alternative: threshold-based SLO (e.g., response time < 500ms)
    threshold_metric: Optional[str] = None
    threshold_value: Optional[float] = None
    threshold_operator: str = "<"  # "<", "<=", ">", ">="


class SLOMonitor:
    """Monitors Service Level Objectives and alerts on violations."""
    
    def __init__(self):
        self.targets: Dict[str, SLOTarget] = {}
        self.status_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._alert_callbacks: Dict[str, List[Callable]] = defaultdict(list)
    
    def register_slo(self, target: SLOTarget):
        """Register a new SLO target."""
        self.targets[target.name] = target
        logger.info(
            f"Registered SLO target: {target.name}",
            slo_name=target.name,
            target_percentage=target.target_percentage,
            measurement_window_minutes=target.measurement_window_minutes
        )
    
    def add_alert_callback(self, slo_name: str, callback: Callable[[str, SLOStatus, float], None]):
        """Add callback for SLO status changes."""
        self._alert_callbacks[slo_name].append(callback)
    
    def evaluate_slo(self, slo_name: str, metrics_collector: 'MetricsCollector') -> Tuple[SLOStatus, float]:
        """Evaluate SLO compliance and return status and actual percentage."""
        if slo_name not in self.targets:
            logger.warning(f"Unknown SLO target: {slo_name}")
            return SLOStatus.UNKNOWN, 0.0
        
        target = self.targets[slo_name]
        
        try:
            if target.threshold_metric:
                # Threshold-based SLO (e.g., response time)
                actual_percentage = self._evaluate_threshold_slo(target, metrics_collector)
            else:
                # Ratio-based SLO (e.g., success rate)
                actual_percentage = self._evaluate_ratio_slo(target, metrics_collector)
            
            # Determine status
            if actual_percentage >= target.target_percentage:
                status = SLOStatus.HEALTHY
            elif actual_percentage >= target.alert_threshold_percentage:
                status = SLOStatus.WARNING
            else:
                status = SLOStatus.VIOLATED
            
            # Record status history
            self.status_history[slo_name].append({
                "timestamp": datetime.utcnow(),
                "status": status,
                "actual_percentage": actual_percentage,
                "target_percentage": target.target_percentage
            })
            
            # Trigger alerts if needed
            self._check_alerts(slo_name, status, actual_percentage)
            
            return status, actual_percentage
            
        except Exception as e:
            logger.error(f"Error evaluating SLO {slo_name}", error=e)
            return SLOStatus.UNKNOWN, 0.0
    
    def _evaluate_ratio_slo(self, target: SLOTarget, metrics_collector: 'MetricsCollector') -> float:
        """Evaluate ratio-based SLO (success_count / total_count)."""
        success_values = metrics_collector.get_metric_values(
            target.success_metric, target.measurement_window_minutes
        )
        total_values = metrics_collector.get_metric_values(
            target.total_metric, target.measurement_window_minutes
        )
        
        if not total_values or not success_values:
            return 0.0
        
        total_success = sum(v.value for v in success_values)
        total_requests = sum(v.value for v in total_values)
        
        if total_requests == 0:
            return 100.0  # No requests = perfect success rate
        
        return (total_success / total_requests) * 100.0
    
    def _evaluate_threshold_slo(self, target: SLOTarget, metrics_collector: 'MetricsCollector') -> float:
        """Evaluate threshold-based SLO (e.g., response_time < 500ms)."""
        values = metrics_collector.get_metric_values(
            target.threshold_metric, target.measurement_window_minutes
        )
        
        if not values:
            return 0.0
        
        # Count values that meet the threshold
        meeting_threshold = 0
        total_values = len(values)
        
        for value in values:
            if target.threshold_operator == "<":
                if value.value < target.threshold_value:
                    meeting_threshold += 1
            elif target.threshold_operator == "<=":
                if value.value <= target.threshold_value:
                    meeting_threshold += 1
            elif target.threshold_operator == ">":
                if value.value > target.threshold_value:
                    meeting_threshold += 1
            elif target.threshold_operator == ">=":
                if value.value >= target.threshold_value:
                    meeting_threshold += 1
        
        return (meeting_threshold / total_values) * 100.0
    
    def _check_alerts(self, slo_name: str, status: SLOStatus, actual_percentage: float):
        """Check if alerts need to be triggered."""
        if slo_name in self._alert_callbacks:
            for callback in self._alert_callbacks[slo_name]:
                try:
                    callback(slo_name, status, actual_percentage)
                except Exception as e:
                    logger.error(f"Error in SLO alert callback for {slo_name}", error=e)
    
    def get_slo_status_history(self, slo_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get historical SLO status."""
        return list(self.status_history[slo_name])[-limit:]


class MetricsCollector:
    """
    Central metrics collection system supporting various metric types
    and providing aggregation, analysis, and alerting capabilities.
    """
    
    def __init__(self):
        self.metrics: Dict[str, MetricSeries] = {}
        self.slo_monitor = SLOMonitor()
        self._running = False
        self._collection_task: Optional[asyncio.Task] = None
        
        # Built-in metrics for system health
        self._setup_system_metrics()
    
    def _setup_system_metrics(self):
        """Setup built-in system metrics."""
        
        # Business metrics
        self.register_metric("requests_total", MetricType.COUNTER, "Total API requests")
        self.register_metric("requests_successful", MetricType.COUNTER, "Successful API requests")
        self.register_metric("requests_failed", MetricType.COUNTER, "Failed API requests")
        
        # Performance metrics
        self.register_metric("response_time_ms", MetricType.HISTOGRAM, "Response time in milliseconds")
        self.register_metric("model_inference_time_ms", MetricType.HISTOGRAM, "Model inference time")
        self.register_metric("context_processing_time_ms", MetricType.HISTOGRAM, "Context processing time")
        
        # Resource metrics
        self.register_metric("active_connections", MetricType.GAUGE, "Active connections")
        self.register_metric("memory_usage_bytes", MetricType.GAUGE, "Memory usage in bytes")
        self.register_metric("cache_hit_rate", MetricType.GAUGE, "Cache hit rate percentage")
        
        # Model metrics
        self.register_metric("model_requests_total", MetricType.COUNTER, "Total model requests")
        self.register_metric("model_errors_total", MetricType.COUNTER, "Total model errors")
        self.register_metric("model_tokens_processed", MetricType.COUNTER, "Tokens processed by models")
        
        # SLO targets
        self._setup_default_slos()
    
    def _setup_default_slos(self):
        """Setup default SLO targets."""
        
        # API Availability SLO (99.9% success rate)
        self.slo_monitor.register_slo(SLOTarget(
            name="api_availability",
            description="API requests should succeed 99.9% of the time",
            target_percentage=99.9,
            success_metric="requests_successful",
            total_metric="requests_total",
            measurement_window_minutes=60,
            alert_threshold_percentage=99.0
        ))
        
        # Response Time SLO (95% of requests under 2 seconds)
        self.slo_monitor.register_slo(SLOTarget(
            name="response_time",
            description="95% of API requests should complete within 2 seconds",
            target_percentage=95.0,
            success_metric="",  # Not used for threshold-based SLO
            total_metric="",    # Not used for threshold-based SLO
            measurement_window_minutes=15,
            alert_threshold_percentage=90.0,
            threshold_metric="response_time_ms",
            threshold_value=2000.0,
            threshold_operator="<"
        ))
        
        # Model Inference SLO (90% of inferences under 5 seconds)
        self.slo_monitor.register_slo(SLOTarget(
            name="model_inference_time",
            description="90% of model inferences should complete within 5 seconds",
            target_percentage=90.0,
            success_metric="",  # Not used for threshold-based SLO
            total_metric="",    # Not used for threshold-based SLO
            measurement_window_minutes=30,
            alert_threshold_percentage=85.0,
            threshold_metric="model_inference_time_ms",
            threshold_value=5000.0,
            threshold_operator="<"
        ))
    
    def register_metric(self, name: str, metric_type: MetricType, help_text: str = "", labels: Optional[Dict[str, str]] = None):
        """Register a new metric."""
        self.metrics[name] = MetricSeries(
            name=name,
            metric_type=metric_type,
            help_text=help_text,
            labels=labels or {}
        )
        
        logger.debug(
            f"Registered metric: {name}",
            metric_name=name,
            metric_type=metric_type.value,
            help_text=help_text
        )
    
    def increment_counter(self, name: str, value: Union[int, float] = 1, labels: Optional[Dict[str, str]] = None, **metadata):
        """Increment a counter metric."""
        if name not in self.metrics:
            self.register_metric(name, MetricType.COUNTER)
        
        current_value = 0
        latest = self.metrics[name].get_latest()
        if latest:
            current_value = latest.value
        
        self.metrics[name].add_value(current_value + value, labels, **metadata)
    
    def set_gauge(self, name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None, **metadata):
        """Set a gauge metric value."""
        if name not in self.metrics:
            self.register_metric(name, MetricType.GAUGE)
        
        self.metrics[name].add_value(value, labels, **metadata)
    
    def record_histogram(self, name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None, **metadata):
        """Record a value in a histogram metric."""
        if name not in self.metrics:
            self.register_metric(name, MetricType.HISTOGRAM)
        
        self.metrics[name].add_value(value, labels, **metadata)
    
    def start_timer(self, name: str, labels: Optional[Dict[str, str]] = None, **metadata) -> 'TimerContext':
        """Start a timer for measuring durations."""
        return TimerContext(self, name, labels, **metadata)
    
    def get_metric_values(self, name: str, duration_minutes: int = 60) -> List[MetricValue]:
        """Get metric values within a time window."""
        if name not in self.metrics:
            return []
        
        return self.metrics[name].get_values_in_window(duration_minutes)
    
    def get_metric_summary(self, name: str, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        if name not in self.metrics:
            return {}
        
        metric = self.metrics[name]
        values = metric.get_values_in_window(duration_minutes)
        
        if not values:
            return {"name": name, "type": metric.metric_type.value, "values": 0}
        
        numeric_values = [v.value for v in values]
        
        summary = {
            "name": name,
            "type": metric.metric_type.value,
            "count": len(values),
            "latest": values[-1].value,
            "min": min(numeric_values),
            "max": max(numeric_values),
            "average": sum(numeric_values) / len(numeric_values)
        }
        
        if metric.metric_type in [MetricType.HISTOGRAM, MetricType.TIMER]:
            summary.update({
                "p50": metric.calculate_percentile(50, duration_minutes),
                "p95": metric.calculate_percentile(95, duration_minutes),
                "p99": metric.calculate_percentile(99, duration_minutes)
            })
        
        if metric.metric_type == MetricType.COUNTER:
            summary["rate_per_minute"] = metric.calculate_rate(duration_minutes)
        
        return summary
    
    def get_all_metrics_summary(self, duration_minutes: int = 60) -> Dict[str, Dict[str, Any]]:
        """Get summary for all metrics."""
        return {
            name: self.get_metric_summary(name, duration_minutes)
            for name in self.metrics.keys()
        }
    
    def evaluate_all_slos(self) -> Dict[str, Tuple[SLOStatus, float]]:
        """Evaluate all registered SLOs."""
        results = {}
        for slo_name in self.slo_monitor.targets.keys():
            results[slo_name] = self.slo_monitor.evaluate_slo(slo_name, self)
        return results
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health based on metrics and SLOs."""
        slo_results = self.evaluate_all_slos()
        
        healthy_slos = sum(1 for status, _ in slo_results.values() if status == SLOStatus.HEALTHY)
        warning_slos = sum(1 for status, _ in slo_results.values() if status == SLOStatus.WARNING)
        violated_slos = sum(1 for status, _ in slo_results.values() if status == SLOStatus.VIOLATED)
        total_slos = len(slo_results)
        
        overall_status = "healthy"
        if violated_slos > 0:
            overall_status = "unhealthy"
        elif warning_slos > 0:
            overall_status = "degraded"
        
        return {
            "overall_status": overall_status,
            "total_slos": total_slos,
            "healthy_slos": healthy_slos,
            "warning_slos": warning_slos,
            "violated_slos": violated_slos,
            "slo_results": {name: {"status": status.value, "percentage": pct} for name, (status, pct) in slo_results.items()},
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def start_collection(self, interval_seconds: int = 60):
        """Start automatic metrics collection and SLO evaluation."""
        self._running = True
        
        async def collection_loop():
            while self._running:
                try:
                    # Evaluate SLOs
                    slo_results = self.evaluate_all_slos()
                    
                    # Log health status
                    health = self.get_health_status()
                    logger.business(
                        "System health check completed",
                        event_type="health_check",
                        overall_status=health["overall_status"],
                        slo_results=health["slo_results"]
                    )
                    
                    await asyncio.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error("Error in metrics collection loop", error=e)
                    await asyncio.sleep(interval_seconds)
        
        self._collection_task = asyncio.create_task(collection_loop())
        logger.info("Started metrics collection", interval_seconds=interval_seconds)
    
    def stop_collection(self):
        """Stop automatic metrics collection."""
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
        logger.info("Stopped metrics collection")


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, name: str, labels: Optional[Dict[str, str]] = None, **metadata):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.metadata = metadata
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        self.collector.record_histogram(self.name, duration_ms, self.labels, **self.metadata)


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector