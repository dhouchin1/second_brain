"""
Comprehensive Monitoring and Observability System for Autom8

Provides enterprise-grade monitoring capabilities including:
- Structured logging with correlation tracking
- Real-time metrics collection and aggregation
- Distributed tracing across components
- Performance monitoring and SLI/SLO tracking
- Alerting and anomaly detection
"""

from .logger import (
    ObservabilityLogger,
    CorrelationContext,
    TraceContext,
    get_observability_logger,
    trace_context
)

from .metrics import (
    MetricsCollector,
    MetricType,
    SLOMonitor,
    SLOTarget,
    get_metrics_collector
)

from .tracing import (
    TracingManager,
    SpanContext,
    SpanType,
    get_tracing_manager,
    traced_operation
)

from .dashboard import (
    MonitoringDashboard,
    get_monitoring_dashboard
)

__all__ = [
    "ObservabilityLogger",
    "CorrelationContext", 
    "TraceContext",
    "get_observability_logger",
    "trace_context",
    "MetricsCollector",
    "MetricType",
    "SLOMonitor",
    "SLOTarget",
    "get_metrics_collector",
    "TracingManager",
    "SpanContext",
    "SpanType", 
    "get_tracing_manager",
    "traced_operation",
    "MonitoringDashboard",
    "get_monitoring_dashboard"
]