"""
Distributed Tracing System

Provides comprehensive distributed tracing capabilities for tracking
requests across components, measuring performance, and debugging issues.
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from autom8.monitoring.logger import get_observability_logger

logger = get_observability_logger(__name__)


class SpanType(str, Enum):
    """Types of spans in distributed tracing."""
    ROOT = "root"              # Root span for a request
    CHILD = "child"            # Child span within a request
    RPC = "rpc"                # Remote procedure call
    DATABASE = "database"      # Database operation
    CACHE = "cache"            # Cache operation
    MODEL = "model"            # Model inference
    CONTEXT = "context"        # Context processing
    ROUTING = "routing"        # Model routing decision


class SpanStatus(str, Enum):
    """Status of span execution."""
    OK = "ok"                  # Successful completion
    ERROR = "error"            # Error occurred
    TIMEOUT = "timeout"        # Operation timed out
    CANCELLED = "cancelled"    # Operation was cancelled


@dataclass
class SpanEvent:
    """Event within a span (log entry, annotation, etc.)."""
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    name: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "name": self.name,
            "attributes": self.attributes
        }


@dataclass
class SpanContext:
    """Context for distributed tracing span."""
    
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    component: str = ""
    span_type: SpanType = SpanType.CHILD
    
    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # Status
    status: SpanStatus = SpanStatus.OK
    error_message: Optional[str] = None
    
    # Metadata
    tags: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    references: List[str] = field(default_factory=list)  # References to other spans
    
    def add_tag(self, key: str, value: Any):
        """Add a tag to this span."""
        self.tags[key] = value
    
    def add_event(self, name: str, **attributes):
        """Add an event to this span."""
        self.events.append(SpanEvent(name=name, attributes=attributes))
    
    def set_status(self, status: SpanStatus, error_message: Optional[str] = None):
        """Set the status of this span."""
        self.status = status
        if error_message:
            self.error_message = error_message
    
    def finish(self):
        """Mark the span as finished."""
        self.end_time = time.time()
    
    def get_duration_ms(self) -> float:
        """Get the duration of this span in milliseconds."""
        end_time = self.end_time or time.time()
        return (end_time - self.start_time) * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "component": self.component,
            "span_type": self.span_type.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.get_duration_ms(),
            "status": self.status.value,
            "error_message": self.error_message,
            "tags": self.tags,
            "events": [event.to_dict() for event in self.events],
            "references": self.references
        }


class TraceData(BaseModel):
    """Complete trace data with all spans."""
    
    trace_id: str = Field(description="Unique trace identifier")
    root_span_id: str = Field(description="Root span identifier")
    spans: List[Dict[str, Any]] = Field(default_factory=list, description="All spans in trace")
    start_time: datetime = Field(default_factory=datetime.utcnow, description="Trace start time")
    end_time: Optional[datetime] = Field(default=None, description="Trace end time")
    total_duration_ms: float = Field(default=0.0, description="Total trace duration")
    
    # Metadata
    service: str = Field(default="autom8", description="Service name")
    operation: str = Field(default="", description="Root operation name")
    user_id: Optional[str] = Field(default=None, description="User ID")
    tags: Dict[str, Any] = Field(default_factory=dict, description="Trace-level tags")
    
    # Status
    has_errors: bool = Field(default=False, description="Whether trace contains errors")
    error_count: int = Field(default=0, description="Number of errors in trace")
    
    def add_span(self, span_context: SpanContext):
        """Add a span to this trace."""
        self.spans.append(span_context.to_dict())
        
        # Update trace metadata
        if span_context.status == SpanStatus.ERROR:
            self.has_errors = True
            self.error_count += 1
    
    def finalize(self):
        """Finalize the trace with summary information."""
        if self.spans:
            start_times = [span["start_time"] for span in self.spans]
            end_times = [span["end_time"] for span in self.spans if span["end_time"]]
            
            if start_times and end_times:
                self.start_time = datetime.fromtimestamp(min(start_times))
                self.end_time = datetime.fromtimestamp(max(end_times))
                self.total_duration_ms = (max(end_times) - min(start_times)) * 1000


class TracingManager:
    """
    Central tracing manager for distributed request tracing.
    
    Manages trace collection, storage, and analysis across components.
    """
    
    def __init__(self, max_traces: int = 1000):
        self.max_traces = max_traces
        self.active_traces: Dict[str, TraceData] = {}
        self.completed_traces: List[TraceData] = []
        self.active_spans: Dict[str, SpanContext] = {}
        
        # Configuration
        self.sample_rate = 1.0  # Sample 100% of traces by default
        self.max_span_duration_ms = 300000  # 5 minutes max span duration
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    def should_trace(self) -> bool:
        """Determine if this request should be traced based on sampling."""
        import random
        return random.random() < self.sample_rate
    
    def start_trace(
        self,
        operation_name: str,
        service: str = "autom8",
        user_id: Optional[str] = None,
        **tags
    ) -> TraceData:
        """Start a new distributed trace."""
        
        if not self.should_trace():
            # Return a no-op trace for sampling
            return TraceData(trace_id="no-trace", root_span_id="no-span")
        
        trace_id = str(uuid.uuid4())
        root_span_id = str(uuid.uuid4())
        
        trace = TraceData(
            trace_id=trace_id,
            root_span_id=root_span_id,
            service=service,
            operation=operation_name,
            user_id=user_id,
            tags=tags
        )
        
        # Create root span
        root_span = SpanContext(
            trace_id=trace_id,
            span_id=root_span_id,
            operation_name=operation_name,
            component=service,
            span_type=SpanType.ROOT
        )
        
        for key, value in tags.items():
            root_span.add_tag(key, value)
        
        if user_id:
            root_span.add_tag("user.id", user_id)
        
        self.active_traces[trace_id] = trace
        self.active_spans[root_span_id] = root_span
        
        logger.trace(
            f"Started trace: {operation_name}",
            trace_id=trace_id,
            span_id=root_span_id,
            operation=operation_name,
            service=service
        )
        
        return trace
    
    def start_span(
        self,
        trace_id: str,
        operation_name: str,
        parent_span_id: Optional[str] = None,
        component: str = "",
        span_type: SpanType = SpanType.CHILD,
        **tags
    ) -> SpanContext:
        """Start a new span within an existing trace."""
        
        if trace_id == "no-trace":
            # Return a no-op span for non-traced requests
            return SpanContext(trace_id="no-trace", span_id="no-span")
        
        span_id = str(uuid.uuid4())
        
        span = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            component=component,
            span_type=span_type
        )
        
        for key, value in tags.items():
            span.add_tag(key, value)
        
        self.active_spans[span_id] = span
        
        logger.trace(
            f"Started span: {operation_name}",
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation=operation_name,
            component=component
        )
        
        return span
    
    def finish_span(
        self,
        span_id: str,
        status: SpanStatus = SpanStatus.OK,
        error_message: Optional[str] = None,
        **tags
    ):
        """Finish a span and add it to the trace."""
        
        if span_id == "no-span":
            return  # No-op for non-traced spans
        
        if span_id not in self.active_spans:
            logger.warning(f"Attempted to finish unknown span: {span_id}")
            return
        
        span = self.active_spans[span_id]
        span.set_status(status, error_message)
        span.finish()
        
        # Add any final tags
        for key, value in tags.items():
            span.add_tag(key, value)
        
        # Add span to trace
        trace_id = span.trace_id
        if trace_id in self.active_traces:
            self.active_traces[trace_id].add_span(span)
        
        # Clean up active span
        del self.active_spans[span_id]
        
        logger.trace(
            f"Finished span: {span.operation_name}",
            trace_id=trace_id,
            span_id=span_id,
            duration_ms=span.get_duration_ms(),
            status=status.value
        )
        
        # Check if this was the root span to finalize trace
        if trace_id in self.active_traces:
            trace = self.active_traces[trace_id]
            if span.span_id == trace.root_span_id:
                self._finalize_trace(trace_id)
    
    def _finalize_trace(self, trace_id: str):
        """Finalize a completed trace."""
        
        if trace_id not in self.active_traces:
            return
        
        trace = self.active_traces[trace_id]
        trace.finalize()
        
        # Move to completed traces
        self.completed_traces.append(trace)
        del self.active_traces[trace_id]
        
        # Limit completed traces
        if len(self.completed_traces) > self.max_traces:
            self.completed_traces.pop(0)
        
        logger.info(
            f"Completed trace: {trace.operation}",
            trace_id=trace_id,
            duration_ms=trace.total_duration_ms,
            span_count=len(trace.spans),
            has_errors=trace.has_errors,
            error_count=trace.error_count
        )
    
    def get_trace(self, trace_id: str) -> Optional[TraceData]:
        """Get a trace by ID (active or completed)."""
        
        # Check active traces first
        if trace_id in self.active_traces:
            return self.active_traces[trace_id]
        
        # Check completed traces
        for trace in self.completed_traces:
            if trace.trace_id == trace_id:
                return trace
        
        return None
    
    def get_recent_traces(self, limit: int = 50, include_active: bool = True) -> List[TraceData]:
        """Get recent traces."""
        traces = []
        
        if include_active:
            traces.extend(self.active_traces.values())
        
        traces.extend(self.completed_traces[-limit:])
        
        # Sort by start time (most recent first)
        traces.sort(key=lambda t: t.start_time, reverse=True)
        
        return traces[:limit]
    
    def get_trace_statistics(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get trace statistics for the specified time window."""
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=duration_minutes)
        recent_traces = [
            trace for trace in self.completed_traces
            if trace.start_time >= cutoff_time
        ]
        
        if not recent_traces:
            return {
                "total_traces": 0,
                "error_rate": 0.0,
                "average_duration_ms": 0.0,
                "p95_duration_ms": 0.0,
                "operations": {}
            }
        
        # Calculate statistics
        total_traces = len(recent_traces)
        error_traces = sum(1 for trace in recent_traces if trace.has_errors)
        error_rate = (error_traces / total_traces) * 100.0
        
        durations = [trace.total_duration_ms for trace in recent_traces]
        average_duration = sum(durations) / len(durations)
        
        # Calculate P95
        sorted_durations = sorted(durations)
        p95_index = int(0.95 * len(sorted_durations))
        p95_duration = sorted_durations[p95_index] if sorted_durations else 0
        
        # Operation breakdown
        operations = {}
        for trace in recent_traces:
            op = trace.operation
            if op not in operations:
                operations[op] = {"count": 0, "errors": 0, "avg_duration": 0}
            
            operations[op]["count"] += 1
            if trace.has_errors:
                operations[op]["errors"] += 1
        
        # Calculate averages for operations
        for op_stats in operations.values():
            if op_stats["count"] > 0:
                op_traces = [t for t in recent_traces if t.operation == op_stats]
                op_durations = [t.total_duration_ms for t in op_traces]
                op_stats["avg_duration"] = sum(op_durations) / len(op_durations) if op_durations else 0
        
        return {
            "total_traces": total_traces,
            "error_rate": error_rate,
            "average_duration_ms": average_duration,
            "p95_duration_ms": p95_duration,
            "operations": operations
        }
    
    async def start_cleanup_task(self, interval_seconds: int = 300):  # 5 minutes
        """Start background cleanup task for old traces and spans."""
        
        self._running = True
        
        async def cleanup_loop():
            while self._running:
                try:
                    await self._cleanup_old_data()
                    await asyncio.sleep(interval_seconds)
                except Exception as e:
                    logger.error("Error in tracing cleanup loop", error=e)
                    await asyncio.sleep(interval_seconds)
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info("Started tracing cleanup task", interval_seconds=interval_seconds)
    
    def stop_cleanup_task(self):
        """Stop the cleanup task."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
        logger.info("Stopped tracing cleanup task")
    
    async def _cleanup_old_data(self):
        """Clean up old traces and spans."""
        
        current_time = time.time()
        cutoff_time = current_time - self.max_span_duration_ms / 1000.0
        
        # Clean up old active spans
        old_spans = [
            span_id for span_id, span in self.active_spans.items()
            if span.start_time < cutoff_time
        ]
        
        for span_id in old_spans:
            span = self.active_spans[span_id]
            logger.warning(
                f"Cleaning up old active span: {span.operation_name}",
                trace_id=span.trace_id,
                span_id=span_id,
                duration_ms=span.get_duration_ms()
            )
            
            self.finish_span(span_id, SpanStatus.TIMEOUT, "Span exceeded maximum duration")
        
        # Clean up old active traces (shouldn't happen often)
        old_traces = [
            trace_id for trace_id, trace in self.active_traces.items()
            if trace.start_time < datetime.fromtimestamp(cutoff_time)
        ]
        
        for trace_id in old_traces:
            trace = self.active_traces[trace_id]
            logger.warning(
                f"Cleaning up old active trace: {trace.operation}",
                trace_id=trace_id,
                span_count=len(trace.spans)
            )
            
            trace.finalize()
            self.completed_traces.append(trace)
            del self.active_traces[trace_id]


@asynccontextmanager
async def traced_operation(
    operation_name: str,
    trace_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    component: str = "",
    span_type: SpanType = SpanType.CHILD,
    **tags
):
    """Context manager for tracing operations."""
    
    tracing_manager = get_tracing_manager()
    
    if trace_id is None:
        # Start a new trace
        trace = tracing_manager.start_trace(operation_name, **tags)
        trace_id = trace.trace_id
        span_id = trace.root_span_id
    else:
        # Start a child span
        span = tracing_manager.start_span(
            trace_id=trace_id,
            operation_name=operation_name,
            parent_span_id=parent_span_id,
            component=component,
            span_type=span_type,
            **tags
        )
        span_id = span.span_id
    
    start_time = time.time()
    
    try:
        # Yield the span context
        if span_id != "no-span":
            yield tracing_manager.active_spans.get(span_id)
        else:
            yield None
        
        # Successful completion
        tracing_manager.finish_span(span_id, SpanStatus.OK)
        
    except Exception as e:
        # Error occurred
        tracing_manager.finish_span(
            span_id,
            SpanStatus.ERROR,
            error_message=str(e),
            error_type=type(e).__name__
        )
        raise


# Global tracing manager instance
_tracing_manager: Optional[TracingManager] = None


def get_tracing_manager() -> TracingManager:
    """Get or create the global tracing manager."""
    global _tracing_manager
    if _tracing_manager is None:
        _tracing_manager = TracingManager()
    return _tracing_manager