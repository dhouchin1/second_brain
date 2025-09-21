"""
Enhanced Error Management System for Autom8

Provides centralized error handling, automatic recovery strategies, and comprehensive
error analytics using the existing event bus infrastructure.
"""

import asyncio
import time
import traceback
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field

from autom8.models.events import EventType, EventPriority
from autom8.storage.redis.events import EventBus
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels for classification and handling priority."""
    LOW = "low"           # Non-critical errors, log and continue
    MEDIUM = "medium"     # Errors that affect functionality but don't break system
    HIGH = "high"         # Critical errors that require immediate attention
    EMERGENCY = "emergency"  # System-breaking errors requiring immediate response


class ErrorCategory(str, Enum):
    """Error categories for specialized handling and recovery strategies."""
    NETWORK = "network"              # Network connectivity issues
    MODEL = "model"                  # Model-related errors (unavailable, failed requests)
    CONTEXT = "context"              # Context processing and optimization errors
    ROUTING = "routing"              # Model routing and selection errors
    STORAGE = "storage"              # Database and storage errors
    AUTHENTICATION = "authentication"  # Auth and permission errors
    BUDGET = "budget"                # Budget limit and cost-related errors
    VALIDATION = "validation"        # Data validation and schema errors
    INTEGRATION = "integration"      # External service integration errors
    TEMPLATE = "template"            # Template processing errors
    MEMORY = "memory"               # Memory management and embeddings errors
    CONFIGURATION = "configuration" # Configuration and setup errors


class RecoveryStrategy:
    """Defines how to recover from specific error types."""
    
    def __init__(
        self,
        name: str,
        handler: Callable[['ErrorRecord'], 'RecoveryResult'],
        max_retries: int = 3,
        backoff_multiplier: float = 1.5,
        base_delay: float = 1.0,
        conditions: Optional[List[Callable[['ErrorRecord'], bool]]] = None,
        priority: int = 0
    ):
        """
        Initialize recovery strategy.
        
        Args:
            name: Human-readable strategy name
            handler: Async function that attempts recovery
            max_retries: Maximum retry attempts
            backoff_multiplier: Exponential backoff multiplier
            base_delay: Base delay between retries in seconds
            conditions: Optional conditions that must be met to apply this strategy
            priority: Strategy priority (higher = tried first)
        """
        self.name = name
        self.handler = handler
        self.max_retries = max_retries
        self.backoff_multiplier = backoff_multiplier
        self.base_delay = base_delay
        self.conditions = conditions or []
        self.priority = priority
        self.success_count = 0
        self.failure_count = 0
        self.total_attempts = 0


class ErrorRecord(BaseModel):
    """Comprehensive error record with context and metadata."""
    
    id: UUID = Field(default_factory=uuid4)
    error_type: str = Field(description="Exception type name")
    error_message: str = Field(description="Error message")
    category: ErrorCategory = Field(description="Error category")
    severity: ErrorSeverity = Field(description="Error severity level")
    component: str = Field(description="Component where error occurred")
    
    # Context information
    context: Dict[str, Any] = Field(default_factory=dict, description="Error context")
    stack_trace: Optional[str] = Field(default=None, description="Exception stack trace")
    user_id: Optional[str] = Field(default=None, description="Associated user ID")
    request_id: Optional[str] = Field(default=None, description="Associated request ID")
    
    # Timing information
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    first_occurrence: datetime = Field(default_factory=datetime.utcnow)
    occurrence_count: int = Field(default=1, description="Number of times this error occurred")
    
    # Recovery tracking
    recovery_attempted: bool = Field(default=False)
    recovery_successful: bool = Field(default=False)
    recovery_strategy: Optional[str] = Field(default=None)
    recovery_attempts: int = Field(default=0)
    
    # Metadata
    tags: Set[str] = Field(default_factory=set, description="Error tags for classification")
    resolved: bool = Field(default=False, description="Whether error has been resolved")
    resolution_notes: Optional[str] = Field(default=None)

    @computed_field
    @property
    def error_signature(self) -> str:
        """Generate unique signature for error deduplication."""
        components = [
            self.error_type,
            self.component,
            self.category.value
        ]
        return "|".join(components)


class RecoveryResult(BaseModel):
    """Result of a recovery attempt."""
    
    success: bool = Field(description="Whether recovery was successful")
    strategy_name: Optional[str] = Field(default=None)
    attempts: int = Field(default=1)
    execution_time: float = Field(default=0.0, description="Recovery execution time in seconds")
    error_message: Optional[str] = Field(default=None, description="Error during recovery")
    recovery_data: Dict[str, Any] = Field(default_factory=dict, description="Recovery-specific data")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for preventing recurrence")


class ErrorHandlingResult(BaseModel):
    """Complete result of error handling process."""
    
    error_record: ErrorRecord
    recovery_result: Optional[RecoveryResult] = None
    handled: bool = Field(description="Whether error was successfully handled")
    escalated: bool = Field(default=False, description="Whether error was escalated")
    notifications_sent: List[str] = Field(default_factory=list, description="Notifications sent")
    processing_time: float = Field(default=0.0, description="Total handling time in seconds")


class ErrorAnalytics(BaseModel):
    """Analytics data for error patterns and trends."""
    
    timeframe: str
    total_errors: int
    errors_by_category: Dict[str, int]
    errors_by_severity: Dict[str, int]
    errors_by_component: Dict[str, int]
    recovery_success_rate: float
    mean_time_to_recovery: float
    top_error_patterns: List[Dict[str, Any]]
    trends: Dict[str, float]  # Growth/decline rates
    recommendations: List[str]


class SystemHealthImpact(BaseModel):
    """Represents impact of errors on system health."""
    
    availability_score: float = Field(ge=0.0, le=1.0)
    performance_impact: float = Field(ge=0.0, le=1.0)
    user_experience_score: float = Field(ge=0.0, le=1.0)
    component_health: Dict[str, float]
    critical_issues: List[str]
    recommended_actions: List[str]


class EnhancedErrorManager:
    """
    Centralized error management system leveraging the existing event infrastructure.
    
    Provides automatic error recovery, comprehensive analytics, and intelligent
    escalation strategies for robust system operation.
    """
    
    def __init__(self, event_bus: EventBus, max_error_history: int = 10000):
        """
        Initialize enhanced error manager.
        
        Args:
            event_bus: Event bus for error event publishing
            max_error_history: Maximum number of error records to keep in memory
        """
        self.event_bus = event_bus
        self.max_error_history = max_error_history
        
        # Error storage and tracking
        self.error_history: List[ErrorRecord] = []
        self.error_signatures: Dict[str, ErrorRecord] = {}  # For deduplication
        
        # Recovery strategies by category
        self.recovery_strategies: Dict[ErrorCategory, List[RecoveryStrategy]] = {
            category: [] for category in ErrorCategory
        }
        
        # Analytics and metrics
        self.error_metrics: Dict[str, Any] = {}
        self.component_health_scores: Dict[str, float] = {}
        
        # Configuration
        self.escalation_thresholds = {
            ErrorSeverity.LOW: 10,      # Escalate after 10 occurrences
            ErrorSeverity.MEDIUM: 5,    # Escalate after 5 occurrences
            ErrorSeverity.HIGH: 2,      # Escalate after 2 occurrences
            ErrorSeverity.EMERGENCY: 1  # Escalate immediately
        }
        
        # Initialize default recovery strategies
        self._initialize_default_strategies()
        
        logger.info("Enhanced error manager initialized with event bus integration")
    
    def _initialize_default_strategies(self) -> None:
        """Initialize default recovery strategies for common error patterns."""
        
        # Network error strategies
        self.register_recovery_strategy(
            ErrorCategory.NETWORK,
            RecoveryStrategy(
                name="exponential_backoff_retry",
                handler=self._exponential_backoff_retry,
                max_retries=3,
                backoff_multiplier=2.0,
                priority=10
            )
        )
        
        # Model error strategies
        self.register_recovery_strategy(
            ErrorCategory.MODEL,
            RecoveryStrategy(
                name="fallback_model_selection",
                handler=self._fallback_model_recovery,
                max_retries=2,
                priority=20
            )
        )
        
        # Storage error strategies
        self.register_recovery_strategy(
            ErrorCategory.STORAGE,
            RecoveryStrategy(
                name="connection_reset",
                handler=self._storage_connection_recovery,
                max_retries=3,
                priority=15
            )
        )
        
        # Budget error strategies
        self.register_recovery_strategy(
            ErrorCategory.BUDGET,
            RecoveryStrategy(
                name="budget_optimization",
                handler=self._budget_optimization_recovery,
                max_retries=1,
                priority=25
            )
        )
        
        # Context error strategies
        self.register_recovery_strategy(
            ErrorCategory.CONTEXT,
            RecoveryStrategy(
                name="context_simplification",
                handler=self._context_simplification_recovery,
                max_retries=2,
                priority=18
            )
        )
    
    async def handle_error(
        self,
        error: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity,
        component: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        tags: Optional[Set[str]] = None
    ) -> ErrorHandlingResult:
        """
        Comprehensive error handling with automatic recovery and analytics.
        
        Args:
            error: The exception that occurred
            category: Error category for specialized handling
            severity: Error severity level
            component: Component where the error occurred
            context: Additional context information
            user_id: Associated user ID if applicable
            request_id: Associated request ID if applicable
            tags: Additional tags for error classification
            
        Returns:
            Complete error handling result with recovery information
        """
        start_time = time.perf_counter()
        
        # Create comprehensive error record
        error_record = self._create_error_record(
            error, category, severity, component, context, user_id, request_id, tags
        )
        
        # Check for duplicate errors and update occurrence count
        signature = error_record.error_signature
        if signature in self.error_signatures:
            existing_record = self.error_signatures[signature]
            existing_record.occurrence_count += 1
            existing_record.timestamp = datetime.utcnow()
            error_record = existing_record
        else:
            self.error_signatures[signature] = error_record
            self.error_history.append(error_record)
            
            # Maintain history size limit
            if len(self.error_history) > self.max_error_history:
                removed = self.error_history.pop(0)
                if removed.error_signature in self.error_signatures:
                    del self.error_signatures[removed.error_signature]
        
        # Emit error event through event bus
        await self._emit_error_event(error_record)
        
        # Attempt automatic recovery
        recovery_result = None
        if not error_record.recovery_attempted or error_record.recovery_attempts < 3:
            recovery_result = await self._attempt_recovery(error_record)
        
        # Check for escalation
        escalated = await self._check_escalation(error_record)
        
        # Update component health scores
        await self._update_component_health(component, severity, recovery_result)
        
        # Calculate processing time
        processing_time = time.perf_counter() - start_time
        
        # Create result
        result = ErrorHandlingResult(
            error_record=error_record,
            recovery_result=recovery_result,
            handled=recovery_result.success if recovery_result else False,
            escalated=escalated,
            processing_time=processing_time
        )
        
        logger.info(
            f"Error handled: {error_record.error_type} in {component} "
            f"(severity: {severity}, handled: {result.handled}, "
            f"escalated: {escalated}, time: {processing_time:.3f}s)"
        )
        
        return result
    
    def _create_error_record(
        self,
        error: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity,
        component: str,
        context: Optional[Dict[str, Any]],
        user_id: Optional[str],
        request_id: Optional[str],
        tags: Optional[Set[str]]
    ) -> ErrorRecord:
        """Create comprehensive error record from exception."""
        
        return ErrorRecord(
            error_type=type(error).__name__,
            error_message=str(error),
            category=category,
            severity=severity,
            component=component,
            context=context or {},
            stack_trace=traceback.format_exc(),
            user_id=user_id,
            request_id=request_id,
            tags=tags or set()
        )
    
    async def _emit_error_event(self, error_record: ErrorRecord) -> None:
        """Emit error event through the event bus."""
        
        try:
            await self.event_bus.emit_event(
                event_type=EventType.ERROR_OCCURRED,
                priority=self._severity_to_event_priority(error_record.severity),
                data={
                    'error_id': str(error_record.id),
                    'error_type': error_record.error_type,
                    'category': error_record.category.value,
                    'severity': error_record.severity.value,
                    'component': error_record.component,
                    'message': error_record.error_message,
                    'occurrence_count': error_record.occurrence_count,
                    'context': error_record.context,
                    'user_id': error_record.user_id,
                    'request_id': error_record.request_id,
                    'tags': list(error_record.tags)
                }
            )
        except Exception as emit_error:
            logger.error(f"Failed to emit error event: {emit_error}")
    
    def _severity_to_event_priority(self, severity: ErrorSeverity) -> EventPriority:
        """Convert error severity to event priority."""
        mapping = {
            ErrorSeverity.LOW: EventPriority.LOW,
            ErrorSeverity.MEDIUM: EventPriority.MEDIUM,
            ErrorSeverity.HIGH: EventPriority.HIGH,
            ErrorSeverity.EMERGENCY: EventPriority.CRITICAL
        }
        return mapping.get(severity, EventPriority.MEDIUM)
    
    async def _attempt_recovery(self, error_record: ErrorRecord) -> RecoveryResult:
        """Attempt to recover from the error using registered strategies."""
        
        error_record.recovery_attempted = True
        error_record.recovery_attempts += 1
        
        # Get applicable recovery strategies
        strategies = self.recovery_strategies.get(error_record.category, [])
        
        # Sort strategies by priority (highest first)
        strategies.sort(key=lambda s: s.priority, reverse=True)
        
        for strategy in strategies:
            if self._should_apply_strategy(strategy, error_record):
                try:
                    start_time = time.perf_counter()
                    logger.debug(f"Attempting recovery strategy: {strategy.name}")
                    
                    # Execute recovery strategy
                    result = await self._execute_recovery_strategy(strategy, error_record)
                    result.execution_time = time.perf_counter() - start_time
                    
                    # Update strategy metrics
                    strategy.total_attempts += 1
                    
                    if result.success:
                        strategy.success_count += 1
                        error_record.recovery_successful = True
                        error_record.recovery_strategy = strategy.name
                        
                        # Emit recovery success event
                        await self._emit_recovery_event(error_record, result, True)
                        
                        logger.info(
                            f"Recovery successful using strategy '{strategy.name}' "
                            f"for error {error_record.id} ({result.execution_time:.3f}s)"
                        )
                        return result
                    else:
                        strategy.failure_count += 1
                        logger.debug(f"Recovery strategy '{strategy.name}' failed: {result.error_message}")
                        
                except Exception as recovery_error:
                    strategy.failure_count += 1
                    logger.error(f"Recovery strategy '{strategy.name}' raised exception: {recovery_error}")
                    continue
        
        # All strategies failed
        result = RecoveryResult(
            success=False,
            strategy_name=None,
            error_message="All recovery strategies failed"
        )
        
        await self._emit_recovery_event(error_record, result, False)
        return result
    
    def _should_apply_strategy(self, strategy: RecoveryStrategy, error_record: ErrorRecord) -> bool:
        """Determine if a recovery strategy should be applied to this error."""
        
        # Check all conditions
        for condition in strategy.conditions:
            try:
                if not condition(error_record):
                    return False
            except Exception as condition_error:
                logger.error(f"Strategy condition check failed: {condition_error}")
                return False
        
        # Check retry limits
        if error_record.recovery_attempts >= strategy.max_retries:
            return False
        
        return True
    
    async def _execute_recovery_strategy(
        self,
        strategy: RecoveryStrategy,
        error_record: ErrorRecord
    ) -> RecoveryResult:
        """Execute a recovery strategy with retry logic."""
        
        for attempt in range(strategy.max_retries):
            try:
                # Calculate delay for exponential backoff
                if attempt > 0:
                    delay = strategy.base_delay * (strategy.backoff_multiplier ** (attempt - 1))
                    await asyncio.sleep(delay)
                
                # Execute the recovery handler
                if asyncio.iscoroutinefunction(strategy.handler):
                    result = await strategy.handler(error_record)
                else:
                    result = strategy.handler(error_record)
                
                if isinstance(result, RecoveryResult):
                    result.attempts = attempt + 1
                    result.strategy_name = strategy.name
                    return result
                else:
                    # Assume success if no explicit result
                    return RecoveryResult(
                        success=True,
                        strategy_name=strategy.name,
                        attempts=attempt + 1
                    )
                
            except Exception as execution_error:
                if attempt == strategy.max_retries - 1:  # Last attempt
                    return RecoveryResult(
                        success=False,
                        strategy_name=strategy.name,
                        attempts=attempt + 1,
                        error_message=str(execution_error)
                    )
                else:
                    logger.debug(f"Recovery attempt {attempt + 1} failed: {execution_error}")
                    continue
        
        return RecoveryResult(
            success=False,
            strategy_name=strategy.name,
            attempts=strategy.max_retries,
            error_message="All retry attempts exhausted"
        )
    
    async def _emit_recovery_event(
        self,
        error_record: ErrorRecord,
        recovery_result: RecoveryResult,
        success: bool
    ) -> None:
        """Emit recovery event through the event bus."""
        
        try:
            event_type = EventType.RECOVERY_COMPLETED if success else EventType.RECOVERY_STARTED
            
            await self.event_bus.emit_event(
                event_type=event_type,
                priority=EventPriority.MEDIUM,
                data={
                    'error_id': str(error_record.id),
                    'recovery_success': success,
                    'strategy': recovery_result.strategy_name,
                    'attempts': recovery_result.attempts,
                    'execution_time': recovery_result.execution_time,
                    'error_message': recovery_result.error_message
                }
            )
        except Exception as emit_error:
            logger.error(f"Failed to emit recovery event: {emit_error}")
    
    async def _check_escalation(self, error_record: ErrorRecord) -> bool:
        """Check if error should be escalated based on occurrence patterns."""
        
        threshold = self.escalation_thresholds.get(error_record.severity, 5)
        
        if error_record.occurrence_count >= threshold:
            await self._escalate_error(error_record)
            return True
        
        return False
    
    async def _escalate_error(self, error_record: ErrorRecord) -> None:
        """Escalate error to higher-level monitoring or alerting systems."""
        
        try:
            # Emit escalation event
            await self.event_bus.emit_event(
                event_type=EventType.SYSTEM_ERROR,
                priority=EventPriority.CRITICAL,
                data={
                    'error_id': str(error_record.id),
                    'escalation_reason': f"Error occurred {error_record.occurrence_count} times",
                    'severity': error_record.severity.value,
                    'component': error_record.component,
                    'error_type': error_record.error_type,
                    'message': error_record.error_message,
                    'requires_attention': True
                }
            )
            
            logger.warning(
                f"Error escalated: {error_record.error_type} in {error_record.component} "
                f"(occurred {error_record.occurrence_count} times)"
            )
            
        except Exception as escalation_error:
            logger.error(f"Failed to escalate error: {escalation_error}")
    
    async def _update_component_health(
        self,
        component: str,
        severity: ErrorSeverity,
        recovery_result: Optional[RecoveryResult]
    ) -> None:
        """Update component health scores based on error patterns."""
        
        current_score = self.component_health_scores.get(component, 1.0)
        
        # Calculate impact based on severity and recovery success
        severity_impact = {
            ErrorSeverity.LOW: 0.01,
            ErrorSeverity.MEDIUM: 0.05,
            ErrorSeverity.HIGH: 0.15,
            ErrorSeverity.EMERGENCY: 0.30
        }
        
        impact = severity_impact.get(severity, 0.05)
        
        # Reduce impact if recovery was successful
        if recovery_result and recovery_result.success:
            impact *= 0.5
        
        # Update score (with some recovery over time)
        new_score = max(0.0, current_score - impact)
        new_score = min(1.0, new_score + 0.001)  # Slow recovery
        
        self.component_health_scores[component] = new_score
        
        if new_score < 0.7:  # Health threshold
            logger.warning(f"Component {component} health degraded: {new_score:.3f}")
    
    def register_recovery_strategy(
        self,
        category: ErrorCategory,
        strategy: RecoveryStrategy
    ) -> None:
        """Register a new recovery strategy for a specific error category."""
        
        self.recovery_strategies[category].append(strategy)
        logger.info(f"Registered recovery strategy '{strategy.name}' for category {category}")
    
    async def get_error_analytics(
        self,
        timeframe: str = "24h",
        include_predictions: bool = False
    ) -> ErrorAnalytics:
        """Generate comprehensive error analytics report."""
        
        # Parse timeframe
        hours = self._parse_timeframe(timeframe)
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Filter errors by timeframe
        recent_errors = [
            error for error in self.error_history
            if error.timestamp >= cutoff_time
        ]
        
        # Calculate analytics
        total_errors = len(recent_errors)
        
        # Categorize errors
        errors_by_category = {}
        errors_by_severity = {}
        errors_by_component = {}
        
        for error in recent_errors:
            # By category
            cat = error.category.value
            errors_by_category[cat] = errors_by_category.get(cat, 0) + 1
            
            # By severity
            sev = error.severity.value
            errors_by_severity[sev] = errors_by_severity.get(sev, 0) + 1
            
            # By component
            comp = error.component
            errors_by_component[comp] = errors_by_component.get(comp, 0) + 1
        
        # Calculate recovery success rate
        recovery_attempts = [e for e in recent_errors if e.recovery_attempted]
        recovery_successes = [e for e in recovery_attempts if e.recovery_successful]
        recovery_success_rate = (
            len(recovery_successes) / len(recovery_attempts)
            if recovery_attempts else 0.0
        )
        
        # Calculate mean time to recovery
        recovery_times = []
        for error in recovery_successes:
            # Approximate recovery time based on attempts (simplified)
            recovery_times.append(error.recovery_attempts * 2.0)
        
        mean_time_to_recovery = (
            sum(recovery_times) / len(recovery_times)
            if recovery_times else 0.0
        )
        
        # Identify top error patterns
        signature_counts = {}
        for error in recent_errors:
            sig = error.error_signature
            signature_counts[sig] = signature_counts.get(sig, 0) + 1
        
        top_patterns = sorted(
            [{"pattern": sig, "count": count} for sig, count in signature_counts.items()],
            key=lambda x: x["count"],
            reverse=True
        )[:10]
        
        # Generate recommendations
        recommendations = self._generate_error_recommendations(
            recent_errors, errors_by_category, errors_by_component
        )
        
        # Calculate trends by comparing with previous period
        trends = self._calculate_error_trends(
            recent_errors, hours, errors_by_category, errors_by_severity
        )
        
        return ErrorAnalytics(
            timeframe=timeframe,
            total_errors=total_errors,
            errors_by_category=errors_by_category,
            errors_by_severity=errors_by_severity,
            errors_by_component=errors_by_component,
            recovery_success_rate=recovery_success_rate,
            mean_time_to_recovery=mean_time_to_recovery,
            top_error_patterns=top_patterns,
            trends=trends,
            recommendations=recommendations
        )
    
    def _parse_timeframe(self, timeframe: str) -> int:
        """Parse timeframe string to hours."""
        if timeframe.endswith('h'):
            return int(timeframe[:-1])
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 24
        elif timeframe.endswith('w'):
            return int(timeframe[:-1]) * 24 * 7
        else:
            return 24  # Default to 24 hours
    
    def _generate_error_recommendations(
        self,
        recent_errors: List[ErrorRecord],
        errors_by_category: Dict[str, int],
        errors_by_component: Dict[str, int]
    ) -> List[str]:
        """Generate actionable recommendations based on error patterns."""
        
        recommendations = []
        
        # Check for high error categories
        total_errors = len(recent_errors)
        if total_errors > 0:
            for category, count in errors_by_category.items():
                if count / total_errors > 0.3:  # >30% of errors
                    recommendations.append(
                        f"High frequency of {category} errors ({count}/{total_errors}). "
                        f"Review {category} error handling and infrastructure."
                    )
            
            # Check for problematic components
            for component, count in errors_by_component.items():
                if count / total_errors > 0.4:  # >40% of errors
                    recommendations.append(
                        f"Component '{component}' has high error rate ({count}/{total_errors}). "
                        f"Consider reviewing and refactoring this component."
                    )
        
        # Check recovery success rate
        recovery_attempts = [e for e in recent_errors if e.recovery_attempted]
        if recovery_attempts:
            successes = [e for e in recovery_attempts if e.recovery_successful]
            success_rate = len(successes) / len(recovery_attempts)
            
            if success_rate < 0.5:
                recommendations.append(
                    f"Low recovery success rate ({success_rate:.1%}). "
                    f"Review and improve recovery strategies."
                )
        
        return recommendations
    
    def _calculate_error_trends(
        self,
        recent_errors: List[ErrorRecord],
        timeframe_hours: int,
        errors_by_category: Dict[str, int],
        errors_by_severity: Dict[str, int]
    ) -> Dict[str, float]:
        """Calculate error trends by comparing current period with previous period."""
        
        trends = {}
        
        # Calculate time boundaries
        now = datetime.utcnow()
        current_period_start = now - timedelta(hours=timeframe_hours)
        previous_period_start = current_period_start - timedelta(hours=timeframe_hours)
        previous_period_end = current_period_start
        
        # Get errors from previous period for comparison
        previous_errors = [
            error for error in self.error_history
            if previous_period_start <= error.timestamp < previous_period_end
        ]
        
        # Calculate previous period statistics
        previous_total = len(previous_errors)
        previous_by_category = {}
        previous_by_severity = {}
        
        for error in previous_errors:
            category = error.category.value
            severity = error.severity.value
            
            previous_by_category[category] = previous_by_category.get(category, 0) + 1
            previous_by_severity[severity] = previous_by_severity.get(severity, 0) + 1
        
        # Calculate overall trend
        current_total = len(recent_errors)
        if previous_total > 0:
            trends["total_errors"] = ((current_total - previous_total) / previous_total) * 100.0
        else:
            trends["total_errors"] = 100.0 if current_total > 0 else 0.0
        
        # Calculate category trends
        for category in set(list(errors_by_category.keys()) + list(previous_by_category.keys())):
            current_count = errors_by_category.get(category, 0)
            previous_count = previous_by_category.get(category, 0)
            
            if previous_count > 0:
                trends[f"category_{category}"] = ((current_count - previous_count) / previous_count) * 100.0
            else:
                trends[f"category_{category}"] = 100.0 if current_count > 0 else 0.0
        
        # Calculate severity trends
        for severity in set(list(errors_by_severity.keys()) + list(previous_by_severity.keys())):
            current_count = errors_by_severity.get(severity, 0)
            previous_count = previous_by_severity.get(severity, 0)
            
            if previous_count > 0:
                trends[f"severity_{severity}"] = ((current_count - previous_count) / previous_count) * 100.0
            else:
                trends[f"severity_{severity}"] = 100.0 if current_count > 0 else 0.0
        
        # Calculate recovery trends
        current_recovery_attempts = [e for e in recent_errors if e.recovery_attempted]
        current_recovery_successes = [e for e in current_recovery_attempts if e.recovery_successful]
        current_success_rate = (
            len(current_recovery_successes) / len(current_recovery_attempts)
            if current_recovery_attempts else 0.0
        )
        
        previous_recovery_attempts = [e for e in previous_errors if e.recovery_attempted]
        previous_recovery_successes = [e for e in previous_recovery_attempts if e.recovery_successful]
        previous_success_rate = (
            len(previous_recovery_successes) / len(previous_recovery_attempts)
            if previous_recovery_attempts else 0.0
        )
        
        if previous_success_rate > 0:
            trends["recovery_success_rate"] = ((current_success_rate - previous_success_rate) / previous_success_rate) * 100.0
        else:
            trends["recovery_success_rate"] = 100.0 if current_success_rate > 0 else 0.0
        
        # Calculate frequency trend (errors per hour)
        current_frequency = current_total / timeframe_hours if timeframe_hours > 0 else 0
        previous_frequency = previous_total / timeframe_hours if timeframe_hours > 0 else 0
        
        if previous_frequency > 0:
            trends["error_frequency"] = ((current_frequency - previous_frequency) / previous_frequency) * 100.0
        else:
            trends["error_frequency"] = 100.0 if current_frequency > 0 else 0.0
        
        return trends
    
    async def get_system_health_impact(self) -> SystemHealthImpact:
        """Calculate the impact of errors on overall system health."""
        
        # Calculate component health scores
        component_health = self.component_health_scores.copy()
        
        # Calculate overall availability score
        if component_health:
            availability_score = sum(component_health.values()) / len(component_health)
        else:
            availability_score = 1.0
        
        # Recent error analysis for performance impact
        recent_errors = [
            error for error in self.error_history
            if error.timestamp >= datetime.utcnow() - timedelta(hours=1)
        ]
        
        performance_impact = min(1.0, len(recent_errors) / 100.0)  # Simplified calculation
        user_experience_score = max(0.0, 1.0 - performance_impact)
        
        # Identify critical issues
        critical_issues = []
        for component, health in component_health.items():
            if health < 0.5:
                critical_issues.append(f"Component '{component}' critically unhealthy ({health:.2f})")
        
        # Generate recommended actions
        recommended_actions = []
        if availability_score < 0.8:
            recommended_actions.append("System availability is degraded - immediate attention required")
        if len(critical_issues) > 0:
            recommended_actions.append("Critical component issues require immediate resolution")
        if performance_impact > 0.3:
            recommended_actions.append("High error rate is impacting performance")
        
        return SystemHealthImpact(
            availability_score=availability_score,
            performance_impact=performance_impact,
            user_experience_score=user_experience_score,
            component_health=component_health,
            critical_issues=critical_issues,
            recommended_actions=recommended_actions
        )
    
    # Default recovery strategy implementations
    
    async def _exponential_backoff_retry(self, error_record: ErrorRecord) -> RecoveryResult:
        """Generic exponential backoff retry strategy."""
        
        # This is a placeholder - in practice, this would retry the original operation
        await asyncio.sleep(0.1)  # Simulate recovery work
        
        return RecoveryResult(
            success=True,
            recommendations=["Implement proper retry logic in the component"]
        )
    
    async def _fallback_model_recovery(self, error_record: ErrorRecord) -> RecoveryResult:
        """Fallback to alternative model when primary model fails."""
        
        # This would integrate with the ModelRouter to select an alternative
        await asyncio.sleep(0.1)  # Simulate model switching
        
        return RecoveryResult(
            success=True,
            recovery_data={"fallback_model": "alternative_model"},
            recommendations=["Update model routing to prefer more reliable models"]
        )
    
    async def _storage_connection_recovery(self, error_record: ErrorRecord) -> RecoveryResult:
        """Attempt to recover from storage connection issues."""
        
        # This would attempt to reconnect to database/Redis
        await asyncio.sleep(0.1)  # Simulate connection recovery
        
        return RecoveryResult(
            success=True,
            recommendations=["Implement connection pooling and health checks"]
        )
    
    async def _budget_optimization_recovery(self, error_record: ErrorRecord) -> RecoveryResult:
        """Attempt to recover from budget-related errors through optimization."""
        
        # This would trigger budget optimization or downgrade to cheaper models
        await asyncio.sleep(0.1)  # Simulate budget optimization
        
        return RecoveryResult(
            success=True,
            recovery_data={"optimization_applied": True},
            recommendations=["Review budget allocation and usage patterns"]
        )
    
    async def _context_simplification_recovery(self, error_record: ErrorRecord) -> RecoveryResult:
        """Attempt to recover from context-related errors by simplifying context."""
        
        # This would trigger context optimization to reduce complexity
        await asyncio.sleep(0.1)  # Simulate context simplification
        
        return RecoveryResult(
            success=True,
            recovery_data={"context_reduced": True},
            recommendations=["Implement automatic context optimization"]
        )
    
    async def resolve_error(
        self,
        error_id: UUID,
        resolution_notes: Optional[str] = None
    ) -> bool:
        """Mark an error as resolved with optional notes."""
        
        # Find error by ID
        for error_record in self.error_history:
            if error_record.id == error_id:
                error_record.resolved = True
                error_record.resolution_notes = resolution_notes
                
                # Emit resolution event
                await self.event_bus.emit_event(
                    event_type=EventType.ERROR_RESOLVED,
                    priority=EventPriority.MEDIUM,
                    data={
                        'error_id': str(error_id),
                        'resolution_notes': resolution_notes
                    }
                )
                
                logger.info(f"Error {error_id} marked as resolved")
                return True
        
        logger.warning(f"Error {error_id} not found for resolution")
        return False
    
    def get_strategy_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all recovery strategies."""
        
        performance = {}
        
        for category, strategies in self.recovery_strategies.items():
            category_performance = {}
            
            for strategy in strategies:
                if strategy.total_attempts > 0:
                    success_rate = strategy.success_count / strategy.total_attempts
                    category_performance[strategy.name] = {
                        'success_rate': success_rate,
                        'total_attempts': strategy.total_attempts,
                        'success_count': strategy.success_count,
                        'failure_count': strategy.failure_count
                    }
            
            if category_performance:
                performance[category.value] = category_performance
        
        return performance


# Utility functions for easy error handling integration

async def handle_error_async(
    error: Exception,
    category: ErrorCategory,
    severity: ErrorSeverity,
    component: str,
    context: Optional[Dict[str, Any]] = None,
    error_manager: Optional[EnhancedErrorManager] = None
) -> ErrorHandlingResult:
    """
    Convenience function for async error handling.
    
    Can be used as a decorator or direct function call for streamlined error handling.
    """
    if error_manager is None:
        # In practice, this would get the global error manager instance
        raise RuntimeError("Error manager not available")
    
    return await error_manager.handle_error(
        error=error,
        category=category,
        severity=severity,
        component=component,
        context=context
    )


def error_handler(
    category: ErrorCategory,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    component: Optional[str] = None
):
    """
    Decorator for automatic error handling in functions.
    
    Usage:
        @error_handler(ErrorCategory.MODEL, ErrorSeverity.HIGH, "model_router")
        async def route_model(self, query):
            # Function implementation
            pass
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Get error manager from global context or first arg if available
                error_manager = getattr(args[0], 'error_manager', None) if args else None
                
                if error_manager:
                    result = await error_manager.handle_error(
                        error=e,
                        category=category,
                        severity=severity,
                        component=component or func.__name__,
                        context={'function': func.__name__, 'args': str(args), 'kwargs': str(kwargs)}
                    )
                    
                    # Re-raise if not handled
                    if not result.handled:
                        raise
                    
                    # Return recovery data if available
                    if result.recovery_result and result.recovery_result.recovery_data:
                        return result.recovery_result.recovery_data
                else:
                    # Fallback to normal exception handling
                    raise
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # For sync functions, just log and re-raise
                logger.error(f"Error in {func.__name__}: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator