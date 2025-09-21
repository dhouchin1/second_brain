"""
BudgetManager - Comprehensive cost control and budgeting system.

Provides per-user, per-project, and time-based budget tracking with real-time 
monitoring, alerts, forecasting, and automatic policy enforcement.
"""

import asyncio
import json
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, computed_field

from autom8.models.routing import Model, ModelProvider
from autom8.storage.redis.client import RedisClient, get_redis_client
from autom8.storage.redis.events import EventBus
from autom8.storage.sqlite.manager import SQLiteManager
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class BudgetType(str, Enum):
    """Types of budget constraints"""
    USER = "user"           # Per-user spending limits
    PROJECT = "project"     # Per-project spending limits
    MODEL = "model"         # Per-model spending limits
    PROVIDER = "provider"   # Per-provider spending limits
    DAILY = "daily"         # Daily spending limits
    WEEKLY = "weekly"       # Weekly spending limits
    MONTHLY = "monthly"     # Monthly spending limits
    QUERY = "query"         # Per-query spending limits


class AlertLevel(str, Enum):
    """Alert severity levels"""
    INFO = "info"           # Informational alerts
    WARNING = "warning"     # Warning alerts (e.g., 75% of budget)
    CRITICAL = "critical"   # Critical alerts (e.g., 90% of budget)
    EMERGENCY = "emergency" # Emergency alerts (budget exceeded)


class EnforcementAction(str, Enum):
    """Actions to take when budget limits are reached"""
    WARN = "warn"                    # Send warning only
    BLOCK = "block"                  # Block further requests
    DOWNGRADE = "downgrade"          # Switch to cheaper models
    APPROVE = "approve"              # Require approval for continuation
    THROTTLE = "throttle"            # Rate limit requests


class Budget(BaseModel):
    model_config = {"protected_namespaces": ()}
    """Budget definition with limits and policies"""
    
    # Basic info
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(description="Human-readable budget name")
    description: Optional[str] = Field(default=None, description="Budget description")
    
    # Budget scope
    budget_type: BudgetType = Field(description="Type of budget constraint")
    scope: Dict[str, Any] = Field(default_factory=dict, description="Budget scope (user_id, project_id, etc.)")
    
    # Limits
    limit_amount: float = Field(gt=0, description="Budget limit in USD")
    spent_amount: float = Field(default=0.0, ge=0, description="Amount spent so far")
    
    # Time constraints
    start_date: datetime = Field(default_factory=datetime.utcnow)
    end_date: Optional[datetime] = Field(default=None, description="Budget expiration date")
    reset_period: Optional[str] = Field(default=None, description="Reset period (daily, weekly, monthly)")
    
    # Alert thresholds
    alert_thresholds: List[float] = Field(default=[0.5, 0.75, 0.9], description="Alert thresholds (0-1)")
    triggered_alerts: Set[float] = Field(default_factory=set, description="Already triggered alert levels")
    
    # Enforcement
    enforcement_action: EnforcementAction = Field(default=EnforcementAction.WARN)
    enforce_hard_limit: bool = Field(default=True, description="Enforce hard limit vs soft warning")
    
    # Status
    is_active: bool = Field(default=True, description="Budget is active")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @computed_field
    @property
    def utilization_percentage(self) -> float:
        """Calculate budget utilization as percentage"""
        if self.limit_amount <= 0:
            return 0.0
        return min(100.0, (self.spent_amount / self.limit_amount) * 100)
    
    @computed_field
    @property
    def remaining_amount(self) -> float:
        """Calculate remaining budget amount"""
        return max(0.0, self.limit_amount - self.spent_amount)
    
    @computed_field
    @property
    def is_exhausted(self) -> bool:
        """Check if budget is exhausted"""
        return self.spent_amount >= self.limit_amount
    
    @computed_field
    @property
    def days_remaining(self) -> Optional[int]:
        """Calculate days remaining in budget period"""
        if not self.end_date:
            return None
        delta = self.end_date - datetime.utcnow()
        return max(0, delta.days)
    
    def should_trigger_alert(self, threshold: float) -> bool:
        """Check if alert should be triggered for given threshold"""
        if threshold in self.triggered_alerts:
            return False
        return self.utilization_percentage / 100 >= threshold
    
    def trigger_alert(self, threshold: float) -> None:
        """Mark alert as triggered for given threshold"""
        self.triggered_alerts.add(threshold)
        self.updated_at = datetime.utcnow()


class SpendingRecord(BaseModel):
    model_config = {"protected_namespaces": ()}
    """Record of spending transaction"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Transaction details
    amount: float = Field(gt=0, description="Amount spent in USD")
    currency: str = Field(default="USD", description="Currency code")
    
    # Model/service details
    model_name: str = Field(description="Model used for this transaction")
    provider: ModelProvider = Field(description="Model provider")
    
    # Token usage
    input_tokens: int = Field(default=0, ge=0, description="Input tokens used")
    output_tokens: int = Field(default=0, ge=0, description="Output tokens generated")
    
    # Context
    user_id: Optional[str] = Field(default=None, description="User who incurred the cost")
    project_id: Optional[str] = Field(default=None, description="Project associated with cost")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    query_id: Optional[str] = Field(default=None, description="Query identifier")
    
    # Timing
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: Optional[int] = Field(default=None, description="Request duration in milliseconds")
    
    # Status
    success: bool = Field(default=True, description="Whether the request was successful")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BudgetAlert(BaseModel):
    model_config = {"protected_namespaces": ()}
    """Budget alert notification"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Alert details
    budget_id: str = Field(description="Budget that triggered the alert")
    level: AlertLevel = Field(description="Alert severity level")
    threshold: float = Field(description="Threshold that was crossed")
    
    # Context
    current_spending: float = Field(description="Current spending amount")
    budget_limit: float = Field(description="Budget limit")
    utilization_percentage: float = Field(description="Budget utilization percentage")
    
    # Message
    title: str = Field(description="Alert title")
    message: str = Field(description="Alert message")
    
    # Actions
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")
    enforcement_actions: List[EnforcementAction] = Field(default_factory=list, description="Enforcement actions taken")
    
    # Timing
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    acknowledged: bool = Field(default=False, description="Whether alert has been acknowledged")
    acknowledged_at: Optional[datetime] = Field(default=None)
    acknowledged_by: Optional[str] = Field(default=None)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CostForecast(BaseModel):
    model_config = {"protected_namespaces": ()}
    """Cost forecasting result"""
    
    # Forecast details
    budget_id: str = Field(description="Budget being forecasted")
    forecast_period_days: int = Field(description="Forecast period in days")
    
    # Predictions
    predicted_cost: float = Field(description="Predicted cost for the period")
    confidence_interval: Tuple[float, float] = Field(description="95% confidence interval")
    confidence_score: float = Field(ge=0, le=1, description="Confidence in prediction")
    
    # Trends
    daily_trend: float = Field(description="Daily spending trend")
    weekly_trend: float = Field(description="Weekly spending trend")
    growth_rate: float = Field(description="Spending growth rate")
    
    # Risk assessment
    risk_level: str = Field(description="Risk level (low, medium, high)")
    budget_exhaustion_date: Optional[datetime] = Field(default=None, description="Predicted budget exhaustion")
    
    # Model details
    model_type: str = Field(description="Forecasting model used")
    data_points: int = Field(description="Number of data points used")
    
    # Timing
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BudgetPolicy(BaseModel):
    model_config = {"protected_namespaces": ()}
    """Budget policy configuration"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(description="Policy name")
    description: Optional[str] = Field(default=None)
    
    # Scope
    applies_to: Dict[str, Any] = Field(default_factory=dict, description="What this policy applies to")
    
    # Rules
    auto_create_budgets: bool = Field(default=False, description="Automatically create budgets")
    default_budget_limits: Dict[str, float] = Field(default_factory=dict, description="Default budget limits by type")
    max_budget_limits: Dict[str, float] = Field(default_factory=dict, description="Maximum allowable budget limits")
    
    # Approval workflows
    require_approval_over: Optional[float] = Field(default=None, description="Require approval for budgets over this amount")
    approval_required_for: List[EnforcementAction] = Field(default_factory=list, description="Actions requiring approval")
    
    # Automatic actions
    auto_downgrade_models: bool = Field(default=True, description="Automatically downgrade to cheaper models")
    model_downgrade_thresholds: Dict[str, float] = Field(default_factory=dict, description="Thresholds for model downgrades")
    
    # Notification settings
    notification_channels: List[str] = Field(default_factory=list, description="Notification channels")
    alert_frequency_limits: Dict[AlertLevel, int] = Field(default_factory=dict, description="Max alerts per time period")
    
    # Status
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class BudgetManager:
    """
    Comprehensive budget management system for cost control.
    
    Provides per-user, per-project, and time-based budget tracking with
    real-time monitoring, alerts, forecasting, and automatic enforcement.
    """
    
    def __init__(self, redis_client: Optional[RedisClient] = None, 
                 sqlite_manager: Optional[SQLiteManager] = None,
                 event_bus: Optional[EventBus] = None):
        self.redis_client = redis_client
        self.sqlite_manager = sqlite_manager
        self.event_bus = event_bus
        
        # In-memory state
        self.budgets: Dict[str, Budget] = {}
        self.policies: Dict[str, BudgetPolicy] = {}
        self.active_spending_cache: Dict[str, List[SpendingRecord]] = defaultdict(list)
        
        # Forecasting models
        self.forecasting_models: Dict[str, Any] = {}
        
        # Tracking
        self._initialized = False
        self._last_cleanup = time.time()
        
    async def initialize(self) -> bool:
        """Initialize the budget manager"""
        try:
            # Initialize dependencies if not provided
            if not self.redis_client:
                self.redis_client = await get_redis_client()
            if not self.sqlite_manager:
                self.sqlite_manager = SQLiteManager()
                await self.sqlite_manager.initialize()
            if not self.event_bus:
                from autom8.storage.redis.events import EventBus
                self.event_bus = EventBus(redis_client=self.redis_client)
                await self.event_bus.start()
            
            # Set up database schema
            await self._setup_database_schema()
            
            # Load existing budgets and policies
            await self._load_budgets_from_storage()
            await self._load_policies_from_storage()
            
            # Initialize forecasting models
            self._initialize_forecasting_models()
            
            # Start background tasks
            asyncio.create_task(self._background_monitoring())
            asyncio.create_task(self._background_cleanup())
            
            self._initialized = True
            logger.info("BudgetManager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize BudgetManager: {e}")
            return False
    
    async def _setup_database_schema(self):
        """Set up database tables for budget management"""
        conn = await self.sqlite_manager._get_connection()
        
        # Budgets table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS budgets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                budget_type TEXT NOT NULL,
                scope TEXT NOT NULL,
                limit_amount REAL NOT NULL,
                spent_amount REAL DEFAULT 0.0,
                start_date TIMESTAMP NOT NULL,
                end_date TIMESTAMP,
                reset_period TEXT,
                alert_thresholds TEXT NOT NULL,
                triggered_alerts TEXT DEFAULT '[]',
                enforcement_action TEXT NOT NULL,
                enforce_hard_limit BOOLEAN DEFAULT TRUE,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT DEFAULT '{}'
            )
        """)
        
        # Spending records table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS spending_records (
                id TEXT PRIMARY KEY,
                amount REAL NOT NULL,
                currency TEXT DEFAULT 'USD',
                model_name TEXT NOT NULL,
                provider TEXT NOT NULL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                user_id TEXT,
                project_id TEXT,
                session_id TEXT,
                query_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                duration_ms INTEGER,
                success BOOLEAN DEFAULT TRUE,
                error_message TEXT,
                metadata TEXT DEFAULT '{}'
            )
        """)
        
        # Budget alerts table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS budget_alerts (
                id TEXT PRIMARY KEY,
                budget_id TEXT NOT NULL,
                level TEXT NOT NULL,
                threshold REAL NOT NULL,
                current_spending REAL NOT NULL,
                budget_limit REAL NOT NULL,
                utilization_percentage REAL NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                recommended_actions TEXT DEFAULT '[]',
                enforcement_actions TEXT DEFAULT '[]',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                acknowledged BOOLEAN DEFAULT FALSE,
                acknowledged_at TIMESTAMP,
                acknowledged_by TEXT,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (budget_id) REFERENCES budgets (id)
            )
        """)
        
        # Budget policies table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS budget_policies (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                applies_to TEXT DEFAULT '{}',
                auto_create_budgets BOOLEAN DEFAULT FALSE,
                default_budget_limits TEXT DEFAULT '{}',
                max_budget_limits TEXT DEFAULT '{}',
                require_approval_over REAL,
                approval_required_for TEXT DEFAULT '[]',
                auto_downgrade_models BOOLEAN DEFAULT TRUE,
                model_downgrade_thresholds TEXT DEFAULT '{}',
                notification_channels TEXT DEFAULT '[]',
                alert_frequency_limits TEXT DEFAULT '{}',
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for performance
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_budgets_type_scope ON budgets (budget_type, scope)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_spending_user_timestamp ON spending_records (user_id, timestamp)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_spending_project_timestamp ON spending_records (project_id, timestamp)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_spending_model_timestamp ON spending_records (model_name, timestamp)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_budget_timestamp ON budget_alerts (budget_id, timestamp)")
        
        await conn.commit()
        
    async def _load_budgets_from_storage(self):
        """Load existing budgets from storage"""
        conn = await self.sqlite_manager._get_connection()
        cursor = await conn.execute("SELECT * FROM budgets WHERE is_active = TRUE")
        rows = await cursor.fetchall()
        
        for row in rows:
            try:
                budget_data = dict(row)
                # Deserialize JSON fields
                budget_data['scope'] = json.loads(budget_data['scope'])
                budget_data['alert_thresholds'] = json.loads(budget_data['alert_thresholds'])
                budget_data['triggered_alerts'] = set(json.loads(budget_data['triggered_alerts']))
                budget_data['metadata'] = json.loads(budget_data['metadata'])
                
                budget = Budget(**budget_data)
                self.budgets[budget.id] = budget
                
            except Exception as e:
                logger.error(f"Failed to load budget {row['id']}: {e}")
    
    async def _load_policies_from_storage(self):
        """Load existing policies from storage"""
        conn = await self.sqlite_manager._get_connection()
        cursor = await conn.execute("SELECT * FROM budget_policies WHERE is_active = TRUE")
        rows = await cursor.fetchall()
        
        for row in rows:
            try:
                policy_data = dict(row)
                # Deserialize JSON fields
                policy_data['applies_to'] = json.loads(policy_data['applies_to'])
                policy_data['default_budget_limits'] = json.loads(policy_data['default_budget_limits'])
                policy_data['max_budget_limits'] = json.loads(policy_data['max_budget_limits'])
                policy_data['approval_required_for'] = json.loads(policy_data['approval_required_for'])
                policy_data['model_downgrade_thresholds'] = json.loads(policy_data['model_downgrade_thresholds'])
                policy_data['notification_channels'] = json.loads(policy_data['notification_channels'])
                policy_data['alert_frequency_limits'] = json.loads(policy_data['alert_frequency_limits'])
                
                policy = BudgetPolicy(**policy_data)
                self.policies[policy.id] = policy
                
            except Exception as e:
                logger.error(f"Failed to load policy {row['id']}: {e}")
    
    def _initialize_forecasting_models(self):
        """Initialize cost forecasting models"""
        # Simple models for now - can be enhanced with ML later
        self.forecasting_models = {
            'linear_trend': self._linear_trend_forecast,
            'moving_average': self._moving_average_forecast,
            'exponential_smoothing': self._exponential_smoothing_forecast
        }
    
    async def _background_monitoring(self):
        """Background task for continuous budget monitoring"""
        while True:
            try:
                await self._check_budget_thresholds()
                await self._update_budget_utilization()
                await self._enforce_budget_policies()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in background monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _background_cleanup(self):
        """Background task for cleanup and maintenance"""
        while True:
            try:
                current_time = time.time()
                if current_time - self._last_cleanup > 3600:  # Cleanup every hour
                    await self._cleanup_expired_budgets()
                    await self._archive_old_spending_records()
                    await self._reset_periodic_budgets()
                    self._last_cleanup = current_time
                
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Error in background cleanup: {e}")
                await asyncio.sleep(300)
    
    # Budget Management Methods
    
    async def create_budget(self, budget: Budget) -> str:
        """Create a new budget"""
        # Validate budget
        if budget.id in self.budgets:
            raise ValueError(f"Budget with ID {budget.id} already exists")
        
        # Check policies
        applicable_policies = await self._get_applicable_policies(budget)
        for policy in applicable_policies:
            await self._validate_budget_against_policy(budget, policy)
        
        # Store budget
        await self._save_budget_to_storage(budget)
        self.budgets[budget.id] = budget
        
        # Emit event
        if self.event_bus:
            await self.event_bus.publish(
                event_type="budget.created",
                data={
                    "budget_id": budget.id,
                    "budget_type": budget.budget_type,
                    "limit_amount": budget.limit_amount,
                    "scope": budget.scope
                },
                source_agent="budget_manager"
            )
        
        logger.info(f"Created budget {budget.id}: {budget.name}")
        return budget.id
    
    async def update_budget(self, budget_id: str, updates: Dict[str, Any]) -> None:
        """Update an existing budget"""
        if budget_id not in self.budgets:
            raise ValueError(f"Budget {budget_id} not found")
        
        budget = self.budgets[budget_id]
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(budget, key):
                setattr(budget, key, value)
        
        budget.updated_at = datetime.utcnow()
        
        # Save to storage
        await self._save_budget_to_storage(budget)
        
        logger.info(f"Updated budget {budget_id}")
    
    async def delete_budget(self, budget_id: str) -> None:
        """Delete a budget"""
        if budget_id not in self.budgets:
            raise ValueError(f"Budget {budget_id} not found")
        
        # Mark as inactive instead of deleting
        budget = self.budgets[budget_id]
        budget.is_active = False
        budget.updated_at = datetime.utcnow()
        
        await self._save_budget_to_storage(budget)
        del self.budgets[budget_id]
        
        logger.info(f"Deleted budget {budget_id}")
    
    async def get_budget(self, budget_id: str) -> Optional[Budget]:
        """Get a budget by ID"""
        return self.budgets.get(budget_id)
    
    async def list_budgets(self, filters: Optional[Dict[str, Any]] = None) -> List[Budget]:
        """List budgets with optional filters"""
        budgets = list(self.budgets.values())
        
        if filters:
            # Apply filters
            if 'budget_type' in filters:
                budgets = [b for b in budgets if b.budget_type == filters['budget_type']]
            if 'user_id' in filters:
                budgets = [b for b in budgets if b.scope.get('user_id') == filters['user_id']]
            if 'project_id' in filters:
                budgets = [b for b in budgets if b.scope.get('project_id') == filters['project_id']]
            if 'is_active' in filters:
                budgets = [b for b in budgets if b.is_active == filters['is_active']]
        
        return budgets
    
    # Cost Tracking Methods
    
    async def record_spending(self, spending: SpendingRecord) -> None:
        """Record a spending transaction"""
        # Save to storage
        await self._save_spending_to_storage(spending)
        
        # Update relevant budgets
        await self._update_budgets_for_spending(spending)
        
        # Cache for real-time monitoring
        cache_key = f"recent_spending:{spending.user_id or 'system'}"
        self.active_spending_cache[cache_key].append(spending)
        
        # Keep only recent records in cache
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.active_spending_cache[cache_key] = [
            s for s in self.active_spending_cache[cache_key] 
            if s.timestamp > cutoff_time
        ]
        
        logger.debug(f"Recorded spending: ${spending.amount} for {spending.model_name}")
    
    async def estimate_query_cost(self, model: Model, input_tokens: int, 
                                estimated_output_tokens: int = 0) -> float:
        """Estimate cost for a query before execution"""
        return model.estimate_cost(input_tokens, estimated_output_tokens)
    
    async def can_afford_query(self, user_id: Optional[str], project_id: Optional[str],
                             estimated_cost: float) -> Tuple[bool, List[str]]:
        """Check if a query can be afforded within budget constraints"""
        reasons = []
        
        # Check user budgets
        if user_id:
            user_budgets = await self.list_budgets({'user_id': user_id})
            for budget in user_budgets:
                if budget.remaining_amount < estimated_cost:
                    reasons.append(f"User budget '{budget.name}' insufficient")
        
        # Check project budgets
        if project_id:
            project_budgets = await self.list_budgets({'project_id': project_id})
            for budget in project_budgets:
                if budget.remaining_amount < estimated_cost:
                    reasons.append(f"Project budget '{budget.name}' insufficient")
        
        # Check daily/weekly/monthly budgets
        daily_budgets = await self.list_budgets({'budget_type': BudgetType.DAILY})
        for budget in daily_budgets:
            if self._budget_applies_to_context(budget, user_id, project_id):
                if budget.remaining_amount < estimated_cost:
                    reasons.append(f"Daily budget '{budget.name}' insufficient")
        
        return len(reasons) == 0, reasons
    
    # Alert and Monitoring Methods
    
    async def _check_budget_thresholds(self):
        """Check all budgets for threshold violations"""
        for budget in self.budgets.values():
            if not budget.is_active:
                continue
            
            # Check each alert threshold
            for threshold in budget.alert_thresholds:
                if budget.should_trigger_alert(threshold):
                    await self._trigger_budget_alert(budget, threshold)
    
    async def _trigger_budget_alert(self, budget: Budget, threshold: float):
        """Trigger a budget alert"""
        # Determine alert level
        if threshold >= 1.0:
            level = AlertLevel.EMERGENCY
        elif threshold >= 0.9:
            level = AlertLevel.CRITICAL
        elif threshold >= 0.75:
            level = AlertLevel.WARNING
        else:
            level = AlertLevel.INFO
        
        # Create alert
        alert = BudgetAlert(
            budget_id=budget.id,
            level=level,
            threshold=threshold,
            current_spending=budget.spent_amount,
            budget_limit=budget.limit_amount,
            utilization_percentage=budget.utilization_percentage,
            title=f"Budget Alert: {budget.name}",
            message=f"Budget '{budget.name}' has reached {threshold*100:.1f}% utilization",
            recommended_actions=self._get_recommended_actions(budget, threshold)
        )
        
        # Save alert
        await self._save_alert_to_storage(alert)
        
        # Mark threshold as triggered
        budget.trigger_alert(threshold)
        await self._save_budget_to_storage(budget)
        
        # Send notifications
        await self._send_alert_notifications(alert, budget)
        
        # Execute enforcement actions
        await self._execute_enforcement_actions(budget, alert)
        
        logger.warning(f"Budget alert triggered: {alert.title}")
    
    def _get_recommended_actions(self, budget: Budget, threshold: float) -> List[str]:
        """Get recommended actions for a budget alert"""
        actions = []
        
        if threshold >= 0.9:
            actions.extend([
                "Consider switching to more cost-effective models",
                "Review recent spending patterns",
                "Implement request throttling",
                "Request budget increase if necessary"
            ])
        elif threshold >= 0.75:
            actions.extend([
                "Monitor spending more closely",
                "Consider optimizing query complexity",
                "Review model selection policies"
            ])
        else:
            actions.append("Continue monitoring budget utilization")
        
        return actions
    
    async def _send_alert_notifications(self, alert: BudgetAlert, budget: Budget):
        """Send alert notifications through configured channels"""
        if self.event_bus:
            await self.event_bus.publish(
                event_type="budget.alert",
                data={
                    "alert_id": alert.id,
                    "budget_id": budget.id,
                    "level": alert.level,
                    "message": alert.message,
                    "utilization": alert.utilization_percentage
                },
                source_agent="budget_manager"
            )
    
    async def _execute_enforcement_actions(self, budget: Budget, alert: BudgetAlert):
        """Execute enforcement actions based on budget policy"""
        action = budget.enforcement_action
        
        if action == EnforcementAction.WARN:
            # Already handled by alert
            return
        elif action == EnforcementAction.BLOCK:
            # Set budget to block further requests
            await self._set_budget_blocked(budget.id, True)
        elif action == EnforcementAction.DOWNGRADE:
            # Signal model router to use cheaper models
            await self._signal_model_downgrade(budget)
        elif action == EnforcementAction.THROTTLE:
            # Implement rate limiting
            await self._implement_rate_limiting(budget)
        
        alert.enforcement_actions.append(action)
    
    # Forecasting Methods
    
    async def generate_cost_forecast(self, budget_id: str,
                                   forecast_days: int = 30) -> CostForecast:
        """Generate cost forecast for a budget with robust error handling"""
        # Input validation
        if not budget_id:
            raise ValueError("Budget ID is required")

        if forecast_days <= 0:
            raise ValueError("Forecast days must be positive")

        budget = self.budgets.get(budget_id)
        if not budget:
            raise ValueError(f"Budget {budget_id} not found")

        try:
            # Get historical spending data
            historical_data = await self._get_historical_spending(budget, days=90)

            # Validate historical data
            if not isinstance(historical_data, list):
                historical_data = []

            # Initialize defaults
            predicted_cost = 0.0
            confidence = 0.1
            daily_trend = 0.0
            weekly_trend = 0.0
            growth_rate = 0.0

            if len(historical_data) < 7:  # Need at least a week of data
                try:
                    # Use simple linear projection based on current spending
                    days_elapsed = max(1, (datetime.utcnow() - budget.start_date).days)
                    if days_elapsed > 0 and budget.spent_amount >= 0:
                        daily_avg = budget.spent_amount / days_elapsed
                        predicted_cost = daily_avg * forecast_days
                        confidence = 0.3  # Low confidence
                    else:
                        predicted_cost = 0.0
                        confidence = 0.1
                except Exception as e:
                    logger.warning(f"Simple forecast calculation failed: {e}")
                    predicted_cost = 0.0
                    confidence = 0.1
            else:
                try:
                    # Use more sophisticated forecasting
                    predicted_cost, confidence = await self._advanced_forecast(historical_data, forecast_days)
                except Exception as e:
                    logger.warning(f"Advanced forecast failed: {e}. Using fallback.")
                    # Fallback to simple average
                    try:
                        valid_data = [x for x in historical_data if isinstance(x, (int, float)) and not np.isnan(x) and not np.isinf(x)]
                        if valid_data:
                            daily_avg = sum(valid_data) / len(valid_data)
                            predicted_cost = daily_avg * forecast_days
                            confidence = 0.2
                    except Exception:
                        predicted_cost = 0.0
                        confidence = 0.1

            # Calculate confidence interval with bounds checking
            try:
                margin = predicted_cost * (1 - confidence) * 0.5
                if np.isnan(margin) or np.isinf(margin):
                    margin = predicted_cost * 0.25  # Default 25% margin
                confidence_interval = (max(0.0, predicted_cost - margin), predicted_cost + margin)
            except Exception as e:
                logger.warning(f"Confidence interval calculation failed: {e}")
                confidence_interval = (0.0, predicted_cost * 1.5)

            # Calculate trends with error handling
            try:
                daily_trend = await self._calculate_daily_trend(historical_data)
            except Exception as e:
                logger.warning(f"Daily trend calculation failed: {e}")
                daily_trend = 0.0

            try:
                weekly_trend = await self._calculate_weekly_trend(historical_data)
            except Exception as e:
                logger.warning(f"Weekly trend calculation failed: {e}")
                weekly_trend = 0.0

            try:
                growth_rate = await self._calculate_growth_rate(historical_data)
            except Exception as e:
                logger.warning(f"Growth rate calculation failed: {e}")
                growth_rate = 0.0

            # Risk assessment
            try:
                risk_level = self._assess_forecast_risk(predicted_cost, budget, growth_rate)
            except Exception as e:
                logger.warning(f"Risk assessment failed: {e}")
                risk_level = "medium"

            # Predict budget exhaustion
            exhaustion_date = None
            try:
                if predicted_cost > 0 and forecast_days > 0:
                    remaining_budget = budget.remaining_amount
                    if remaining_budget > 0:
                        daily_predicted_cost = predicted_cost / forecast_days
                        if daily_predicted_cost > 0:
                            days_to_exhaustion = remaining_budget / daily_predicted_cost
                            if days_to_exhaustion > 0 and not np.isinf(days_to_exhaustion):
                                exhaustion_date = datetime.utcnow() + timedelta(days=days_to_exhaustion)
            except Exception as e:
                logger.warning(f"Exhaustion date calculation failed: {e}")
                exhaustion_date = None

            # Validate all numeric values before creating forecast
            predicted_cost = max(0.0, float(predicted_cost)) if not np.isnan(predicted_cost) and not np.isinf(predicted_cost) else 0.0
            confidence = max(0.0, min(1.0, float(confidence))) if not np.isnan(confidence) and not np.isinf(confidence) else 0.1
            daily_trend = float(daily_trend) if not np.isnan(daily_trend) and not np.isinf(daily_trend) else 0.0
            weekly_trend = float(weekly_trend) if not np.isnan(weekly_trend) and not np.isinf(weekly_trend) else 0.0
            growth_rate = float(growth_rate) if not np.isnan(growth_rate) and not np.isinf(growth_rate) else 0.0

            forecast = CostForecast(
                budget_id=budget_id,
                forecast_period_days=forecast_days,
                predicted_cost=predicted_cost,
                confidence_interval=confidence_interval,
                confidence_score=confidence,
                daily_trend=daily_trend,
                weekly_trend=weekly_trend,
                growth_rate=growth_rate,
                risk_level=risk_level,
                budget_exhaustion_date=exhaustion_date,
                model_type="hybrid",
                data_points=len(historical_data)
            )

            return forecast

        except Exception as e:
            logger.error(f"Cost forecast generation failed for budget {budget_id}: {e}")
            # Return a safe default forecast
            return CostForecast(
                budget_id=budget_id,
                forecast_period_days=forecast_days,
                predicted_cost=0.0,
                confidence_interval=(0.0, 0.0),
                confidence_score=0.1,
                daily_trend=0.0,
                weekly_trend=0.0,
                growth_rate=0.0,
                risk_level="low",
                budget_exhaustion_date=None,
                model_type="fallback",
                data_points=0
            )
    
    def _linear_trend_forecast(self, data: List[float], forecast_days: int) -> Tuple[float, float]:
        """Simple linear trend forecasting with robust error handling"""
        # Input validation
        if not data or len(data) < 2:
            return 0.0, 0.1

        if forecast_days <= 0:
            return 0.0, 0.1

        try:
            # Calculate linear trend
            x = np.arange(len(data))
            y = np.array(data, dtype=float)

            # Check for edge cases
            if len(x) == 0 or len(y) == 0:
                return 0.0, 0.1

            # Check for all zeros or constant values
            if np.all(y == 0) or np.std(y) == 0:
                return np.mean(y) * forecast_days, 0.3

            # Check for NaN or inf values
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                # Filter out invalid values
                valid_mask = np.isfinite(y)
                if not np.any(valid_mask):
                    return 0.0, 0.1
                x = x[valid_mask]
                y = y[valid_mask]

                if len(x) < 2:
                    return 0.0, 0.1

            # Simple linear regression with error handling
            try:
                coeffs = np.polyfit(x, y, 1)
                slope = coeffs[0]
            except (np.linalg.LinAlgError, ValueError, RuntimeWarning) as e:
                logger.warning(f"Linear regression failed: {e}. Using mean as fallback.")
                # Fallback to simple mean projection
                slope = np.mean(y)
                predicted_cost = slope * forecast_days
                confidence = 0.2  # Low confidence due to fallback
                return max(0.0, predicted_cost), confidence

            # Project forward
            predicted_cost = slope * forecast_days
            confidence = min(0.8, len(data) / 30)  # Higher confidence with more data

            # Ensure reasonable bounds
            if np.isnan(predicted_cost) or np.isinf(predicted_cost):
                predicted_cost = np.mean(y) * forecast_days
                confidence = 0.2

            return max(0.0, predicted_cost), confidence

        except Exception as e:
            logger.error(f"Linear trend forecasting failed: {e}")
            return 0.0, 0.1
    
    def _moving_average_forecast(self, data: List[float], forecast_days: int) -> Tuple[float, float]:
        """Moving average forecasting with robust error handling"""
        # Input validation
        if not data or len(data) < 1:
            return 0.0, 0.1

        if forecast_days <= 0:
            return 0.0, 0.1

        try:
            # Convert to numpy array for safety
            data_array = np.array(data, dtype=float)

            # Check for edge cases
            if len(data_array) == 0:
                return 0.0, 0.1

            # Filter out NaN and inf values
            valid_data = data_array[np.isfinite(data_array)]

            if len(valid_data) == 0:
                return 0.0, 0.1

            # Determine window size based on available data
            window_size = min(7, len(valid_data))

            if window_size < 1:
                return 0.0, 0.1

            # Use last N days as moving average
            recent_data = valid_data[-window_size:]

            try:
                recent_avg = np.mean(recent_data)
            except (RuntimeWarning, ValueError) as e:
                logger.warning(f"Moving average calculation failed: {e}")
                recent_avg = 0.0

            # Check for invalid average
            if np.isnan(recent_avg) or np.isinf(recent_avg):
                recent_avg = 0.0

            predicted_cost = recent_avg * forecast_days

            # Adjust confidence based on data quality and quantity
            if len(data) >= 7:
                confidence = min(0.7, len(data) / 21)
            else:
                confidence = min(0.5, len(data) / 14)  # Lower confidence for less data

            return max(0.0, predicted_cost), confidence

        except Exception as e:
            logger.error(f"Moving average forecasting failed: {e}")
            return 0.0, 0.1
    
    def _exponential_smoothing_forecast(self, data: List[float], forecast_days: int) -> Tuple[float, float]:
        """Exponential smoothing forecasting with robust error handling"""
        # Input validation
        if not data or len(data) < 1:
            return 0.0, 0.1

        if forecast_days <= 0:
            return 0.0, 0.1

        try:
            # Convert to numpy array and filter invalid values
            data_array = np.array(data, dtype=float)
            valid_data = data_array[np.isfinite(data_array)]

            if len(valid_data) == 0:
                return 0.0, 0.1

            # For very small datasets, use simple average
            if len(valid_data) < 3:
                try:
                    avg = np.mean(valid_data)
                    if np.isnan(avg) or np.isinf(avg):
                        avg = 0.0
                    return max(0.0, avg * forecast_days), 0.2
                except Exception:
                    return 0.0, 0.1

            alpha = 0.3  # Smoothing parameter

            try:
                # Initialize with first valid value
                smoothed = [float(valid_data[0])]

                # Apply exponential smoothing
                for i in range(1, len(valid_data)):
                    current_value = float(valid_data[i])
                    previous_smoothed = smoothed[i-1]

                    # Check for invalid values
                    if np.isnan(current_value) or np.isinf(current_value):
                        smoothed.append(previous_smoothed)
                        continue

                    if np.isnan(previous_smoothed) or np.isinf(previous_smoothed):
                        smoothed.append(current_value)
                        continue

                    new_smoothed = alpha * current_value + (1 - alpha) * previous_smoothed

                    # Validate result
                    if np.isnan(new_smoothed) or np.isinf(new_smoothed):
                        new_smoothed = current_value

                    smoothed.append(new_smoothed)

                # Use last smoothed value for prediction
                if not smoothed:
                    return 0.0, 0.1

                daily_avg = smoothed[-1]

                # Validate final average
                if np.isnan(daily_avg) or np.isinf(daily_avg):
                    daily_avg = np.mean(valid_data) if len(valid_data) > 0 else 0.0
                    if np.isnan(daily_avg) or np.isinf(daily_avg):
                        daily_avg = 0.0

                predicted_cost = daily_avg * forecast_days
                confidence = min(0.8, len(data) / 30)

                return max(0.0, predicted_cost), confidence

            except (ValueError, RuntimeWarning, OverflowError) as e:
                logger.warning(f"Exponential smoothing calculation failed: {e}. Using mean as fallback.")
                # Fallback to simple mean
                try:
                    mean_value = np.mean(valid_data)
                    if np.isnan(mean_value) or np.isinf(mean_value):
                        mean_value = 0.0
                    return max(0.0, mean_value * forecast_days), 0.2
                except Exception:
                    return 0.0, 0.1

        except Exception as e:
            logger.error(f"Exponential smoothing forecasting failed: {e}")
            return 0.0, 0.1
    
    # Helper Methods
    
    async def _save_budget_to_storage(self, budget: Budget):
        """Save budget to SQLite storage"""
        conn = await self.sqlite_manager._get_connection()
        
        await conn.execute("""
            INSERT OR REPLACE INTO budgets 
            (id, name, description, budget_type, scope, limit_amount, spent_amount,
             start_date, end_date, reset_period, alert_thresholds, triggered_alerts,
             enforcement_action, enforce_hard_limit, is_active, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            budget.id, budget.name, budget.description, budget.budget_type.value,
            json.dumps(budget.scope), budget.limit_amount, budget.spent_amount,
            budget.start_date, budget.end_date, budget.reset_period,
            json.dumps(budget.alert_thresholds), json.dumps(list(budget.triggered_alerts)),
            budget.enforcement_action.value, budget.enforce_hard_limit,
            budget.is_active, budget.created_at, budget.updated_at,
            json.dumps(budget.metadata)
        ))
        
        await conn.commit()
    
    async def _save_spending_to_storage(self, spending: SpendingRecord):
        """Save spending record to SQLite storage"""
        conn = await self.sqlite_manager._get_connection()
        
        await conn.execute("""
            INSERT INTO spending_records 
            (id, amount, currency, model_name, provider, input_tokens, output_tokens,
             user_id, project_id, session_id, query_id, timestamp, duration_ms,
             success, error_message, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            spending.id, spending.amount, spending.currency, spending.model_name,
            spending.provider.value, spending.input_tokens, spending.output_tokens,
            spending.user_id, spending.project_id, spending.session_id, spending.query_id,
            spending.timestamp, spending.duration_ms, spending.success,
            spending.error_message, json.dumps(spending.metadata)
        ))
        
        await conn.commit()
    
    async def _save_alert_to_storage(self, alert: BudgetAlert):
        """Save budget alert to SQLite storage"""
        conn = await self.sqlite_manager._get_connection()
        
        await conn.execute("""
            INSERT INTO budget_alerts 
            (id, budget_id, level, threshold, current_spending, budget_limit,
             utilization_percentage, title, message, recommended_actions,
             enforcement_actions, timestamp, acknowledged, acknowledged_at,
             acknowledged_by, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.id, alert.budget_id, alert.level.value, alert.threshold,
            alert.current_spending, alert.budget_limit, alert.utilization_percentage,
            alert.title, alert.message, json.dumps(alert.recommended_actions),
            json.dumps([a.value for a in alert.enforcement_actions]),
            alert.timestamp, alert.acknowledged, alert.acknowledged_at,
            alert.acknowledged_by, json.dumps(alert.metadata)
        ))
        
        await conn.commit()
    
    async def _update_budgets_for_spending(self, spending: SpendingRecord):
        """Update relevant budgets when spending occurs"""
        # Find applicable budgets
        applicable_budgets = []
        
        for budget in self.budgets.values():
            if not budget.is_active:
                continue
                
            if self._budget_applies_to_spending(budget, spending):
                applicable_budgets.append(budget)
        
        # Update budget spending amounts
        for budget in applicable_budgets:
            budget.spent_amount += spending.amount
            budget.updated_at = datetime.utcnow()
            await self._save_budget_to_storage(budget)
    
    def _budget_applies_to_spending(self, budget: Budget, spending: SpendingRecord) -> bool:
        """Check if a budget applies to a spending record"""
        # Check budget type and scope
        if budget.budget_type == BudgetType.USER:
            return budget.scope.get('user_id') == spending.user_id
        elif budget.budget_type == BudgetType.PROJECT:
            return budget.scope.get('project_id') == spending.project_id
        elif budget.budget_type == BudgetType.MODEL:
            return budget.scope.get('model_name') == spending.model_name
        elif budget.budget_type == BudgetType.PROVIDER:
            return budget.scope.get('provider') == spending.provider.value
        elif budget.budget_type in [BudgetType.DAILY, BudgetType.WEEKLY, BudgetType.MONTHLY]:
            # Time-based budgets apply based on time window
            return self._spending_in_budget_timeframe(budget, spending)
        
        return False
    
    def _budget_applies_to_context(self, budget: Budget, user_id: Optional[str], 
                                 project_id: Optional[str]) -> bool:
        """Check if a budget applies to a query context"""
        if budget.budget_type == BudgetType.USER:
            return budget.scope.get('user_id') == user_id
        elif budget.budget_type == BudgetType.PROJECT:
            return budget.scope.get('project_id') == project_id
        elif budget.budget_type in [BudgetType.DAILY, BudgetType.WEEKLY, BudgetType.MONTHLY]:
            # Time-based budgets apply to current time
            return True
        
        return False
    
    def _spending_in_budget_timeframe(self, budget: Budget, spending: SpendingRecord) -> bool:
        """Check if spending falls within budget timeframe"""
        now = datetime.utcnow()
        
        if budget.budget_type == BudgetType.DAILY:
            return spending.timestamp.date() == now.date()
        elif budget.budget_type == BudgetType.WEEKLY:
            # Check if in same week
            week_start = now - timedelta(days=now.weekday())
            return spending.timestamp >= week_start
        elif budget.budget_type == BudgetType.MONTHLY:
            # Check if in same month
            return (spending.timestamp.year == now.year and 
                   spending.timestamp.month == now.month)
        
        return False
    
    async def _get_applicable_policies(self, budget: Budget) -> List[BudgetPolicy]:
        """Get policies that apply to a budget"""
        applicable = []
        
        for policy in self.policies.values():
            if not policy.is_active:
                continue
            
            # Check if policy applies
            applies_to = policy.applies_to
            if not applies_to:  # Empty means applies to all
                applicable.append(policy)
                continue
            
            # Check specific criteria
            if applies_to.get('budget_type') == budget.budget_type.value:
                applicable.append(policy)
            elif applies_to.get('user_id') == budget.scope.get('user_id'):
                applicable.append(policy)
            elif applies_to.get('project_id') == budget.scope.get('project_id'):
                applicable.append(policy)
        
        return applicable
    
    async def _validate_budget_against_policy(self, budget: Budget, policy: BudgetPolicy):
        """Validate budget against policy constraints"""
        # Check maximum limits
        max_limits = policy.max_budget_limits
        budget_type_key = budget.budget_type.value
        
        if budget_type_key in max_limits:
            max_limit = max_limits[budget_type_key]
            if budget.limit_amount > max_limit:
                raise ValueError(f"Budget limit ${budget.limit_amount} exceeds policy maximum ${max_limit}")
        
        # Check approval requirements
        if policy.require_approval_over and budget.limit_amount > policy.require_approval_over:
            # Would need approval workflow integration here
            logger.warning(f"Budget {budget.id} requires approval due to policy")
    
    async def _get_historical_spending(self, budget: Budget, days: int = 30) -> List[float]:
        """Get historical daily spending for a budget"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        conn = await self.sqlite_manager._get_connection()
        
        # Build query based on budget type
        where_clause = "WHERE timestamp >= ? AND timestamp <= ?"
        params = [start_date, end_date]
        
        if budget.budget_type == BudgetType.USER:
            where_clause += " AND user_id = ?"
            params.append(budget.scope.get('user_id'))
        elif budget.budget_type == BudgetType.PROJECT:
            where_clause += " AND project_id = ?"
            params.append(budget.scope.get('project_id'))
        elif budget.budget_type == BudgetType.MODEL:
            where_clause += " AND model_name = ?"
            params.append(budget.scope.get('model_name'))
        elif budget.budget_type == BudgetType.PROVIDER:
            where_clause += " AND provider = ?"
            params.append(budget.scope.get('provider'))
        
        cursor = await conn.execute(f"""
            SELECT DATE(timestamp) as day, SUM(amount) as daily_total
            FROM spending_records 
            {where_clause}
            GROUP BY DATE(timestamp)
            ORDER BY day
        """, params)
        
        rows = await cursor.fetchall()
        
        # Convert to list of daily totals
        daily_spending = {}
        for row in rows:
            daily_spending[row[0]] = row[1]
        
        # Fill in missing days with 0
        result = []
        current_date = start_date.date()
        while current_date <= end_date.date():
            day_str = current_date.isoformat()
            result.append(daily_spending.get(day_str, 0.0))
            current_date += timedelta(days=1)
        
        return result
    
    async def _advanced_forecast(self, historical_data: List[float],
                               forecast_days: int) -> Tuple[float, float]:
        """Advanced forecasting using multiple methods with robust error handling"""
        # Input validation
        if not historical_data or forecast_days <= 0:
            return 0.0, 0.1

        forecasts = []
        confidences = []

        # Try each forecasting method
        for method_name, method in self.forecasting_models.items():
            try:
                pred, conf = method(historical_data, forecast_days)

                # Validate results
                if (isinstance(pred, (int, float)) and isinstance(conf, (int, float)) and
                    not np.isnan(pred) and not np.isinf(pred) and
                    not np.isnan(conf) and not np.isinf(conf) and
                    pred >= 0 and 0 <= conf <= 1):
                    forecasts.append(float(pred))
                    confidences.append(float(conf))
                else:
                    logger.warning(f"Forecasting method {method_name} returned invalid results: pred={pred}, conf={conf}")

            except Exception as e:
                logger.warning(f"Forecasting method {method_name} failed: {e}")

        if not forecasts:
            logger.warning("All forecasting methods failed, using fallback")
            return 0.0, 0.1

        try:
            # Ensemble prediction (weighted average)
            confidence_sum = sum(confidences)

            if confidence_sum > 0:
                weights = np.array(confidences, dtype=float) / confidence_sum
            else:
                weights = np.ones(len(forecasts), dtype=float) / len(forecasts)

            # Validate weights
            if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                weights = np.ones(len(forecasts), dtype=float) / len(forecasts)

            forecasts_array = np.array(forecasts, dtype=float)
            ensemble_prediction = np.average(forecasts_array, weights=weights)
            ensemble_confidence = np.mean(confidences)

            # Final validation
            if np.isnan(ensemble_prediction) or np.isinf(ensemble_prediction):
                ensemble_prediction = np.mean(forecasts_array)
                if np.isnan(ensemble_prediction) or np.isinf(ensemble_prediction):
                    ensemble_prediction = 0.0

            if np.isnan(ensemble_confidence) or np.isinf(ensemble_confidence):
                ensemble_confidence = 0.1

            return max(0.0, float(ensemble_prediction)), max(0.0, min(1.0, float(ensemble_confidence)))

        except Exception as e:
            logger.error(f"Ensemble forecasting failed: {e}")
            # Fallback to simple average
            try:
                if forecasts:
                    simple_avg = sum(forecasts) / len(forecasts)
                    simple_conf = sum(confidences) / len(confidences) if confidences else 0.1
                    return max(0.0, simple_avg), max(0.0, min(1.0, simple_conf))
            except Exception:
                pass

            return 0.0, 0.1
    
    async def _calculate_daily_trend(self, data: List[float]) -> float:
        """Calculate daily spending trend with robust error handling"""
        if not data or len(data) < 2:
            return 0.0

        try:
            # Convert to numpy array and filter invalid values
            data_array = np.array(data, dtype=float)
            valid_data = data_array[np.isfinite(data_array)]

            if len(valid_data) < 2:
                return 0.0

            # Need at least 7 days for meaningful trend
            if len(valid_data) < 7:
                return 0.0

            # Compare recent week to previous week
            try:
                recent_week = np.mean(valid_data[-7:])
                if len(valid_data) >= 14:
                    previous_week = np.mean(valid_data[-14:-7])
                else:
                    previous_week = np.mean(valid_data[:-7]) if len(valid_data) > 7 else 0.0

                # Validate calculated means
                if (np.isnan(recent_week) or np.isinf(recent_week) or
                    np.isnan(previous_week) or np.isinf(previous_week)):
                    return 0.0

                if previous_week == 0:
                    return 0.0

                trend = (recent_week - previous_week) / previous_week

                # Validate result
                if np.isnan(trend) or np.isinf(trend):
                    return 0.0

                return float(trend)

            except (ValueError, RuntimeWarning, ZeroDivisionError) as e:
                logger.warning(f"Daily trend calculation failed: {e}")
                return 0.0

        except Exception as e:
            logger.error(f"Daily trend calculation error: {e}")
            return 0.0
    
    async def _calculate_weekly_trend(self, data: List[float]) -> float:
        """Calculate weekly spending trend with robust error handling"""
        if not data or len(data) < 14:
            return 0.0

        try:
            # Convert to numpy array and filter invalid values
            data_array = np.array(data, dtype=float)
            valid_data = data_array[np.isfinite(data_array)]

            if len(valid_data) < 14:
                return 0.0

            # Split into weeks and compare
            weeks = []
            for i in range(0, len(valid_data), 7):
                week_data = valid_data[i:i+7]
                if len(week_data) > 0:  # Only add non-empty weeks
                    weeks.append(week_data)

            if len(weeks) < 2:
                return 0.0

            try:
                # Calculate averages for last two weeks
                recent_week_avg = np.mean(weeks[-1])
                previous_week_avg = np.mean(weeks[-2])

                # Validate calculated means
                if (np.isnan(recent_week_avg) or np.isinf(recent_week_avg) or
                    np.isnan(previous_week_avg) or np.isinf(previous_week_avg)):
                    return 0.0

                if previous_week_avg == 0:
                    return 0.0

                trend = (recent_week_avg - previous_week_avg) / previous_week_avg

                # Validate result
                if np.isnan(trend) or np.isinf(trend):
                    return 0.0

                return float(trend)

            except (ValueError, RuntimeWarning, ZeroDivisionError) as e:
                logger.warning(f"Weekly trend calculation failed: {e}")
                return 0.0

        except Exception as e:
            logger.error(f"Weekly trend calculation error: {e}")
            return 0.0
    
    async def _calculate_growth_rate(self, data: List[float]) -> float:
        """Calculate overall growth rate with robust error handling"""
        if not data or len(data) < 7:
            return 0.0

        try:
            # Convert to numpy array and filter invalid values
            data_array = np.array(data, dtype=float)
            valid_data = data_array[np.isfinite(data_array)]

            if len(valid_data) < 7:
                return 0.0

            try:
                first_week = np.mean(valid_data[:7])
                last_week = np.mean(valid_data[-7:])

                # Validate calculated means
                if (np.isnan(first_week) or np.isinf(first_week) or
                    np.isnan(last_week) or np.isinf(last_week)):
                    return 0.0

                if first_week <= 0:
                    return 0.0

                weeks = len(valid_data) / 7
                if weeks <= 0:
                    return 0.0

                # Calculate growth rate with bounds checking
                ratio = last_week / first_week
                if ratio <= 0 or np.isnan(ratio) or np.isinf(ratio):
                    return 0.0

                # Avoid domain errors in power calculation
                try:
                    growth_rate = (ratio ** (1 / weeks)) - 1
                except (ValueError, OverflowError, ZeroDivisionError):
                    return 0.0

                # Validate result
                if np.isnan(growth_rate) or np.isinf(growth_rate):
                    return 0.0

                # Cap extreme growth rates
                return max(-1.0, min(10.0, float(growth_rate)))

            except (ValueError, RuntimeWarning, ZeroDivisionError, OverflowError) as e:
                logger.warning(f"Growth rate calculation failed: {e}")
                return 0.0

        except Exception as e:
            logger.error(f"Growth rate calculation error: {e}")
            return 0.0
    
    def _assess_forecast_risk(self, predicted_cost: float, budget: Budget, 
                            growth_rate: float) -> str:
        """Assess risk level of forecast"""
        remaining_budget = budget.remaining_amount
        
        if predicted_cost > remaining_budget * 1.2:
            return "high"
        elif predicted_cost > remaining_budget or growth_rate > 0.5:
            return "medium"
        else:
            return "low"
    
    # Additional utility methods would go here...
    
    async def _cleanup_expired_budgets(self):
        """Clean up expired budgets"""
        current_time = datetime.utcnow()
        expired_budgets = []
        
        for budget_id, budget in self.budgets.items():
            if budget.end_date and budget.end_date < current_time:
                expired_budgets.append(budget_id)
        
        for budget_id in expired_budgets:
            budget = self.budgets[budget_id]
            budget.is_active = False
            await self._save_budget_to_storage(budget)
            del self.budgets[budget_id]
            logger.info(f"Expired budget {budget_id}")
    
    async def _archive_old_spending_records(self):
        """Archive old spending records"""
        # Could implement archiving logic here
        pass
    
    async def _reset_periodic_budgets(self):
        """Reset periodic budgets (daily, weekly, monthly)"""
        current_time = datetime.utcnow()
        
        for budget in self.budgets.values():
            if not budget.is_active or not budget.reset_period:
                continue
            
            should_reset = False
            
            if budget.reset_period == "daily":
                # Reset if it's a new day
                last_reset = budget.updated_at.date()
                should_reset = current_time.date() > last_reset
            elif budget.reset_period == "weekly":
                # Reset if it's a new week (Monday)
                days_since_monday = current_time.weekday()
                week_start = current_time - timedelta(days=days_since_monday)
                should_reset = budget.updated_at < week_start
            elif budget.reset_period == "monthly":
                # Reset if it's a new month
                should_reset = (current_time.month != budget.updated_at.month or 
                              current_time.year != budget.updated_at.year)
            
            if should_reset:
                budget.spent_amount = 0.0
                budget.triggered_alerts.clear()
                budget.updated_at = current_time
                await self._save_budget_to_storage(budget)
                logger.info(f"Reset periodic budget {budget.id}")
    
    async def _update_budget_utilization(self):
        """Update budget utilization from storage"""
        # This would sync in-memory budget state with storage
        # to handle cases where spending is recorded by other instances
        pass
    
    async def _enforce_budget_policies(self):
        """Enforce budget policies and automated actions"""
        # Implementation for policy enforcement
        pass
    
    async def _set_budget_blocked(self, budget_id: str, blocked: bool):
        """Set budget as blocked/unblocked"""
        # Store in Redis for real-time access
        if self.redis_client:
            await self.redis_client.set(f"budget_blocked:{budget_id}", str(blocked))
    
    async def _signal_model_downgrade(self, budget: Budget):
        """Signal model router to use cheaper models"""
        if self.event_bus:
            await self.event_bus.publish(
                event_type="budget.downgrade_models",
                data={
                    "budget_id": budget.id,
                    "scope": budget.scope,
                    "utilization": budget.utilization_percentage
                },
                source_agent="budget_manager"
            )
    
    async def _implement_rate_limiting(self, budget: Budget):
        """Implement rate limiting for budget"""
        # Store rate limit in Redis
        if self.redis_client:
            rate_limit = max(1, int(10 * (1 - budget.utilization_percentage / 100)))
            await self.redis_client.set(
                f"rate_limit:{budget.id}", 
                rate_limit, 
                ex=3600  # 1 hour TTL
            )