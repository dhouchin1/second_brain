"""
Ecology Tracker for Carbon Footprint Monitoring.

Implements comprehensive tracking and reporting of environmental impact
from AI model usage, supporting the PRD's ecological responsibility goals.
"""

import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass

from pydantic import BaseModel, Field
from autom8.models.routing import Model, ModelProvider
from autom8.storage.redis.client import RedisClient, get_redis_client
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class EcologyMetric(str, Enum):
    """Types of ecological metrics tracked."""
    ENERGY_CONSUMPTION = "energy_consumption"  # kWh
    CARBON_FOOTPRINT = "carbon_footprint"     # kg CO2
    WATER_USAGE = "water_usage"               # liters
    COMPUTE_TIME = "compute_time"             # seconds
    TOKEN_EFFICIENCY = "token_efficiency"     # tokens per kWh


class TimeWindow(str, Enum):
    """Time windows for ecological reporting."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class ModelEcologyProfile(BaseModel):
    model_config = {"protected_namespaces": ()}
    """Ecological profile for a specific model."""
    model_name: str = Field(description="Name of the model")
    provider: ModelProvider = Field(description="Model provider")
    
    # Energy consumption (kWh per 1000 tokens)
    energy_per_1k_tokens: float = Field(description="Energy consumption per 1000 tokens")
    
    # Carbon intensity (kg CO2 per kWh)
    carbon_intensity: float = Field(description="Carbon intensity per kWh")
    
    # Water usage (liters per kWh)
    water_intensity: float = Field(description="Water usage per kWh")
    
    # Hardware efficiency metrics
    tflops_per_watt: float = 0.0
    memory_efficiency: float = 1.0
    
    # Location-based factors
    grid_carbon_intensity: float = 0.5  # kg CO2/kWh (global average)
    renewable_percentage: float = 0.0   # Percentage renewable energy
    
    def calculate_energy_consumption(self, tokens: int) -> float:
        """Calculate energy consumption for given token count."""
        return (tokens / 1000) * self.energy_per_1k_tokens
    
    def calculate_carbon_footprint(self, energy_kwh: float) -> float:
        """Calculate carbon footprint from energy consumption."""
        effective_carbon_intensity = self.carbon_intensity * (1.0 - self.renewable_percentage)
        return energy_kwh * effective_carbon_intensity
    
    def calculate_water_usage(self, energy_kwh: float) -> float:
        """Calculate water usage from energy consumption."""
        return energy_kwh * self.water_intensity


class EcologyRecord(BaseModel):
    model_config = {"protected_namespaces": ()}
    """Single ecology tracking record."""
    
    # Identity
    id: str = Field(description="Unique record identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Model information
    model_name: str = Field(description="Name of the model used")
    provider: ModelProvider = Field(description="Model provider")
    
    # Usage metrics
    input_tokens: int = Field(description="Number of input tokens")
    output_tokens: int = Field(description="Number of output tokens")
    total_tokens: int = Field(description="Total token count")
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    
    # Ecological impact
    energy_consumption_kwh: float = Field(description="Energy consumed in kWh")
    carbon_footprint_kg: float = Field(description="Carbon footprint in kg CO2")
    water_usage_liters: float = Field(description="Water usage in liters")
    
    # Efficiency metrics
    tokens_per_kwh: float = Field(default=0.0, description="Token efficiency")
    carbon_per_token: float = Field(default=0.0, description="Carbon per token (g CO2)")
    
    # Context
    user_id: Optional[str] = Field(default=None, description="User identifier")
    project_id: Optional[str] = Field(default=None, description="Project identifier")
    task_type: Optional[str] = Field(default=None, description="Type of task")


class EcologyInsight(BaseModel):
    model_config = {"protected_namespaces": ()}
    """Ecological insight or recommendation."""
    
    type: str = Field(description="Type of insight")
    title: str = Field(description="Insight title")
    description: str = Field(description="Detailed description")
    impact_reduction: float = Field(description="Potential impact reduction (0-1)")
    urgency: int = Field(description="Urgency level (1-5)")
    actionable: bool = Field(description="Whether this is actionable")
    recommendations: List[str] = Field(default_factory=list)


class EcologyReport(BaseModel):
    model_config = {"protected_namespaces": ()}
    """Comprehensive ecology impact report."""
    
    # Report metadata
    id: str = Field(description="Report ID")
    period_start: datetime = Field(description="Report period start")
    period_end: datetime = Field(description="Report period end")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Summary metrics
    total_queries: int = Field(description="Total number of queries")
    total_tokens: int = Field(description="Total tokens processed")
    total_energy_kwh: float = Field(description="Total energy consumption")
    total_carbon_kg: float = Field(description="Total carbon footprint")
    total_water_liters: float = Field(description="Total water usage")
    
    # Efficiency metrics
    average_tokens_per_kwh: float = Field(description="Average token efficiency")
    carbon_intensity: float = Field(description="Average carbon intensity")
    
    # Breakdown by model
    model_breakdown: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    # Time series data
    time_series: Dict[str, List[float]] = Field(default_factory=dict)
    
    # Comparisons and benchmarks
    comparison_data: Dict[str, float] = Field(default_factory=dict)
    
    # Insights and recommendations
    insights: List[EcologyInsight] = Field(default_factory=list)
    
    # Carbon offset information
    offset_required_kg: float = Field(description="Carbon offset required")
    offset_cost_usd: float = Field(description="Estimated offset cost")


class EcologyTracker:
    """
    Comprehensive ecological impact tracking system.
    
    Monitors energy consumption, carbon footprint, and environmental impact
    of AI model usage with detailed analytics and optimization recommendations.
    """
    
    def __init__(self, redis_client: Optional[RedisClient] = None):
        self.redis_client = redis_client
        self._initialized = False
        
        # Namespaces for Redis storage
        self.namespaces = {
            "records": "autom8:ecology:records:",
            "aggregates": "autom8:ecology:aggregates:",
            "insights": "autom8:ecology:insights:",
            "profiles": "autom8:ecology:profiles:"
        }
        
        # Model ecology profiles
        self.model_profiles: Dict[str, ModelEcologyProfile] = {}
        self._initialize_default_profiles()
        
        # Global settings
        self.carbon_offset_cost_per_kg = 15.0  # USD per kg CO2
        self.sustainability_targets = {
            "daily_carbon_limit_kg": 1.0,
            "energy_efficiency_target": 1000.0,  # tokens per kWh
            "renewable_percentage_target": 0.5
        }
    
    async def initialize(self) -> bool:
        """Initialize the ecology tracker."""
        try:
            if not self.redis_client:
                self.redis_client = await get_redis_client()
            
            self._initialized = True
            logger.info("EcologyTracker initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize EcologyTracker: {e}")
            return False
    
    def _initialize_default_profiles(self):
        """Initialize default ecological profiles for common models."""
        
        # Local models (more efficient, no data center overhead)
        self.model_profiles.update({
            "llama3.2:3b": ModelEcologyProfile(
                model_name="llama3.2:3b",
                provider=ModelProvider.OLLAMA,
                energy_per_1k_tokens=0.02,  # Very efficient for small model
                carbon_intensity=0.4,       # Local grid average
                water_intensity=2.0,        # Minimal cooling needs
                tflops_per_watt=15.0,
                renewable_percentage=0.0    # Depends on local grid
            ),
            "llama3.2:7b": ModelEcologyProfile(
                model_name="llama3.2:7b",
                provider=ModelProvider.OLLAMA,
                energy_per_1k_tokens=0.05,
                carbon_intensity=0.4,
                water_intensity=2.0,
                tflops_per_watt=12.0,
                renewable_percentage=0.0
            ),
            "mixtral:8x7b": ModelEcologyProfile(
                model_name="mixtral:8x7b",
                provider=ModelProvider.OLLAMA,
                energy_per_1k_tokens=0.15,  # Larger model, more energy
                carbon_intensity=0.4,
                water_intensity=2.5,
                tflops_per_watt=10.0,
                renewable_percentage=0.0
            ),
            "phi-3:3.8b": ModelEcologyProfile(
                model_name="phi-3:3.8b",
                provider=ModelProvider.OLLAMA,
                energy_per_1k_tokens=0.03,
                carbon_intensity=0.4,
                water_intensity=2.0,
                tflops_per_watt=14.0,
                renewable_percentage=0.0
            )
        })
        
        # Cloud models (data center scale, potentially more renewable energy)
        self.model_profiles.update({
            "claude-haiku": ModelEcologyProfile(
                model_name="claude-haiku",
                provider=ModelProvider.ANTHROPIC,
                energy_per_1k_tokens=0.08,
                carbon_intensity=0.3,       # Data centers often greener
                water_intensity=3.0,        # More cooling needed
                tflops_per_watt=20.0,       # Efficient data center hardware
                renewable_percentage=0.6    # Many providers use renewables
            ),
            "claude-sonnet": ModelEcologyProfile(
                model_name="claude-sonnet",
                provider=ModelProvider.ANTHROPIC,
                energy_per_1k_tokens=0.25,
                carbon_intensity=0.3,
                water_intensity=4.0,
                tflops_per_watt=18.0,
                renewable_percentage=0.6
            ),
            "claude-opus": ModelEcologyProfile(
                model_name="claude-opus",
                provider=ModelProvider.ANTHROPIC,
                energy_per_1k_tokens=0.8,   # High-capability model
                carbon_intensity=0.3,
                water_intensity=6.0,
                tflops_per_watt=16.0,
                renewable_percentage=0.6
            ),
            "gpt-3.5-turbo": ModelEcologyProfile(
                model_name="gpt-3.5-turbo",
                provider=ModelProvider.OPENAI,
                energy_per_1k_tokens=0.12,
                carbon_intensity=0.35,
                water_intensity=3.5,
                tflops_per_watt=17.0,
                renewable_percentage=0.5
            ),
            "gpt-4": ModelEcologyProfile(
                model_name="gpt-4",
                provider=ModelProvider.OPENAI,
                energy_per_1k_tokens=0.6,
                carbon_intensity=0.35,
                water_intensity=5.5,
                tflops_per_watt=15.0,
                renewable_percentage=0.5
            )
        })
    
    async def record_usage(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        processing_time_ms: float,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> EcologyRecord:
        """Record ecological impact of model usage."""
        
        if not self._initialized:
            await self.initialize()
        
        # Get model profile
        profile = self.model_profiles.get(model_name)
        if not profile:
            logger.warning(f"No ecology profile for model {model_name}, using defaults")
            profile = self._create_default_profile(model_name)
        
        # Calculate ecological impact
        total_tokens = input_tokens + output_tokens
        energy_consumption = profile.calculate_energy_consumption(total_tokens)
        carbon_footprint = profile.calculate_carbon_footprint(energy_consumption)
        water_usage = profile.calculate_water_usage(energy_consumption)
        
        # Calculate efficiency metrics
        tokens_per_kwh = total_tokens / energy_consumption if energy_consumption > 0 else 0
        carbon_per_token = (carbon_footprint * 1000) / total_tokens if total_tokens > 0 else 0  # g CO2 per token
        
        # Create record
        record = EcologyRecord(
            id=f"{model_name}_{int(time.time() * 1000)}",
            model_name=model_name,
            provider=profile.provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            processing_time_ms=processing_time_ms,
            energy_consumption_kwh=energy_consumption,
            carbon_footprint_kg=carbon_footprint,
            water_usage_liters=water_usage,
            tokens_per_kwh=tokens_per_kwh,
            carbon_per_token=carbon_per_token,
            user_id=user_id,
            project_id=project_id,
            task_type=task_type
        )
        
        # Store record
        await self._store_record(record)
        
        # Update real-time aggregates
        await self._update_aggregates(record)
        
        logger.debug(f"Recorded ecology usage: {model_name}, {total_tokens} tokens, "
                    f"{energy_consumption:.4f} kWh, {carbon_footprint:.4f} kg CO2")
        
        return record
    
    def _create_default_profile(self, model_name: str) -> ModelEcologyProfile:
        """Create a default ecology profile for unknown models."""
        
        # Determine provider from model name
        if "claude" in model_name.lower():
            provider = ModelProvider.ANTHROPIC
        elif "gpt" in model_name.lower():
            provider = ModelProvider.OPENAI
        else:
            provider = ModelProvider.OLLAMA
        
        # Use conservative estimates
        return ModelEcologyProfile(
            model_name=model_name,
            provider=provider,
            energy_per_1k_tokens=0.1,  # Conservative estimate
            carbon_intensity=0.4,
            water_intensity=3.0,
            tflops_per_watt=10.0,
            renewable_percentage=0.3
        )
    
    async def _store_record(self, record: EcologyRecord):
        """Store ecology record in Redis."""
        
        if not self.redis_client or not self.redis_client.is_connected:
            logger.warning("Redis not available, ecology record not stored")
            return
        
        try:
            key = f"{self.namespaces['records']}{record.id}"
            data = record.model_dump_json()
            
            # Store with 30-day TTL
            await self.redis_client.setex(key, 30 * 24 * 3600, data)
            
            # Add to time-based sorted sets for efficient querying
            timestamp = record.timestamp.timestamp()
            
            # Daily aggregation key
            day_key = f"{self.namespaces['records']}daily:{record.timestamp.strftime('%Y-%m-%d')}"
            await self.redis_client.zadd(day_key, {record.id: timestamp})
            await self.redis_client.expire(day_key, 30 * 24 * 3600)
            
            # Model-based aggregation
            model_key = f"{self.namespaces['records']}model:{record.model_name}"
            await self.redis_client.zadd(model_key, {record.id: timestamp})
            await self.redis_client.expire(model_key, 30 * 24 * 3600)
            
        except Exception as e:
            logger.error(f"Failed to store ecology record: {e}")
    
    async def _update_aggregates(self, record: EcologyRecord):
        """Update real-time aggregated metrics."""
        
        if not self.redis_client or not self.redis_client.is_connected:
            return
        
        try:
            # Current hour and day keys
            now = record.timestamp
            hour_key = f"{self.namespaces['aggregates']}hour:{now.strftime('%Y-%m-%d-%H')}"
            day_key = f"{self.namespaces['aggregates']}day:{now.strftime('%Y-%m-%d')}"
            
            # Update aggregates
            for key in [hour_key, day_key]:
                # Increment counters
                await self.redis_client.hincrbyfloat(key, "total_queries", 1)
                await self.redis_client.hincrbyfloat(key, "total_tokens", record.total_tokens)
                await self.redis_client.hincrbyfloat(key, "total_energy_kwh", record.energy_consumption_kwh)
                await self.redis_client.hincrbyfloat(key, "total_carbon_kg", record.carbon_footprint_kg)
                await self.redis_client.hincrbyfloat(key, "total_water_liters", record.water_usage_liters)
                
                # Set expiration
                ttl = 24 * 3600 if "hour" in key else 30 * 24 * 3600
                await self.redis_client.expire(key, ttl)
            
            # Model-specific aggregates
            model_key = f"{self.namespaces['aggregates']}model:{record.model_name}:{now.strftime('%Y-%m-%d')}"
            await self.redis_client.hincrbyfloat(model_key, "queries", 1)
            await self.redis_client.hincrbyfloat(model_key, "energy_kwh", record.energy_consumption_kwh)
            await self.redis_client.hincrbyfloat(model_key, "carbon_kg", record.carbon_footprint_kg)
            await self.redis_client.expire(model_key, 30 * 24 * 3600)
            
        except Exception as e:
            logger.error(f"Failed to update aggregates: {e}")
    
    async def get_current_impact(self, time_window: TimeWindow = TimeWindow.DAILY) -> Dict[str, float]:
        """Get current ecological impact for specified time window."""
        
        if not self._initialized:
            await self.initialize()
        
        now = datetime.utcnow()
        
        if time_window == TimeWindow.HOURLY:
            key = f"{self.namespaces['aggregates']}hour:{now.strftime('%Y-%m-%d-%H')}"
        else:  # Daily default
            key = f"{self.namespaces['aggregates']}day:{now.strftime('%Y-%m-%d')}"
        
        try:
            if self.redis_client and self.redis_client.is_connected:
                data = await self.redis_client.hgetall(key)
                
                return {
                    "total_queries": float(data.get("total_queries", 0)),
                    "total_tokens": float(data.get("total_tokens", 0)),
                    "total_energy_kwh": float(data.get("total_energy_kwh", 0)),
                    "total_carbon_kg": float(data.get("total_carbon_kg", 0)),
                    "total_water_liters": float(data.get("total_water_liters", 0))
                }
            else:
                return {"total_queries": 0, "total_tokens": 0, "total_energy_kwh": 0, 
                       "total_carbon_kg": 0, "total_water_liters": 0}
                
        except Exception as e:
            logger.error(f"Failed to get current impact: {e}")
            return {}
    
    async def generate_report(
        self,
        start_date: datetime,
        end_date: datetime,
        include_insights: bool = True
    ) -> EcologyReport:
        """Generate comprehensive ecology report for specified period."""
        
        if not self._initialized:
            await self.initialize()
        
        report = EcologyReport(
            id=f"ecology_report_{int(time.time())}",
            period_start=start_date,
            period_end=end_date
        )
        
        try:
            # Aggregate data for the period
            records = await self._get_records_for_period(start_date, end_date)
            
            if records:
                # Calculate summary metrics
                report.total_queries = len(records)
                report.total_tokens = sum(r.total_tokens for r in records)
                report.total_energy_kwh = sum(r.energy_consumption_kwh for r in records)
                report.total_carbon_kg = sum(r.carbon_footprint_kg for r in records)
                report.total_water_liters = sum(r.water_usage_liters for r in records)
                
                # Calculate efficiency metrics
                if report.total_energy_kwh > 0:
                    report.average_tokens_per_kwh = report.total_tokens / report.total_energy_kwh
                
                if report.total_tokens > 0:
                    report.carbon_intensity = (report.total_carbon_kg * 1000) / report.total_tokens  # g CO2 per token
                
                # Model breakdown
                model_stats = {}
                for record in records:
                    if record.model_name not in model_stats:
                        model_stats[record.model_name] = {
                            "queries": 0, "tokens": 0, "energy_kwh": 0, 
                            "carbon_kg": 0, "water_liters": 0
                        }
                    
                    stats = model_stats[record.model_name]
                    stats["queries"] += 1
                    stats["tokens"] += record.total_tokens
                    stats["energy_kwh"] += record.energy_consumption_kwh
                    stats["carbon_kg"] += record.carbon_footprint_kg
                    stats["water_liters"] += record.water_usage_liters
                
                report.model_breakdown = model_stats
                
                # Time series data (daily aggregation)
                daily_data = self._create_time_series(records, start_date, end_date)
                report.time_series = daily_data
                
                # Comparison data
                report.comparison_data = await self._get_comparison_data(report)
                
                # Carbon offset calculation
                report.offset_required_kg = report.total_carbon_kg
                report.offset_cost_usd = report.total_carbon_kg * self.carbon_offset_cost_per_kg
            
            # Generate insights
            if include_insights:
                report.insights = await self._generate_insights(report)
            
            logger.info(f"Generated ecology report: {report.total_queries} queries, "
                       f"{report.total_carbon_kg:.3f} kg CO2")
            
        except Exception as e:
            logger.error(f"Failed to generate ecology report: {e}")
        
        return report
    
    async def _get_records_for_period(self, start_date: datetime, end_date: datetime) -> List[EcologyRecord]:
        """Retrieve ecology records for specified time period."""
        
        records = []
        
        if not self.redis_client or not self.redis_client.is_connected:
            return records
        
        try:
            # Get records by day
            current = start_date.date()
            end = end_date.date()
            
            while current <= end:
                day_key = f"{self.namespaces['records']}daily:{current.strftime('%Y-%m-%d')}"
                
                # Get record IDs for this day
                start_ts = datetime.combine(current, datetime.min.time()).timestamp()
                end_ts = datetime.combine(current, datetime.max.time()).timestamp()
                
                record_ids = await self.redis_client.zrangebyscore(
                    day_key, start_ts, end_ts
                )
                
                # Fetch individual records
                for record_id in record_ids:
                    key = f"{self.namespaces['records']}{record_id}"
                    data = await self.redis_client.get(key)
                    if data:
                        try:
                            record = EcologyRecord.model_validate_json(data)
                            # Filter by exact time range
                            if start_date <= record.timestamp <= end_date:
                                records.append(record)
                        except Exception as e:
                            logger.warning(f"Failed to parse ecology record {record_id}: {e}")
                
                current += timedelta(days=1)
            
        except Exception as e:
            logger.error(f"Failed to retrieve records for period: {e}")
        
        return records
    
    def _create_time_series(self, records: List[EcologyRecord], start_date: datetime, end_date: datetime) -> Dict[str, List[float]]:
        """Create time series data from records."""
        
        # Group by day
        daily_data = {}
        current = start_date.date()
        end = end_date.date()
        
        # Initialize all days with zeros
        while current <= end:
            day_str = current.strftime('%Y-%m-%d')
            daily_data[day_str] = {
                "queries": 0, "tokens": 0, "energy_kwh": 0, 
                "carbon_kg": 0, "water_liters": 0
            }
            current += timedelta(days=1)
        
        # Aggregate records by day
        for record in records:
            day_str = record.timestamp.strftime('%Y-%m-%d')
            if day_str in daily_data:
                daily_data[day_str]["queries"] += 1
                daily_data[day_str]["tokens"] += record.total_tokens
                daily_data[day_str]["energy_kwh"] += record.energy_consumption_kwh
                daily_data[day_str]["carbon_kg"] += record.carbon_footprint_kg
                daily_data[day_str]["water_liters"] += record.water_usage_liters
        
        # Convert to lists for time series
        time_series = {}
        sorted_days = sorted(daily_data.keys())
        
        for metric in ["queries", "tokens", "energy_kwh", "carbon_kg", "water_liters"]:
            time_series[metric] = [daily_data[day][metric] for day in sorted_days]
        
        time_series["dates"] = sorted_days
        
        return time_series
    
    async def _get_comparison_data(self, report: EcologyReport) -> Dict[str, float]:
        """Get comparison data for benchmarking."""
        
        # Industry benchmarks (approximate values)
        benchmarks = {
            "industry_avg_carbon_per_query": 0.01,    # kg CO2 per query
            "industry_avg_energy_per_1k_tokens": 0.1, # kWh per 1k tokens
            "cloud_carbon_intensity": 0.35,           # kg CO2 per kWh
            "local_carbon_intensity": 0.5,            # kg CO2 per kWh
            "renewable_energy_target": 0.8            # 80% renewable target
        }
        
        comparison = {}
        
        if report.total_queries > 0:
            our_carbon_per_query = report.total_carbon_kg / report.total_queries
            comparison["carbon_efficiency_vs_industry"] = benchmarks["industry_avg_carbon_per_query"] / our_carbon_per_query
        
        if report.total_tokens > 0:
            our_energy_per_1k_tokens = (report.total_energy_kwh * 1000) / report.total_tokens
            comparison["energy_efficiency_vs_industry"] = benchmarks["industry_avg_energy_per_1k_tokens"] / our_energy_per_1k_tokens
        
        # Calculate renewable percentage (estimated from model mix)
        renewable_percentage = 0.0
        if report.model_breakdown:
            total_energy = sum(stats["energy_kwh"] for stats in report.model_breakdown.values())
            renewable_energy = 0.0
            
            for model_name, stats in report.model_breakdown.items():
                profile = self.model_profiles.get(model_name)
                if profile:
                    renewable_energy += stats["energy_kwh"] * profile.renewable_percentage
            
            if total_energy > 0:
                renewable_percentage = renewable_energy / total_energy
        
        comparison["renewable_percentage"] = renewable_percentage
        comparison["renewable_vs_target"] = renewable_percentage / benchmarks["renewable_energy_target"]
        
        return comparison
    
    async def _generate_insights(self, report: EcologyReport) -> List[EcologyInsight]:
        """Generate ecological insights and recommendations."""
        
        insights = []
        
        # Carbon footprint analysis
        if report.total_carbon_kg > self.sustainability_targets["daily_carbon_limit_kg"]:
            daily_carbon = report.total_carbon_kg / max(1, (report.period_end - report.period_start).days)
            
            if daily_carbon > self.sustainability_targets["daily_carbon_limit_kg"]:
                insights.append(EcologyInsight(
                    type="carbon_warning",
                    title="High Carbon Footprint",
                    description=f"Daily carbon footprint ({daily_carbon:.3f} kg CO2) exceeds sustainability target ({self.sustainability_targets['daily_carbon_limit_kg']} kg CO2)",
                    impact_reduction=0.3,
                    urgency=4,
                    actionable=True,
                    recommendations=[
                        "Prioritize local models over cloud models",
                        "Optimize context length to reduce token usage",
                        "Consider carbon offset programs"
                    ]
                ))
        
        # Energy efficiency analysis
        if report.average_tokens_per_kwh < self.sustainability_targets["energy_efficiency_target"]:
            efficiency_gap = (self.sustainability_targets["energy_efficiency_target"] - report.average_tokens_per_kwh) / self.sustainability_targets["energy_efficiency_target"]
            
            insights.append(EcologyInsight(
                type="efficiency_opportunity",
                title="Energy Efficiency Opportunity",
                description=f"Current efficiency ({report.average_tokens_per_kwh:.0f} tokens/kWh) is {efficiency_gap:.1%} below target",
                impact_reduction=efficiency_gap * 0.5,
                urgency=3,
                actionable=True,
                recommendations=[
                    "Use smaller, more efficient models for simple tasks",
                    "Implement context optimization",
                    "Batch similar queries for processing efficiency"
                ]
            ))
        
        # Model usage optimization
        if report.model_breakdown:
            # Find most carbon-intensive model
            model_carbon_intensity = {}
            for model, stats in report.model_breakdown.items():
                if stats["tokens"] > 0:
                    model_carbon_intensity[model] = (stats["carbon_kg"] * 1000) / stats["tokens"]  # g CO2 per token
            
            if model_carbon_intensity:
                worst_model = max(model_carbon_intensity.items(), key=lambda x: x[1])
                best_model = min(model_carbon_intensity.items(), key=lambda x: x[1])
                
                if worst_model[1] > best_model[1] * 2:  # More than 2x difference
                    insights.append(EcologyInsight(
                        type="model_optimization",
                        title="Model Selection Optimization",
                        description=f"{worst_model[0]} is {worst_model[1]/best_model[1]:.1f}x more carbon intensive than {best_model[0]}",
                        impact_reduction=0.4,
                        urgency=3,
                        actionable=True,
                        recommendations=[
                            f"Use {best_model[0]} for tasks where capability allows",
                            "Implement intelligent model routing based on complexity",
                            "Set carbon budget limits for high-impact models"
                        ]
                    ))
        
        # Water usage awareness
        if report.total_water_liters > 50:  # Arbitrary threshold
            insights.append(EcologyInsight(
                type="water_usage",
                title="Water Usage Awareness",
                description=f"AI processing consumed {report.total_water_liters:.1f} liters of water for cooling",
                impact_reduction=0.1,
                urgency=2,
                actionable=False,
                recommendations=[
                    "Consider water impact in model selection",
                    "Support providers with water-efficient cooling",
                    "Advocate for renewable energy in data centers"
                ]
            ))
        
        # Positive feedback
        if report.comparison_data.get("carbon_efficiency_vs_industry", 0) > 1.2:
            insights.append(EcologyInsight(
                type="positive_feedback",
                title="Excellent Carbon Efficiency",
                description="Your AI usage is 20% more carbon efficient than industry average",
                impact_reduction=0.0,
                urgency=1,
                actionable=False,
                recommendations=[
                    "Continue current optimization practices",
                    "Share best practices with team",
                    "Consider further optimization opportunities"
                ]
            ))
        
        return insights
    
    async def get_sustainability_score(self) -> Dict[str, float]:
        """Calculate overall sustainability score."""
        
        current_impact = await self.get_current_impact(TimeWindow.DAILY)
        
        scores = {
            "carbon_efficiency": 0.0,
            "energy_efficiency": 0.0,
            "water_efficiency": 0.0,
            "renewable_usage": 0.0,
            "overall": 0.0
        }
        
        if current_impact.get("total_carbon_kg", 0) > 0:
            # Carbon efficiency (lower is better)
            daily_target = self.sustainability_targets["daily_carbon_limit_kg"]
            carbon_score = max(0, min(1, daily_target / current_impact["total_carbon_kg"]))
            scores["carbon_efficiency"] = carbon_score
        else:
            scores["carbon_efficiency"] = 1.0
        
        if current_impact.get("total_energy_kwh", 0) > 0 and current_impact.get("total_tokens", 0) > 0:
            # Energy efficiency
            tokens_per_kwh = current_impact["total_tokens"] / current_impact["total_energy_kwh"]
            target_efficiency = self.sustainability_targets["energy_efficiency_target"]
            energy_score = min(1, tokens_per_kwh / target_efficiency)
            scores["energy_efficiency"] = energy_score
        else:
            scores["energy_efficiency"] = 1.0
        
        # Renewable usage (estimated from model profiles)
        renewable_score = 0.5  # Default moderate score
        scores["renewable_usage"] = renewable_score
        
        # Water efficiency (placeholder)
        scores["water_efficiency"] = 0.8
        
        # Overall score (weighted average)
        weights = {"carbon_efficiency": 0.4, "energy_efficiency": 0.3, 
                  "renewable_usage": 0.2, "water_efficiency": 0.1}
        
        scores["overall"] = sum(scores[key] * weights[key] for key in weights.keys())
        
        return scores
    
    async def suggest_green_alternative(self, model_name: str) -> Optional[str]:
        """Suggest a more environmentally friendly model alternative."""
        
        current_profile = self.model_profiles.get(model_name)
        if not current_profile:
            return None
        
        # Find models with similar capability but better ecology
        alternatives = []
        current_carbon_per_token = current_profile.energy_per_1k_tokens * current_profile.carbon_intensity
        
        for alt_name, alt_profile in self.model_profiles.items():
            if alt_name == model_name:
                continue
            
            alt_carbon_per_token = alt_profile.energy_per_1k_tokens * alt_profile.carbon_intensity
            
            # Look for models with significantly lower carbon footprint
            if alt_carbon_per_token < current_carbon_per_token * 0.7:
                efficiency_gain = (current_carbon_per_token - alt_carbon_per_token) / current_carbon_per_token
                alternatives.append((alt_name, efficiency_gain))
        
        if alternatives:
            # Return the most efficient alternative
            best_alternative = max(alternatives, key=lambda x: x[1])
            return best_alternative[0]
        
        return None


# Global ecology tracker instance
_ecology_tracker: Optional[EcologyTracker] = None


async def get_ecology_tracker() -> EcologyTracker:
    """Get global ecology tracker instance."""
    global _ecology_tracker
    
    if _ecology_tracker is None:
        _ecology_tracker = EcologyTracker()
        await _ecology_tracker.initialize()
    
    return _ecology_tracker