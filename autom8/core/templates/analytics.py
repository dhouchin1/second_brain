"""
Template Analytics Engine

This module provides comprehensive analytics and insights for template usage,
performance, and adoption patterns.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Set, Tuple
from collections import defaultdict, Counter
import statistics

from autom8.models.template import (
    ContextTemplate,
    TemplateExecutionResult,
    TemplateAnalytics,
    TemplateType,
    TemplateStatus,
)
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class AnalyticsCalculator:
    """
    Core analytics calculation engine for template metrics.
    """
    
    def __init__(self):
        self._execution_history = defaultdict(list)  # template_id -> list of executions
        self._usage_patterns = defaultdict(dict)     # template_id -> usage patterns
    
    def record_execution(
        self,
        template: ContextTemplate,
        result: TemplateExecutionResult
    ):
        """Record a template execution for analytics."""
        execution_data = {
            "timestamp": result.execution_context.completed_at or datetime.utcnow(),
            "success": result.success,
            "render_time_ms": result.render_time_ms,
            "token_count_time_ms": result.token_count_time_ms,
            "total_tokens": result.total_tokens,
            "source_count": len(result.rendered_sources),
            "variables": result.execution_context.variables.copy(),
            "executor": result.execution_context.executor,
            "validation_errors": len(result.validation_errors),
            "warnings": len(result.warnings)
        }
        
        self._execution_history[template.template_id].append(execution_data)
        
        # Keep only recent executions (last 1000 per template)
        if len(self._execution_history[template.template_id]) > 1000:
            self._execution_history[template.template_id] = \
                self._execution_history[template.template_id][-1000:]
    
    def calculate_template_analytics(
        self,
        template_id: str,
        template: Optional[ContextTemplate] = None
    ) -> TemplateAnalytics:
        """Calculate analytics for a specific template."""
        
        executions = self._execution_history.get(template_id, [])
        
        if not executions:
            # Return default analytics for templates with no executions
            return TemplateAnalytics(
                template_id=template_id,
                total_executions=0,
                successful_executions=0,
                failed_executions=0
            )
        
        # Basic execution stats
        total_executions = len(executions)
        successful_executions = sum(1 for ex in executions if ex["success"])
        failed_executions = total_executions - successful_executions
        
        # Performance metrics
        successful_execs = [ex for ex in executions if ex["success"]]
        
        if successful_execs:
            render_times = [ex["render_time_ms"] for ex in successful_execs]
            token_counts = [ex["total_tokens"] for ex in successful_execs]
            source_counts = [ex["source_count"] for ex in successful_execs]
            
            avg_render_time_ms = statistics.mean(render_times)
            avg_token_count = statistics.mean(token_counts)
            avg_source_count = statistics.mean(source_counts)
        else:
            avg_render_time_ms = 0.0
            avg_token_count = 0.0
            avg_source_count = 0.0
        
        # Variable usage patterns
        most_common_variables = self._analyze_variable_usage(executions)
        
        # Execution frequency patterns
        execution_frequency = self._analyze_execution_frequency(executions)
        
        # User adoption patterns
        user_adoption = self._analyze_user_adoption(executions)
        
        # Quality metrics
        validation_errors = sum(ex["validation_errors"] for ex in executions)
        validation_error_rate = validation_errors / total_executions if total_executions > 0 else 0.0
        
        # Calculate quality score (0.0 to 1.0)
        avg_quality_score = self._calculate_quality_score(
            successful_executions / total_executions if total_executions > 0 else 0.0,
            validation_error_rate,
            avg_render_time_ms
        )
        
        # Time tracking
        timestamps = [ex["timestamp"] for ex in executions]
        first_used = min(timestamps) if timestamps else None
        last_used = max(timestamps) if timestamps else None
        
        return TemplateAnalytics(
            template_id=template_id,
            total_executions=total_executions,
            successful_executions=successful_executions,
            failed_executions=failed_executions,
            avg_render_time_ms=avg_render_time_ms,
            avg_token_count=avg_token_count,
            avg_source_count=avg_source_count,
            most_common_variables=most_common_variables,
            execution_frequency=execution_frequency,
            user_adoption=user_adoption,
            avg_quality_score=avg_quality_score,
            validation_error_rate=validation_error_rate,
            first_used=first_used,
            last_used=last_used,
            updated_at=datetime.utcnow()
        )
    
    def _analyze_variable_usage(self, executions: List[Dict]) -> Dict[str, int]:
        """Analyze most common variable values."""
        variable_usage = defaultdict(Counter)
        
        for execution in executions:
            for var_name, var_value in execution["variables"].items():
                # Convert value to string for counting
                str_value = str(var_value)[:50]  # Truncate long values
                variable_usage[var_name][str_value] += 1
        
        # Get top 3 values for each variable
        result = {}
        for var_name, counter in variable_usage.items():
            top_values = counter.most_common(3)
            result[var_name] = {value: count for value, count in top_values}
        
        return result
    
    def _analyze_execution_frequency(self, executions: List[Dict]) -> Dict[str, int]:
        """Analyze execution frequency patterns."""
        if not executions:
            return {}
        
        # Group by time periods
        daily_counts = defaultdict(int)
        hourly_counts = defaultdict(int)
        
        for execution in executions:
            timestamp = execution["timestamp"]
            date_key = timestamp.strftime("%Y-%m-%d")
            hour_key = timestamp.strftime("%H")
            
            daily_counts[date_key] += 1
            hourly_counts[hour_key] += 1
        
        return {
            "daily_executions": dict(daily_counts),
            "hourly_distribution": dict(hourly_counts),
            "total_days": len(daily_counts),
            "avg_daily": sum(daily_counts.values()) / len(daily_counts) if daily_counts else 0
        }
    
    def _analyze_user_adoption(self, executions: List[Dict]) -> Dict[str, int]:
        """Analyze user adoption patterns."""
        user_counts = Counter()
        
        for execution in executions:
            executor = execution.get("executor")
            if executor:
                user_counts[executor] += 1
        
        return {
            "unique_users": len(user_counts),
            "top_users": dict(user_counts.most_common(5)),
            "total_executions": sum(user_counts.values())
        }
    
    def _calculate_quality_score(
        self,
        success_rate: float,
        error_rate: float,
        avg_render_time: float
    ) -> float:
        """Calculate a quality score for the template (0.0 to 1.0)."""
        
        # Base score from success rate
        score = success_rate * 0.6
        
        # Penalty for high error rate
        error_penalty = min(error_rate, 0.5) * 0.2
        score -= error_penalty
        
        # Performance score (faster is better, but with diminishing returns)
        if avg_render_time > 0:
            # Normalize render time (consider 1000ms as baseline)
            normalized_time = min(avg_render_time / 1000.0, 2.0)  # Cap at 2x baseline
            perf_score = max(0, 1.0 - (normalized_time - 0.5)) * 0.2
            score += perf_score
        else:
            score += 0.1  # Small bonus for very fast templates
        
        return max(0.0, min(1.0, score))
    
    def get_system_analytics(self) -> Dict[str, Any]:
        """Calculate system-wide analytics."""
        all_executions = []
        template_stats = {}
        
        for template_id, executions in self._execution_history.items():
            all_executions.extend(executions)
            
            template_stats[template_id] = {
                "execution_count": len(executions),
                "success_rate": sum(1 for ex in executions if ex["success"]) / len(executions),
                "last_used": max(ex["timestamp"] for ex in executions) if executions else None
            }
        
        if not all_executions:
            return {"total_executions": 0, "templates_tracked": 0}
        
        # System-wide metrics
        total_executions = len(all_executions)
        successful_executions = sum(1 for ex in all_executions if ex["success"])
        
        # Performance distribution
        render_times = [ex["render_time_ms"] for ex in all_executions if ex["success"]]
        token_counts = [ex["total_tokens"] for ex in all_executions if ex["success"]]
        
        analytics = {
            "total_executions": total_executions,
            "templates_tracked": len(self._execution_history),
            "system_success_rate": successful_executions / total_executions,
            "avg_render_time_ms": statistics.mean(render_times) if render_times else 0,
            "median_render_time_ms": statistics.median(render_times) if render_times else 0,
            "avg_token_count": statistics.mean(token_counts) if token_counts else 0,
            "median_token_count": statistics.median(token_counts) if token_counts else 0,
            "execution_time_range": {
                "start": min(ex["timestamp"] for ex in all_executions),
                "end": max(ex["timestamp"] for ex in all_executions)
            },
            "template_stats": template_stats
        }
        
        # Top performers
        sorted_templates = sorted(
            template_stats.items(),
            key=lambda x: (x[1]["execution_count"], x[1]["success_rate"]),
            reverse=True
        )
        
        analytics["top_templates"] = [
            {
                "template_id": template_id,
                "executions": stats["execution_count"],
                "success_rate": stats["success_rate"]
            }
            for template_id, stats in sorted_templates[:10]
        ]
        
        return analytics
    
    def get_popular_templates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular templates by usage."""
        template_popularity = []
        
        for template_id, executions in self._execution_history.items():
            if not executions:
                continue
            
            recent_executions = [
                ex for ex in executions
                if (datetime.utcnow() - ex["timestamp"]).days <= 30
            ]
            
            popularity_score = len(recent_executions) * 2 + len(executions)
            success_rate = sum(1 for ex in executions if ex["success"]) / len(executions)
            
            template_popularity.append({
                "template_id": template_id,
                "popularity_score": popularity_score,
                "total_executions": len(executions),
                "recent_executions": len(recent_executions),
                "success_rate": success_rate,
                "last_used": max(ex["timestamp"] for ex in executions)
            })
        
        # Sort by popularity score
        template_popularity.sort(key=lambda x: x["popularity_score"], reverse=True)
        
        return template_popularity[:limit]


class TemplateAnalyticsEngine:
    """
    Main analytics engine that coordinates analytics calculation and storage.
    """
    
    def __init__(self):
        self.calculator = AnalyticsCalculator()
        self._storage = None
        self._initialized = False
        
        # Background analytics processing
        self._analytics_queue = asyncio.Queue()
        self._processing_task = None
    
    async def initialize(self, template_storage) -> bool:
        """Initialize the analytics engine."""
        if self._initialized:
            return True
        
        try:
            logger.info("Initializing TemplateAnalyticsEngine...")
            self._storage = template_storage
            
            # Start background processing
            self._processing_task = asyncio.create_task(self._process_analytics_queue())
            
            self._initialized = True
            logger.info("TemplateAnalyticsEngine initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize TemplateAnalyticsEngine: {e}")
            return False
    
    async def record_execution(
        self,
        template: ContextTemplate,
        result: TemplateExecutionResult
    ):
        """Record a template execution for analytics."""
        try:
            # Record in calculator for real-time analytics
            self.calculator.record_execution(template, result)
            
            # Queue for persistent storage
            await self._analytics_queue.put({
                "template": template,
                "result": result,
                "timestamp": datetime.utcnow()
            })
            
        except Exception as e:
            logger.error(f"Failed to record execution analytics: {e}")
    
    async def get_template_analytics(self, template_id: str) -> Dict[str, Any]:
        """Get analytics for a specific template."""
        try:
            # Get real-time analytics
            analytics = self.calculator.calculate_template_analytics(template_id)
            
            # Add stored analytics if available
            stored_analytics = await self._storage.storage_engine.get_template_analytics(template_id)
            
            result = {
                "template_id": template_id,
                "real_time": analytics.dict(),
                "stored": stored_analytics.dict() if stored_analytics else None,
                "generated_at": datetime.utcnow()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get template analytics: {e}")
            return {"error": str(e)}
    
    async def get_system_analytics(self) -> Dict[str, Any]:
        """Get system-wide analytics."""
        try:
            return self.calculator.get_system_analytics()
        except Exception as e:
            logger.error(f"Failed to get system analytics: {e}")
            return {"error": str(e)}
    
    async def get_popular_templates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular templates."""
        try:
            return self.calculator.get_popular_templates(limit)
        except Exception as e:
            logger.error(f"Failed to get popular templates: {e}")
            return []
    
    async def generate_analytics_report(
        self,
        template_id: Optional[str] = None,
        time_range_days: int = 30
    ) -> Dict[str, Any]:
        """Generate a comprehensive analytics report."""
        try:
            report = {
                "report_type": "template_analytics",
                "generated_at": datetime.utcnow(),
                "time_range_days": time_range_days,
                "summary": {},
                "details": {}
            }
            
            if template_id:
                # Single template report
                analytics = await self.get_template_analytics(template_id)
                report["template_id"] = template_id
                report["summary"] = self._create_template_summary(analytics)
                report["details"] = analytics
                
            else:
                # System-wide report
                system_analytics = await self.get_system_analytics()
                popular_templates = await self.get_popular_templates(10)
                
                report["summary"] = {
                    "total_templates": system_analytics.get("templates_tracked", 0),
                    "total_executions": system_analytics.get("total_executions", 0),
                    "system_success_rate": system_analytics.get("system_success_rate", 0),
                    "avg_performance": {
                        "render_time_ms": system_analytics.get("avg_render_time_ms", 0),
                        "token_count": system_analytics.get("avg_token_count", 0)
                    }
                }
                
                report["details"] = {
                    "system_analytics": system_analytics,
                    "popular_templates": popular_templates,
                    "recommendations": self._generate_recommendations(system_analytics, popular_templates)
                }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate analytics report: {e}")
            return {"error": str(e)}
    
    def _create_template_summary(self, analytics: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary from template analytics."""
        real_time = analytics.get("real_time", {})
        
        return {
            "usage_level": self._categorize_usage(real_time.get("total_executions", 0)),
            "performance_rating": self._rate_performance(real_time.get("avg_render_time_ms", 0)),
            "quality_score": real_time.get("avg_quality_score", 0),
            "success_rate": real_time.get("success_rate", 0),
            "last_used": real_time.get("last_used"),
            "health_status": self._assess_template_health(real_time)
        }
    
    def _categorize_usage(self, execution_count: int) -> str:
        """Categorize template usage level."""
        if execution_count == 0:
            return "unused"
        elif execution_count < 5:
            return "low"
        elif execution_count < 20:
            return "moderate"
        elif execution_count < 100:
            return "high"
        else:
            return "very_high"
    
    def _rate_performance(self, avg_render_time: float) -> str:
        """Rate template performance."""
        if avg_render_time < 100:
            return "excellent"
        elif avg_render_time < 500:
            return "good"
        elif avg_render_time < 1000:
            return "fair"
        else:
            return "slow"
    
    def _assess_template_health(self, analytics: Dict[str, Any]) -> str:
        """Assess overall template health."""
        success_rate = analytics.get("success_rate", 0)
        quality_score = analytics.get("avg_quality_score", 0)
        error_rate = analytics.get("validation_error_rate", 0)
        
        if success_rate > 0.9 and quality_score > 0.8 and error_rate < 0.1:
            return "excellent"
        elif success_rate > 0.8 and quality_score > 0.6 and error_rate < 0.2:
            return "good"
        elif success_rate > 0.6 and quality_score > 0.4:
            return "fair"
        else:
            return "needs_attention"
    
    def _generate_recommendations(
        self,
        system_analytics: Dict[str, Any],
        popular_templates: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Generate recommendations based on analytics."""
        recommendations = []
        
        # System-level recommendations
        success_rate = system_analytics.get("system_success_rate", 0)
        if success_rate < 0.8:
            recommendations.append({
                "type": "system_health",
                "priority": "high",
                "title": "Low System Success Rate",
                "description": f"System success rate is {success_rate:.1%}. Review failing templates.",
                "action": "investigate_failures"
            })
        
        avg_render_time = system_analytics.get("avg_render_time_ms", 0)
        if avg_render_time > 1000:
            recommendations.append({
                "type": "performance",
                "priority": "medium",
                "title": "Slow Template Performance",
                "description": f"Average render time is {avg_render_time:.0f}ms. Optimize templates.",
                "action": "optimize_performance"
            })
        
        # Template-specific recommendations
        low_usage_count = sum(1 for t in popular_templates if t["total_executions"] < 5)
        if low_usage_count > len(popular_templates) * 0.5:
            recommendations.append({
                "type": "adoption",
                "priority": "low",
                "title": "Low Template Adoption",
                "description": "Many templates have low usage. Consider consolidation or promotion.",
                "action": "improve_adoption"
            })
        
        return recommendations
    
    async def _process_analytics_queue(self):
        """Background task to process analytics queue."""
        while True:
            try:
                # Get analytics item with timeout
                try:
                    item = await asyncio.wait_for(self._analytics_queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                
                # Process the analytics
                await self._persist_analytics(item)
                
            except Exception as e:
                logger.error(f"Error processing analytics queue: {e}")
                await asyncio.sleep(1)  # Brief pause on error
    
    async def _persist_analytics(self, item: Dict[str, Any]):
        """Persist analytics to storage."""
        try:
            template = item["template"]
            
            # Calculate current analytics
            analytics = self.calculator.calculate_template_analytics(template.template_id, template)
            
            # Store in database
            await self._storage.storage_engine.store_template_analytics(analytics)
            
        except Exception as e:
            logger.error(f"Failed to persist analytics: {e}")
    
    async def shutdown(self):
        """Shutdown the analytics engine."""
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        logger.info("TemplateAnalyticsEngine shutdown complete")


# Global instance
_template_analytics = None


async def get_template_analytics() -> TemplateAnalyticsEngine:
    """Get global template analytics instance."""
    global _template_analytics
    
    if _template_analytics is None:
        _template_analytics = TemplateAnalyticsEngine()
        
        # Initialize with storage
        from .storage import get_template_storage
        storage = await get_template_storage()
        await _template_analytics.initialize(storage)
    
    return _template_analytics