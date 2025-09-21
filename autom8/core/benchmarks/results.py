"""
Benchmark Results and Reporting System.

Handles benchmark execution results, performance metrics,
and comprehensive reporting capabilities.
"""

import time
import statistics
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from pydantic import BaseModel, Field, computed_field


class ExecutionStatus(str, Enum):
    """Status of benchmark execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class QualityRating(str, Enum):
    """Quality rating for benchmark responses."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"


class PerformanceMetrics(BaseModel):
    model_config = {"protected_namespaces": ()}
    """Performance metrics for a benchmark execution."""
    # Timing
    total_time_ms: float
    inference_time_ms: float
    preprocessing_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0
    
    # Throughput
    tokens_per_second: float = 0.0
    chars_per_second: float = 0.0
    
    # Cost
    estimated_cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    
    # Quality
    accuracy_score: float = 0.0
    quality_score: float = 0.0
    completeness_score: float = 0.0
    
    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    @computed_field
    @property
    def efficiency_score(self) -> float:
        """Calculate overall efficiency score."""
        if self.total_time_ms == 0:
            return 0.0
        
        # Combine quality and speed
        time_score = min(1.0, 5000 / self.total_time_ms)  # 5 seconds baseline
        quality_weight = 0.7
        speed_weight = 0.3
        
        return (self.quality_score * quality_weight) + (time_score * speed_weight)


class BenchmarkResult(BaseModel):
    model_config = {"protected_namespaces": ()}
    """
    Results from executing a single benchmark scenario.
    """
    
    # Identity
    id: str = Field(description="Unique result identifier")
    scenario_id: str = Field(description="ID of the benchmark scenario")
    model_name: str = Field(description="Name of the tested model")
    
    # Execution details
    status: ExecutionStatus = Field(description="Execution status")
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = Field(default=None)
    
    # Input/Output
    input_prompt: str = Field(description="The actual prompt sent to model")
    model_response: str = Field(default="", description="Model's response")
    expected_output: Optional[str] = Field(default=None, description="Expected response")
    
    # Performance
    metrics: PerformanceMetrics = Field(default_factory=lambda: PerformanceMetrics(total_time_ms=0.0, inference_time_ms=0.0), description="Performance metrics")
    
    # Evaluation
    quality_rating: QualityRating = Field(default=QualityRating.FAILED)
    evaluation_scores: Dict[str, float] = Field(default_factory=dict)
    evaluation_notes: List[str] = Field(default_factory=list)
    
    # Error handling
    error_message: Optional[str] = Field(default=None)
    error_type: Optional[str] = Field(default=None)
    
    # Metadata
    model_config: Dict[str, Any] = Field(default_factory=dict)
    environment_info: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    
    @computed_field
    @property
    def execution_time_ms(self) -> float:
        """Total execution time in milliseconds."""
        if self.end_time and self.start_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds() * 1000
        return 0.0
    
    @computed_field
    @property
    def success(self) -> bool:
        """Whether the benchmark executed successfully."""
        return self.status == ExecutionStatus.COMPLETED and self.quality_rating != QualityRating.FAILED
    
    @computed_field
    @property
    def overall_score(self) -> float:
        """Calculate overall performance score (0-1)."""
        if not self.success:
            return 0.0
        
        # Weight different aspects
        quality_weight = 0.4
        efficiency_weight = 0.3
        accuracy_weight = 0.3
        
        quality_scores = {
            QualityRating.EXCELLENT: 1.0,
            QualityRating.GOOD: 0.8,
            QualityRating.ACCEPTABLE: 0.6,
            QualityRating.POOR: 0.3,
            QualityRating.FAILED: 0.0
        }
        
        quality_score = quality_scores.get(self.quality_rating, 0.0)
        efficiency_score = self.metrics.efficiency_score
        accuracy_score = self.metrics.accuracy_score
        
        return (
            quality_score * quality_weight +
            efficiency_score * efficiency_weight +
            accuracy_score * accuracy_weight
        )
    
    def mark_completed(self, response: str, quality_rating: QualityRating = QualityRating.GOOD):
        """Mark the benchmark as completed."""
        self.status = ExecutionStatus.COMPLETED
        self.end_time = datetime.utcnow()
        self.model_response = response
        self.quality_rating = quality_rating
    
    def mark_failed(self, error_message: str, error_type: str = "unknown"):
        """Mark the benchmark as failed."""
        self.status = ExecutionStatus.FAILED
        self.end_time = datetime.utcnow()
        self.error_message = error_message
        self.error_type = error_type
        self.quality_rating = QualityRating.FAILED


class BenchmarkReport(BaseModel):
    model_config = {"protected_namespaces": ()}
    """
    Comprehensive report from benchmark execution.
    """
    
    # Identity
    id: str = Field(description="Unique report identifier")
    name: str = Field(description="Report name")
    description: str = Field(default="", description="Report description")
    
    # Execution details
    created_at: datetime = Field(default_factory=datetime.utcnow)
    execution_time_total_ms: float = Field(default=0.0)
    
    # Results
    results: List[BenchmarkResult] = Field(default_factory=list)
    
    # Summary statistics
    total_scenarios: int = Field(default=0)
    completed_scenarios: int = Field(default=0)
    failed_scenarios: int = Field(default=0)
    
    # Model performance summary
    model_rankings: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    scenario_type_performance: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    difficulty_level_performance: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Overall success rate across all benchmarks."""
        if self.total_scenarios == 0:
            return 0.0
        return self.completed_scenarios / self.total_scenarios
    
    @computed_field
    @property
    def average_score(self) -> float:
        """Average performance score across all successful benchmarks."""
        successful_results = [r for r in self.results if r.success]
        if not successful_results:
            return 0.0
        return statistics.mean(r.overall_score for r in successful_results)
    
    @computed_field
    @property
    def average_latency_ms(self) -> float:
        """Average response latency across all benchmarks."""
        completed_results = [r for r in self.results if r.status == ExecutionStatus.COMPLETED]
        if not completed_results:
            return 0.0
        return statistics.mean(r.execution_time_ms for r in completed_results)
    
    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result to the report."""
        self.results.append(result)
        self.total_scenarios = len(self.results)
        self.completed_scenarios = len([r for r in self.results if r.success])
        self.failed_scenarios = len([r for r in self.results if not r.success])
    
    def get_results_by_model(self, model_name: str) -> List[BenchmarkResult]:
        """Get all results for a specific model."""
        return [r for r in self.results if r.model_name == model_name]
    
    def get_results_by_scenario(self, scenario_id: str) -> List[BenchmarkResult]:
        """Get all results for a specific scenario."""
        return [r for r in self.results if r.scenario_id == scenario_id]
    
    def generate_model_rankings(self):
        """Generate model performance rankings."""
        model_stats = {}
        
        for result in self.results:
            if result.model_name not in model_stats:
                model_stats[result.model_name] = {
                    'scores': [],
                    'latencies': [],
                    'success_count': 0,
                    'total_count': 0,
                    'costs': []
                }
            
            stats = model_stats[result.model_name]
            stats['total_count'] += 1
            
            if result.success:
                stats['success_count'] += 1
                stats['scores'].append(result.overall_score)
                stats['latencies'].append(result.execution_time_ms)
                stats['costs'].append(result.metrics.estimated_cost)
        
        # Calculate rankings
        rankings = {}
        for model, stats in model_stats.items():
            if stats['total_count'] > 0:
                rankings[model] = {
                    'average_score': statistics.mean(stats['scores']) if stats['scores'] else 0.0,
                    'success_rate': stats['success_count'] / stats['total_count'],
                    'average_latency_ms': statistics.mean(stats['latencies']) if stats['latencies'] else 0.0,
                    'average_cost': statistics.mean(stats['costs']) if stats['costs'] else 0.0,
                    'total_tests': stats['total_count'],
                    'successful_tests': stats['success_count']
                }
        
        # Sort by average score
        self.model_rankings = dict(sorted(
            rankings.items(),
            key=lambda x: x[1]['average_score'],
            reverse=True
        ))
    
    def generate_insights(self):
        """Generate insights and recommendations from the benchmark results."""
        insights = []
        recommendations = []
        
        if not self.results:
            insights.append("No benchmark results available for analysis.")
            return
        
        # Success rate analysis
        if self.success_rate < 0.5:
            insights.append(f"Low overall success rate ({self.success_rate:.1%}) indicates model capability issues.")
            recommendations.append("Consider using more capable models or adjusting complexity expectations.")
        elif self.success_rate > 0.9:
            insights.append(f"Excellent success rate ({self.success_rate:.1%}) suggests good model-task alignment.")
        
        # Latency analysis
        avg_latency = self.average_latency_ms
        if avg_latency > 10000:  # 10 seconds
            insights.append(f"High average latency ({avg_latency:.0f}ms) may impact user experience.")
            recommendations.append("Consider using faster models or implementing caching strategies.")
        elif avg_latency < 1000:  # 1 second
            insights.append(f"Excellent response times ({avg_latency:.0f}ms) provide good user experience.")
        
        # Cost analysis
        costs = [r.metrics.estimated_cost for r in self.results if r.metrics.estimated_cost > 0]
        if costs:
            total_cost = sum(costs)
            avg_cost = statistics.mean(costs)
            
            if avg_cost > 0.05:  # 5 cents per query
                insights.append(f"High average cost per query (${avg_cost:.3f}) may be unsustainable.")
                recommendations.append("Consider more cost-effective models or optimize context length.")
            
            insights.append(f"Total benchmark cost: ${total_cost:.3f}")
        
        # Model comparison
        if len(self.model_rankings) > 1:
            best_model = list(self.model_rankings.keys())[0]
            worst_model = list(self.model_rankings.keys())[-1]
            
            best_score = self.model_rankings[best_model]['average_score']
            worst_score = self.model_rankings[worst_model]['average_score']
            
            insights.append(f"Performance gap: {best_model} outperforms {worst_model} by {(best_score - worst_score):.2f} points.")
            
            if best_score - worst_score > 0.3:
                recommendations.append(f"Consider standardizing on {best_model} for better consistency.")
        
        # Quality trends
        quality_counts = {}
        for result in self.results:
            if result.success:
                quality_counts[result.quality_rating] = quality_counts.get(result.quality_rating, 0) + 1
        
        if quality_counts:
            excellent_pct = quality_counts.get(QualityRating.EXCELLENT, 0) / len(self.results)
            if excellent_pct < 0.2:
                insights.append(f"Low excellent quality rate ({excellent_pct:.1%}) suggests room for improvement.")
                recommendations.append("Consider fine-tuning or using more capable models for higher quality.")
        
        self.insights = insights
        self.recommendations = recommendations
    
    def generate_summary(self) -> str:
        """Generate a human-readable summary of the benchmark report."""
        summary_parts = [
            f"Benchmark Report: {self.name}",
            f"Generated: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"📊 Overall Results:",
            f"  • Total scenarios: {self.total_scenarios}",
            f"  • Success rate: {self.success_rate:.1%}",
            f"  • Average score: {self.average_score:.2f}/1.0",
            f"  • Average latency: {self.average_latency_ms:.0f}ms",
            ""
        ]
        
        if self.model_rankings:
            summary_parts.extend([
                "🏆 Model Rankings:",
                *[f"  {i+1}. {model}: {stats['average_score']:.2f} score, {stats['success_rate']:.1%} success"
                  for i, (model, stats) in enumerate(list(self.model_rankings.items())[:5])]
            ])
            summary_parts.append("")
        
        if self.insights:
            summary_parts.extend([
                "💡 Key Insights:",
                *[f"  • {insight}" for insight in self.insights[:3]]
            ])
            summary_parts.append("")
        
        if self.recommendations:
            summary_parts.extend([
                "📋 Recommendations:",
                *[f"  • {rec}" for rec in self.recommendations[:3]]
            ])
        
        return "\n".join(summary_parts)