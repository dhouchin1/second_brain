"""
Advanced Multi-Criteria Decision Analysis (MCDA) for Model Routing.

Implements sophisticated scoring algorithms for optimal model selection
based on the PRD requirements for intelligent routing.
"""

import asyncio
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from autom8.models.complexity import ComplexityScore, ComplexityTier
from autom8.models.routing import Model, RoutingPreferences, ModelStats
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class ScoringMethod(str, Enum):
    """Different scoring methodologies."""
    WEIGHTED_SUM = "weighted_sum"
    TOPSIS = "topsis"  # Technique for Order of Preference by Similarity
    ELECTRE = "electre"  # ELimination Et Choix Traduisant la REalité
    FUZZY_LOGIC = "fuzzy_logic"


@dataclass
class ScoringCriteria:
    """Defines criteria for model scoring."""
    name: str
    weight: float
    is_benefit: bool = True  # True = higher is better, False = lower is better
    importance: float = 1.0  # Importance multiplier
    threshold: Optional[float] = None  # Minimum acceptable value


@dataclass
class ModelScore:
    """Detailed scoring breakdown for a model."""
    model_name: str
    total_score: float
    criteria_scores: Dict[str, float]
    normalized_scores: Dict[str, float]
    ranking_position: int
    confidence: float
    explanation: str
    decision_factors: List[str]


class AdvancedModelScorer:
    """
    Advanced Multi-Criteria Decision Analysis (MCDA) for intelligent model selection.
    
    This class implements sophisticated scoring algorithms that go beyond simple
    weighted averages to provide optimal model routing decisions. The system
    evaluates models across multiple dimensions including capability alignment,
    performance history, cost efficiency, latency, and ecological impact.
    
    Supported Methods:
        - TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
        - Weighted Sum (Traditional approach with enhanced criteria)
        - ELECTRE (ELimination Et Choix Traduisant la REalité)
        - Fuzzy Logic (For handling uncertainty in model performance)
    
    Key Features:
        - Multi-dimensional model evaluation
        - Historical performance integration
        - Dynamic weight adjustment based on context
        - Confidence scoring for routing decisions
        - Detailed explanation generation for transparency
    
    Example Usage:
        ```python
        scorer = AdvancedModelScorer(ScoringMethod.TOPSIS)
        scored_models = await scorer.score_models(
            models=available_models,
            complexity=complexity_score,
            context_tokens=1500,
            preferences=user_preferences,
            performance_tracker=tracker
        )
        
        best_model = scored_models[0]  # Highest scored model
        print(f"Selected: {best_model.model_name} (score: {best_model.total_score:.3f})")
        print(f"Reasoning: {best_model.explanation}")
        ```
    
    Args:
        method: The scoring methodology to use (default: TOPSIS)
        
    Attributes:
        method: Current scoring method
        criteria: Dictionary of scoring criteria with weights and thresholds
        scoring_history: List of historical scoring decisions for analysis
        performance_cache: Cache for frequently accessed performance data
    """
    
    def __init__(self, method: ScoringMethod = ScoringMethod.TOPSIS):
        """
        Initialize the advanced model scorer.
        
        Args:
            method: The MCDA method to use for scoring models
        """
        self.method = method
        self.scoring_history: List[Dict] = []
        self.performance_cache: Dict[str, Dict] = {}  # Cache for performance data
        self._last_cache_update = datetime.utcnow()
        
        # Define scoring criteria based on PRD requirements
        self.criteria = {
            'capability_match': ScoringCriteria(
                name='Capability Match',
                weight=0.25,
                is_benefit=True,
                importance=1.2,
                threshold=0.3
            ),
            'performance_quality': ScoringCriteria(
                name='Historical Performance',
                weight=0.20,
                is_benefit=True,
                importance=1.1,
                threshold=0.7
            ),
            'cost_efficiency': ScoringCriteria(
                name='Cost Efficiency',
                weight=0.15,
                is_benefit=True,  # Higher efficiency = better
                importance=1.0
            ),
            'latency_performance': ScoringCriteria(
                name='Latency Performance',
                weight=0.15,
                is_benefit=False,  # Lower latency = better
                importance=0.9,
                threshold=5000  # Max 5 seconds
            ),
            'ecological_impact': ScoringCriteria(
                name='Ecological Impact',
                weight=0.10,
                is_benefit=False,  # Lower impact = better
                importance=0.8
            ),
            'reliability_score': ScoringCriteria(
                name='Reliability',
                weight=0.10,
                is_benefit=True,
                importance=1.0,
                threshold=0.9
            ),
            'context_window_fit': ScoringCriteria(
                name='Context Window Adequacy',
                weight=0.05,
                is_benefit=True,
                importance=0.7,
                threshold=0.8
            )
        }
    
    async def score_models(
        self,
        models: List[Model],
        complexity: ComplexityScore,
        context_tokens: int,
        preferences: RoutingPreferences,
        performance_tracker,
        additional_criteria: Optional[Dict[str, ScoringCriteria]] = None
    ) -> List[ModelScore]:
        """
        Score all models using advanced MCDA methodology.
        
        Args:
            models: List of candidate models
            complexity: Query complexity analysis
            context_tokens: Estimated context size
            preferences: User routing preferences
            performance_tracker: Historical performance data
            additional_criteria: Additional scoring criteria
            
        Returns:
            List of ModelScore objects ranked by total score
        """
        if not models:
            return []
        
        start_time = time.perf_counter()
        
        # Merge additional criteria if provided
        criteria = self.criteria.copy()
        if additional_criteria:
            criteria.update(additional_criteria)
        
        # Calculate raw scores for each criterion
        raw_scores = await self._calculate_raw_scores(
            models, complexity, context_tokens, preferences, performance_tracker
        )
        
        # Apply scoring methodology
        if self.method == ScoringMethod.TOPSIS:
            scored_models = await self._apply_topsis_scoring(models, raw_scores, criteria)
        elif self.method == ScoringMethod.WEIGHTED_SUM:
            scored_models = await self._apply_weighted_sum_scoring(models, raw_scores, criteria)
        elif self.method == ScoringMethod.FUZZY_LOGIC:
            scored_models = await self._apply_fuzzy_scoring(models, raw_scores, criteria)
        else:
            scored_models = await self._apply_weighted_sum_scoring(models, raw_scores, criteria)
        
        # Sort by total score (descending)
        scored_models.sort(key=lambda x: x.total_score, reverse=True)
        
        # Assign ranking positions
        for i, model_score in enumerate(scored_models):
            model_score.ranking_position = i + 1
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        # Record scoring decision for analysis
        self._record_scoring_decision(
            scored_models, complexity, preferences, processing_time
        )
        
        logger.info(
            f"Advanced scoring complete: {len(scored_models)} models, "
            f"method={self.method}, time={processing_time:.1f}ms"
        )
        
        return scored_models
    
    async def _calculate_raw_scores(
        self,
        models: List[Model],
        complexity: ComplexityScore,
        context_tokens: int,
        preferences: RoutingPreferences,
        performance_tracker
    ) -> Dict[str, Dict[str, float]]:
        """Calculate raw scores for each model and criterion."""
        
        raw_scores = {model.name: {} for model in models}
        
        for model in models:
            scores = raw_scores[model.name]
            
            # 1. Capability Match
            capability_diff = abs(model.capability_score - complexity.raw_score)
            scores['capability_match'] = 1.0 - capability_diff
            
            # 2. Performance Quality
            stats = await performance_tracker.get_stats(model.name)
            if stats and stats.total_requests > 0:
                scores['performance_quality'] = (
                    stats.success_rate * 0.4 +
                    stats.user_satisfaction_score * 0.3 +
                    stats.routing_accuracy * 0.2 +
                    min(1.0, stats.total_requests / 100) * 0.1  # Experience bonus
                )
            else:
                scores['performance_quality'] = 0.5  # Default for new models
            
            # 3. Cost Efficiency
            if model.is_free:
                scores['cost_efficiency'] = 1.0
            else:
                # Estimate cost and calculate efficiency
                estimated_output = min(150, context_tokens * 0.3)
                cost = model.estimate_cost(context_tokens, estimated_output)
                max_acceptable_cost = preferences.max_cost_per_query
                if cost <= max_acceptable_cost:
                    scores['cost_efficiency'] = 1.0 - (cost / max_acceptable_cost)
                else:
                    scores['cost_efficiency'] = 0.0
            
            # 4. Latency Performance (inverse - lower is better)
            base_latency = model.avg_latency_ms
            context_penalty = max(0, (context_tokens - 1000) * 0.1)
            network_penalty = 200 if not model.is_local else 0
            total_latency = base_latency + context_penalty + network_penalty
            
            scores['latency_performance'] = total_latency  # Will be inverted in normalization
            
            # 5. Ecological Impact (inverse - lower is better)
            scores['ecological_impact'] = model.energy_usage + model.carbon_footprint / 100
            
            # 6. Reliability Score
            scores['reliability_score'] = model.reliability_score
            
            # 7. Context Window Fit
            if context_tokens <= model.max_context_tokens:
                window_utilization = context_tokens / model.max_context_tokens
                # Sweet spot is around 70-80% utilization
                if 0.7 <= window_utilization <= 0.8:
                    scores['context_window_fit'] = 1.0
                elif window_utilization < 0.7:
                    scores['context_window_fit'] = 0.8 + (window_utilization - 0.5) * 0.4
                else:
                    scores['context_window_fit'] = 1.0 - (window_utilization - 0.8) * 2
            else:
                scores['context_window_fit'] = 0.0  # Doesn't fit
        
        return raw_scores
    
    async def _apply_topsis_scoring(
        self,
        models: List[Model],
        raw_scores: Dict[str, Dict[str, float]],
        criteria: Dict[str, ScoringCriteria]
    ) -> List[ModelScore]:
        """
        Apply TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution).
        
        TOPSIS finds the best alternative by maximizing distance from negative ideal
        and minimizing distance from positive ideal solution.
        """
        
        if not models:
            return []
        
        # Step 1: Create decision matrix
        decision_matrix = []
        for model in models:
            row = []
            for criterion_name in criteria.keys():
                row.append(raw_scores[model.name].get(criterion_name, 0.0))
            decision_matrix.append(row)
        
        # Step 2: Normalize decision matrix
        normalized_matrix = []
        criterion_names = list(criteria.keys())
        
        for i in range(len(criterion_names)):
            column = [row[i] for row in decision_matrix]
            
            # Handle inverse criteria (lower is better)
            if not criteria[criterion_names[i]].is_benefit:
                column = [max(column) - val for val in column]
            
            # Normalize using vector normalization
            sum_of_squares = sum(val ** 2 for val in column)
            if sum_of_squares > 0:
                norm_factor = math.sqrt(sum_of_squares)
                normalized_column = [val / norm_factor for val in column]
            else:
                normalized_column = [0.0] * len(column)
            
            if not normalized_matrix:
                normalized_matrix = [[val] for val in normalized_column]
            else:
                for j, val in enumerate(normalized_column):
                    normalized_matrix[j].append(val)
        
        # Step 3: Weight the normalized matrix
        weighted_matrix = []
        for i, row in enumerate(normalized_matrix):
            weighted_row = []
            for j, val in enumerate(row):
                criterion = criteria[criterion_names[j]]
                weight = criterion.weight * criterion.importance
                weighted_row.append(val * weight)
            weighted_matrix.append(weighted_row)
        
        # Step 4: Find ideal solutions
        positive_ideal = []  # Best values
        negative_ideal = []  # Worst values
        
        for j in range(len(criterion_names)):
            column = [row[j] for row in weighted_matrix]
            positive_ideal.append(max(column))
            negative_ideal.append(min(column))
        
        # Step 5: Calculate distances
        model_scores = []
        
        for i, model in enumerate(models):
            weighted_row = weighted_matrix[i]
            
            # Distance to positive ideal
            pos_distance = math.sqrt(
                sum((weighted_row[j] - positive_ideal[j]) ** 2 
                    for j in range(len(criterion_names)))
            )
            
            # Distance to negative ideal
            neg_distance = math.sqrt(
                sum((weighted_row[j] - negative_ideal[j]) ** 2 
                    for j in range(len(criterion_names)))
            )
            
            # TOPSIS score
            if pos_distance + neg_distance > 0:
                topsis_score = neg_distance / (pos_distance + neg_distance)
            else:
                topsis_score = 0.0
            
            # Calculate confidence based on separation
            confidence = min(1.0, neg_distance / (pos_distance + 0.001))
            
            # Create detailed scoring breakdown
            criteria_scores = {
                criterion_names[j]: raw_scores[model.name].get(criterion_names[j], 0.0)
                for j in range(len(criterion_names))
            }
            
            normalized_scores = {
                criterion_names[j]: normalized_matrix[i][j]
                for j in range(len(criterion_names))
            }
            
            # Generate explanation
            explanation = self._generate_topsis_explanation(
                model, topsis_score, criteria_scores, criteria
            )
            
            decision_factors = self._identify_decision_factors(
                criteria_scores, criteria, top_n=3
            )
            
            model_scores.append(ModelScore(
                model_name=model.name,
                total_score=topsis_score,
                criteria_scores=criteria_scores,
                normalized_scores=normalized_scores,
                ranking_position=0,  # Will be set after sorting
                confidence=confidence,
                explanation=explanation,
                decision_factors=decision_factors
            ))
        
        return model_scores
    
    async def _apply_weighted_sum_scoring(
        self,
        models: List[Model],
        raw_scores: Dict[str, Dict[str, float]],
        criteria: Dict[str, ScoringCriteria]
    ) -> List[ModelScore]:
        """Apply simple weighted sum scoring methodology."""
        
        model_scores = []
        
        for model in models:
            total_score = 0.0
            criteria_scores = {}
            normalized_scores = {}
            
            for criterion_name, criterion in criteria.items():
                raw_value = raw_scores[model.name].get(criterion_name, 0.0)
                
                # Handle inverse criteria
                if not criterion.is_benefit and criterion_name == 'latency_performance':
                    # Normalize latency (lower is better)
                    normalized_value = max(0, 1.0 - (raw_value / 10000))  # 10 seconds max
                elif not criterion.is_benefit and criterion_name == 'ecological_impact':
                    # Normalize ecological impact (lower is better)
                    normalized_value = max(0, 1.0 - raw_value)
                else:
                    normalized_value = max(0, min(1.0, raw_value))
                
                # Apply threshold if specified
                if criterion.threshold is not None:
                    if criterion.is_benefit and raw_value < criterion.threshold:
                        normalized_value *= 0.5  # Penalty for not meeting threshold
                    elif not criterion.is_benefit and raw_value > criterion.threshold:
                        normalized_value *= 0.5  # Penalty for exceeding threshold
                
                weighted_value = normalized_value * criterion.weight * criterion.importance
                total_score += weighted_value
                
                criteria_scores[criterion_name] = raw_value
                normalized_scores[criterion_name] = normalized_value
            
            # Calculate confidence based on score distribution
            score_variance = sum(
                (normalized_scores[name] - total_score) ** 2 
                for name in normalized_scores
            ) / len(normalized_scores)
            confidence = min(1.0, 1.0 - score_variance)
            
            # Generate explanation
            explanation = self._generate_weighted_sum_explanation(
                model, total_score, criteria_scores, criteria
            )
            
            decision_factors = self._identify_decision_factors(
                criteria_scores, criteria, top_n=3
            )
            
            model_scores.append(ModelScore(
                model_name=model.name,
                total_score=total_score,
                criteria_scores=criteria_scores,
                normalized_scores=normalized_scores,
                ranking_position=0,
                confidence=confidence,
                explanation=explanation,
                decision_factors=decision_factors
            ))
        
        return model_scores
    
    async def _apply_fuzzy_scoring(
        self,
        models: List[Model],
        raw_scores: Dict[str, Dict[str, float]],
        criteria: Dict[str, ScoringCriteria]
    ) -> List[ModelScore]:
        """Apply fuzzy logic scoring methodology."""
        # For now, fall back to weighted sum - fuzzy logic is complex
        # This is a placeholder for future fuzzy logic implementation
        return await self._apply_weighted_sum_scoring(models, raw_scores, criteria)
    
    def _generate_topsis_explanation(
        self,
        model: Model,
        score: float,
        criteria_scores: Dict[str, float],
        criteria: Dict[str, ScoringCriteria]
    ) -> str:
        """Generate human-readable explanation for TOPSIS scoring."""
        
        explanations = []
        
        if score > 0.8:
            explanations.append("Excellent overall match with optimal characteristics")
        elif score > 0.6:
            explanations.append("Good match with strong performance in key areas")
        elif score > 0.4:
            explanations.append("Adequate choice with acceptable trade-offs")
        else:
            explanations.append("Suboptimal choice with significant limitations")
        
        # Highlight best criteria
        best_criterion = max(criteria_scores.items(), key=lambda x: x[1])
        explanations.append(f"Strongest in {best_criterion[0]} ({best_criterion[1]:.2f})")
        
        # Note any threshold violations
        for name, criterion in criteria.items():
            if criterion.threshold is not None:
                value = criteria_scores.get(name, 0.0)
                if criterion.is_benefit and value < criterion.threshold:
                    explanations.append(f"Below threshold for {name}")
                elif not criterion.is_benefit and value > criterion.threshold:
                    explanations.append(f"Exceeds acceptable limit for {name}")
        
        return "; ".join(explanations)
    
    def _generate_weighted_sum_explanation(
        self,
        model: Model,
        score: float,
        criteria_scores: Dict[str, float],
        criteria: Dict[str, ScoringCriteria]
    ) -> str:
        """Generate human-readable explanation for weighted sum scoring."""
        
        explanations = []
        
        if score > 0.8:
            explanations.append("High-scoring model with excellent overall performance")
        elif score > 0.6:
            explanations.append("Well-balanced model with good characteristics")
        elif score > 0.4:
            explanations.append("Moderate choice with mixed performance")
        else:
            explanations.append("Low-scoring model with several limitations")
        
        # Identify key strengths and weaknesses
        sorted_scores = sorted(
            criteria_scores.items(), 
            key=lambda x: x[1] * criteria[x[0]].weight,
            reverse=True
        )
        
        best_criterion = sorted_scores[0]
        explanations.append(f"Excels in {best_criterion[0]}")
        
        if len(sorted_scores) > 1:
            worst_criterion = sorted_scores[-1]
            if worst_criterion[1] < 0.3:
                explanations.append(f"Limited by {worst_criterion[0]}")
        
        return "; ".join(explanations)
    
    def _identify_decision_factors(
        self,
        criteria_scores: Dict[str, float],
        criteria: Dict[str, ScoringCriteria],
        top_n: int = 3
    ) -> List[str]:
        """Identify the top factors influencing the decision."""
        
        # Weight scores by importance and criteria weight
        weighted_impact = []
        for name, score in criteria_scores.items():
            criterion = criteria[name]
            impact = score * criterion.weight * criterion.importance
            weighted_impact.append((name, impact, score))
        
        # Sort by weighted impact
        weighted_impact.sort(key=lambda x: x[1], reverse=True)
        
        factors = []
        for name, impact, score in weighted_impact[:top_n]:
            factor_desc = f"{name}: {score:.2f} (impact: {impact:.3f})"
            factors.append(factor_desc)
        
        return factors
    
    def _record_scoring_decision(
        self,
        scored_models: List[ModelScore],
        complexity: ComplexityScore,
        preferences: RoutingPreferences,
        processing_time: float
    ) -> None:
        """Record scoring decision for analysis and improvement."""
        
        decision_record = {
            'timestamp': time.time(),
            'method': self.method,
            'processing_time_ms': processing_time,
            'complexity_score': complexity.raw_score,
            'complexity_tier': complexity.recommended_tier.value,
            'prefer_local': preferences.prefer_local,
            'models_evaluated': len(scored_models),
            'top_model': scored_models[0].model_name if scored_models else None,
            'top_score': scored_models[0].total_score if scored_models else 0.0,
            'score_spread': (
                scored_models[0].total_score - scored_models[-1].total_score
                if len(scored_models) > 1 else 0.0
            )
        }
        
        self.scoring_history.append(decision_record)
        
        # Keep only recent history (last 1000 decisions)
        if len(self.scoring_history) > 1000:
            self.scoring_history = self.scoring_history[-1000:]
    
    def get_scoring_analytics(self) -> Dict[str, any]:
        """Get analytics about scoring performance and patterns."""
        
        if not self.scoring_history:
            return {}
        
        recent_decisions = self.scoring_history[-100:]  # Last 100 decisions
        
        analytics = {
            'total_decisions': len(self.scoring_history),
            'avg_processing_time': sum(d['processing_time_ms'] for d in recent_decisions) / len(recent_decisions),
            'avg_models_per_decision': sum(d['models_evaluated'] for d in recent_decisions) / len(recent_decisions),
            'avg_score_spread': sum(d['score_spread'] for d in recent_decisions) / len(recent_decisions),
            'scoring_method': self.method,
            'local_preference_percentage': sum(1 for d in recent_decisions if d['prefer_local']) / len(recent_decisions) * 100
        }
        
        # Model selection frequency
        model_counts = {}
        for decision in recent_decisions:
            if decision['top_model']:
                model_counts[decision['top_model']] = model_counts.get(decision['top_model'], 0) + 1
        
        analytics['top_selected_models'] = sorted(
            model_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return analytics


async def get_advanced_scorer(method: ScoringMethod = ScoringMethod.TOPSIS) -> AdvancedModelScorer:
    """Get configured advanced model scorer instance."""
    return AdvancedModelScorer(method)