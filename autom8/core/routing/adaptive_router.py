"""
Adaptive Model Router with Dynamic Preference Learning

This module extends the ModelRouter with dynamic preference learning capabilities,
integrating machine learning algorithms to continuously improve routing decisions
based on user behavior and outcomes.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
import json

from autom8.core.routing.router import ModelRouter, PerformanceTracker
from autom8.core.routing.preference_learning import (
    PreferenceLearningEngine, LearningAlgorithm, UserProfile, UserInteraction,
    OutcomeType, FeedbackType
)
from autom8.core.routing.interaction_tracker import InteractionTracker, get_interaction_tracker
from autom8.core.routing.advanced_scoring import AdvancedModelScorer, ScoringCriteria, ModelScore
from autom8.models.complexity import ComplexityScore, ComplexityTier
from autom8.models.routing import (
    Model, ModelSelection, RoutingPreferences, ModelStats
)
from autom8.services.budget import BudgetManager
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AdaptiveRoutingConfig:
    """Configuration for adaptive routing behavior"""
    # Learning parameters
    enable_learning: bool = True
    learning_algorithm: LearningAlgorithm = LearningAlgorithm.ENSEMBLE
    min_interactions_for_learning: int = 5
    learning_confidence_threshold: float = 0.6
    
    # Adaptation parameters
    adaptation_rate: float = 0.1  # How quickly to adapt to new patterns
    exploration_rate: float = 0.15  # Rate of trying non-optimal choices for learning
    override_learning_weight: float = 0.3  # Weight for user override signals
    
    # A/B testing
    enable_ab_testing: bool = True
    ab_test_ratio: float = 0.1  # Percentage of users in experimental groups
    ab_test_duration_days: int = 7
    
    # Safety parameters
    max_cost_increase: float = 0.2  # Max 20% cost increase from learning
    max_latency_increase: float = 0.3  # Max 30% latency increase from learning
    fallback_to_base_threshold: float = 0.3  # Fall back if confidence < 30%


class ConfidenceScorer:
    """Calculates confidence scores for routing decisions"""
    
    def __init__(self):
        self.confidence_factors = {
            'model_reliability': 0.25,
            'historical_performance': 0.25,
            'user_preference_strength': 0.20,
            'prediction_certainty': 0.15,
            'data_quality': 0.15
        }
    
    def calculate_confidence(self, model: Model, model_score: ModelScore, 
                           user_profile: Optional[UserProfile],
                           prediction_data: Dict[str, Any]) -> float:
        """Calculate overall confidence in routing decision"""
        
        confidence_components = {}
        
        # Model reliability
        confidence_components['model_reliability'] = model.reliability_score
        
        # Historical performance (from model score)
        confidence_components['historical_performance'] = model_score.confidence
        
        # User preference strength
        if user_profile and user_profile.interaction_count > 0:
            preference_strength = user_profile.confidence_score
            interaction_bonus = min(0.3, user_profile.interaction_count / 100)
            confidence_components['user_preference_strength'] = preference_strength + interaction_bonus
        else:
            confidence_components['user_preference_strength'] = 0.5  # Neutral for new users
        
        # Prediction certainty (how certain are our ML predictions)
        prediction_variance = prediction_data.get('prediction_variance', 0.5)
        confidence_components['prediction_certainty'] = 1.0 - prediction_variance
        
        # Data quality (how much relevant data we have)
        data_points = prediction_data.get('data_points', 0)
        if data_points > 50:
            confidence_components['data_quality'] = 1.0
        elif data_points > 20:
            confidence_components['data_quality'] = 0.8
        elif data_points > 5:
            confidence_components['data_quality'] = 0.6
        else:
            confidence_components['data_quality'] = 0.4
        
        # Calculate weighted confidence
        total_confidence = 0.0
        for factor, weight in self.confidence_factors.items():
            total_confidence += confidence_components[factor] * weight
        
        return max(0.0, min(1.0, total_confidence))


class ABTestingFramework:
    """A/B testing framework for routing decisions"""
    
    def __init__(self, config: AdaptiveRoutingConfig):
        self.config = config
        self.active_tests = {}
        self.test_results = {}
        self.user_assignments = {}  # user_id -> test_group
    
    def should_participate_in_test(self, user_id: str) -> bool:
        """Determine if user should participate in A/B testing"""
        if not self.config.enable_ab_testing:
            return False
        
        # Simple hash-based assignment for consistency
        user_hash = hash(user_id) % 100
        return user_hash < (self.config.ab_test_ratio * 100)
    
    def assign_test_group(self, user_id: str, test_id: str) -> str:
        """Assign user to test group"""
        if user_id in self.user_assignments:
            return self.user_assignments[user_id]
        
        # Assign based on hash for consistency
        user_hash = hash(f"{user_id}_{test_id}") % 2
        group = "control" if user_hash == 0 else "treatment"
        self.user_assignments[user_id] = group
        return group
    
    def create_test(self, test_id: str, description: str, treatment_config: Dict[str, Any]):
        """Create a new A/B test"""
        self.active_tests[test_id] = {
            "description": description,
            "treatment_config": treatment_config,
            "start_date": datetime.utcnow(),
            "end_date": datetime.utcnow() + timedelta(days=self.config.ab_test_duration_days),
            "control_results": [],
            "treatment_results": []
        }
    
    def record_test_result(self, user_id: str, test_id: str, outcome_data: Dict[str, Any]):
        """Record result for A/B test"""
        if test_id not in self.active_tests:
            return
        
        group = self.user_assignments.get(user_id, "control")
        result = {
            "user_id": user_id,
            "timestamp": datetime.utcnow(),
            "outcome": outcome_data
        }
        
        self.active_tests[test_id][f"{group}_results"].append(result)
    
    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get results for an A/B test"""
        if test_id not in self.active_tests:
            return {}
        
        test = self.active_tests[test_id]
        control_results = test["control_results"]
        treatment_results = test["treatment_results"]
        
        if not control_results or not treatment_results:
            return {"status": "insufficient_data"}
        
        # Calculate key metrics
        control_satisfaction = sum(r["outcome"].get("satisfaction", 0.5) for r in control_results) / len(control_results)
        treatment_satisfaction = sum(r["outcome"].get("satisfaction", 0.5) for r in treatment_results) / len(treatment_results)
        
        control_cost = sum(r["outcome"].get("cost", 0.0) for r in control_results) / len(control_results)
        treatment_cost = sum(r["outcome"].get("cost", 0.0) for r in treatment_results) / len(treatment_results)
        
        return {
            "test_id": test_id,
            "control_group_size": len(control_results),
            "treatment_group_size": len(treatment_results),
            "satisfaction_improvement": treatment_satisfaction - control_satisfaction,
            "cost_change": treatment_cost - control_cost,
            "is_significant": abs(treatment_satisfaction - control_satisfaction) > 0.05,
            "recommendation": "adopt" if treatment_satisfaction > control_satisfaction else "reject"
        }


class AdaptiveModelRouter(ModelRouter):
    """
    Enhanced ModelRouter with dynamic preference learning and adaptation.
    
    Extends the base ModelRouter with machine learning capabilities to:
    - Learn user preferences from interactions
    - Adapt routing decisions based on outcomes
    - A/B test routing strategies
    - Provide confidence scores for decisions
    """
    
    def __init__(self, budget_manager: Optional[BudgetManager] = None, 
                 config: Optional[AdaptiveRoutingConfig] = None):
        super().__init__(budget_manager)
        
        self.config = config or AdaptiveRoutingConfig()
        self.confidence_scorer = ConfidenceScorer()
        self.ab_testing = ABTestingFramework(self.config)
        
        # Learning components
        self.learning_engine = PreferenceLearningEngine(self.config.learning_algorithm)
        self.interaction_tracker: Optional[InteractionTracker] = None
        
        # Adaptation state
        self.adaptation_history = []
        self.performance_baselines = {}
        self.learning_metrics = {
            "adaptations_made": 0,
            "user_satisfaction_improvement": 0.0,
            "cost_optimization": 0.0,
            "routing_accuracy": 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize the adaptive router with learning components"""
        # Initialize base router
        if not await super().initialize():
            return False
        
        try:
            # Initialize learning components
            if self.config.enable_learning:
                self.interaction_tracker = await get_interaction_tracker(self.learning_engine)
                logger.info("Adaptive routing learning components initialized")
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize adaptive router: {e}")
            return False
    
    async def route(
        self,
        query: str,
        complexity: ComplexityScore,
        context_tokens: int = 500,
        preferences: Optional[RoutingPreferences] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> ModelSelection:
        """
        Route query with adaptive learning and preference adjustment.
        
        Args:
            query: The query to route
            complexity: Complexity analysis results
            context_tokens: Estimated context size
            preferences: Base user preferences
            user_id: User identifier for learning
            session_id: Session identifier for tracking
            
        Returns:
            Enhanced model selection with confidence and learning data
        """
        start_time = time.perf_counter()
        
        # Initialize tracking
        query_id = None
        if self.interaction_tracker and user_id and session_id:
            query_id = await self.interaction_tracker.track_routing_request(
                user_id, session_id, query, complexity, context_tokens, preferences or RoutingPreferences()
            )
        
        # Get base routing preferences
        base_preferences = preferences or RoutingPreferences()
        
        # Apply learned preferences if available
        adapted_preferences = await self._adapt_preferences(
            user_id, base_preferences, complexity, context_tokens
        )
        
        # Check if user should participate in A/B testing
        use_experimental = (user_id and self.ab_testing.should_participate_in_test(user_id))
        
        if use_experimental:
            # Apply A/B test modifications
            adapted_preferences = await self._apply_ab_test_modifications(
                user_id, adapted_preferences, query_id
            )
        
        # Get capable models
        capable_models = self._get_capable_models(complexity.recommended_tier, adapted_preferences)
        
        if not capable_models:
            capable_models = self.model_registry.get_available_models()
            if not capable_models:
                raise RuntimeError("No models available for routing")
        
        # Apply learned model preferences
        if self.config.enable_learning and user_id:
            model_preferences = await self.learning_engine.predict_preferences(
                user_id, capable_models, complexity, context_tokens, adapted_preferences
            )
        else:
            model_preferences = {model.name: 0.5 for model in capable_models}
        
        # Enhanced MCDA scoring with learned preferences
        if self.use_advanced_scoring and self.advanced_scorer:
            scored_models = await self._score_with_learned_preferences(
                capable_models, complexity, context_tokens, adapted_preferences, 
                model_preferences, user_id
            )
        else:
            # Fallback to base scoring with preference adjustment
            scored_models = await self._fallback_scoring_with_preferences(
                capable_models, complexity, context_tokens, adapted_preferences, model_preferences
            )
        
        # Select model with exploration consideration
        primary_model, model_score = await self._select_with_exploration(
            scored_models, user_id, complexity
        )
        
        # Calculate enhanced confidence
        user_profile = self.learning_engine.user_profiles.get(user_id) if user_id else None
        prediction_data = {
            "prediction_variance": self._calculate_prediction_variance(scored_models),
            "data_points": user_profile.interaction_count if user_profile else 0
        }
        
        enhanced_confidence = self.confidence_scorer.calculate_confidence(
            primary_model, model_score, user_profile, prediction_data
        )
        
        # Create enhanced model selection
        alternatives = [score.model_name for score in scored_models[1:4]]
        alternative_models = [next(m for m in capable_models if m.name == name) for name in alternatives]
        
        fallback_model = self._find_fallback_model(primary_model, capable_models)
        
        selection = ModelSelection(
            primary_model=primary_model,
            alternatives=alternative_models,
            fallback_model=fallback_model,
            selection_reasoning=self._generate_adaptive_reasoning(
                primary_model, complexity, model_score, adapted_preferences, user_profile
            ),
            confidence=enhanced_confidence,
            routing_factors=self._get_enhanced_routing_factors(
                primary_model, complexity, adapted_preferences, model_score, model_preferences
            ),
            estimated_quality=self._estimate_quality(primary_model, complexity),
            estimated_latency_ms=self._estimate_latency(primary_model, context_tokens),
            estimated_cost=self._estimate_cost(primary_model, context_tokens),
            complexity_tier=complexity.recommended_tier,
            user_preferences=adapted_preferences.dict(),
            constraints=self._get_active_constraints(adapted_preferences)
        )
        
        # Add adaptive-specific metadata
        selection.selection_id = query_id
        
        # Track routing decision
        if self.interaction_tracker and query_id:
            await self.interaction_tracker.track_routing_decision(query_id, selection)
        
        # Record routing decision for base router
        self._record_routing_decision(query, complexity, selection)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        logger.info(
            f"Adaptive routing complete: {primary_model.name} (confidence: {enhanced_confidence:.2f}, "
            f"adapted: {user_id is not None}, time: {processing_time:.1f}ms)"
        )
        
        return selection
    
    async def record_interaction_outcome(
        self, 
        selection: ModelSelection,
        outcome_type: str,
        actual_metrics: Dict[str, Any],
        user_feedback: Optional[Dict[str, Any]] = None
    ):
        """Record the outcome of a routing decision for learning"""
        
        if not self.config.enable_learning or not self.interaction_tracker:
            return
        
        try:
            # Create interaction record
            interaction = UserInteraction(
                user_id=selection.metadata.get("user_id", "unknown"),
                session_id=selection.metadata.get("session_id", "unknown"),
                timestamp=datetime.utcnow(),
                query_text=selection.metadata.get("query", ""),
                query_length=len(selection.metadata.get("query", "")),
                complexity=selection.metadata.get("complexity"),
                context_tokens=selection.metadata.get("context_tokens", 0),
                selected_model=selection.primary_model.name,
                alternative_models=[m.name for m in selection.alternative_models],
                routing_confidence=selection.confidence_score,
                routing_factors=selection.selection_factors,
                user_preferences=selection.metadata.get("preferences", {}),
                outcome_type=OutcomeType(outcome_type),
                actual_latency_ms=actual_metrics.get("latency_ms"),
                actual_cost=actual_metrics.get("cost"),
                actual_quality=actual_metrics.get("quality"),
                error_occurred=actual_metrics.get("error", False),
                retry_count=actual_metrics.get("retry_count", 0)
            )
            
            # Add user feedback if provided
            if user_feedback:
                interaction.feedback_type = FeedbackType(user_feedback.get("type", "implicit_satisfaction"))
                interaction.user_rating = user_feedback.get("rating")
                interaction.satisfaction_score = user_feedback.get("satisfaction")
                interaction.user_override = user_feedback.get("override_model")
                interaction.edit_preferences = user_feedback.get("edited_preferences", False)
                interaction.abandoned_session = user_feedback.get("abandoned", False)
            
            # Record with learning engine
            await self.learning_engine.record_interaction(interaction)
            
            # Update routing performance metrics
            await self._update_performance_metrics(interaction)
            
            # Trigger adaptation if needed
            if self._should_trigger_adaptation(interaction):
                await self._trigger_adaptation(interaction)
            
            logger.debug(f"Recorded interaction outcome for model {selection.primary_model.name}")
            
        except Exception as e:
            logger.error(f"Failed to record interaction outcome: {e}")
    
    async def _update_performance_metrics(self, interaction: UserInteraction):
        """Update internal performance tracking"""
        
        model_name = interaction.selected_model
        
        # Initialize model metrics if not exists
        if model_name not in self.performance_baselines:
            self.performance_baselines[model_name] = {
                "total_uses": 0,
                "successful_uses": 0,
                "avg_satisfaction": 0.5,
                "avg_cost": 0.0,
                "avg_latency": 0.0,
                "override_count": 0
            }
        
        baseline = self.performance_baselines[model_name]
        baseline["total_uses"] += 1
        
        # Update success metrics
        if interaction.outcome_type in [OutcomeType.SUCCESS, OutcomeType.USER_SATISFIED]:
            baseline["successful_uses"] += 1
        
        # Update satisfaction score
        if interaction.satisfaction_score is not None:
            baseline["avg_satisfaction"] = (baseline["avg_satisfaction"] + interaction.satisfaction_score) / 2
        
        # Update cost and latency
        if interaction.actual_cost is not None:
            baseline["avg_cost"] = (baseline["avg_cost"] + interaction.actual_cost) / 2
        
        if interaction.actual_latency_ms is not None:
            baseline["avg_latency"] = (baseline["avg_latency"] + interaction.actual_latency_ms) / 2
        
        # Track overrides
        if interaction.user_override:
            baseline["override_count"] += 1
        
        # Update global learning metrics
        if interaction.outcome_type in [OutcomeType.SUCCESS, OutcomeType.USER_SATISFIED]:
            self.learning_metrics["successful_predictions"] += 1
        
        self.learning_metrics["override_rate"] = sum(
            baseline["override_count"] for baseline in self.performance_baselines.values()
        ) / sum(
            baseline["total_uses"] for baseline in self.performance_baselines.values()
        ) if self.performance_baselines else 0.0
    
    def _should_trigger_adaptation(self, interaction: UserInteraction) -> bool:
        """Determine if this interaction should trigger system adaptation"""
        
        # Trigger on user overrides
        if interaction.user_override:
            return True
        
        # Trigger on poor satisfaction
        if interaction.satisfaction_score is not None and interaction.satisfaction_score < 0.3:
            return True
        
        # Trigger on cost violations
        if interaction.outcome_type == OutcomeType.COST_EXCEEDED:
            return True
        
        # Trigger on consistent poor performance
        model_baseline = self.performance_baselines.get(interaction.selected_model)
        if model_baseline and model_baseline["total_uses"] > 10:
            success_rate = model_baseline["successful_uses"] / model_baseline["total_uses"]
            if success_rate < 0.6:  # Less than 60% success rate
                return True
        
        return False
    
    async def _trigger_adaptation(self, interaction: UserInteraction):
        """Trigger adaptation based on interaction outcome"""
        
        adaptation = {
            "timestamp": datetime.utcnow(),
            "trigger_interaction": interaction.selected_model,
            "trigger_type": "",
            "adaptation_action": "",
            "parameters": {}
        }
        
        if interaction.user_override:
            adaptation["trigger_type"] = "user_override"
            adaptation["adaptation_action"] = "adjust_model_scoring"
            adaptation["parameters"] = {
                "demote_model": interaction.selected_model,
                "promote_model": interaction.user_override,
                "adjustment_factor": 0.1
            }
            
            # Adjust model scoring in router
            await self._adjust_model_scoring(
                interaction.selected_model, 
                interaction.user_override, 
                0.1
            )
            
        elif interaction.outcome_type == OutcomeType.COST_EXCEEDED:
            adaptation["trigger_type"] = "cost_violation"
            adaptation["adaptation_action"] = "increase_cost_weight"
            adaptation["parameters"] = {
                "user_id": interaction.user_id,
                "cost_weight_increase": 0.1
            }
            
            # Increase cost sensitivity for this user
            await self._adjust_user_cost_sensitivity(interaction.user_id, 0.1)
        
        elif interaction.satisfaction_score is not None and interaction.satisfaction_score < 0.3:
            adaptation["trigger_type"] = "poor_satisfaction"
            adaptation["adaptation_action"] = "reduce_model_preference"
            adaptation["parameters"] = {
                "model": interaction.selected_model,
                "reduction_factor": 0.05
            }
            
            # Reduce preference for this model
            await self._adjust_model_preference(interaction.selected_model, -0.05)
        
        # Record adaptation
        self.adaptation_history.append(adaptation)
        self.learning_metrics["adaptations_made"] += 1
        
        logger.info(f"Triggered adaptation: {adaptation['adaptation_action']} due to {adaptation['trigger_type']}")
    
    async def _adjust_model_scoring(self, demote_model: str, promote_model: str, factor: float):
        """Adjust scoring for models based on user feedback"""
        # This would integrate with the model registry to adjust capability scores
        # For now, we'll store adjustments in memory
        
        if not hasattr(self, 'model_adjustments'):
            self.model_adjustments = {}
        
        self.model_adjustments[demote_model] = self.model_adjustments.get(demote_model, 0.0) - factor
        self.model_adjustments[promote_model] = self.model_adjustments.get(promote_model, 0.0) + factor
        
        logger.debug(f"Adjusted model scoring: {demote_model} (-{factor}), {promote_model} (+{factor})")
    
    async def _adjust_user_cost_sensitivity(self, user_id: str, increase: float):
        """Increase cost sensitivity for a user"""
        if user_id in self.learning_engine.user_profiles:
            profile = self.learning_engine.user_profiles[user_id]
            profile.cost_weight = min(1.0, profile.cost_weight + increase)
            logger.debug(f"Increased cost sensitivity for user {user_id} to {profile.cost_weight}")
    
    async def _adjust_model_preference(self, model_name: str, adjustment: float):
        """Adjust global preference for a model"""
        # This could integrate with global model registry
        if not hasattr(self, 'global_model_preferences'):
            self.global_model_preferences = {}
        
        current = self.global_model_preferences.get(model_name, 0.5)
        self.global_model_preferences[model_name] = max(0.0, min(1.0, current + adjustment))
        
        logger.debug(f"Adjusted global preference for {model_name} to {self.global_model_preferences[model_name]}")
    
    async def get_adaptation_insights(self) -> Dict[str, Any]:
        """Get insights about adaptations and learning performance"""
        
        return {
            "learning_metrics": self.learning_metrics.copy(),
            "model_performance": self.performance_baselines.copy(),
            "recent_adaptations": self.adaptation_history[-10:],  # Last 10 adaptations
            "learning_engine_stats": self.learning_engine.get_learning_statistics(),
            "adaptation_recommendations": await self.learning_engine.get_adaptation_recommendations(),
            "confidence_trends": {
                "average_confidence": sum(
                    baseline["avg_satisfaction"] for baseline in self.performance_baselines.values()
                ) / len(self.performance_baselines) if self.performance_baselines else 0.5,
                "models_improving": len([
                    model for model, baseline in self.performance_baselines.items()
                    if baseline["avg_satisfaction"] > 0.7
                ]),
                "models_struggling": len([
                    model for model, baseline in self.performance_baselines.items()
                    if baseline["avg_satisfaction"] < 0.4 and baseline["total_uses"] > 5
                ])
            }
        }
    
    async def save_adaptation_state(self, storage_path: str = "autom8_adaptation_state.pkl"):
        """Save adaptation and learning state"""
        try:
            state = {
                "learning_metrics": self.learning_metrics,
                "performance_baselines": self.performance_baselines,
                "adaptation_history": self.adaptation_history,
                "model_adjustments": getattr(self, 'model_adjustments', {}),
                "global_model_preferences": getattr(self, 'global_model_preferences', {}),
                "config": asdict(self.config),
                "saved_at": datetime.utcnow().isoformat()
            }
            
            import pickle
            with open(storage_path, 'wb') as f:
                pickle.dump(state, f)
            
            # Also save learning engine state
            await self.learning_engine.save_learning_state(
                storage_path.replace('.pkl', '_learning.pkl')
            )
            
            logger.info(f"Adaptation state saved to {storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save adaptation state: {e}")
            return False
    
    async def load_adaptation_state(self, storage_path: str = "autom8_adaptation_state.pkl"):
        """Load adaptation and learning state"""
        try:
            import pickle
            with open(storage_path, 'rb') as f:
                state = pickle.load(f)
            
            self.learning_metrics = state.get("learning_metrics", self.learning_metrics)
            self.performance_baselines = state.get("performance_baselines", {})
            self.adaptation_history = state.get("adaptation_history", [])
            self.model_adjustments = state.get("model_adjustments", {})
            self.global_model_preferences = state.get("global_model_preferences", {})
            
            # Load learning engine state
            await self.learning_engine.load_learning_state(
                storage_path.replace('.pkl', '_learning.pkl')
            )
            
            logger.info(f"Adaptation state loaded from {storage_path}")
            return True
            
        except FileNotFoundError:
            logger.info(f"No existing adaptation state found at {storage_path}")
            return False
        except Exception as e:
            logger.error(f"Failed to load adaptation state: {e}")
            return False
    
    async def _adapt_preferences(self, user_id: Optional[str], base_preferences: RoutingPreferences,
                               complexity: ComplexityScore, context_tokens: int) -> RoutingPreferences:
        """Adapt routing preferences based on learned user behavior"""
        
        if not self.config.enable_learning or not user_id:
            return base_preferences
        
        user_profile = self.learning_engine.user_profiles.get(user_id)
        if not user_profile or user_profile.interaction_count < self.config.min_interactions_for_learning:
            return base_preferences
        
        # Create adapted preferences
        adapted = RoutingPreferences(**base_preferences.dict())
        
        # Apply learned preference weights
        adaptation_strength = min(1.0, user_profile.confidence_score * self.config.adaptation_rate)
        
        # Adapt cost sensitivity
        if user_profile.max_acceptable_cost != base_preferences.max_cost_per_query:
            cost_adjustment = (user_profile.max_acceptable_cost - base_preferences.max_cost_per_query) * adaptation_strength
            adapted.max_cost_per_query = max(0.001, base_preferences.max_cost_per_query + cost_adjustment)
        
        # Adapt latency sensitivity
        if user_profile.max_acceptable_latency != base_preferences.max_latency_ms:
            latency_adjustment = (user_profile.max_acceptable_latency - base_preferences.max_latency_ms) * adaptation_strength
            adapted.max_latency_ms = max(500, base_preferences.max_latency_ms + latency_adjustment)
        
        # Adapt quality threshold
        if user_profile.min_acceptable_quality != base_preferences.quality_threshold:
            quality_adjustment = (user_profile.min_acceptable_quality - base_preferences.quality_threshold) * adaptation_strength
            adapted.quality_threshold = max(0.3, min(1.0, base_preferences.quality_threshold + quality_adjustment))
        
        # Adapt preferred models based on learning
        if user_profile.preferred_models:
            # Add top preferred models to preferences
            top_models = sorted(
                user_profile.preferred_models.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            for model_name, preference_score in top_models:
                if preference_score > 0.7 and model_name not in adapted.preferred_models:
                    adapted.preferred_models.append(model_name)
        
        # Add avoided models
        for model_name in user_profile.avoided_models:
            if model_name not in adapted.blocked_models:
                adapted.blocked_models.append(model_name)
        
        return adapted
    
    async def _apply_ab_test_modifications(self, user_id: str, preferences: RoutingPreferences,
                                         query_id: Optional[str]) -> RoutingPreferences:
        """Apply A/B test modifications to preferences"""
        
        # Example A/B test: more aggressive cost optimization
        test_id = "cost_optimization_v1"
        if test_id not in self.ab_testing.active_tests:
            self.ab_testing.create_test(
                test_id,
                "Test more aggressive cost optimization",
                {"cost_weight_multiplier": 1.5}
            )
        
        test_group = self.ab_testing.assign_test_group(user_id, test_id)
        
        if test_group == "treatment":
            # Apply treatment modifications
            modified_preferences = RoutingPreferences(**preferences.dict())
            modified_preferences.max_cost_per_query *= 0.8  # More cost-sensitive
            modified_preferences.prefer_cost_over_quality = True
            return modified_preferences
        
        return preferences
    
    async def _score_with_learned_preferences(
        self, models: List[Model], complexity: ComplexityScore, context_tokens: int,
        preferences: RoutingPreferences, model_preferences: Dict[str, float], user_id: Optional[str]
    ) -> List[ModelScore]:
        """Enhanced MCDA scoring incorporating learned preferences"""
        
        # Get base MCDA scores
        base_scores = await self.advanced_scorer.score_models(
            models, complexity, context_tokens, preferences, self.performance_tracker
        )
        
        # Adjust scores based on learned preferences
        enhanced_scores = []
        for score in base_scores:
            model_name = score.model_name
            learned_preference = model_preferences.get(model_name, 0.5)
            
            # Blend MCDA score with learned preference
            if user_id and user_id in self.learning_engine.user_profiles:
                profile = self.learning_engine.user_profiles[user_id]
                blend_weight = min(0.4, profile.confidence_score * 0.4)  # Max 40% learning weight
            else:
                blend_weight = 0.1  # Small weight for new users
            
            enhanced_total = (
                score.total_score * (1 - blend_weight) + 
                learned_preference * blend_weight
            )
            
            # Create enhanced score
            enhanced_score = ModelScore(
                model_name=score.model_name,
                total_score=enhanced_total,
                criteria_scores=score.criteria_scores.copy(),
                normalized_scores=score.normalized_scores.copy(),
                ranking_position=0,  # Will be set after sorting
                confidence=score.confidence,
                explanation=f"{score.explanation}; Learned preference: {learned_preference:.2f}",
                decision_factors=score.decision_factors + [f"learned_preference: {learned_preference:.2f}"]
            )
            
            # Add learned preference to criteria scores
            enhanced_score.criteria_scores["learned_preference"] = learned_preference
            
            enhanced_scores.append(enhanced_score)
        
        # Sort by enhanced total score
        enhanced_scores.sort(key=lambda x: x.total_score, reverse=True)
        
        # Update ranking positions
        for i, score in enumerate(enhanced_scores):
            score.ranking_position = i + 1
        
        return enhanced_scores
    
    async def _fallback_scoring_with_preferences(
        self, models: List[Model], complexity: ComplexityScore, context_tokens: int,
        preferences: RoutingPreferences, model_preferences: Dict[str, float]
    ) -> List[ModelScore]:
        """Fallback scoring with learned preferences when advanced scoring unavailable"""
        
        enhanced_scores = []
        
        for model in models:
            # Get base score using parent method
            base_score = await self._score_model(model, complexity, context_tokens, preferences)
            learned_preference = model_preferences.get(model.name, 0.5)
            
            # Blend scores
            enhanced_score = base_score * 0.8 + learned_preference * 0.2
            
            model_score = ModelScore(
                model_name=model.name,
                total_score=enhanced_score,
                criteria_scores={"base_score": base_score, "learned_preference": learned_preference},
                normalized_scores={"normalized_score": enhanced_score},
                ranking_position=0,
                confidence=0.7,  # Default confidence for fallback
                explanation=f"Base score: {base_score:.2f}, Learned: {learned_preference:.2f}",
                decision_factors=[f"base_score: {base_score:.2f}", f"learned_preference: {learned_preference:.2f}"]
            )
            
            enhanced_scores.append(model_score)
        
        enhanced_scores.sort(key=lambda x: x.total_score, reverse=True)
        
        for i, score in enumerate(enhanced_scores):
            score.ranking_position = i + 1
        
        return enhanced_scores
    
    async def _select_with_exploration(self, scored_models: List[ModelScore], 
                                     user_id: Optional[str], complexity: ComplexityScore) -> Tuple[Model, ModelScore]:
        """Select model with exploration for learning"""
        
        if not scored_models:
            raise RuntimeError("No scored models available")
        
        # Determine if we should explore
        should_explore = False
        if self.config.enable_learning and user_id:
            user_profile = self.learning_engine.user_profiles.get(user_id)
            if user_profile:
                exploration_probability = min(
                    self.config.exploration_rate,
                    user_profile.exploration_rate
                )
                should_explore = (hash(f"{user_id}_{time.time()}") % 100) < (exploration_probability * 100)
        
        if should_explore and len(scored_models) > 1:
            # Select second or third best model for exploration
            exploration_candidates = scored_models[1:min(3, len(scored_models))]
            selected_score = exploration_candidates[hash(user_id or "default") % len(exploration_candidates)]
            logger.debug(f"Exploring with {selected_score.model_name} instead of top choice")
        else:
            # Select best model
            selected_score = scored_models[0]
        
        # Find the actual model object
        selected_model = next(
            model for model in self.model_registry.models.values()
            if model.name == selected_score.model_name
        )
        
        return selected_model, selected_score
    
    def _calculate_prediction_variance(self, scored_models: List[ModelScore]) -> float:
        """Calculate variance in model scores as measure of prediction uncertainty"""
        if len(scored_models) < 2:
            return 0.0
        
        scores = [model.total_score for model in scored_models]
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        
        return min(1.0, variance)  # Normalize to 0-1
    
    def _generate_adaptive_reasoning(self, model: Model, complexity: ComplexityScore, 
                                   model_score: ModelScore, preferences: RoutingPreferences,
                                   user_profile: Optional[UserProfile]) -> str:
        """Generate reasoning that includes adaptive learning factors"""
        
        base_reasoning = self._generate_routing_reasoning(model, complexity, model_score.total_score, preferences)
        
        adaptive_factors = []
        
        if user_profile:
            if user_profile.interaction_count > 0:
                adaptive_factors.append(f"Based on {user_profile.interaction_count} previous interactions")
            
            if model.name in user_profile.preferred_models:
                preference_strength = user_profile.preferred_models[model.name]
                adaptive_factors.append(f"Strong user preference (score: {preference_strength:.2f})")
            
            if user_profile.confidence_score > 0.8:
                adaptive_factors.append("High confidence in user preference model")
            elif user_profile.confidence_score < 0.4:
                adaptive_factors.append("Low confidence - emphasizing exploration")
        
        if "learned_preference" in model_score.criteria_scores:
            learned_score = model_score.criteria_scores["learned_preference"]
            adaptive_factors.append(f"ML preference prediction: {learned_score:.2f}")
        
        if adaptive_factors:
            return f"{base_reasoning}; Learning factors: {'; '.join(adaptive_factors)}"
        else:
            return base_reasoning
    
    def _get_enhanced_routing_factors(self, model: Model, complexity: ComplexityScore,
                                    preferences: RoutingPreferences, model_score: ModelScore,
                                    model_preferences: Dict[str, float]) -> Dict[str, float]:
        """Get enhanced routing factors including learning components"""
        
        base_factors = self._get_routing_factors(model, complexity, preferences, model_score.total_score)
        
        # Add learning factors
        base_factors.update({
            'learned_preference': model_preferences.get(model.name, 0.5),
            'ml_confidence': model_score.confidence,
            'exploration_factor': 0.0 if model_score.ranking_position == 1 else 0.2,
            'adaptation_strength': getattr(self, '_last_adaptation_strength', 0.0)
        })
        
        return base_factors
    
    async def record_performance_feedback(self, query_id: str, actual_latency: float,
                                        actual_cost: float, actual_quality: Optional[float] = None,
                                        error_occurred: bool = False) -> None:
        """Record actual performance for learning"""
        
        if self.interaction_tracker:
            await self.interaction_tracker.track_performance(
                query_id, actual_latency, actual_cost, actual_quality, error_occurred
            )
        
        # Update learning metrics
        if not error_occurred:
            self.learning_metrics["routing_accuracy"] += 0.01  # Increment success
        
        # Record for A/B testing if applicable
        # This would be enhanced based on query tracking
        
    async def record_user_override(self, query_id: str, preferred_model: str, reason: str = "") -> None:
        """Record when user overrides routing decision"""
        
        if self.interaction_tracker:
            await self.interaction_tracker.track_user_override(query_id, preferred_model, reason)
        
        self.learning_metrics["adaptations_made"] += 1
    
    async def record_explicit_feedback(self, query_id: str, rating: float, feedback: str = "") -> None:
        """Record explicit user feedback"""
        
        if self.interaction_tracker:
            from autom8.core.routing.interaction_tracker import FeedbackType
            await self.interaction_tracker.collect_explicit_feedback(
                query_id, 
                FeedbackType.EXPLICIT_RATING,
                {"rating": rating, "feedback": feedback}
            )
    
    async def get_adaptive_analytics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get analytics about adaptive learning performance"""
        
        analytics = {
            "base_routing_stats": self.get_routing_stats(),
            "learning_metrics": self.learning_metrics.copy(),
            "config": asdict(self.config)
        }
        
        # Add learning insights
        if self.config.enable_learning:
            learning_insights = await self.learning_engine.get_learning_insights(user_id)
            analytics["learning_insights"] = learning_insights
        
        # Add tracking analytics
        if self.interaction_tracker:
            tracking_analytics = await self.interaction_tracker.get_tracking_analytics(user_id)
            analytics["interaction_analytics"] = tracking_analytics
        
        # Add A/B testing results
        if self.config.enable_ab_testing:
            test_results = {}
            for test_id in self.ab_testing.active_tests:
                test_results[test_id] = self.ab_testing.get_test_results(test_id)
            analytics["ab_test_results"] = test_results
        
        return analytics
    
    async def export_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Export learned preferences for a user"""
        return await self.learning_engine.export_learned_preferences(user_id)
    
    async def import_user_preferences(self, user_id: str, preferences_data: Dict[str, Any]) -> None:
        """Import learned preferences for a user"""
        await self.learning_engine.import_learned_preferences(user_id, preferences_data)
        
    async def start_user_session(self, user_id: str, preferences: Optional[RoutingPreferences] = None) -> str:
        """Start a tracked user session"""
        if self.interaction_tracker:
            return await self.interaction_tracker.start_session(user_id, preferences)
        return str(hash(f"{user_id}_{time.time()}"))
    
    async def end_user_session(self, session_id: str, reason: str = "normal") -> None:
        """End a tracked user session"""
        if self.interaction_tracker:
            await self.interaction_tracker.end_session(session_id, reason)