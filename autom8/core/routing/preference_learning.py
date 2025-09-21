"""
Dynamic Preference Learning for ModelRouter

This module implements machine learning capabilities to adapt routing decisions
based on user behavior, outcomes, and feedback. It includes collaborative filtering,
reinforcement learning, and pattern recognition algorithms to continuously improve
model selection accuracy.
"""

import asyncio
import json
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import numpy as np
from collections import defaultdict, deque

from autom8.models.complexity import ComplexityScore, ComplexityTier
from autom8.models.routing import Model, RoutingPreferences, ModelStats, ModelSelection
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class LearningAlgorithm(str, Enum):
    """Machine learning algorithms for preference learning"""
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    REINFORCEMENT_LEARNING = "reinforcement_learning" 
    PATTERN_RECOGNITION = "pattern_recognition"
    ENSEMBLE = "ensemble"  # Combines multiple algorithms


class FeedbackType(str, Enum):
    """Types of user feedback"""
    EXPLICIT_RATING = "explicit_rating"  # User explicitly rates the response
    IMPLICIT_SATISFACTION = "implicit_satisfaction"  # Inferred from behavior
    OVERRIDE = "override"  # User overrides routing decision
    APPROVAL = "approval"  # User approves/rejects routing
    PERFORMANCE = "performance"  # Objective performance metrics


class OutcomeType(str, Enum):
    """Types of routing outcomes"""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    COST_EXCEEDED = "cost_exceeded"
    QUALITY_POOR = "quality_poor"
    USER_SATISFIED = "user_satisfied"
    USER_DISSATISFIED = "user_dissatisfied"


@dataclass
class UserInteraction:
    """Records a user interaction with the routing system"""
    user_id: str
    session_id: str
    timestamp: datetime
    query_text: str
    query_length: int
    complexity: ComplexityScore
    context_tokens: int
    
    # Routing decision
    selected_model: str
    alternative_models: List[str]
    routing_confidence: float
    routing_factors: Dict[str, float]
    
    # User preferences at time of routing
    user_preferences: Dict[str, Any]
    
    # Outcome and feedback
    outcome_type: OutcomeType
    feedback_type: Optional[FeedbackType] = None
    user_rating: Optional[float] = None  # 1-5 scale
    satisfaction_score: Optional[float] = None  # 0-1 scale
    
    # Performance metrics
    actual_latency_ms: Optional[float] = None
    actual_cost: Optional[float] = None
    actual_quality: Optional[float] = None
    error_occurred: bool = False
    retry_count: int = 0
    
    # User actions
    user_override: Optional[str] = None  # Model user preferred instead
    edit_preferences: bool = False
    abandoned_session: bool = False


@dataclass
class UserProfile:
    """User's learned preferences and behavioral patterns"""
    user_id: str
    created_at: datetime
    last_updated: datetime
    
    # Behavioral patterns
    interaction_count: int = 0
    total_sessions: int = 0
    avg_session_length: float = 0.0
    
    # Preference weights (learned)
    capability_weight: float = 0.25
    cost_weight: float = 0.20
    latency_weight: float = 0.15
    privacy_weight: float = 0.15
    ecological_weight: float = 0.10
    reliability_weight: float = 0.15
    
    # Model preferences (learned)
    preferred_models: Dict[str, float] = None  # model -> preference score
    avoided_models: Set[str] = None
    
    # Context-aware preferences
    complexity_preferences: Dict[str, Dict[str, float]] = None  # tier -> preferences
    time_based_preferences: Dict[str, Dict[str, float]] = None  # time_period -> preferences
    
    # Quality standards
    min_acceptable_quality: float = 0.7
    max_acceptable_cost: float = 0.10
    max_acceptable_latency: float = 5000
    
    # Learning metadata
    confidence_score: float = 0.5
    learning_rate: float = 0.1
    exploration_rate: float = 0.15
    
    def __post_init__(self):
        if self.preferred_models is None:
            self.preferred_models = {}
        if self.avoided_models is None:
            self.avoided_models = set()
        if self.complexity_preferences is None:
            self.complexity_preferences = {}
        if self.time_based_preferences is None:
            self.time_based_preferences = {}


@dataclass
class LearningPattern:
    """Discovered pattern in user behavior"""
    pattern_id: str
    pattern_type: str  # e.g., "time_based", "complexity_based", "cost_sensitive"
    confidence: float
    support: int  # Number of interactions supporting this pattern
    created_at: datetime
    last_seen: datetime
    
    # Pattern specifics
    conditions: Dict[str, Any]  # Conditions under which pattern applies
    preferences: Dict[str, float]  # Preference adjustments for this pattern
    
    # Validation
    success_rate: float = 0.0
    user_satisfaction: float = 0.0


class CollaborativeFilter:
    """Collaborative filtering for user preference learning"""
    
    def __init__(self, min_interactions: int = 5):
        self.min_interactions = min_interactions
        self.user_item_matrix = defaultdict(dict)  # user -> item -> rating
        self.item_features = defaultdict(dict)  # item -> features
        self.user_similarities = {}
        
    def add_interaction(self, user_id: str, item_id: str, rating: float, features: Dict[str, float]):
        """Add a user-item interaction"""
        self.user_item_matrix[user_id][item_id] = rating
        self.item_features[item_id].update(features)
        
    def calculate_user_similarity(self, user1: str, user2: str) -> float:
        """Calculate similarity between two users using cosine similarity"""
        items1 = set(self.user_item_matrix[user1].keys())
        items2 = set(self.user_item_matrix[user2].keys())
        common_items = items1.intersection(items2)
        
        if len(common_items) < 2:
            return 0.0
        
        # Calculate cosine similarity
        ratings1 = [self.user_item_matrix[user1][item] for item in common_items]
        ratings2 = [self.user_item_matrix[user2][item] for item in common_items]
        
        dot_product = sum(r1 * r2 for r1, r2 in zip(ratings1, ratings2))
        magnitude1 = math.sqrt(sum(r1 ** 2 for r1 in ratings1))
        magnitude2 = math.sqrt(sum(r2 ** 2 for r2 in ratings2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def predict_preference(self, user_id: str, item_id: str, k: int = 5) -> float:
        """Predict user preference for an item using k-nearest neighbors"""
        if user_id not in self.user_item_matrix:
            return 0.5  # Default neutral preference
        
        # Find similar users
        similarities = []
        for other_user in self.user_item_matrix:
            if other_user != user_id and item_id in self.user_item_matrix[other_user]:
                sim = self.calculate_user_similarity(user_id, other_user)
                if sim > 0:
                    rating = self.user_item_matrix[other_user][item_id]
                    similarities.append((sim, rating))
        
        if not similarities:
            return 0.5
        
        # Sort by similarity and take top k
        similarities.sort(reverse=True)
        top_similarities = similarities[:k]
        
        # Weighted average prediction
        numerator = sum(sim * rating for sim, rating in top_similarities)
        denominator = sum(sim for sim, _ in top_similarities)
        
        if denominator == 0:
            return 0.5
        
        return max(0.0, min(1.0, numerator / denominator))
    
    def get_recommendations(self, user_id: str, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N item recommendations for a user"""
        if user_id not in self.user_item_matrix:
            return []
        
        user_items = set(self.user_item_matrix[user_id].keys())
        all_items = set()
        for items in self.user_item_matrix.values():
            all_items.update(items.keys())
        
        # Items user hasn't interacted with
        candidate_items = all_items - user_items
        
        predictions = []
        for item in candidate_items:
            pred = self.predict_preference(user_id, item)
            predictions.append((item, pred))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]


class ReinforcementLearner:
    """Q-Learning based reinforcement learning for routing decisions"""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9, 
                 epsilon: float = 0.1, epsilon_decay: float = 0.995):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        # Q-table: (state, action) -> Q-value
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.state_visits = defaultdict(int)
        self.action_counts = defaultdict(lambda: defaultdict(int))
        
    def get_state_key(self, complexity: ComplexityScore, context_tokens: int, 
                     preferences: Dict[str, Any], time_of_day: int = None) -> str:
        """Convert routing context to state key"""
        # Discretize continuous values
        complexity_bin = int(complexity.raw_score * 10)
        token_bin = min(int(context_tokens / 500), 10)  # Bins of 500 tokens
        cost_sensitive = preferences.get('max_cost_per_query', 0.10) < 0.05
        latency_sensitive = preferences.get('max_latency_ms', 5000) < 2000
        prefer_local = preferences.get('prefer_local', False)
        
        hour_bin = (time_of_day // 6) if time_of_day else 0  # 4 time periods per day
        
        return f"c{complexity_bin}_t{token_bin}_cs{int(cost_sensitive)}_ls{int(latency_sensitive)}_pl{int(prefer_local)}_h{hour_bin}"
    
    def choose_action(self, state: str, available_actions: List[str]) -> str:
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.choice(available_actions)
        else:
            # Exploitation: best known action
            q_values = {action: self.q_table[state][action] for action in available_actions}
            return max(q_values, key=q_values.get)
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str, 
                      next_actions: List[str]):
        """Update Q-value using Q-learning update rule"""
        current_q = self.q_table[state][action]
        
        if next_actions:
            max_next_q = max(self.q_table[next_state][a] for a in next_actions)
        else:
            max_next_q = 0.0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
        self.action_counts[state][action] += 1
        self.state_visits[state] += 1
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def calculate_reward(self, interaction: UserInteraction) -> float:
        """Calculate reward based on interaction outcome"""
        reward = 0.0
        
        # Base reward from outcome
        if interaction.outcome_type == OutcomeType.SUCCESS:
            reward += 1.0
        elif interaction.outcome_type == OutcomeType.USER_SATISFIED:
            reward += 1.5
        elif interaction.outcome_type == OutcomeType.FAILURE:
            reward -= 1.0
        elif interaction.outcome_type == OutcomeType.USER_DISSATISFIED:
            reward -= 1.5
        
        # Performance-based adjustments
        if interaction.actual_quality is not None:
            reward += (interaction.actual_quality - 0.7) * 2  # Quality above 0.7 is good
        
        if interaction.actual_cost is not None:
            max_cost = interaction.user_preferences.get('max_cost_per_query', 0.10)
            if interaction.actual_cost <= max_cost:
                reward += 0.5
            else:
                reward -= (interaction.actual_cost - max_cost) * 10
        
        if interaction.actual_latency_ms is not None:
            max_latency = interaction.user_preferences.get('max_latency_ms', 5000)
            if interaction.actual_latency_ms <= max_latency:
                reward += 0.3
            else:
                reward -= (interaction.actual_latency_ms - max_latency) / 1000
        
        # User override penalty
        if interaction.user_override:
            reward -= 2.0  # Strong penalty for wrong routing
        
        # Explicit rating bonus
        if interaction.user_rating is not None:
            reward += (interaction.user_rating - 3.0) / 2.0  # Scale 1-5 to -1 to +1
        
        return reward
    
    def get_action_preferences(self, state: str, available_actions: List[str]) -> Dict[str, float]:
        """Get preference scores for available actions"""
        preferences = {}
        for action in available_actions:
            preferences[action] = self.q_table[state][action]
        
        # Normalize to 0-1 range
        if preferences:
            min_val = min(preferences.values())
            max_val = max(preferences.values())
            if max_val > min_val:
                for action in preferences:
                    preferences[action] = (preferences[action] - min_val) / (max_val - min_val)
            else:
                for action in preferences:
                    preferences[action] = 0.5
        
        return preferences


class PatternRecognition:
    """Pattern recognition for identifying user behavior patterns"""
    
    def __init__(self, min_support: int = 3, min_confidence: float = 0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.patterns = {}
        
    def extract_patterns(self, interactions: List[UserInteraction]) -> List[LearningPattern]:
        """Extract behavioral patterns from user interactions"""
        patterns = []
        
        # Time-based patterns
        time_patterns = self._extract_time_patterns(interactions)
        patterns.extend(time_patterns)
        
        # Complexity-based patterns
        complexity_patterns = self._extract_complexity_patterns(interactions)
        patterns.extend(complexity_patterns)
        
        # Cost-sensitivity patterns
        cost_patterns = self._extract_cost_patterns(interactions)
        patterns.extend(cost_patterns)
        
        # Quality preference patterns
        quality_patterns = self._extract_quality_patterns(interactions)
        patterns.extend(quality_patterns)
        
        return [p for p in patterns if p.support >= self.min_support and p.confidence >= self.min_confidence]
    
    def _extract_time_patterns(self, interactions: List[UserInteraction]) -> List[LearningPattern]:
        """Extract time-based behavioral patterns"""
        patterns = []
        
        # Group by hour of day
        hour_groups = defaultdict(list)
        for interaction in interactions:
            hour = interaction.timestamp.hour
            hour_groups[hour].append(interaction)
        
        for hour, hour_interactions in hour_groups.items():
            if len(hour_interactions) >= self.min_support:
                # Analyze model preferences for this hour
                model_preferences = defaultdict(list)
                for interaction in hour_interactions:
                    if interaction.satisfaction_score:
                        model_preferences[interaction.selected_model].append(interaction.satisfaction_score)
                
                # Create pattern if strong preference exists
                for model, scores in model_preferences.items():
                    avg_score = sum(scores) / len(scores)
                    if avg_score > 0.7:
                        pattern = LearningPattern(
                            pattern_id=f"time_{hour}_{model}",
                            pattern_type="time_based",
                            confidence=avg_score,
                            support=len(scores),
                            created_at=datetime.utcnow(),
                            last_seen=max(i.timestamp for i in hour_interactions),
                            conditions={"hour": hour},
                            preferences={"preferred_model": model, "preference_strength": avg_score}
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _extract_complexity_patterns(self, interactions: List[UserInteraction]) -> List[LearningPattern]:
        """Extract complexity-based patterns"""
        patterns = []
        
        # Group by complexity tier
        tier_groups = defaultdict(list)
        for interaction in interactions:
            tier = interaction.complexity.recommended_tier.value
            tier_groups[tier].append(interaction)
        
        for tier, tier_interactions in tier_groups.items():
            if len(tier_interactions) >= self.min_support:
                # Analyze preferences for this complexity
                successful_interactions = [i for i in tier_interactions 
                                         if i.outcome_type in [OutcomeType.SUCCESS, OutcomeType.USER_SATISFIED]]
                
                if len(successful_interactions) / len(tier_interactions) > self.min_confidence:
                    # Find most successful model
                    model_success = defaultdict(lambda: {"count": 0, "success": 0})
                    for interaction in tier_interactions:
                        model_success[interaction.selected_model]["count"] += 1
                        if interaction.outcome_type in [OutcomeType.SUCCESS, OutcomeType.USER_SATISFIED]:
                            model_success[interaction.selected_model]["success"] += 1
                    
                    for model, stats in model_success.items():
                        if stats["count"] >= self.min_support:
                            success_rate = stats["success"] / stats["count"]
                            if success_rate > self.min_confidence:
                                pattern = LearningPattern(
                                    pattern_id=f"complexity_{tier}_{model}",
                                    pattern_type="complexity_based",
                                    confidence=success_rate,
                                    support=stats["count"],
                                    created_at=datetime.utcnow(),
                                    last_seen=max(i.timestamp for i in tier_interactions),
                                    conditions={"complexity_tier": tier},
                                    preferences={"preferred_model": model, "success_rate": success_rate}
                                )
                                patterns.append(pattern)
        
        return patterns
    
    def _extract_cost_patterns(self, interactions: List[UserInteraction]) -> List[LearningPattern]:
        """Extract cost-sensitivity patterns"""
        patterns = []
        
        # Analyze cost overrides and dissatisfaction
        cost_sensitive_interactions = [
            i for i in interactions 
            if i.user_override or i.outcome_type == OutcomeType.COST_EXCEEDED
        ]
        
        if len(cost_sensitive_interactions) >= self.min_support:
            # Calculate cost thresholds where user becomes dissatisfied
            costs = [i.actual_cost for i in cost_sensitive_interactions if i.actual_cost is not None]
            if costs:
                threshold = np.percentile(costs, 75)  # 75th percentile
                pattern = LearningPattern(
                    pattern_id="cost_sensitivity",
                    pattern_type="cost_sensitive",
                    confidence=len(cost_sensitive_interactions) / len(interactions),
                    support=len(cost_sensitive_interactions),
                    created_at=datetime.utcnow(),
                    last_seen=max(i.timestamp for i in cost_sensitive_interactions),
                    conditions={"cost_threshold": threshold},
                    preferences={"cost_weight": min(1.0, 0.3 + len(cost_sensitive_interactions) * 0.1)}
                )
                patterns.append(pattern)
        
        return patterns
    
    def _extract_quality_patterns(self, interactions: List[UserInteraction]) -> List[LearningPattern]:
        """Extract quality preference patterns"""
        patterns = []
        
        # Analyze quality-related feedback
        quality_interactions = [
            i for i in interactions 
            if i.outcome_type in [OutcomeType.QUALITY_POOR, OutcomeType.USER_DISSATISFIED] 
            and i.actual_quality is not None
        ]
        
        if len(quality_interactions) >= self.min_support:
            # Find quality threshold
            quality_scores = [i.actual_quality for i in quality_interactions]
            threshold = np.percentile(quality_scores, 75)  # User expects better than 75% of poor quality
            
            pattern = LearningPattern(
                pattern_id="quality_standards",
                pattern_type="quality_sensitive",
                confidence=len(quality_interactions) / len(interactions),
                support=len(quality_interactions),
                created_at=datetime.utcnow(),
                last_seen=max(i.timestamp for i in quality_interactions),
                conditions={"min_quality_threshold": threshold},
                preferences={"capability_weight": min(1.0, 0.25 + len(quality_interactions) * 0.05)}
            )
            patterns.append(pattern)
        
        return patterns


class PreferenceLearningEngine:
    """Main engine for dynamic preference learning"""
    
    def __init__(self, algorithm: LearningAlgorithm = LearningAlgorithm.ENSEMBLE):
        self.algorithm = algorithm
        self.user_profiles = {}
        self.interaction_history = defaultdict(list)
        self.patterns = {}
        
        # Learning components
        self.collaborative_filter = CollaborativeFilter()
        self.reinforcement_learner = ReinforcementLearner()
        self.pattern_recognition = PatternRecognition()
        
        # Performance tracking
        self.learning_metrics = {
            "total_interactions": 0,
            "successful_predictions": 0,
            "override_rate": 0.0,
            "satisfaction_improvement": 0.0,
            "last_updated": datetime.utcnow()
        }
        
    async def record_interaction(self, interaction: UserInteraction):
        """Record a user interaction for learning"""
        self.interaction_history[interaction.user_id].append(interaction)
        self.learning_metrics["total_interactions"] += 1
        
        # Update user profile
        await self._update_user_profile(interaction)
        
        # Update learning models
        await self._update_learning_models(interaction)
        
        # Check for new patterns
        if len(self.interaction_history[interaction.user_id]) % 10 == 0:
            await self._update_patterns(interaction.user_id)
        
        logger.debug(f"Recorded interaction for user {interaction.user_id}: {interaction.outcome_type}")
    
    async def _update_user_profile(self, interaction: UserInteraction):
        """Update user profile based on interaction"""
        user_id = interaction.user_id
        
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
        
        profile = self.user_profiles[user_id]
        profile.interaction_count += 1
        profile.last_updated = datetime.utcnow()
        
        # Update preference weights based on outcome
        learning_rate = profile.learning_rate
        
        if interaction.user_override:
            # Strong signal - user explicitly disagrees with routing
            if interaction.actual_cost and interaction.actual_cost > profile.max_acceptable_cost:
                profile.cost_weight = min(1.0, profile.cost_weight + learning_rate * 0.2)
            if interaction.actual_latency_ms and interaction.actual_latency_ms > profile.max_acceptable_latency:
                profile.latency_weight = min(1.0, profile.latency_weight + learning_rate * 0.2)
        
        elif interaction.outcome_type == OutcomeType.USER_SATISFIED:
            # Positive signal - reinforce current preferences
            selected_model = interaction.selected_model
            profile.preferred_models[selected_model] = profile.preferred_models.get(selected_model, 0.5) + learning_rate * 0.1
        
        elif interaction.outcome_type == OutcomeType.USER_DISSATISFIED:
            # Negative signal - adjust preferences away from this choice
            selected_model = interaction.selected_model
            if selected_model in profile.preferred_models:
                profile.preferred_models[selected_model] = max(0.0, profile.preferred_models[selected_model] - learning_rate * 0.1)
        
        # Update quality standards
        if interaction.actual_quality is not None:
            if interaction.outcome_type == OutcomeType.USER_SATISFIED and interaction.actual_quality < profile.min_acceptable_quality:
                profile.min_acceptable_quality = max(0.5, profile.min_acceptable_quality - learning_rate * 0.05)
            elif interaction.outcome_type == OutcomeType.QUALITY_POOR:
                profile.min_acceptable_quality = min(0.9, profile.min_acceptable_quality + learning_rate * 0.1)
        
        # Update confidence score
        if interaction.outcome_type in [OutcomeType.SUCCESS, OutcomeType.USER_SATISFIED]:
            profile.confidence_score = min(1.0, profile.confidence_score + learning_rate * 0.05)
        else:
            profile.confidence_score = max(0.1, profile.confidence_score - learning_rate * 0.02)
    
    async def _update_learning_models(self, interaction: UserInteraction):
        """Update all learning models with interaction data"""
        user_id = interaction.user_id
        
        # Update collaborative filter
        rating = self._calculate_implicit_rating(interaction)
        features = {
            "capability": interaction.routing_factors.get("capability_match", 0.5),
            "cost_efficiency": interaction.routing_factors.get("cost_efficiency", 0.5),
            "latency": 1.0 - (interaction.actual_latency_ms or 1000) / 10000,
            "quality": interaction.actual_quality or 0.5
        }
        self.collaborative_filter.add_interaction(user_id, interaction.selected_model, rating, features)
        
        # Update reinforcement learner
        state = self.reinforcement_learner.get_state_key(
            interaction.complexity, 
            interaction.context_tokens,
            interaction.user_preferences,
            interaction.timestamp.hour
        )
        reward = self.reinforcement_learner.calculate_reward(interaction)
        
        # For simplicity, assume next state is similar (this could be improved)
        next_state = state
        available_models = [interaction.selected_model] + interaction.alternative_models
        
        self.reinforcement_learner.update_q_value(
            state, interaction.selected_model, reward, next_state, available_models
        )
    
    def _calculate_implicit_rating(self, interaction: UserInteraction) -> float:
        """Calculate implicit rating from interaction outcome"""
        if interaction.user_rating:
            return interaction.user_rating / 5.0  # Convert 1-5 to 0-1
        
        rating = 0.5  # Neutral
        
        if interaction.outcome_type == OutcomeType.SUCCESS:
            rating = 0.7
        elif interaction.outcome_type == OutcomeType.USER_SATISFIED:
            rating = 0.9
        elif interaction.outcome_type == OutcomeType.FAILURE:
            rating = 0.2
        elif interaction.outcome_type == OutcomeType.USER_DISSATISFIED:
            rating = 0.1
        
        # Adjust based on performance
        if interaction.actual_quality:
            rating = (rating + interaction.actual_quality) / 2
        
        if interaction.user_override:
            rating = max(0.1, rating - 0.3)
        
        return max(0.0, min(1.0, rating))
    
    async def _update_patterns(self, user_id: str):
        """Update behavioral patterns for a user"""
        interactions = self.interaction_history[user_id]
        if len(interactions) < 5:
            return
        
        # Extract patterns from recent interactions
        recent_interactions = interactions[-50:]  # Last 50 interactions
        new_patterns = self.pattern_recognition.extract_patterns(recent_interactions)
        
        for pattern in new_patterns:
            pattern_key = f"{user_id}_{pattern.pattern_id}"
            if pattern_key in self.patterns:
                # Update existing pattern
                existing = self.patterns[pattern_key]
                existing.support += pattern.support
                existing.confidence = (existing.confidence + pattern.confidence) / 2
                existing.last_seen = pattern.last_seen
            else:
                # Add new pattern
                self.patterns[pattern_key] = pattern
        
        logger.debug(f"Updated patterns for user {user_id}: {len(new_patterns)} new patterns")
    
    async def predict_preferences(self, user_id: str, available_models: List[Model],
                                complexity: ComplexityScore, context_tokens: int,
                                base_preferences: RoutingPreferences) -> Dict[str, float]:
        """Predict user preferences for available models"""
        
        if user_id not in self.user_profiles:
            # No data for user, return neutral preferences
            return {model.name: 0.5 for model in available_models}
        
        profile = self.user_profiles[user_id]
        predictions = {}
        
        for model in available_models:
            # Start with base preference
            base_pref = profile.preferred_models.get(model.name, 0.5)
            
            # Collaborative filtering prediction
            cf_pred = self.collaborative_filter.predict_preference(user_id, model.name)
            
            # Reinforcement learning prediction
            state = self.reinforcement_learner.get_state_key(
                complexity, context_tokens, base_preferences.dict(), datetime.now().hour
            )
            rl_prefs = self.reinforcement_learner.get_action_preferences(state, [model.name])
            rl_pred = rl_prefs.get(model.name, 0.5)
            
            # Pattern-based adjustment
            pattern_adjustment = self._get_pattern_adjustment(
                user_id, model.name, complexity, datetime.now().hour
            )
            
            # Ensemble prediction
            if self.algorithm == LearningAlgorithm.ENSEMBLE:
                # Weight different predictions based on confidence and data availability
                cf_weight = 0.3 if profile.interaction_count > 10 else 0.1
                rl_weight = 0.4 if profile.interaction_count > 20 else 0.2
                pattern_weight = 0.2 if len(self.patterns) > 0 else 0.0
                base_weight = 1.0 - cf_weight - rl_weight - pattern_weight
                
                final_pred = (
                    base_pref * base_weight +
                    cf_pred * cf_weight +
                    rl_pred * rl_weight +
                    pattern_adjustment * pattern_weight
                )
            else:
                # Use single algorithm
                if self.algorithm == LearningAlgorithm.COLLABORATIVE_FILTERING:
                    final_pred = cf_pred
                elif self.algorithm == LearningAlgorithm.REINFORCEMENT_LEARNING:
                    final_pred = rl_pred
                else:  # PATTERN_RECOGNITION
                    final_pred = base_pref + pattern_adjustment
            
            predictions[model.name] = max(0.0, min(1.0, final_pred))
        
        return predictions
    
    def _get_pattern_adjustment(self, user_id: str, model_name: str, 
                              complexity: ComplexityScore, hour: int) -> float:
        """Get preference adjustment based on learned patterns"""
        adjustment = 0.0
        
        # Check time-based patterns
        time_pattern_key = f"{user_id}_time_{hour}_{model_name}"
        if time_pattern_key in self.patterns:
            pattern = self.patterns[time_pattern_key]
            adjustment += pattern.preferences.get("preference_strength", 0.0) * pattern.confidence
        
        # Check complexity-based patterns
        complexity_pattern_key = f"{user_id}_complexity_{complexity.recommended_tier.value}_{model_name}"
        if complexity_pattern_key in self.patterns:
            pattern = self.patterns[complexity_pattern_key]
            adjustment += pattern.preferences.get("success_rate", 0.5) * pattern.confidence
        
        return max(-0.5, min(0.5, adjustment))
    
    async def get_learning_insights(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get insights about learning performance and patterns"""
        insights = {
            "total_users": len(self.user_profiles),
            "total_interactions": self.learning_metrics["total_interactions"],
            "discovered_patterns": len(self.patterns),
            "learning_performance": self.learning_metrics.copy()
        }
        
        if user_id and user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            user_patterns = [p for key, p in self.patterns.items() if key.startswith(user_id)]
            
            insights["user_specific"] = {
                "interaction_count": profile.interaction_count,
                "confidence_score": profile.confidence_score,
                "preferred_models": dict(profile.preferred_models),
                "learning_rate": profile.learning_rate,
                "discovered_patterns": len(user_patterns),
                "pattern_types": list(set(p.pattern_type for p in user_patterns))
            }
        
        return insights
    
    async def save_learning_state(self, storage_path: str = "autom8_learning_state.pkl"):
        """Persist learning state to disk"""
        try:
            learning_state = {
                "user_profiles": {uid: asdict(profile) for uid, profile in self.user_profiles.items()},
                "interaction_history": {uid: [asdict(i) for i in interactions] 
                                     for uid, interactions in self.interaction_history.items()},
                "patterns": {pid: asdict(pattern) for pid, pattern in self.patterns.items()},
                "learning_metrics": self.learning_metrics,
                "collaborative_filter_state": {
                    "user_item_matrix": dict(self.collaborative_filter.user_item_matrix),
                    "item_features": dict(self.collaborative_filter.item_features)
                },
                "q_table": {
                    state: dict(actions) for state, actions in self.reinforcement_learner.q_table.items()
                },
                "state_visits": dict(self.reinforcement_learner.state_visits),
                "algorithm": self.algorithm.value,
                "saved_at": datetime.utcnow().isoformat()
            }
            
            import pickle
            with open(storage_path, 'wb') as f:
                pickle.dump(learning_state, f)
            
            logger.info(f"Learning state saved to {storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save learning state: {e}")
            return False
    
    async def load_learning_state(self, storage_path: str = "autom8_learning_state.pkl"):
        """Load learning state from disk"""
        try:
            import pickle
            with open(storage_path, 'rb') as f:
                learning_state = pickle.load(f)
            
            # Restore user profiles
            self.user_profiles = {}
            for uid, profile_data in learning_state.get("user_profiles", {}).items():
                profile_data["created_at"] = datetime.fromisoformat(profile_data["created_at"])
                profile_data["last_interaction"] = datetime.fromisoformat(profile_data["last_interaction"])
                self.user_profiles[uid] = UserProfile(**profile_data)
            
            # Restore interaction history
            self.interaction_history = defaultdict(list)
            for uid, interactions_data in learning_state.get("interaction_history", {}).items():
                for interaction_data in interactions_data:
                    interaction_data["timestamp"] = datetime.fromisoformat(interaction_data["timestamp"])
                    interaction_data["complexity"] = ComplexityScore(**interaction_data["complexity"])
                    interaction_data["outcome_type"] = OutcomeType(interaction_data["outcome_type"])
                    if interaction_data.get("feedback_type"):
                        interaction_data["feedback_type"] = FeedbackType(interaction_data["feedback_type"])
                    self.interaction_history[uid].append(UserInteraction(**interaction_data))
            
            # Restore patterns
            self.patterns = {}
            for pid, pattern_data in learning_state.get("patterns", {}).items():
                pattern_data["created_at"] = datetime.fromisoformat(pattern_data["created_at"])
                pattern_data["last_seen"] = datetime.fromisoformat(pattern_data["last_seen"])
                self.patterns[pid] = LearningPattern(**pattern_data)
            
            # Restore learning metrics
            self.learning_metrics.update(learning_state.get("learning_metrics", {}))
            
            # Restore collaborative filter state
            cf_state = learning_state.get("collaborative_filter_state", {})
            self.collaborative_filter.user_item_matrix = defaultdict(dict, cf_state.get("user_item_matrix", {}))
            self.collaborative_filter.item_features = defaultdict(dict, cf_state.get("item_features", {}))
            
            # Restore Q-table
            q_table_data = learning_state.get("q_table", {})
            self.reinforcement_learner.q_table = defaultdict(lambda: defaultdict(float))
            for state, actions in q_table_data.items():
                self.reinforcement_learner.q_table[state] = defaultdict(float, actions)
            
            # Restore state visits
            self.reinforcement_learner.state_visits = defaultdict(int, learning_state.get("state_visits", {}))
            
            logger.info(f"Learning state loaded from {storage_path}")
            logger.info(f"Restored {len(self.user_profiles)} user profiles, {len(self.patterns)} patterns")
            return True
            
        except FileNotFoundError:
            logger.info(f"No existing learning state found at {storage_path}")
            return False
        except Exception as e:
            logger.error(f"Failed to load learning state: {e}")
            return False
    
    async def get_adaptation_recommendations(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recommendations for system adaptation based on learning"""
        recommendations = []
        
        if user_id:
            # User-specific recommendations
            if user_id in self.user_profiles:
                profile = self.user_profiles[user_id]
                
                # Check if user strongly prefers local models
                local_preference = profile.preferred_models.get("local_average", 0.5)
                if local_preference > 0.8:
                    recommendations.append({
                        "type": "user_preference",
                        "user_id": user_id,
                        "recommendation": "increase_local_routing",
                        "confidence": local_preference,
                        "details": "User strongly prefers local models"
                    })
                
                # Check cost sensitivity
                if profile.cost_weight > 0.7:
                    recommendations.append({
                        "type": "user_preference", 
                        "user_id": user_id,
                        "recommendation": "prioritize_cost_optimization",
                        "confidence": profile.cost_weight,
                        "details": "User is highly cost-sensitive"
                    })
        else:
            # System-wide recommendations
            
            # Analyze override patterns
            total_overrides = 0
            total_interactions = 0
            for interactions in self.interaction_history.values():
                for interaction in interactions:
                    total_interactions += 1
                    if interaction.user_override:
                        total_overrides += 1
            
            if total_interactions > 50:
                override_rate = total_overrides / total_interactions
                if override_rate > 0.15:  # More than 15% override rate
                    recommendations.append({
                        "type": "system_optimization",
                        "recommendation": "improve_routing_accuracy",
                        "confidence": override_rate,
                        "details": f"High override rate: {override_rate:.1%}"
                    })
            
            # Analyze cost patterns
            cost_violations = sum(1 for interactions in self.interaction_history.values()
                                for interaction in interactions
                                if interaction.outcome_type == OutcomeType.COST_EXCEEDED)
            
            if cost_violations > total_interactions * 0.05:  # More than 5% cost violations
                recommendations.append({
                    "type": "system_optimization",
                    "recommendation": "strengthen_cost_controls",
                    "confidence": cost_violations / total_interactions,
                    "details": f"Cost violations: {cost_violations}/{total_interactions}"
                })
            
            # Model distribution recommendations
            model_usage = defaultdict(int)
            for interactions in self.interaction_history.values():
                for interaction in interactions:
                    model_usage[interaction.selected_model] += 1
            
            if model_usage:
                total_usage = sum(model_usage.values())
                local_usage = sum(count for model, count in model_usage.items() 
                               if "local" in model.lower() or "ollama" in model.lower())
                local_percentage = local_usage / total_usage
                
                if local_percentage < 0.7:  # Target 70% local usage
                    recommendations.append({
                        "type": "ecological_optimization",
                        "recommendation": "increase_local_model_usage",
                        "confidence": 1.0 - local_percentage,
                        "details": f"Current local usage: {local_percentage:.1%}, target: 70%"
                    })
        
        return recommendations
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        stats = {
            "users": {
                "total_users": len(self.user_profiles),
                "active_users": len([p for p in self.user_profiles.values() 
                                   if p.last_interaction > datetime.utcnow() - timedelta(days=7)]),
                "experienced_users": len([p for p in self.user_profiles.values() 
                                        if p.interaction_count > 20])
            },
            "interactions": {
                "total_interactions": self.learning_metrics["total_interactions"],
                "successful_predictions": self.learning_metrics["successful_predictions"],
                "override_rate": self.learning_metrics["override_rate"],
                "satisfaction_improvement": self.learning_metrics["satisfaction_improvement"]
            },
            "patterns": {
                "total_patterns": len(self.patterns),
                "time_patterns": len([p for p in self.patterns.values() if p.pattern_type == "time_based"]),
                "complexity_patterns": len([p for p in self.patterns.values() if p.pattern_type == "complexity_based"]),
                "cost_patterns": len([p for p in self.patterns.values() if p.pattern_type == "cost_based"])
            },
            "learning_performance": {
                "q_table_size": len(self.reinforcement_learner.q_table),
                "state_space_coverage": len(self.reinforcement_learner.state_visits),
                "exploration_rate": self.reinforcement_learner.epsilon,
                "collaborative_filter_users": len(self.collaborative_filter.user_item_matrix),
                "collaborative_filter_items": len(set().union(*self.collaborative_filter.user_item_matrix.values()))
            }
        }
        
        return stats
    
    async def export_learned_preferences(self, user_id: str) -> Dict[str, Any]:
        """Export learned preferences for a user"""
        if user_id not in self.user_profiles:
            return {}
        
        profile = self.user_profiles[user_id]
        user_patterns = [p for key, p in self.patterns.items() if key.startswith(user_id)]
        
        return {
            "user_profile": asdict(profile),
            "patterns": [asdict(p) for p in user_patterns],
            "interaction_count": len(self.interaction_history[user_id]),
            "exported_at": datetime.utcnow().isoformat()
        }
    
    async def import_learned_preferences(self, user_id: str, preferences_data: Dict[str, Any]):
        """Import learned preferences for a user"""
        if "user_profile" in preferences_data:
            profile_data = preferences_data["user_profile"]
            self.user_profiles[user_id] = UserProfile(**profile_data)
        
        if "patterns" in preferences_data:
            for pattern_data in preferences_data["patterns"]:
                pattern = LearningPattern(**pattern_data)
                self.patterns[f"{user_id}_{pattern.pattern_id}"] = pattern
        
        logger.info(f"Imported preferences for user {user_id}")