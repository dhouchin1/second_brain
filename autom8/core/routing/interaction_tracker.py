"""
User Interaction Tracking and Feedback Collection System

This module handles tracking user interactions with the routing system,
collecting explicit and implicit feedback, and managing user overrides
to enable continuous learning and improvement.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from collections import defaultdict, deque

from autom8.core.routing.preference_learning import (
    UserInteraction, FeedbackType, OutcomeType, PreferenceLearningEngine
)
from autom8.models.complexity import ComplexityScore, ComplexityTier
from autom8.models.routing import Model, RoutingPreferences, ModelSelection
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class InteractionEvent(str, Enum):
    """Types of interaction events to track"""
    ROUTING_REQUEST = "routing_request"
    ROUTING_DECISION = "routing_decision"
    USER_OVERRIDE = "user_override"
    EXPLICIT_FEEDBACK = "explicit_feedback"
    IMPLICIT_FEEDBACK = "implicit_feedback"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    PREFERENCE_UPDATE = "preference_update"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_MEASURED = "performance_measured"


@dataclass
class TrackingEvent:
    """A single tracking event"""
    event_id: str
    event_type: InteractionEvent
    user_id: str
    session_id: str
    timestamp: datetime
    
    # Event data
    data: Dict[str, Any]
    
    # Context
    query_id: Optional[str] = None
    model_name: Optional[str] = None
    complexity_score: Optional[float] = None
    
    # Metadata
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    platform: Optional[str] = None


@dataclass
class UserSession:
    """Represents a user session for tracking interactions"""
    session_id: str
    user_id: str
    started_at: datetime
    last_activity: datetime
    
    # Session metrics
    query_count: int = 0
    override_count: int = 0
    feedback_count: int = 0
    total_cost: float = 0.0
    avg_satisfaction: float = 0.0
    
    # Context
    user_preferences: Optional[Dict[str, Any]] = None
    session_tags: List[str] = None
    
    def __post_init__(self):
        if self.session_tags is None:
            self.session_tags = []


class FeedbackCollector:
    """Collects and processes user feedback"""
    
    def __init__(self):
        self.feedback_handlers = {}
        self.implicit_signals = {
            "response_time": 0.2,      # Weight for response time satisfaction
            "retry_attempts": 0.3,     # Weight for retry behavior
            "session_duration": 0.1,   # Weight for session duration
            "query_refinement": 0.2,   # Weight for query refinement patterns
            "abandonment": 0.2         # Weight for session abandonment
        }
    
    def register_feedback_handler(self, feedback_type: FeedbackType, handler: Callable):
        """Register a handler for specific feedback type"""
        self.feedback_handlers[feedback_type] = handler
    
    async def collect_explicit_feedback(self, user_id: str, session_id: str, 
                                      query_id: str, feedback_data: Dict[str, Any]) -> TrackingEvent:
        """Collect explicit user feedback"""
        event = TrackingEvent(
            event_id=str(uuid.uuid4()),
            event_type=InteractionEvent.EXPLICIT_FEEDBACK,
            user_id=user_id,
            session_id=session_id,
            query_id=query_id,
            timestamp=datetime.utcnow(),
            data=feedback_data
        )
        
        # Process feedback through registered handlers
        if FeedbackType.EXPLICIT_RATING in self.feedback_handlers:
            await self.feedback_handlers[FeedbackType.EXPLICIT_RATING](event)
        
        logger.debug(f"Collected explicit feedback from user {user_id}: {feedback_data}")
        return event
    
    async def infer_implicit_feedback(self, interaction_data: Dict[str, Any]) -> Dict[str, float]:
        """Infer satisfaction from implicit signals"""
        implicit_feedback = {}
        
        # Response time satisfaction
        response_time = interaction_data.get('actual_latency_ms', 0)
        expected_time = interaction_data.get('estimated_latency_ms', 2000)
        if response_time > 0 and expected_time > 0:
            time_satisfaction = max(0, 1 - (response_time - expected_time) / expected_time)
            implicit_feedback['time_satisfaction'] = time_satisfaction
        
        # Cost satisfaction  
        actual_cost = interaction_data.get('actual_cost', 0)
        expected_cost = interaction_data.get('estimated_cost', 0.05)
        user_max_cost = interaction_data.get('user_max_cost', 0.10)
        if actual_cost <= user_max_cost:
            cost_satisfaction = 1.0 - (actual_cost / user_max_cost)
            implicit_feedback['cost_satisfaction'] = cost_satisfaction
        else:
            implicit_feedback['cost_satisfaction'] = 0.0
        
        # Quality satisfaction (if available)
        actual_quality = interaction_data.get('actual_quality')
        if actual_quality is not None:
            implicit_feedback['quality_satisfaction'] = actual_quality
        
        # Retry behavior (negative signal)
        retry_count = interaction_data.get('retry_count', 0)
        implicit_feedback['retry_satisfaction'] = max(0, 1.0 - retry_count * 0.3)
        
        # Session continuation (positive signal)
        continued_session = interaction_data.get('continued_session', True)
        implicit_feedback['session_satisfaction'] = 1.0 if continued_session else 0.3
        
        # Calculate overall implicit satisfaction
        weights = self.implicit_signals
        total_weight = 0
        weighted_sum = 0
        
        for signal, weight in weights.items():
            satisfaction_key = f"{signal.replace('_', '_')}satisfaction".replace("__", "_")
            if satisfaction_key in implicit_feedback:
                weighted_sum += implicit_feedback[satisfaction_key] * weight
                total_weight += weight
        
        if total_weight > 0:
            implicit_feedback['overall_satisfaction'] = weighted_sum / total_weight
        else:
            implicit_feedback['overall_satisfaction'] = 0.5  # Neutral
        
        return implicit_feedback


class InteractionTracker:
    """Main interaction tracking system"""
    
    def __init__(self, learning_engine: PreferenceLearningEngine):
        self.learning_engine = learning_engine
        self.feedback_collector = FeedbackCollector()
        
        # Storage
        self.events = deque(maxlen=10000)  # Keep last 10k events in memory
        self.active_sessions = {}
        self.session_history = defaultdict(list)
        
        # Tracking state
        self.active_queries = {}  # query_id -> query data
        self.pending_feedback = defaultdict(list)  # user_id -> pending feedback requests
        
        # Performance metrics
        self.tracking_metrics = {
            "total_events": 0,
            "active_users": 0,
            "feedback_rate": 0.0,
            "override_rate": 0.0,
            "avg_session_duration": 0.0
        }
        
        # Setup feedback handlers
        self._setup_feedback_handlers()
    
    def _setup_feedback_handlers(self):
        """Setup default feedback handlers"""
        async def handle_explicit_rating(event: TrackingEvent):
            """Handle explicit user ratings"""
            data = event.data
            rating = data.get('rating', 3)  # 1-5 scale
            query_id = event.query_id
            
            if query_id in self.active_queries:
                query_data = self.active_queries[query_id]
                query_data['user_rating'] = rating
                query_data['feedback_timestamp'] = event.timestamp
                
                # Create user interaction for learning
                await self._create_learning_interaction(event.user_id, query_data)
        
        self.feedback_collector.register_feedback_handler(
            FeedbackType.EXPLICIT_RATING, handle_explicit_rating
        )
    
    async def start_session(self, user_id: str, preferences: Optional[RoutingPreferences] = None,
                          context: Optional[Dict[str, Any]] = None) -> str:
        """Start a new user session"""
        session_id = str(uuid.uuid4())
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            started_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            user_preferences=preferences.dict() if preferences else None
        )
        
        self.active_sessions[session_id] = session
        
        # Track session start event
        event = TrackingEvent(
            event_id=str(uuid.uuid4()),
            event_type=InteractionEvent.SESSION_START,
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.utcnow(),
            data={
                "preferences": session.user_preferences,
                "context": context or {}
            }
        )
        
        await self._record_event(event)
        
        logger.info(f"Started session {session_id} for user {user_id}")
        return session_id
    
    async def end_session(self, session_id: str, reason: str = "normal"):
        """End a user session"""
        if session_id not in self.active_sessions:
            logger.warning(f"Attempted to end unknown session: {session_id}")
            return
        
        session = self.active_sessions[session_id]
        duration = (datetime.utcnow() - session.started_at).total_seconds()
        
        # Calculate session metrics
        avg_satisfaction = session.avg_satisfaction / max(1, session.query_count)
        override_rate = session.override_count / max(1, session.query_count)
        
        # Track session end event
        event = TrackingEvent(
            event_id=str(uuid.uuid4()),
            event_type=InteractionEvent.SESSION_END,
            user_id=session.user_id,
            session_id=session_id,
            timestamp=datetime.utcnow(),
            data={
                "duration_seconds": duration,
                "query_count": session.query_count,
                "override_count": session.override_count,
                "feedback_count": session.feedback_count,
                "total_cost": session.total_cost,
                "avg_satisfaction": avg_satisfaction,
                "override_rate": override_rate,
                "reason": reason
            }
        )
        
        await self._record_event(event)
        
        # Move to history and clean up
        self.session_history[session.user_id].append(session)
        del self.active_sessions[session_id]
        
        logger.info(f"Ended session {session_id} after {duration:.0f}s ({session.query_count} queries)")
    
    async def track_routing_request(self, user_id: str, session_id: str, query: str,
                                  complexity: ComplexityScore, context_tokens: int,
                                  preferences: RoutingPreferences) -> str:
        """Track a routing request"""
        query_id = str(uuid.uuid4())
        
        # Store query data for later correlation
        self.active_queries[query_id] = {
            "user_id": user_id,
            "session_id": session_id,
            "query": query,
            "query_length": len(query),
            "complexity": complexity,
            "context_tokens": context_tokens,
            "preferences": preferences.dict(),
            "timestamp": datetime.utcnow(),
            "status": "pending"
        }
        
        # Track request event
        event = TrackingEvent(
            event_id=str(uuid.uuid4()),
            event_type=InteractionEvent.ROUTING_REQUEST,
            user_id=user_id,
            session_id=session_id,
            query_id=query_id,
            timestamp=datetime.utcnow(),
            complexity_score=complexity.raw_score,
            data={
                "query_length": len(query),
                "complexity_tier": complexity.recommended_tier.value,
                "context_tokens": context_tokens,
                "preferences": preferences.dict()
            }
        )
        
        await self._record_event(event)
        return query_id
    
    async def track_routing_decision(self, query_id: str, selection: ModelSelection) -> TrackingEvent:
        """Track a routing decision"""
        if query_id not in self.active_queries:
            logger.warning(f"Routing decision for unknown query: {query_id}")
            return None
        
        query_data = self.active_queries[query_id]
        query_data.update({
            "selected_model": selection.primary_model.name,
            "alternative_models": [m.name for m in selection.alternatives],
            "routing_confidence": selection.confidence,
            "routing_factors": selection.routing_factors,
            "estimated_cost": selection.estimated_cost,
            "estimated_latency": selection.estimated_latency_ms,
            "estimated_quality": selection.estimated_quality,
            "status": "routed"
        })
        
        # Update session
        if query_data["session_id"] in self.active_sessions:
            session = self.active_sessions[query_data["session_id"]]
            session.query_count += 1
            session.last_activity = datetime.utcnow()
        
        # Track decision event
        event = TrackingEvent(
            event_id=str(uuid.uuid4()),
            event_type=InteractionEvent.ROUTING_DECISION,
            user_id=query_data["user_id"],
            session_id=query_data["session_id"],
            query_id=query_id,
            model_name=selection.primary_model.name,
            timestamp=datetime.utcnow(),
            data={
                "selected_model": selection.primary_model.name,
                "alternatives": [m.name for m in selection.alternatives],
                "confidence": selection.confidence,
                "estimated_cost": selection.estimated_cost,
                "estimated_latency": selection.estimated_latency_ms,
                "estimated_quality": selection.estimated_quality,
                "reasoning": selection.selection_reasoning
            }
        )
        
        await self._record_event(event)
        return event
    
    async def track_user_override(self, query_id: str, preferred_model: str, reason: str = "") -> TrackingEvent:
        """Track when user overrides routing decision"""
        if query_id not in self.active_queries:
            logger.warning(f"Override for unknown query: {query_id}")
            return None
        
        query_data = self.active_queries[query_id]
        query_data.update({
            "user_override": preferred_model,
            "override_reason": reason,
            "override_timestamp": datetime.utcnow()
        })
        
        # Update session
        if query_data["session_id"] in self.active_sessions:
            session = self.active_sessions[query_data["session_id"]]
            session.override_count += 1
        
        # Track override event
        event = TrackingEvent(
            event_id=str(uuid.uuid4()),
            event_type=InteractionEvent.USER_OVERRIDE,
            user_id=query_data["user_id"],
            session_id=query_data["session_id"],
            query_id=query_id,
            model_name=preferred_model,
            timestamp=datetime.utcnow(),
            data={
                "original_model": query_data.get("selected_model"),
                "preferred_model": preferred_model,
                "reason": reason,
                "routing_confidence": query_data.get("routing_confidence", 0.0)
            }
        )
        
        await self._record_event(event)
        
        # Create learning interaction immediately for overrides (strong signal)
        await self._create_learning_interaction(query_data["user_id"], query_data, OutcomeType.USER_DISSATISFIED)
        
        return event
    
    async def track_performance(self, query_id: str, actual_latency: float, 
                              actual_cost: float, actual_quality: Optional[float] = None,
                              error_occurred: bool = False) -> TrackingEvent:
        """Track actual performance metrics"""
        if query_id not in self.active_queries:
            logger.warning(f"Performance tracking for unknown query: {query_id}")
            return None
        
        query_data = self.active_queries[query_id]
        query_data.update({
            "actual_latency_ms": actual_latency,
            "actual_cost": actual_cost,
            "actual_quality": actual_quality,
            "error_occurred": error_occurred,
            "performance_timestamp": datetime.utcnow(),
            "status": "completed" if not error_occurred else "failed"
        })
        
        # Update session
        if query_data["session_id"] in self.active_sessions:
            session = self.active_sessions[query_data["session_id"]]
            session.total_cost += actual_cost
        
        # Infer implicit feedback
        implicit_feedback = await self.feedback_collector.infer_implicit_feedback(query_data)
        query_data["implicit_feedback"] = implicit_feedback
        
        # Track performance event
        event = TrackingEvent(
            event_id=str(uuid.uuid4()),
            event_type=InteractionEvent.PERFORMANCE_MEASURED,
            user_id=query_data["user_id"],
            session_id=query_data["session_id"],
            query_id=query_id,
            model_name=query_data.get("selected_model"),
            timestamp=datetime.utcnow(),
            data={
                "actual_latency_ms": actual_latency,
                "actual_cost": actual_cost,
                "actual_quality": actual_quality,
                "error_occurred": error_occurred,
                "estimated_latency_ms": query_data.get("estimated_latency"),
                "estimated_cost": query_data.get("estimated_cost"),
                "estimated_quality": query_data.get("estimated_quality"),
                "implicit_feedback": implicit_feedback
            }
        )
        
        await self._record_event(event)
        
        # Create learning interaction for performance data
        outcome = OutcomeType.SUCCESS if not error_occurred else OutcomeType.FAILURE
        await self._create_learning_interaction(query_data["user_id"], query_data, outcome)
        
        return event
    
    async def collect_explicit_feedback(self, query_id: str, feedback_type: FeedbackType,
                                      feedback_data: Dict[str, Any]) -> TrackingEvent:
        """Collect explicit user feedback"""
        if query_id not in self.active_queries:
            logger.warning(f"Feedback for unknown query: {query_id}")
            return None
        
        query_data = self.active_queries[query_id]
        
        # Update session
        if query_data["session_id"] in self.active_sessions:
            session = self.active_sessions[query_data["session_id"]]
            session.feedback_count += 1
            
            # Update session satisfaction
            if 'rating' in feedback_data:
                rating = feedback_data['rating'] / 5.0  # Normalize to 0-1
                session.avg_satisfaction = (session.avg_satisfaction * (session.feedback_count - 1) + rating) / session.feedback_count
        
        # Collect feedback through collector
        event = await self.feedback_collector.collect_explicit_feedback(
            query_data["user_id"], query_data["session_id"], query_id, feedback_data
        )
        
        await self._record_event(event)
        return event
    
    async def _create_learning_interaction(self, user_id: str, query_data: Dict[str, Any], 
                                         outcome: Optional[OutcomeType] = None) -> None:
        """Create a UserInteraction for the learning engine"""
        
        # Determine outcome if not provided
        if outcome is None:
            if query_data.get("error_occurred", False):
                outcome = OutcomeType.FAILURE
            elif query_data.get("user_override"):
                outcome = OutcomeType.USER_DISSATISFIED
            elif query_data.get("implicit_feedback", {}).get("overall_satisfaction", 0.5) > 0.7:
                outcome = OutcomeType.USER_SATISFIED
            else:
                outcome = OutcomeType.SUCCESS
        
        # Determine feedback type
        feedback_type = None
        if query_data.get("user_rating"):
            feedback_type = FeedbackType.EXPLICIT_RATING
        elif query_data.get("user_override"):
            feedback_type = FeedbackType.OVERRIDE
        elif query_data.get("implicit_feedback"):
            feedback_type = FeedbackType.IMPLICIT_SATISFACTION
        
        interaction = UserInteraction(
            user_id=user_id,
            session_id=query_data.get("session_id", ""),
            timestamp=query_data.get("timestamp", datetime.utcnow()),
            query_text=query_data.get("query", ""),
            query_length=query_data.get("query_length", 0),
            complexity=query_data.get("complexity"),
            context_tokens=query_data.get("context_tokens", 0),
            selected_model=query_data.get("selected_model", ""),
            alternative_models=query_data.get("alternative_models", []),
            routing_confidence=query_data.get("routing_confidence", 0.5),
            routing_factors=query_data.get("routing_factors", {}),
            user_preferences=query_data.get("preferences", {}),
            outcome_type=outcome,
            feedback_type=feedback_type,
            user_rating=query_data.get("user_rating"),
            satisfaction_score=query_data.get("implicit_feedback", {}).get("overall_satisfaction"),
            actual_latency_ms=query_data.get("actual_latency_ms"),
            actual_cost=query_data.get("actual_cost"),
            actual_quality=query_data.get("actual_quality"),
            error_occurred=query_data.get("error_occurred", False),
            user_override=query_data.get("user_override")
        )
        
        await self.learning_engine.record_interaction(interaction)
    
    async def _record_event(self, event: TrackingEvent):
        """Record a tracking event"""
        self.events.append(event)
        self.tracking_metrics["total_events"] += 1
        
        # Update metrics
        active_user_sessions = len(set(s.user_id for s in self.active_sessions.values()))
        self.tracking_metrics["active_users"] = active_user_sessions
        
        logger.debug(f"Recorded event: {event.event_type} for user {event.user_id}")
    
    async def get_tracking_analytics(self, user_id: Optional[str] = None, 
                                   time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get analytics about tracking and interactions"""
        time_filter = datetime.utcnow() - time_window if time_window else None
        
        # Filter events
        if time_filter:
            filtered_events = [e for e in self.events if e.timestamp >= time_filter]
        else:
            filtered_events = list(self.events)
        
        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]
        
        # Calculate analytics
        total_events = len(filtered_events)
        unique_users = len(set(e.user_id for e in filtered_events))
        unique_sessions = len(set(e.session_id for e in filtered_events))
        
        event_counts = defaultdict(int)
        for event in filtered_events:
            event_counts[event.event_type] += 1
        
        # Feedback and override rates
        feedback_events = event_counts[InteractionEvent.EXPLICIT_FEEDBACK]
        override_events = event_counts[InteractionEvent.USER_OVERRIDE]
        routing_decisions = event_counts[InteractionEvent.ROUTING_DECISION]
        
        feedback_rate = feedback_events / max(1, routing_decisions)
        override_rate = override_events / max(1, routing_decisions)
        
        # Session analytics
        completed_sessions = [s for sessions in self.session_history.values() for s in sessions]
        if time_filter:
            completed_sessions = [s for s in completed_sessions if s.started_at >= time_filter]
        
        if user_id:
            completed_sessions = [s for s in completed_sessions if s.user_id == user_id]
        
        if completed_sessions:
            avg_session_duration = sum(
                (s.last_activity - s.started_at).total_seconds() for s in completed_sessions
            ) / len(completed_sessions)
            avg_queries_per_session = sum(s.query_count for s in completed_sessions) / len(completed_sessions)
        else:
            avg_session_duration = 0.0
            avg_queries_per_session = 0.0
        
        analytics = {
            "time_window": time_window.total_seconds() if time_window else "all_time",
            "total_events": total_events,
            "unique_users": unique_users,
            "unique_sessions": unique_sessions,
            "event_breakdown": dict(event_counts),
            "feedback_rate": feedback_rate,
            "override_rate": override_rate,
            "avg_session_duration_seconds": avg_session_duration,
            "avg_queries_per_session": avg_queries_per_session,
            "active_sessions": len(self.active_sessions),
            "tracking_metrics": self.tracking_metrics.copy()
        }
        
        if user_id:
            # Add user-specific analytics
            user_sessions = [s for s in completed_sessions if s.user_id == user_id]
            if user_sessions:
                user_analytics = {
                    "total_sessions": len(user_sessions),
                    "total_queries": sum(s.query_count for s in user_sessions),
                    "total_overrides": sum(s.override_count for s in user_sessions),
                    "total_cost": sum(s.total_cost for s in user_sessions),
                    "avg_satisfaction": sum(s.avg_satisfaction for s in user_sessions) / len(user_sessions)
                }
                analytics["user_specific"] = user_analytics
        
        return analytics
    
    async def cleanup_old_data(self, retention_days: int = 30):
        """Clean up old tracking data"""
        cutoff_time = datetime.utcnow() - timedelta(days=retention_days)
        
        # Clean up completed queries
        old_queries = [qid for qid, data in self.active_queries.items() 
                      if data.get("timestamp", datetime.utcnow()) < cutoff_time]
        for qid in old_queries:
            del self.active_queries[qid]
        
        # Clean up old sessions from history
        for user_id in list(self.session_history.keys()):
            self.session_history[user_id] = [
                s for s in self.session_history[user_id] 
                if s.started_at >= cutoff_time
            ]
            if not self.session_history[user_id]:
                del self.session_history[user_id]
        
        logger.info(f"Cleaned up data older than {retention_days} days")


# Global instance
_interaction_tracker = None


async def get_interaction_tracker(learning_engine: PreferenceLearningEngine) -> InteractionTracker:
    """Get global interaction tracker instance"""
    global _interaction_tracker
    
    if _interaction_tracker is None:
        _interaction_tracker = InteractionTracker(learning_engine)
    
    return _interaction_tracker