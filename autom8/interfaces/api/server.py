"""
FastAPI Web Server for Autom8 Real-time Dashboard

Provides REST API endpoints and WebSocket connectivity for the React frontend,
with real-time data streaming, system metrics, and agent coordination.
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

from autom8.config.settings import get_settings
from autom8.storage.redis.shared_memory import get_shared_memory, RedisSharedMemory
from autom8.storage.redis.events import get_event_bus, EventBus, EventFilter
from autom8.models.memory import AgentEvent, EventType, Priority
from autom8.interfaces.api.data_service import get_data_service, DashboardDataService
from autom8.interfaces.api.auth import (
    get_current_user, check_rate_limit_dependency, get_auth_manager,
    AuthToken, require_permissions
)

logger = logging.getLogger(__name__)

# Pydantic Models for API Responses
class SystemStatus(BaseModel):
    model_config = {"protected_namespaces": ()}
    """System health and status information."""
    status: str = Field(description="Overall system status")
    uptime: float = Field(description="System uptime in seconds")
    version: str = Field(description="Autom8 version")
    redis_connected: bool = Field(description="Redis connection status")
    active_agents: int = Field(description="Number of active agents")
    total_events: int = Field(description="Total events processed")
    memory_usage: Dict[str, Any] = Field(description="Memory usage statistics")
    last_updated: datetime = Field(description="Last update timestamp")


class AgentInfo(BaseModel):
    model_config = {"protected_namespaces": ()}
    """Agent information and status."""
    id: str = Field(description="Agent ID")
    name: str = Field(description="Agent name")
    status: str = Field(description="Agent status")
    last_active: Optional[datetime] = Field(description="Last activity timestamp")
    current_work: Optional[str] = Field(description="Current work description")
    performance_score: float = Field(description="Performance score")
    tasks_completed: int = Field(description="Number of completed tasks")
    error_count: int = Field(description="Number of errors encountered")


class AgentsList(BaseModel):
    model_config = {"protected_namespaces": ()}
    """List of active agents."""
    agents: List[AgentInfo] = Field(description="List of active agents")
    total_count: int = Field(description="Total number of agents")
    active_count: int = Field(description="Number of active agents")
    idle_count: int = Field(description="Number of idle agents")


class ModelInfo(BaseModel):
    model_config = {"protected_namespaces": ()}
    """Model information and performance metrics."""
    name: str = Field(description="Model name")
    provider: str = Field(description="Model provider")
    available: bool = Field(description="Model availability")
    requests_count: int = Field(description="Total requests processed")
    avg_response_time: float = Field(description="Average response time in seconds")
    success_rate: float = Field(description="Success rate percentage")
    cost_per_request: float = Field(description="Average cost per request")
    last_used: Optional[datetime] = Field(description="Last usage timestamp")


class ModelsStatus(BaseModel):
    model_config = {"protected_namespaces": ()}
    """Status of all available models."""
    models: List[ModelInfo] = Field(description="List of available models")
    total_requests: int = Field(description="Total requests across all models")
    total_cost: float = Field(description="Total cost across all models")
    preferred_model: Optional[str] = Field(description="Currently preferred model")


class ComplexityStats(BaseModel):
    model_config = {"protected_namespaces": ()}
    """Complexity analysis statistics."""
    total_analyses: int = Field(description="Total complexity analyses performed")
    avg_complexity_score: float = Field(description="Average complexity score")
    distribution: Dict[str, int] = Field(description="Complexity level distribution")
    trend_data: List[Dict[str, Any]] = Field(description="Historical trend data")
    last_analysis: Optional[datetime] = Field(description="Last analysis timestamp")


class RoutingStats(BaseModel):
    model_config = {"protected_namespaces": ()}
    """Model routing statistics and performance."""
    total_routes: int = Field(description="Total routing decisions made")
    local_routes: int = Field(description="Routes to local models")
    cloud_routes: int = Field(description="Routes to cloud models")
    avg_routing_time: float = Field(description="Average routing decision time")
    routing_accuracy: float = Field(description="Routing accuracy percentage")
    model_distribution: Dict[str, int] = Field(description="Distribution of routes by model")


class ContextStats(BaseModel):
    model_config = {"protected_namespaces": ()}
    """Context usage and optimization metrics."""
    total_contexts: int = Field(description="Total contexts created")
    avg_context_size: float = Field(description="Average context size in tokens")
    optimization_savings: float = Field(description="Tokens saved through optimization")
    cache_hit_rate: float = Field(description="Context cache hit rate")
    memory_efficiency: float = Field(description="Memory efficiency percentage")


class ContextOptimizeRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    """Request to optimize context for a specific model and constraints."""
    content: str = Field(description="Content to optimize")
    model: str = Field(description="Target model name")
    max_tokens: Optional[int] = Field(default=None, description="Maximum token limit")
    preserve_start: int = Field(default=200, description="Characters to preserve from start")
    preserve_end: int = Field(default=200, description="Characters to preserve from end")
    optimization_level: str = Field(default="balanced", description="Optimization level: aggressive, balanced, conservative")


class ContextOptimizeResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    """Response from context optimization."""
    original_content: str = Field(description="Original content")
    optimized_content: str = Field(description="Optimized content")
    original_tokens: int = Field(description="Original token count")
    optimized_tokens: int = Field(description="Optimized token count")
    tokens_saved: int = Field(description="Number of tokens saved")
    savings_percentage: float = Field(description="Percentage of tokens saved")
    optimization_time_ms: float = Field(description="Time taken for optimization in milliseconds")
    optimization_strategy: str = Field(description="Strategy used for optimization")


class ModelSwitchRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    """Request to switch to a different model."""
    current_model: str = Field(description="Current model name")
    target_model: str = Field(description="Target model name")
    reason: Optional[str] = Field(default=None, description="Reason for model switch")
    content: Optional[str] = Field(default=None, description="Content for compatibility analysis")


class ModelSwitchResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    """Response from model switch operation."""
    success: bool = Field(description="Whether the switch was successful")
    from_model: str = Field(description="Original model")
    to_model: str = Field(description="New model")
    compatibility_score: float = Field(description="Compatibility score between models")
    performance_impact: str = Field(description="Expected performance impact")
    cost_impact: str = Field(description="Expected cost impact")
    recommendations: List[str] = Field(description="Recommendations for the switch")
    switch_time_ms: float = Field(description="Time taken for switch analysis")
    warnings: List[str] = Field(default_factory=list, description="Any warnings about the switch")


class EventMessage(BaseModel):
    model_config = {"protected_namespaces": ()}
    """WebSocket event message."""
    type: str = Field(description="Message type")
    data: Dict[str, Any] = Field(description="Message data")
    timestamp: datetime = Field(description="Message timestamp")
    source: Optional[str] = Field(description="Message source")


class MetricsMessage(BaseModel):
    model_config = {"protected_namespaces": ()}
    """WebSocket metrics message."""
    type: str = "metrics"
    system_status: SystemStatus
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# WebSocket Connection Manager
class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {
            "events": set(),
            "metrics": set()
        }
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, connection_type: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[connection_type].add(websocket)
        self.connection_metadata[websocket] = {
            "type": connection_type,
            "connected_at": datetime.utcnow(),
            "last_ping": datetime.utcnow()
        }
        logger.info(f"WebSocket connected: {connection_type}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        metadata = self.connection_metadata.get(websocket, {})
        connection_type = metadata.get("type")
        
        if connection_type and websocket in self.active_connections[connection_type]:
            self.active_connections[connection_type].remove(websocket)
        
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]
        
        logger.info(f"WebSocket disconnected: {connection_type}")
    
    async def send_to_type(self, connection_type: str, message: dict):
        """Send message to all connections of a specific type."""
        if connection_type not in self.active_connections:
            return
        
        disconnected = []
        for websocket in self.active_connections[connection_type]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket: {e}")
                disconnected.append(websocket)
        
        # Clean up disconnected connections
        for websocket in disconnected:
            self.disconnect(websocket)
    
    async def broadcast_event(self, event: AgentEvent):
        """Broadcast an event to all event subscribers."""
        message = EventMessage(
            type="agent_event",
            data=event.model_dump(),
            timestamp=datetime.utcnow(),
            source=event.source_agent
        )
        await self.send_to_type("events", message.model_dump())
    
    async def broadcast_metrics(self, metrics: SystemStatus):
        """Broadcast metrics to all metrics subscribers."""
        message = MetricsMessage(system_status=metrics)
        await self.send_to_type("metrics", message.model_dump())
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics."""
        return {
            "total_connections": sum(len(conns) for conns in self.active_connections.values()),
            "connections_by_type": {
                conn_type: len(conns) for conn_type, conns in self.active_connections.items()
            },
            "oldest_connection": min(
                (metadata["connected_at"] for metadata in self.connection_metadata.values()),
                default=None
            )
        }


# Global instances
websocket_manager = WebSocketManager()
_shared_memory: Optional[RedisSharedMemory] = None
_event_bus: Optional[EventBus] = None
_data_service: Optional[DashboardDataService] = None
_server_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global _shared_memory, _event_bus, _data_service
    
    logger.info("Starting Autom8 API server...")
    
    # Initialize shared memory, event bus, and data service
    try:
        _shared_memory = await get_shared_memory()
        _event_bus = await get_event_bus()
        _data_service = await get_data_service()
        
        # Start background tasks
        asyncio.create_task(metrics_broadcaster())
        asyncio.create_task(event_forwarder())
        asyncio.create_task(cache_cleanup_task())
        
        logger.info("Autom8 API server started successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down Autom8 API server...")


# Create FastAPI application
app = FastAPI(
    title="Autom8 Dashboard API",
    description="Real-time dashboard API for Autom8 agent system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Security
security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Basic authentication - expand this for production."""
    # For now, just check if a token is provided
    # In production, validate JWT tokens or API keys
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"user_id": "default"}  # Placeholder


# Background Tasks
async def metrics_broadcaster():
    """Background task to broadcast system metrics."""
    while True:
        try:
            if _data_service:
                status = await get_system_status()
                await websocket_manager.broadcast_metrics(status)
        except Exception as e:
            logger.error(f"Error in metrics broadcaster: {e}")
        
        await asyncio.sleep(5)  # Update every 5 seconds


async def event_forwarder():
    """Background task to forward events from EventBus to WebSocket."""
    if not _event_bus:
        return
    
    try:
        # Subscribe to all events
        subscription_id = await _event_bus.subscribe("dashboard", include_broadcasts=True)
        
        async for event in _event_bus.listen(subscription_id):
            await websocket_manager.broadcast_event(event)
            
    except Exception as e:
        logger.error(f"Error in event forwarder: {e}")


async def cache_cleanup_task():
    """Background task to clean up expired cache entries."""
    while True:
        try:
            if _data_service:
                await _data_service.cleanup_cache()
        except Exception as e:
            logger.error(f"Error in cache cleanup: {e}")
        
        await asyncio.sleep(300)  # Clean up every 5 minutes


# Helper Functions
async def get_system_status() -> SystemStatus:
    """Get current system status."""
    uptime = time.time() - _server_start_time
    
    # Get comprehensive metrics from data service
    system_metrics = {}
    if _data_service:
        system_metrics = await _data_service.get_system_metrics()
    
    memory_usage = system_metrics.get("memory_stats", {})
    active_agents = len(await _data_service.get_agent_details()) if _data_service else 0
    
    return SystemStatus(
        status="healthy" if system_metrics.get("redis_status", False) else "degraded",
        uptime=uptime,
        version="1.0.0",
        redis_connected=system_metrics.get("redis_status", False),
        active_agents=active_agents,
        total_events=memory_usage.get("events", 0),
        memory_usage=memory_usage,
        last_updated=datetime.utcnow()
    )


# Authentication and rate limiting endpoint
@app.get("/api/auth/status")
async def auth_status():
    """Get authentication system status and statistics."""
    auth_manager = await get_auth_manager()
    stats = await auth_manager.get_auth_stats()
    
    return {
        "authentication_enabled": True,
        "rate_limiting_enabled": True,
        "stats": stats,
        "demo_credentials": {
            "admin_key": "autom8_admin_key_demo_2024",
            "user_key": "autom8_user_key_demo_2024", 
            "power_user_key": "autom8_power_key_demo_2024"
        }
    }


# Protected REST API Endpoints
@app.get("/api/system/status", response_model=SystemStatus)
async def system_status(
    auth_token: AuthToken = Depends(get_current_user),
    _: bool = Depends(check_rate_limit_dependency)
):
    """Get overall system health and metrics."""
    return await get_system_status()


@app.get("/api/agents/list", response_model=AgentsList)
async def agents_list():
    """Get active agents and their status."""
    if not _data_service:
        return AgentsList(agents=[], total_count=0, active_count=0, idle_count=0)
    
    agent_details = await _data_service.get_agent_details()
    
    agents = []
    for agent_data in agent_details:
        agents.append(AgentInfo(
            id=agent_data["id"],
            name=agent_data["name"],
            status=agent_data["status"],
            last_active=datetime.fromisoformat(agent_data["last_active"]) if agent_data["last_active"] else None,
            current_work=agent_data["current_work"],
            performance_score=agent_data["performance_score"],
            tasks_completed=agent_data["tasks_completed"],
            error_count=agent_data["error_count"]
        ))
    
    return AgentsList(
        agents=agents,
        total_count=len(agents),
        active_count=len([a for a in agents if a.status == "active"]),
        idle_count=len([a for a in agents if a.status == "idle"])
    )


@app.get("/api/models/status", response_model=ModelsStatus)
async def models_status():
    """Get available models and their performance metrics."""
    if not _data_service:
        return ModelsStatus(models=[], total_requests=0, total_cost=0.0, preferred_model=None)
    
    performance_data = await _data_service.get_model_performance_data()
    model_list = performance_data.get("models", [])
    
    models = []
    for model_data in model_list:
        models.append(ModelInfo(
            name=model_data["name"],
            provider=model_data["provider"],
            available=model_data["available"],
            requests_count=model_data["requests_count"],
            avg_response_time=model_data["avg_response_time"],
            success_rate=model_data["success_rate"],
            cost_per_request=model_data["cost_per_request"],
            last_used=datetime.fromisoformat(model_data["last_used"]) if model_data["last_used"] else None
        ))
    
    return ModelsStatus(
        models=models,
        total_requests=sum(m.requests_count for m in models),
        total_cost=sum(m.requests_count * m.cost_per_request for m in models),
        preferred_model=model_list[0]["name"] if model_list else None
    )


@app.get("/api/complexity/stats", response_model=ComplexityStats)
async def complexity_stats():
    """Get complexity analysis statistics."""
    if not _data_service:
        return ComplexityStats(
            total_analyses=0, avg_complexity_score=0.0, distribution={}, 
            trend_data=[], last_analysis=None
        )
    
    complexity_data = await _data_service.get_complexity_analysis_data()
    
    return ComplexityStats(
        total_analyses=complexity_data["total_analyses"],
        avg_complexity_score=complexity_data["avg_complexity_score"],
        distribution=complexity_data["distribution"],
        trend_data=complexity_data["trend_data"],
        last_analysis=datetime.fromisoformat(complexity_data["last_analysis"]) if complexity_data["last_analysis"] else None
    )


@app.get("/api/routing/stats", response_model=RoutingStats)
async def routing_stats():
    """Get model routing distribution and performance."""
    if not _data_service:
        return RoutingStats(
            total_routes=0, local_routes=0, cloud_routes=0, avg_routing_time=0.0,
            routing_accuracy=0.0, model_distribution={}
        )
    
    routing_data = await _data_service.get_routing_statistics()
    
    return RoutingStats(
        total_routes=routing_data["total_routes"],
        local_routes=routing_data["local_routes"],
        cloud_routes=routing_data["cloud_routes"],
        avg_routing_time=routing_data["avg_routing_time"],
        routing_accuracy=routing_data["routing_accuracy"],
        model_distribution=routing_data["model_distribution"]
    )


@app.get("/api/context/stats", response_model=ContextStats)
async def context_stats():
    """Get context usage and optimization metrics."""
    if not _data_service:
        return ContextStats(
            total_contexts=0, avg_context_size=0.0, optimization_savings=0.0,
            cache_hit_rate=0.0, memory_efficiency=0.0
        )
    
    context_data = await _data_service.get_context_optimization_data()
    
    return ContextStats(
        total_contexts=context_data["total_contexts"],
        avg_context_size=context_data["avg_context_size"],
        optimization_savings=context_data["optimization_savings"],
        cache_hit_rate=context_data["cache_hit_rate"],
        memory_efficiency=context_data["memory_efficiency"]
    )


@app.post("/api/context/optimize", response_model=ContextOptimizeResponse)
async def optimize_context(
    request: ContextOptimizeRequest,
    auth_token: AuthToken = Depends(get_current_user),
    _: bool = Depends(check_rate_limit_dependency)
):
    """Optimize context content for a specific model and constraints."""
    import time
    from autom8.utils.cached_tokens import count_tokens_cached
    from autom8.utils.tokens import optimize_for_token_budget
    
    start_time = time.perf_counter()
    
    try:
        # Count original tokens
        original_tokens = await count_tokens_cached(request.content, request.model)
        
        # Determine optimization strategy
        if request.max_tokens is None:
            # No token limit - apply general optimization
            if request.optimization_level == "aggressive":
                max_tokens = int(original_tokens * 0.6)  # 40% reduction
            elif request.optimization_level == "balanced":
                max_tokens = int(original_tokens * 0.8)  # 20% reduction
            else:  # conservative
                max_tokens = int(original_tokens * 0.9)  # 10% reduction
        else:
            max_tokens = request.max_tokens
        
        # Apply optimization if needed
        if original_tokens <= max_tokens:
            # No optimization needed
            optimized_content = request.content
            optimized_tokens = original_tokens
            optimization_strategy = "none_needed"
        else:
            # Apply token budget optimization
            optimized_content = optimize_for_token_budget(
                request.content,
                max_tokens,
                request.model,
                request.preserve_start,
                request.preserve_end
            )
            
            # Count optimized tokens
            optimized_tokens = await count_tokens_cached(optimized_content, request.model)
            
            # Determine strategy used
            if len(optimized_content) < len(request.content) * 0.7:
                optimization_strategy = "aggressive_truncation"
            elif len(optimized_content) < len(request.content) * 0.85:
                optimization_strategy = "balanced_reduction"
            else:
                optimization_strategy = "conservative_trim"
        
        # Calculate savings
        tokens_saved = original_tokens - optimized_tokens
        savings_percentage = (tokens_saved / original_tokens * 100) if original_tokens > 0 else 0
        
        optimization_time_ms = (time.perf_counter() - start_time) * 1000
        
        return ContextOptimizeResponse(
            original_content=request.content,
            optimized_content=optimized_content,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            tokens_saved=tokens_saved,
            savings_percentage=savings_percentage,
            optimization_time_ms=optimization_time_ms,
            optimization_strategy=optimization_strategy
        )
        
    except Exception as e:
        logger.error(f"Context optimization failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Context optimization failed: {str(e)}"
        )


@app.post("/api/models/switch", response_model=ModelSwitchResponse)
async def switch_model(
    request: ModelSwitchRequest,
    auth_token: AuthToken = Depends(get_current_user),
    _: bool = Depends(check_rate_limit_dependency)
):
    """Analyze and perform model switching with compatibility assessment."""
    import time
    from autom8.utils.cached_tokens import count_tokens_cached
    from autom8.utils.tokens import get_token_counter
    
    start_time = time.perf_counter()
    
    try:
        warnings = []
        recommendations = []
        
        # Analyze model compatibility
        token_counter = get_token_counter()
        
        # Get model characteristics
        model_families = {
            "gpt": {"family": "openai", "type": "chat", "context_window": 8192},
            "claude": {"family": "anthropic", "type": "chat", "context_window": 100000},
            "llama": {"family": "meta", "type": "instruct", "context_window": 4096},
            "mixtral": {"family": "mistral", "type": "instruct", "context_window": 32768},
            "phi": {"family": "microsoft", "type": "code", "context_window": 2048},
        }
        
        # Determine model families
        current_family = "unknown"
        target_family = "unknown"
        
        for family_name, info in model_families.items():
            if family_name in request.current_model.lower():
                current_family = info["family"]
            if family_name in request.target_model.lower():
                target_family = info["family"]
        
        # Calculate compatibility score
        compatibility_score = 1.0
        
        if current_family != target_family:
            compatibility_score *= 0.8  # Different families
            warnings.append(f"Switching between different model families ({current_family} → {target_family})")
        
        # Analyze content compatibility if provided
        if request.content:
            current_tokens = await count_tokens_cached(request.content, request.current_model)
            target_tokens = await count_tokens_cached(request.content, request.target_model)
            
            token_difference = abs(current_tokens - target_tokens) / current_tokens if current_tokens > 0 else 0
            
            if token_difference > 0.2:
                compatibility_score *= 0.9
                warnings.append(f"Significant tokenization differences: {token_difference:.1%}")
            
            # Check context window compatibility
            current_limit = model_families.get(current_family.split()[0] if current_family != "unknown" else "gpt", {}).get("context_window", 4096)
            target_limit = model_families.get(target_family.split()[0] if target_family != "unknown" else "gpt", {}).get("context_window", 4096)
            
            if target_tokens > target_limit * 0.8:
                compatibility_score *= 0.7
                warnings.append(f"Content may exceed target model's context window ({target_tokens} tokens)")
        
        # Performance impact analysis
        local_models = ["llama", "mixtral", "phi"]
        cloud_models = ["gpt", "claude"]
        
        current_is_local = any(model in request.current_model.lower() for model in local_models)
        target_is_local = any(model in request.target_model.lower() for model in local_models)
        
        if current_is_local and not target_is_local:
            performance_impact = "Higher latency, better quality"
            cost_impact = "Increased cost"
            recommendations.append("Consider if higher quality justifies increased latency and cost")
        elif not current_is_local and target_is_local:
            performance_impact = "Lower latency, potential quality trade-off"
            cost_impact = "Reduced cost"
            recommendations.append("Monitor output quality after switch")
        else:
            performance_impact = "Similar performance characteristics"
            cost_impact = "Similar cost profile"
        
        # Generate additional recommendations
        if request.reason:
            if "cost" in request.reason.lower():
                recommendations.append("Monitor response quality to ensure cost savings don't impact performance")
            elif "performance" in request.reason.lower():
                recommendations.append("Benchmark response times and quality after switching")
            elif "quality" in request.reason.lower():
                recommendations.append("Evaluate output improvements to justify any cost increase")
        
        # General recommendations
        recommendations.append("Test with representative queries before full deployment")
        if compatibility_score < 0.8:
            recommendations.append("Consider gradual rollout due to compatibility concerns")
        
        switch_time_ms = (time.perf_counter() - start_time) * 1000
        
        return ModelSwitchResponse(
            success=True,
            from_model=request.current_model,
            to_model=request.target_model,
            compatibility_score=compatibility_score,
            performance_impact=performance_impact,
            cost_impact=cost_impact,
            recommendations=recommendations,
            switch_time_ms=switch_time_ms,
            warnings=warnings
        )
        
    except Exception as e:
        logger.error(f"Model switch analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Model switch analysis failed: {str(e)}"
        )


# WebSocket Endpoints
@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """WebSocket endpoint for real-time event streaming."""
    await websocket_manager.connect(websocket, "events")
    try:
        while True:
            # Keep connection alive and handle ping/pong
            await websocket.receive_text()
            websocket_manager.connection_metadata[websocket]["last_ping"] = datetime.utcnow()
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """WebSocket endpoint for real-time system metrics."""
    await websocket_manager.connect(websocket, "metrics")
    try:
        while True:
            # Keep connection alive and handle ping/pong
            await websocket.receive_text()
            websocket_manager.connection_metadata[websocket]["last_ping"] = datetime.utcnow()
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


# WebSocket connection info endpoint
@app.get("/api/websocket/stats")
async def websocket_stats():
    """Get WebSocket connection statistics."""
    return websocket_manager.get_connection_stats()


# Dashboard route
@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard HTML page."""
    dashboard_path = Path(__file__).parent / "static" / "dashboard.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    else:
        return HTMLResponse(
            content="<h1>Dashboard not found</h1><p>Static dashboard files are not available.</p>",
            status_code=404
        )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "uptime": time.time() - _server_start_time
    }


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "autom8.interfaces.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )