#!/usr/bin/env python3
"""
Simple Autom8-compatible AI Routing Microservice for Second Brain

Provides intelligent model routing, cost tracking, and performance monitoring
for AI requests with fallback to local Ollama.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security Headers Middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response

# Models for API
class GenerateRequest(BaseModel):
    prompt: str
    context: Optional[str] = None
    task_type: str = "general"
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    complexity_hint: Optional[str] = None

class GenerateResponse(BaseModel):
    content: str
    model_used: str
    cost: float
    response_time: float
    provider: str
    complexity_score: Optional[float] = None
    tokens_used: Optional[int] = None

@dataclass
class ModelStats:
    name: str
    provider: str
    available: bool
    requests_count: int = 0
    avg_response_time: float = 0.0
    success_rate: float = 100.0
    cost_per_request: float = 0.0
    last_used: Optional[datetime] = None
    total_cost: float = 0.0

class SimpleAutom8Service:
    """Simple AI routing service with cost tracking."""

    def __init__(self):
        self.models: Dict[str, ModelStats] = {
            "llama3.2": ModelStats("llama3.2", "ollama", True, cost_per_request=0.0),
            "gpt-4": ModelStats("gpt-4", "openai", False, cost_per_request=0.03),
            "claude-3-sonnet": ModelStats("claude-3-sonnet", "anthropic", False, cost_per_request=0.003),
        }
        self.request_history = []
        self.total_requests = 0
        self.start_time = datetime.now()

    def select_model(self, task_type: str, complexity_hint: str = None) -> str:
        """Select the best available model for the task."""
        # Simple routing logic - use Ollama for now
        if self.models["llama3.2"].available:
            return "llama3.2"
        return "llama3.2"  # Always fallback to Ollama

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate content using selected model."""
        start_time = time.time()
        selected_model = self.select_model(request.task_type, request.complexity_hint)

        try:
            # Use Ollama for now
            response = await self._ollama_generate(request.prompt, selected_model)
            response_time = time.time() - start_time

            # Track stats
            model_stats = self.models[selected_model]
            model_stats.requests_count += 1
            model_stats.last_used = datetime.now()
            model_stats.avg_response_time = (
                (model_stats.avg_response_time * (model_stats.requests_count - 1) + response_time)
                / model_stats.requests_count
            )

            self.total_requests += 1

            # Add to history
            self.request_history.append({
                "timestamp": datetime.now(),
                "model": selected_model,
                "task_type": request.task_type,
                "response_time": response_time,
                "cost": model_stats.cost_per_request
            })

            return GenerateResponse(
                content=response,
                model_used=selected_model,
                cost=model_stats.cost_per_request,
                response_time=response_time,
                provider=model_stats.provider,
                tokens_used=int(len(response.split()) * 1.3)  # Rough estimate
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _ollama_generate(self, prompt: str, model: str) -> str:
        """Generate using local Ollama."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return f"Generation failed: {str(e)}"

    def get_model_stats(self) -> Dict[str, Any]:
        """Get model performance statistics."""
        return {
            "models": [asdict(model) for model in self.models.values()],
            "total_requests": self.total_requests,
            "total_cost": sum(model.total_cost for model in self.models.values()),
            "preferred_model": "llama3.2"
        }

    def get_cost_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get cost summary for specified period."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_requests = [
            req for req in self.request_history
            if req["timestamp"] > cutoff
        ]

        total_cost = sum(req["cost"] for req in recent_requests)

        return {
            "total_cost": total_cost,
            "requests_count": len(recent_requests),
            "cost_by_model": {},
            "cost_by_provider": {"ollama": 0.0},
            "projected_monthly_cost": (total_cost / max(hours, 1)) * 24 * 30,
            "period_hours": hours
        }

# Create the FastAPI app
app = FastAPI(
    title="Simple Autom8 AI Router",
    description="AI routing microservice with cost tracking",
    version="1.0.0"
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add security middleware
app.add_middleware(SecurityHeadersMiddleware)

# Add CORS middleware - Secure configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8082",
        "http://127.0.0.1:8082",
        "http://localhost:3000",  # React dev if needed
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["Authorization", "Content-Type", "Accept", "Origin", "X-Requested-With"],
)

# Global service instance
autom8_service = SimpleAutom8Service()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.post("/api/generate", response_model=GenerateResponse)
@limiter.limit("10/minute")
async def generate(request: Request, generate_request: GenerateRequest):
    """Generate content using AI routing."""
    return await autom8_service.generate(generate_request)

@app.get("/api/models/status")
@limiter.limit("30/minute")
async def get_models_status(request: Request):
    """Get model status and statistics."""
    return autom8_service.get_model_stats()

@app.get("/api/costs/summary")
@limiter.limit("30/minute")
async def get_costs_summary(request: Request, hours: int = 24):
    """Get cost summary for specified period."""
    return autom8_service.get_cost_summary(hours)

@app.get("/api/system/status")
@limiter.limit("60/minute")
async def get_system_status(request: Request):
    """Get system status."""
    uptime = (datetime.now() - autom8_service.start_time).total_seconds()
    return {
        "status": "healthy",
        "uptime": uptime,
        "version": "1.0.0",
        "redis_connected": False,
        "active_agents": 1,
        "total_events": autom8_service.total_requests,
        "memory_usage": {},
        "last_updated": datetime.now()
    }

@app.get("/")
async def root():
    """Root endpoint with basic info."""
    return {
        "service": "Simple Autom8 AI Router",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "generate": "/api/generate",
            "models": "/api/models/status",
            "costs": "/api/costs/summary",
            "system": "/api/system/status"
        }
    }

if __name__ == "__main__":
    print("🚀 Starting Simple Autom8 AI Router on port 8000")
    print("📊 Dashboard available at: http://localhost:8000/")
    print("📚 API endpoints available at: http://localhost:8000/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )