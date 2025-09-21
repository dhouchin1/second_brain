"""
Autom8 Client Service for Second Brain

Provides intelligent AI model routing and cost optimization by integrating
with the Autom8 microservice for model selection and request optimization.
"""

import asyncio
import logging
import httpx
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class Autom8Response:
    """Response from Autom8 API with model selection and cost info."""
    content: str
    model_used: str
    cost: float
    response_time: float
    provider: str
    complexity_score: Optional[float] = None
    tokens_used: Optional[int] = None

@dataclass
class Autom8Request:
    """Request structure for Autom8 API."""
    prompt: str
    context: Optional[str] = None
    task_type: str = "general"  # general, summarization, tagging, title_generation
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    complexity_hint: Optional[str] = None  # simple, medium, complex

class Autom8Client:
    """Client for interacting with Autom8 AI routing service."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=5.0)  # Reduced timeout
        self._available = False  # Start as unavailable
        self._last_health_check = None
        self._health_check_interval = 60  # Check every 60 seconds
        self._consecutive_failures = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def health_check(self) -> bool:
        """Check if Autom8 service is available."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            if response.status_code == 200:
                self._available = True
                self._consecutive_failures = 0
                logger.debug("Autom8 service is available")
            else:
                self._available = False
                self._consecutive_failures += 1
            self._last_health_check = datetime.now()
            return self._available
        except Exception as e:
            self._consecutive_failures += 1
            self._available = False

            # Only log every 10th failure to reduce noise
            if self._consecutive_failures == 1 or self._consecutive_failures % 10 == 0:
                logger.debug(f"Autom8 service unavailable (attempt {self._consecutive_failures})")

            self._last_health_check = datetime.now()
            return False

    async def is_available(self) -> bool:
        """Check if service is available (cached with TTL)."""
        now = datetime.now()
        if (self._last_health_check is None or
            (now - self._last_health_check).seconds > self._health_check_interval):
            return await self.health_check()
        return self._available or False

    async def generate(self, request: Autom8Request) -> Autom8Response:
        """Generate content using Autom8 intelligent routing."""
        if not await self.is_available():
            raise Exception("Autom8 service not available")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "prompt": request.prompt,
            "context": request.context,
            "task_type": request.task_type,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "complexity_hint": request.complexity_hint
        }

        start_time = datetime.now()
        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers=headers
            )
            response.raise_for_status()

            data = response.json()
            response_time = (datetime.now() - start_time).total_seconds()

            return Autom8Response(
                content=data.get("content", ""),
                model_used=data.get("model_used", "unknown"),
                cost=data.get("cost", 0.0),
                response_time=response_time,
                provider=data.get("provider", "unknown"),
                complexity_score=data.get("complexity_score"),
                tokens_used=data.get("tokens_used")
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Autom8 API error: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Autom8 API error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Autom8 request failed: {e}")
            raise

    async def get_model_stats(self) -> Dict[str, Any]:
        """Get model performance and cost statistics."""
        try:
            if not await self.is_available():
                logger.debug("Autom8 service not available, returning mock model data")
                return {
                    "models": [
                        {
                            "name": "llama3.2",
                            "provider": "ollama",
                            "available": True,
                            "requests_count": 0,
                            "avg_response_time": 0.0,
                            "success_rate": 100.0,
                            "cost_per_request": 0.0,
                            "last_used": None
                        }
                    ],
                    "total_requests": 0,
                    "total_cost": 0.0,
                    "preferred_model": "llama3.2",
                    "service_status": "unavailable"
                }

            response = await self.client.get(f"{self.base_url}/api/models/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug(f"Autom8 model API unavailable (expected): {e}")
            return {
                "models": [
                    {
                        "name": "llama3.2",
                        "provider": "ollama",
                        "available": True,
                        "requests_count": 0,
                        "avg_response_time": 0.0,
                        "success_rate": 100.0,
                        "cost_per_request": 0.0,
                        "last_used": None
                    }
                ],
                "total_requests": 0,
                "total_cost": 0.0,
                "preferred_model": "llama3.2",
                "service_status": "unavailable"
            }

    async def get_cost_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get cost summary for the specified time period."""
        try:
            if not await self.is_available():
                logger.debug("Autom8 service not available, returning mock cost data")
                return {
                    "total_cost": 0.0,
                    "requests_count": 0,
                    "cost_by_model": {},
                    "cost_by_provider": {},
                    "projected_monthly_cost": 0.0,
                    "period_hours": hours,
                    "service_status": "unavailable"
                }

            response = await self.client.get(
                f"{self.base_url}/api/costs/summary",
                params={"hours": hours}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug(f"Autom8 cost API unavailable (expected): {e}")
            return {
                "total_cost": 0.0,
                "requests_count": 0,
                "cost_by_model": {},
                "cost_by_provider": {},
                "projected_monthly_cost": 0.0,
                "period_hours": hours,
                "service_status": "unavailable"
            }

class Autom8Service:
    """High-level service for Autom8 integration with fallback to Ollama."""

    def __init__(self, autom8_url: str = "http://localhost:8000",
                 ollama_url: str = "http://localhost:11434"):
        self.autom8_client = Autom8Client(autom8_url)
        self.ollama_url = ollama_url
        self.ollama_client = httpx.AsyncClient(timeout=30.0)
        self.stats = {
            "autom8_requests": 0,
            "ollama_fallbacks": 0,
            "total_cost_saved": 0.0,
            "total_requests": 0
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.autom8_client.client.aclose()
        await self.ollama_client.aclose()

    async def generate_with_fallback(self, prompt: str, context: str = "",
                                   task_type: str = "general",
                                   model: str = "llama3.2") -> Dict[str, Any]:
        """Generate content with Autom8 routing and Ollama fallback."""
        self.stats["total_requests"] += 1

        # Try Autom8 first
        try:
            request = Autom8Request(
                prompt=prompt,
                context=context,
                task_type=task_type,
                complexity_hint=self._get_complexity_hint(prompt, task_type)
            )

            response = await self.autom8_client.generate(request)
            self.stats["autom8_requests"] += 1

            logger.info(f"Autom8 response: model={response.model_used}, "
                       f"cost=${response.cost:.4f}, time={response.response_time:.2f}s")

            return {
                "content": response.content,
                "source": "autom8",
                "model_used": response.model_used,
                "cost": response.cost,
                "provider": response.provider,
                "response_time": response.response_time
            }

        except Exception as e:
            logger.warning(f"Autom8 failed, falling back to Ollama: {e}")
            return await self._ollama_fallback(prompt, model)

    async def _ollama_fallback(self, prompt: str, model: str) -> Dict[str, Any]:
        """Fallback to direct Ollama generation."""
        self.stats["ollama_fallbacks"] += 1

        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }

            start_time = datetime.now()
            response = await self.ollama_client.post(
                f"{self.ollama_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            response_time = (datetime.now() - start_time).total_seconds()

            data = response.json()

            logger.info(f"Ollama fallback: model={model}, time={response_time:.2f}s")

            return {
                "content": data.get("response", ""),
                "source": "ollama",
                "model_used": model,
                "cost": 0.0,  # Ollama is free
                "provider": "ollama",
                "response_time": response_time
            }

        except Exception as e:
            logger.error(f"Ollama fallback failed: {e}")
            raise Exception("Both Autom8 and Ollama unavailable")

    def _get_complexity_hint(self, prompt: str, task_type: str) -> str:
        """Estimate complexity based on prompt and task type."""
        # Simple heuristics for complexity estimation
        if task_type in ["title_generation", "tagging"]:
            return "simple"
        elif len(prompt) > 2000 or "analyze" in prompt.lower() or "complex" in prompt.lower():
            return "complex"
        else:
            return "medium"

    async def get_stats(self) -> Dict[str, Any]:
        """Get service statistics including cost savings."""
        autom8_stats = await self.autom8_client.get_model_stats()

        return {
            **self.stats,
            "autom8_available": await self.autom8_client.is_available(),
            "cost_per_request": (autom8_stats.get("total_cost", 0) /
                               max(self.stats["autom8_requests"], 1)),
            "autom8_usage_rate": (self.stats["autom8_requests"] /
                                max(self.stats["total_requests"], 1)),
            "autom8_models": autom8_stats.get("models", [])
        }

# Global service instance
_autom8_service = None

async def get_autom8_service() -> Autom8Service:
    """Get or create the global Autom8 service instance."""
    global _autom8_service
    if _autom8_service is None:
        _autom8_service = Autom8Service()
    return _autom8_service