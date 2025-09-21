"""
OpenAI API Integration for Autom8.

Provides OpenAI GPT API integration for cloud model routing when local models
are insufficient for complex tasks.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel

from autom8.config.settings import get_settings

logger = logging.getLogger(__name__)


class OpenAIConfig(BaseModel):
    """OpenAI configuration."""
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    timeout: int = 60
    max_retries: int = 3


class OpenAIClient:
    """
    OpenAI GPT API client for cloud model execution.
    
    Integrates with Autom8's model routing system for complex tasks
    that exceed local model capabilities.
    """
    
    def __init__(self, config: Optional[OpenAIConfig] = None):
        self.config = config or self._load_config()
        self._client: Optional[httpx.AsyncClient] = None
        self._available = False
        
    def _load_config(self) -> OpenAIConfig:
        """Load configuration from settings."""
        settings = get_settings()
        return OpenAIConfig(
            api_key=getattr(settings, 'openai_api_key', None),
            timeout=60,
            max_retries=3
        )
    
    async def initialize(self) -> bool:
        """Initialize the OpenAI client."""
        try:
            if not self.config.api_key:
                logger.warning("OpenAI API key not configured, GPT models unavailable")
                return False
                
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(self.config.timeout),
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                }
            )
            
            # Test connection with a simple request
            await self._health_check()
            self._available = True
            
            logger.info("OpenAI GPT client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self._available = False
            return False
    
    async def _health_check(self) -> bool:
        """Quick health check to verify API access."""
        try:
            # Make a minimal request to verify API access
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5
                }
            )
            
            if response.status_code == 200:
                return True
            else:
                logger.warning(f"OpenAI health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False
    
    async def generate(
        self,
        messages: List[Dict[str, str]], 
        model: str = "gpt-4",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response using OpenAI API.
        
        Args:
            messages: Conversation messages
            model: GPT model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generation result with content and metadata
        """
        if not self._available or not self._client:
            raise RuntimeError("OpenAI client not available")
        
        try:
            request_data = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            start_time = asyncio.get_event_loop().time()
            
            response = await self._client.post(
                "/chat/completions",
                json=request_data
            )
            
            end_time = asyncio.get_event_loop().time()
            latency_ms = (end_time - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract content from OpenAI response format
                content = ""
                if "choices" in data and data["choices"]:
                    choice = data["choices"][0]
                    if "message" in choice:
                        content = choice["message"].get("content", "")
                
                return {
                    "content": content,
                    "model": model,
                    "usage": data.get("usage", {}),
                    "latency_ms": latency_ms,
                    "success": True,
                    "provider": "openai"
                }
            else:
                error_detail = response.text
                logger.error(f"OpenAI API error {response.status_code}: {error_detail}")
                
                return {
                    "content": "",
                    "error": f"API error {response.status_code}: {error_detail}",
                    "success": False,
                    "model": model,
                    "latency_ms": latency_ms,
                    "provider": "openai"
                }
                
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return {
                "content": "",
                "error": str(e),
                "success": False,
                "model": model,
                "provider": "openai"
            }
    
    async def chat_completion(
        self,
        prompt: str,
        model: str = "gpt-4",
        max_tokens: int = 1000,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simple chat completion interface.
        
        Args:
            prompt: User prompt
            model: GPT model to use
            max_tokens: Maximum tokens to generate
            context: Optional context to include
            
        Returns:
            Completion result
        """
        # Format as messages
        messages = []
        
        if context:
            messages.append({
                "role": "system", 
                "content": f"Context: {context}"
            })
            messages.append({
                "role": "user", 
                "content": prompt
            })
        else:
            messages.append({
                "role": "user", 
                "content": prompt
            })
        
        return await self.generate(
            messages=messages,
            model=model,
            max_tokens=max_tokens
        )
    
    @property
    def is_available(self) -> bool:
        """Check if client is available."""
        return self._available and self._client is not None
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._available = False


# Global OpenAI client instance
_openai_client: Optional[OpenAIClient] = None


async def get_openai_client() -> OpenAIClient:
    """Get global OpenAI client instance."""
    global _openai_client
    
    if _openai_client is None:
        _openai_client = OpenAIClient()
        await _openai_client.initialize()
    
    return _openai_client