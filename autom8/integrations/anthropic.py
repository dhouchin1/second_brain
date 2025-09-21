"""
Anthropic Claude API Integration for Autom8.

Provides Claude API integration for cloud model routing when local models
are insufficient for complex tasks.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel

from autom8.config.settings import get_settings

logger = logging.getLogger(__name__)


class AnthropicConfig(BaseModel):
    """Anthropic configuration."""
    api_key: Optional[str] = None
    base_url: str = "https://api.anthropic.com"
    timeout: int = 60
    max_retries: int = 3


class AnthropicClient:
    """
    Anthropic Claude API client for cloud model execution.
    
    Integrates with Autom8's model routing system for complex tasks
    that exceed local model capabilities.
    """
    
    def __init__(self, config: Optional[AnthropicConfig] = None):
        self.config = config or self._load_config()
        self._client: Optional[httpx.AsyncClient] = None
        self._available = False
        
    def _load_config(self) -> AnthropicConfig:
        """Load configuration from settings."""
        settings = get_settings()
        return AnthropicConfig(
            api_key=getattr(settings, 'anthropic_api_key', None),
            timeout=60,
            max_retries=3
        )
    
    async def initialize(self) -> bool:
        """Initialize the Anthropic client."""
        try:
            if not self.config.api_key:
                logger.warning("Anthropic API key not configured, Claude models unavailable")
                return False
                
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(self.config.timeout),
                headers={
                    "x-api-key": self.config.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
            )
            
            # Test connection with a simple request
            await self._health_check()
            self._available = True
            
            logger.info("Anthropic Claude client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            self._available = False
            return False
    
    async def _health_check(self) -> bool:
        """Quick health check to verify API access."""
        try:
            # Make a minimal request to verify API access
            response = await self._client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "Hi"}]
                }
            )
            
            if response.status_code == 200:
                return True
            else:
                logger.warning(f"Anthropic health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"Anthropic health check failed: {e}")
            return False
    
    async def generate(
        self,
        messages: List[Dict[str, str]], 
        model: str = "claude-3-sonnet-20240229",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response using Claude API.
        
        Args:
            messages: Conversation messages
            model: Claude model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generation result with content and metadata
        """
        if not self._available or not self._client:
            raise RuntimeError("Anthropic client not available")
        
        try:
            request_data = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
                **kwargs
            }
            
            start_time = asyncio.get_event_loop().time()
            
            response = await self._client.post(
                "/v1/messages",
                json=request_data
            )
            
            end_time = asyncio.get_event_loop().time()
            latency_ms = (end_time - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract content from Claude response format
                content = ""
                if "content" in data and data["content"]:
                    # Claude returns content as a list of content blocks
                    content_blocks = data["content"]
                    content = "\n".join([
                        block.get("text", "") for block in content_blocks 
                        if block.get("type") == "text"
                    ])
                
                return {
                    "content": content,
                    "model": model,
                    "usage": {
                        "input_tokens": data.get("usage", {}).get("input_tokens", 0),
                        "output_tokens": data.get("usage", {}).get("output_tokens", 0),
                    },
                    "latency_ms": latency_ms,
                    "success": True,
                    "provider": "anthropic"
                }
            else:
                error_detail = response.text
                logger.error(f"Claude API error {response.status_code}: {error_detail}")
                
                return {
                    "content": "",
                    "error": f"API error {response.status_code}: {error_detail}",
                    "success": False,
                    "model": model,
                    "latency_ms": latency_ms,
                    "provider": "anthropic"
                }
                
        except Exception as e:
            logger.error(f"Claude generation failed: {e}")
            return {
                "content": "",
                "error": str(e),
                "success": False,
                "model": model,
                "provider": "anthropic"
            }
    
    async def chat_completion(
        self,
        prompt: str,
        model: str = "claude-3-sonnet-20240229",
        max_tokens: int = 1000,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simple chat completion interface.
        
        Args:
            prompt: User prompt
            model: Claude model to use
            max_tokens: Maximum tokens to generate
            context: Optional context to include
            
        Returns:
            Completion result
        """
        # Format as messages
        messages = []
        
        if context:
            messages.append({
                "role": "user", 
                "content": f"Context: {context}\n\nUser: {prompt}"
            })
        else:
            messages.append({"role": "user", "content": prompt})
        
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


# Global Anthropic client instance
_anthropic_client: Optional[AnthropicClient] = None


async def get_anthropic_client() -> AnthropicClient:
    """Get global Anthropic client instance."""
    global _anthropic_client
    
    if _anthropic_client is None:
        _anthropic_client = AnthropicClient()
        await _anthropic_client.initialize()
    
    return _anthropic_client