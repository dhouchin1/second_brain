"""
Ollama Integration

Client for interacting with Ollama local models, including model management,
health checking, and inference.
"""

import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx
from autom8.utils.logging import get_logger
from autom8.utils.tokens import get_token_counter

logger = get_logger(__name__)


class OllamaModel:
    """Represents an Ollama model with metadata."""
    
    def __init__(self, data: Dict[str, Any]):
        self.name = data.get('name', '')
        self.size = data.get('size', 0)
        self.digest = data.get('digest', '')
        self.modified_at = data.get('modified_at', '')
        self.details = data.get('details', {})
        self.family = self.details.get('family', '')
        self.format = self.details.get('format', '')
        self.parameter_size = self.details.get('parameter_size', '')
        
    @property
    def size_gb(self) -> float:
        """Model size in GB."""
        return self.size / (1024 ** 3)
    
    @property
    def is_code_model(self) -> bool:
        """Check if this is a code-specialized model."""
        return any(term in self.name.lower() for term in ['code', 'coding', 'coder'])
    
    @property
    def estimated_capability(self) -> float:
        """Estimate capability score based on model characteristics."""
        # Simple heuristic based on parameter size and model family
        if '3b' in self.name.lower() or '3.8b' in self.name.lower():
            return 0.2
        elif '7b' in self.name.lower():
            return 0.4
        elif '8x7b' in self.name.lower() or 'mixtral' in self.name.lower():
            return 0.6
        elif '13b' in self.name.lower():
            return 0.5
        elif '70b' in self.name.lower():
            return 0.8
        else:
            return 0.3  # Default for unknown models
    
    def __str__(self) -> str:
        return f"OllamaModel(name={self.name}, size={self.size_gb:.1f}GB, family={self.family})"


class OllamaClient:
    """
    Client for interacting with Ollama local models.
    
    Provides model management, health checking, and inference capabilities.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
        self.token_counter = get_token_counter()
        self._models_cache: Dict[str, OllamaModel] = {}
        self._cache_timestamp = 0
        self._cache_ttl = 300  # 5 minutes
        
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def is_available(self) -> bool:
        """Check if Ollama is available and responding."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama availability check failed: {e}")
            return False
    
    async def get_models(self, force_refresh: bool = False) -> List[OllamaModel]:
        """
        Get list of available models.
        
        Args:
            force_refresh: Force refresh of model cache
            
        Returns:
            List of available Ollama models
        """
        current_time = time.time()
        
        # Use cache if valid and not forcing refresh
        if not force_refresh and self._models_cache and (current_time - self._cache_timestamp) < self._cache_ttl:
            return list(self._models_cache.values())
        
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            for model_data in data.get('models', []):
                model = OllamaModel(model_data)
                models.append(model)
                self._models_cache[model.name] = model
            
            self._cache_timestamp = current_time
            logger.debug(f"Retrieved {len(models)} models from Ollama")
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to get Ollama models: {e}")
            return []
    
    async def get_model(self, name: str) -> Optional[OllamaModel]:
        """Get specific model by name."""
        models = await self.get_models()
        return next((m for m in models if m.name == name), None)
    
    async def pull_model(self, name: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Pull a model from Ollama registry.
        
        Args:
            name: Model name to pull
            
        Yields:
            Progress updates during model download
        """
        try:
            async with self.client.stream(
                'POST',
                f"{self.base_url}/api/pull",
                json={'name': name},
                timeout=None  # No timeout for model downloads
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            progress = json.loads(line)
                            yield progress
                        except json.JSONDecodeError:
                            continue
                            
            # Refresh model cache after successful pull
            await self.get_models(force_refresh=True)
            logger.info(f"Successfully pulled model: {name}")
            
        except Exception as e:
            logger.error(f"Failed to pull model {name}: {e}")
            yield {"status": "error", "error": str(e)}
    
    async def delete_model(self, name: str) -> bool:
        """
        Delete a model from Ollama.
        
        Args:
            name: Model name to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            response = await self.client.delete(
                f"{self.base_url}/api/delete",
                json={'name': name}
            )
            response.raise_for_status()
            
            # Remove from cache
            self._models_cache.pop(name, None)
            
            logger.info(f"Successfully deleted model: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {name}: {e}")
            return False
    
    async def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text using an Ollama model.
        
        Args:
            model: Model name to use
            prompt: Input prompt
            system: System message
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Generation result with metadata
        """
        start_time = time.time()
        
        # Prepare request
        request_data = {
            'model': model,
            'prompt': prompt,
            'stream': stream,
            'options': {
                'temperature': temperature,
                'top_p': top_p
            }
        }
        
        if system:
            request_data['system'] = system
        
        if max_tokens:
            request_data['options']['num_predict'] = max_tokens
        
        try:
            if stream:
                return await self._generate_streaming(request_data, start_time)
            else:
                return await self._generate_non_streaming(request_data, start_time)
                
        except Exception as e:
            logger.error(f"Generation failed with model {model}: {e}")
            return {
                'success': False,
                'error': str(e),
                'model': model,
                'latency_ms': (time.time() - start_time) * 1000
            }
    
    async def _generate_non_streaming(self, request_data: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Handle non-streaming generation."""
        response = await self.client.post(
            f"{self.base_url}/api/generate",
            json=request_data,
            timeout=60.0
        )
        response.raise_for_status()
        
        result = response.json()
        latency_ms = (time.time() - start_time) * 1000
        
        # Calculate token counts
        input_tokens = self.token_counter.count_tokens(request_data['prompt'])
        output_tokens = self.token_counter.count_tokens(result.get('response', ''))
        
        return {
            'success': True,
            'response': result.get('response', ''),
            'model': request_data['model'],
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'latency_ms': latency_ms,
            'tokens_per_second': output_tokens / (latency_ms / 1000) if latency_ms > 0 else 0,
            'done': result.get('done', True),
            'context': result.get('context', []),
            'total_duration': result.get('total_duration', 0),
            'load_duration': result.get('load_duration', 0),
            'prompt_eval_count': result.get('prompt_eval_count', 0),
            'prompt_eval_duration': result.get('prompt_eval_duration', 0),
            'eval_count': result.get('eval_count', 0),
            'eval_duration': result.get('eval_duration', 0)
        }
    
    async def _generate_streaming(self, request_data: Dict[str, Any], start_time: float) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle streaming generation."""
        async with self.client.stream(
            'POST',
            f"{self.base_url}/api/generate",
            json=request_data,
            timeout=60.0
        ) as response:
            response.raise_for_status()
            
            full_response = ""
            first_token_time = None
            
            async for line in response.aiter_lines():
                if line.strip():
                    try:
                        chunk = json.loads(line)
                        
                        if first_token_time is None:
                            first_token_time = time.time()
                        
                        if 'response' in chunk:
                            full_response += chunk['response']
                        
                        # Add timing information
                        chunk['latency_ms'] = (time.time() - start_time) * 1000
                        if first_token_time:
                            chunk['time_to_first_token'] = (first_token_time - start_time) * 1000
                        
                        yield chunk
                        
                        if chunk.get('done', False):
                            # Final chunk with complete statistics
                            input_tokens = self.token_counter.count_tokens(request_data['prompt'])
                            output_tokens = self.token_counter.count_tokens(full_response)
                            
                            yield {
                                'success': True,
                                'full_response': full_response,
                                'model': request_data['model'],
                                'input_tokens': input_tokens,
                                'output_tokens': output_tokens,
                                'total_latency_ms': (time.time() - start_time) * 1000,
                                'tokens_per_second': output_tokens / ((time.time() - start_time)) if (time.time() - start_time) > 0 else 0
                            }
                            
                    except json.JSONDecodeError:
                        continue
    
    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Chat with an Ollama model using the chat API.
        
        Args:
            model: Model name to use
            messages: List of chat messages
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Chat result with metadata
        """
        start_time = time.time()
        
        request_data = {
            'model': model,
            'messages': messages,
            'stream': stream,
            'options': {
                'temperature': temperature,
                'top_p': top_p
            }
        }
        
        if max_tokens:
            request_data['options']['num_predict'] = max_tokens
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json=request_data,
                timeout=60.0
            )
            response.raise_for_status()
            
            result = response.json()
            latency_ms = (time.time() - start_time) * 1000
            
            # Calculate token counts
            input_text = '\n'.join([f"{msg['role']}: {msg['content']}" for msg in messages])
            input_tokens = self.token_counter.count_tokens(input_text)
            
            assistant_message = result.get('message', {})
            output_text = assistant_message.get('content', '')
            output_tokens = self.token_counter.count_tokens(output_text)
            
            return {
                'success': True,
                'message': assistant_message,
                'model': model,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'latency_ms': latency_ms,
                'tokens_per_second': output_tokens / (latency_ms / 1000) if latency_ms > 0 else 0,
                'done': result.get('done', True),
                'total_duration': result.get('total_duration', 0),
                'load_duration': result.get('load_duration', 0),
                'prompt_eval_count': result.get('prompt_eval_count', 0),
                'prompt_eval_duration': result.get('prompt_eval_duration', 0),
                'eval_count': result.get('eval_count', 0),
                'eval_duration': result.get('eval_duration', 0)
            }
            
        except Exception as e:
            logger.error(f"Chat failed with model {model}: {e}")
            return {
                'success': False,
                'error': str(e),
                'model': model,
                'latency_ms': (time.time() - start_time) * 1000
            }
    
    async def embeddings(
        self,
        model: str,
        text: str
    ) -> Dict[str, Any]:
        """
        Generate embeddings for text using an Ollama model.
        
        Args:
            model: Model name to use for embeddings
            text: Text to embed
            
        Returns:
            Embeddings result
        """
        start_time = time.time()
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    'model': model,
                    'prompt': text
                },
                timeout=30.0
            )
            response.raise_for_status()
            
            result = response.json()
            latency_ms = (time.time() - start_time) * 1000
            
            embeddings = result.get('embedding', [])
            
            return {
                'success': True,
                'embeddings': embeddings,
                'dimensions': len(embeddings),
                'model': model,
                'latency_ms': latency_ms
            }
            
        except Exception as e:
            logger.error(f"Embeddings failed with model {model}: {e}")
            return {
                'success': False,
                'error': str(e),
                'model': model,
                'latency_ms': (time.time() - start_time) * 1000
            }
    
    async def model_info(self, name: str) -> Dict[str, Any]:
        """
        Get detailed information about a model.
        
        Args:
            name: Model name
            
        Returns:
            Detailed model information
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/api/show",
                json={'name': name}
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get model info for {name}: {e}")
            return {'error': str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of Ollama service.
        
        Returns:
            Health check results
        """
        health_info = {
            'service_available': False,
            'models_count': 0,
            'models': [],
            'response_time_ms': 0,
            'version': None,
            'errors': []
        }
        
        start_time = time.time()
        
        try:
            # Check if service is available
            health_info['service_available'] = await self.is_available()
            health_info['response_time_ms'] = (time.time() - start_time) * 1000
            
            if not health_info['service_available']:
                health_info['errors'].append("Ollama service not available")
                return health_info
            
            # Get models list
            models = await self.get_models()
            health_info['models_count'] = len(models)
            health_info['models'] = [
                {
                    'name': model.name,
                    'size_gb': model.size_gb,
                    'family': model.family,
                    'capability': model.estimated_capability
                }
                for model in models
            ]
            
            # Try to get version info
            try:
                response = await self.client.get(f"{self.base_url}/api/version", timeout=5.0)
                if response.status_code == 200:
                    version_info = response.json()
                    health_info['version'] = version_info.get('version', 'unknown')
            except:
                pass  # Version endpoint might not exist in all versions
            
            # Test a simple generation if models are available
            if models:
                test_model = models[0].name
                try:
                    test_result = await self.generate(
                        model=test_model,
                        prompt="Hello",
                        max_tokens=5
                    )
                    if not test_result.get('success'):
                        health_info['errors'].append(f"Test generation failed: {test_result.get('error')}")
                except Exception as e:
                    health_info['errors'].append(f"Test generation error: {str(e)}")
            
        except Exception as e:
            health_info['errors'].append(f"Health check error: {str(e)}")
        
        return health_info
    
    async def get_recommended_models(self) -> List[Dict[str, Any]]:
        """
        Get recommended models for different use cases.
        
        Returns:
            List of recommended model configurations
        """
        available_models = await self.get_models()
        
        recommendations = []
        
        # Categorize available models
        lightweight_models = [m for m in available_models if '3b' in m.name.lower()]
        balanced_models = [m for m in available_models if '7b' in m.name.lower()]
        powerful_models = [m for m in available_models if any(x in m.name.lower() for x in ['8x7b', '13b', '70b'])]
        code_models = [m for m in available_models if m.is_code_model]
        
        if lightweight_models:
            recommendations.append({
                'category': 'Lightweight',
                'use_case': 'Simple tasks, fast responses',
                'models': [m.name for m in lightweight_models[:2]],
                'complexity_tiers': ['trivial', 'simple']
            })
        
        if balanced_models:
            recommendations.append({
                'category': 'Balanced',
                'use_case': 'General purpose, good quality',
                'models': [m.name for m in balanced_models[:2]],
                'complexity_tiers': ['simple', 'moderate']
            })
        
        if powerful_models:
            recommendations.append({
                'category': 'Powerful',
                'use_case': 'Complex tasks, high quality',
                'models': [m.name for m in powerful_models[:2]],
                'complexity_tiers': ['moderate', 'complex']
            })
        
        if code_models:
            recommendations.append({
                'category': 'Code Specialized',
                'use_case': 'Programming and code tasks',
                'models': [m.name for m in code_models[:2]],
                'complexity_tiers': ['simple', 'moderate', 'complex']
            })
        
        return recommendations


# Global client instance
_ollama_client: Optional[OllamaClient] = None


async def get_ollama_client() -> OllamaClient:
    """Get global Ollama client instance."""
    global _ollama_client
    
    if _ollama_client is None:
        from autom8.config.settings import get_settings
        settings = get_settings()
        _ollama_client = OllamaClient(base_url=settings.ollama_host)
    
    return _ollama_client


async def close_ollama_client():
    """Close global Ollama client."""
    global _ollama_client
    
    if _ollama_client:
        await _ollama_client.close()
        _ollama_client = None
