# ──────────────────────────────────────────────────────────────────────────────
# File: services/local_first_ai.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Local-first AI service manager.
Enforces local AI processing with optional external fallbacks based on configuration.
"""
from __future__ import annotations
import logging
from typing import Optional, List, Dict, Any, Union
from config import settings

logger = logging.getLogger(__name__)


class LocalFirstAIError(Exception):
    """Raised when AI operations fail and external AI is disabled."""
    pass


class LocalFirstAI:
    """
    Local-first AI service manager that respects ai_allow_external setting.
    
    When ai_allow_external=False (default):
    - Only uses local AI services (sentence-transformers, ollama on localhost)
    - Raises LocalFirstAIError if local services fail
    
    When ai_allow_external=True:
    - Tries local services first (if ai_prefer_local=True)
    - Falls back to external services (OpenAI, Anthropic) if local fails
    """
    
    def __init__(self):
        self.allow_external = settings.ai_allow_external
        self.prefer_local = settings.ai_prefer_local
        self.embedding_priorities = settings.embeddings_providers
        self.llm_priorities = settings.llm_providers
        
        # Filter priorities based on ai_allow_external setting
        if not self.allow_external:
            self.embedding_priorities = [p for p in self.embedding_priorities if self._is_local_provider(p, 'embedding')]
            self.llm_priorities = [p for p in self.llm_priorities if self._is_local_provider(p, 'llm')]
            
        logger.info(f"Local-first AI initialized: external_allowed={self.allow_external}, prefer_local={self.prefer_local}")
        logger.info(f"Embedding providers: {self.embedding_priorities}")
        logger.info(f"LLM providers: {self.llm_priorities}")
    
    def _is_local_provider(self, provider: str, service_type: str) -> bool:
        """Check if a provider is considered 'local'."""
        local_embedding_providers = {'sentence_transformers', 'ollama', 'none'}
        local_llm_providers = {'ollama', 'none'}
        
        if service_type == 'embedding':
            return provider in local_embedding_providers
        elif service_type == 'llm':
            return provider in local_llm_providers
        
        return False
    
    def _check_external_allowed(self, provider: str, service_type: str) -> bool:
        """Check if external provider is allowed."""
        if self._is_local_provider(provider, service_type):
            return True
        
        if not self.allow_external:
            logger.warning(f"External AI provider '{provider}' blocked by ai_allow_external=False")
            return False
        
        return True
    
    def get_embeddings_service(self, provider: Optional[str] = None) -> 'Embeddings':
        """Get embeddings service respecting local-first policy."""
        from services.embeddings import Embeddings
        
        if provider:
            if not self._check_external_allowed(provider, 'embedding'):
                raise LocalFirstAIError(f"External embeddings provider '{provider}' not allowed (ai_allow_external=False)")
            return Embeddings(provider=provider)
        
        # Try providers in priority order
        for provider in self.embedding_priorities:
            try:
                if not self._check_external_allowed(provider, 'embedding'):
                    continue
                    
                logger.info(f"Trying embeddings provider: {provider}")
                embedder = Embeddings(provider=provider)
                
                # Test the provider with a simple embedding
                test_result = embedder.embed("test")
                if test_result and len(test_result) > 0:
                    logger.info(f"Successfully initialized embeddings provider: {provider}")
                    return embedder
                
            except Exception as e:
                logger.warning(f"Embeddings provider '{provider}' failed: {e}")
                continue
        
        # If we get here, all providers failed
        available_providers = [p for p in self.embedding_priorities if self._check_external_allowed(p, 'embedding')]
        if not available_providers:
            raise LocalFirstAIError("No embeddings providers available with current ai_allow_external setting")
        else:
            raise LocalFirstAIError(f"All available embeddings providers failed: {available_providers}")
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using LLM providers in priority order."""
        for provider in self.llm_priorities:
            try:
                if not self._check_external_allowed(provider, 'llm'):
                    continue
                    
                logger.info(f"Trying LLM provider: {provider}")
                
                if provider == 'ollama':
                    return self._generate_with_ollama(prompt, **kwargs)
                elif provider == 'openai' and self.allow_external:
                    return self._generate_with_openai(prompt, **kwargs)
                elif provider == 'anthropic' and self.allow_external:
                    return self._generate_with_anthropic(prompt, **kwargs)
                    
            except Exception as e:
                logger.warning(f"LLM provider '{provider}' failed: {e}")
                continue
        
        # If we get here, all providers failed
        available_providers = [p for p in self.llm_priorities if self._check_external_allowed(p, 'llm')]
        if not available_providers:
            raise LocalFirstAIError("No LLM providers available with current ai_allow_external setting")
        else:
            raise LocalFirstAIError(f"All available LLM providers failed: {available_providers}")
    
    def _generate_with_ollama(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama."""
        import requests
        import json
        
        model = kwargs.get('model', settings.ollama_model)
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": self._get_ollama_options(**kwargs)
        }
        
        response = requests.post(
            settings.ollama_api_url,
            json=data,
            timeout=kwargs.get('timeout', 60)
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', '')
    
    def _generate_with_openai(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API (only when external allowed)."""
        if not settings.openai_api_key:
            raise LocalFirstAIError("OpenAI API key not configured")
        
        # This is a placeholder - would need openai client
        raise NotImplementedError("OpenAI integration not implemented yet")
    
    def _generate_with_anthropic(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic API (only when external allowed)."""
        if not settings.anthropic_api_key:
            raise LocalFirstAIError("Anthropic API key not configured")
        
        # This is a placeholder - would need anthropic client
        raise NotImplementedError("Anthropic integration not implemented yet")
    
    def _get_ollama_options(self, **kwargs) -> Dict[str, Any]:
        """Build Ollama options from settings and kwargs."""
        options = {}
        
        # From settings
        if settings.ollama_num_ctx:
            options['num_ctx'] = settings.ollama_num_ctx
        if settings.ollama_num_predict:
            options['num_predict'] = settings.ollama_num_predict
        if settings.ollama_temperature:
            options['temperature'] = settings.ollama_temperature
        if settings.ollama_top_p:
            options['top_p'] = settings.ollama_top_p
        if settings.ollama_num_gpu is not None:
            options['num_gpu'] = settings.ollama_num_gpu
        
        # Override with kwargs
        for key in ['num_ctx', 'num_predict', 'temperature', 'top_p', 'num_gpu']:
            if key in kwargs:
                options[key] = kwargs[key]
        
        return options
    
    def summarize(self, text: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Summarize text and extract structured information."""
        system_prompt = (
            prompt or 
            "Summarize and extract tags and action items from this transcript of conversation snippet or note."
        )
        
        full_prompt = (
            f"{system_prompt}\n\n{text}\n\n"
            "Respond in JSON with keys 'summary', 'tags', and 'actions'."
        )
        
        response = self.generate_text(full_prompt)
        
        try:
            import json
            parsed = json.loads(response)
            
            # Ensure expected structure
            result = {
                "summary": parsed.get("summary", "").strip(),
                "tags": parsed.get("tags", []),
                "actions": parsed.get("actions", [])
            }
            
            # Normalize tags and actions
            if isinstance(result['tags'], str):
                result['tags'] = [t.strip() for t in result['tags'].split(",") if t.strip()]
            if isinstance(result['actions'], str):
                result['actions'] = [a.strip() for a in result['actions'].splitlines() if a.strip()]
                
            return result
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response, returning raw text")
            return {"summary": response, "tags": [], "actions": []}
    
    def generate_title(self, text: str) -> str:
        """Generate a title for content."""
        if not text or not text.strip():
            return "Untitled Note"
        
        prompt = (
            "Generate a concise, descriptive title (max 10 words) for the following note or meeting transcript. "
            "Avoid generic phrases like 'Meeting Transcript' or 'Recording.' "
            "Only respond with the title, no extra commentary.\n\n"
            f"{text}\n\nTitle:"
        )
        
        try:
            response = self.generate_text(prompt, num_predict=50)
            return response.strip().strip('"') or "Untitled Note"
        except Exception as e:
            logger.warning(f"Title generation failed: {e}")
            return "Untitled Note"


# Global instance
local_first_ai = LocalFirstAI()