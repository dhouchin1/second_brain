# ──────────────────────────────────────────────────────────────────────────────
# File: services/embeddings.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Local embedding helpers with sentence-transformers + sqlite-vec integration.
- Primary provider: SentenceTransformers with all-MiniLM-L6-v2 (384 dims)
- Fallback: Ollama embeddings API (http://localhost:11434)
- Dev fallback: deterministic pseudo-embedding
Configure via env:
  EMBEDDINGS_PROVIDER=sentence_transformers|ollama|none
  EMBEDDINGS_MODEL=all-MiniLM-L6-v2 (or other sentence-transformer model)
  SENTENCE_TRANSFORMER_MODEL_PATH=./sentence_transformer_model (local model path)
"""
from __future__ import annotations
import hashlib
import json
import os
import random
import struct
import urllib.request
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Default dimensions for all-MiniLM-L6-v2
DEFAULT_DIM = 384

class Embeddings:
    def __init__(self, provider: str | None = None, model: str | None = None, dim: int = DEFAULT_DIM):
        from config import settings
        
        self.provider = provider or os.getenv('EMBEDDINGS_PROVIDER', 'sentence_transformers')
        self.model = model or os.getenv('EMBEDDINGS_MODEL', 'all-MiniLM-L6-v2')
        self.dim = int(os.getenv('EMBEDDINGS_DIM', str(dim)))
        self.model_path = os.getenv('SENTENCE_TRANSFORMER_MODEL_PATH', './sentence_transformer_model')
        self._sentence_transformer = None
        
        # Check local-first AI policy
        self._check_external_allowed(settings)
        
    def _check_external_allowed(self, settings):
        """Check if the current provider is allowed under local-first policy."""
        local_providers = {'sentence_transformers', 'ollama', 'none'}
        external_providers = {'openai', 'cohere', 'huggingface'}
        
        if self.provider in external_providers and not settings.ai_allow_external:
            logger.warning(f"External embeddings provider '{self.provider}' not allowed (ai_allow_external=False). Switching to sentence_transformers.")
            self.provider = 'sentence_transformers'
        
    def embed(self, text: str) -> list[float]:
        """Generate embeddings with local-first priority."""
        try:
            if self.provider == 'sentence_transformers':
                return self._sentence_transformers_embed(text)
            elif self.provider == 'ollama':
                return self._ollama_embed(text)
            elif self.provider == 'none':
                return self._pseudo_embed(text)
            else:
                # External provider - only allowed if ai_allow_external=True
                from config import settings
                if not settings.ai_allow_external:
                    logger.warning(f"External provider '{self.provider}' blocked. Using sentence_transformers fallback.")
                    return self._sentence_transformers_embed(text)
                else:
                    return self._external_embed(text)
        except Exception as e:
            logger.error(f"Embedding generation failed with provider '{self.provider}': {e}")
            # Always try local fallback
            try:
                logger.info("Attempting sentence_transformers fallback")
                return self._sentence_transformers_embed(text)
            except Exception as fallback_e:
                logger.error(f"Sentence transformers fallback failed: {fallback_e}")
                try:
                    logger.info("Attempting pseudo-embedding fallback")
                    return self._pseudo_embed(text)
                except Exception as final_e:
                    logger.error(f"All embedding methods failed: {final_e}")
                    raise
    
    def _sentence_transformers_embed(self, text: str) -> list[float]:
        """Generate embeddings using sentence-transformers."""
        try:
            if self._sentence_transformer is None:
                self._load_sentence_transformer()
            
            # Generate embedding
            embedding = self._sentence_transformer.encode(text, convert_to_numpy=True)
            return embedding.tolist()
            
        except Exception as e:
            logger.warning(f"SentenceTransformers failed, falling back to Ollama: {e}")
            return self._ollama_embed(text)
    
    def _load_sentence_transformer(self):
        """Load sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Try to load local model first
            local_path = Path(self.model_path)
            if local_path.exists() and local_path.is_dir():
                logger.info(f"Loading local SentenceTransformer model from {local_path}")
                self._sentence_transformer = SentenceTransformer(str(local_path))
            else:
                # Fall back to downloading/caching model
                logger.info(f"Loading SentenceTransformer model: {self.model}")
                self._sentence_transformer = SentenceTransformer(self.model)
                
            # Update dimensions based on loaded model
            if hasattr(self._sentence_transformer, 'get_sentence_embedding_dimension'):
                actual_dim = self._sentence_transformer.get_sentence_embedding_dimension()
                if actual_dim != self.dim:
                    logger.info(f"Updating embedding dimensions from {self.dim} to {actual_dim}")
                    self.dim = actual_dim
                    
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}")
            raise

    def _external_embed(self, text: str) -> list[float]:
        """External embedding providers (only when ai_allow_external=True)."""
        raise NotImplementedError(f"External provider '{self.provider}' not implemented. Use sentence_transformers or ollama for local operation.")

    def _ollama_embed(self, text: str) -> list[float]:
        data = json.dumps({"model": self.model, "input": text}).encode('utf-8')
        req = urllib.request.Request(
            os.getenv('OLLAMA_URL', 'http://localhost:11434/api/embeddings'),
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode('utf-8'))
        vec = payload.get('embedding') or payload.get('data', [{}])[0].get('embedding')
        if not vec:
            raise RuntimeError('No embedding returned from Ollama')
        return vec

    def _pseudo_embed(self, text: str) -> list[float]:
        # Stable pseudo-embedding using a hash; useful for offline dev
        h = hashlib.sha256(text.encode('utf-8')).digest()
        rng = random.Random(h)
        return [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]

    @staticmethod
    def pack_f32(array: list[float]) -> bytes:
        return struct.pack('<%sf' % len(array), *array)