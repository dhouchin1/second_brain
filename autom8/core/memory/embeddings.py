"""
Local embeddings implementation for Autom8.

Provides local embedding generation for semantic search without
sending data to external services.
"""

import logging
from typing import Optional, List
import numpy as np

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """
    Local embedding generator using sentence-transformers.
    
    Provides privacy-preserving embeddings for semantic search
    in the context broker.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the embedding model."""
        try:
            # Try to import sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                self._initialized = True
                logger.info(f"Initialized local embedder with model {self.model_name}")
                return True
                
            except ImportError:
                logger.warning("sentence-transformers not available, embeddings disabled")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}")
            return False
    
    async def embed(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text."""
        if not self._initialized or not self.model:
            return None
            
        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    async def embed_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings for multiple texts."""
        if not self._initialized or not self.model:
            return None
            
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if embedder is available."""
        return self._initialized and self.model is not None
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self.model_name == "all-MiniLM-L6-v2":
            return 384
        return 384  # Default assumption