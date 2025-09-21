"""
SQLite-Vec Vector Manager for Autom8

Implements semantic vector search capabilities using the sqlite-vec extension,
with graceful fallback to standard SQLite when the extension is not available.
Designed for high-performance semantic search and context retrieval.

This is a core component of Autom8's context transparency system, enabling
intelligent context retrieval through semantic similarity rather than just
keyword matching.
"""

import asyncio
import aiosqlite
import json
import numpy as np
import logging
import sqlite3
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from autom8.models.complexity import ComplexityTier
from autom8.utils.logging import get_logger
from autom8.core.memory.embeddings import LocalEmbedder

logger = get_logger(__name__)


class VectorSearchConfig:
    """Configuration for vector search operations."""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dimensions: int = 384,
        similarity_threshold: float = 0.5,
        max_search_results: int = 50,
        chunk_size: int = 1000,
        overlap_size: int = 100,
        auto_cleanup_days: int = 30
    ):
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.similarity_threshold = similarity_threshold
        self.max_search_results = max_search_results
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.auto_cleanup_days = auto_cleanup_days


class VectorSearchResult:
    """Represents a single vector search result."""
    
    def __init__(
        self,
        content_id: str,
        content: str,
        similarity_score: float,
        metadata: Dict[str, Any] = None,
        context: str = None
    ):
        self.content_id = content_id
        self.content = content
        self.similarity_score = similarity_score
        self.metadata = metadata or {}
        self.context = context
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'content_id': self.content_id,
            'content': self.content,
            'similarity_score': self.similarity_score,
            'metadata': self.metadata,
            'context': self.context
        }
    
    def __repr__(self) -> str:
        return f"VectorSearchResult(id={self.content_id}, score={self.similarity_score:.3f})"


class VectorSearchStats:
    """Statistics for vector search operations."""
    
    def __init__(self):
        self.total_embeddings = 0
        self.total_searches = 0
        self.average_search_time_ms = 0.0
        self.vec_extension_available = False
        self.embedding_model = ""
        self.embedding_dimensions = 0
        self.last_updated = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_embeddings': self.total_embeddings,
            'total_searches': self.total_searches,
            'average_search_time_ms': self.average_search_time_ms,
            'vec_extension_available': self.vec_extension_available,
            'embedding_model': self.embedding_model,
            'embedding_dimensions': self.embedding_dimensions,
            'last_updated': self.last_updated.isoformat()
        }


class SQLiteVectorManager:
    """
    High-performance vector manager using SQLite + sqlite-vec extension.
    
    Provides semantic search capabilities with automatic fallback to standard
    SQLite when the vec extension is not available. Designed for the Autom8
    context transparency system.
    
    Key Features:
    - Native sqlite-vec integration for high performance
    - Graceful fallback to cosine similarity in standard SQLite  
    - Local embedding generation (no external API calls)
    - Efficient batch operations
    - Comprehensive search statistics
    - Automatic cleanup and maintenance
    """
    
    def __init__(
        self,
        db_path: str,
        config: VectorSearchConfig = None,
        embedding_manager: LocalEmbedder = None
    ):
        """
        Initialize the vector manager.
        
        Args:
            db_path: Path to SQLite database
            config: Vector search configuration
            embedding_manager: Manager for generating embeddings
        """
        self.db_path = Path(db_path)
        self.config = config or VectorSearchConfig()
        self.embedding_manager = embedding_manager or LocalEmbedder(
            model_name=self.config.embedding_model
        )
        
        # Extension availability
        self.vec_extension_available = False
        self.extension_checked = False
        
        # Performance tracking
        self.search_times = []
        self.total_searches = 0
        
        logger.info(f"Initialized SQLite vector manager at {self.db_path}")
    
    async def initialize(self) -> bool:
        """
        Initialize the vector database and check for extension availability.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Create database directory if needed
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Test database connectivity
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("SELECT 1")
            
            # Check for vec extension
            await self._check_vec_extension()
            
            # Create tables
            await self._create_tables()
            
            # Load configuration
            await self._load_configuration()
            
            logger.info(f"Vector manager initialized successfully. Vec extension: {self.vec_extension_available}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize vector manager: {e}")
            return False
    
    async def _check_vec_extension(self) -> None:
        """Check if sqlite-vec extension is available."""
        
        if self.extension_checked:
            return
        
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.enable_load_extension(True)
                
                # Try different possible extension names
                extension_names = ["vec0", "sqlite_vec", "vec"]
                
                for ext_name in extension_names:
                    try:
                        await conn.load_extension(ext_name)
                        
                        # Test if extension works
                        await conn.execute("SELECT vec_version()")
                        
                        self.vec_extension_available = True
                        logger.info(f"Successfully loaded sqlite-vec extension: {ext_name}")
                        break
                        
                    except Exception as e:
                        logger.debug(f"Could not load extension {ext_name}: {e}")
                        continue
                
                await conn.enable_load_extension(False)
                
        except Exception as e:
            logger.warning(f"Extension loading not supported: {e}")
        
        if not self.vec_extension_available:
            logger.info("sqlite-vec extension not available, using fallback mode")
        
        self.extension_checked = True
    
    async def _create_tables(self) -> None:
        """Create vector tables based on extension availability."""
        
        async with aiosqlite.connect(self.db_path) as conn:
            if self.vec_extension_available:
                await self._create_vec_tables(conn)
            else:
                await self._create_fallback_tables(conn)
            
            await self._create_metadata_tables(conn)
            await conn.commit()
    
    async def _create_vec_tables(self, conn: aiosqlite.Connection) -> None:
        """Create tables using sqlite-vec extension."""
        
        # Enable extension
        await conn.enable_load_extension(True)
        await conn.load_extension("vec0")
        await conn.enable_load_extension(False)
        
        # Create vector table
        await conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
                id TEXT PRIMARY KEY,
                content_id TEXT,
                embedding_model TEXT,
                embedding FLOAT[{self.config.embedding_dimensions}]
            )
        """)
        
        logger.info(f"Created vec_embeddings table with {self.config.embedding_dimensions} dimensions")
    
    async def _create_fallback_tables(self, conn: aiosqlite.Connection) -> None:
        """Create fallback tables for vector storage."""
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                content_id TEXT NOT NULL,
                embedding_model TEXT DEFAULT 'unknown',
                embedding_data BLOB,
                dimensions INTEGER DEFAULT 384,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (content_id) REFERENCES context_registry (id) ON DELETE CASCADE
            )
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_content ON embeddings(content_id)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(embedding_model)
        """)
        
        logger.info("Created fallback embedding tables")
    
    async def _create_metadata_tables(self, conn: aiosqlite.Connection) -> None:
        """Create metadata and statistics tables."""
        
        # Vector search statistics
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS vector_search_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT,
                search_type TEXT DEFAULT 'semantic',
                results_count INTEGER DEFAULT 0,
                search_time_ms REAL DEFAULT 0.0,
                threshold_used REAL DEFAULT 0.5,
                model_used TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Vector configuration
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS vector_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Content chunks for large documents
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS content_chunks (
                id TEXT PRIMARY KEY,
                parent_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_id) REFERENCES context_registry (id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_vector_stats_timestamp ON vector_search_stats(timestamp DESC)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_parent ON content_chunks(parent_id, chunk_index)
        """)
    
    async def _load_configuration(self) -> None:
        """Load vector configuration from database."""
        
        async with aiosqlite.connect(self.db_path) as conn:
            # Insert default configuration if not exists
            config_items = [
                ('default_embedding_model', self.config.embedding_model),
                ('default_similarity_threshold', str(self.config.similarity_threshold)),
                ('max_search_results', str(self.config.max_search_results)),
                ('embedding_dimension', str(self.config.embedding_dimensions)),
                ('auto_cleanup_days', str(self.config.auto_cleanup_days)),
                ('vec_extension_available', str(self.vec_extension_available))
            ]
            
            for key, value in config_items:
                await conn.execute("""
                    INSERT OR IGNORE INTO vector_config (key, value) VALUES (?, ?)
                """, (key, value))
            
            await conn.commit()
    
    async def store_embedding(
        self,
        content_id: str,
        content: str,
        embedding: Optional[np.ndarray] = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Store content with its embedding vector.
        
        Args:
            content_id: Unique identifier for the content
            content: Text content to embed
            embedding: Pre-computed embedding (will generate if None)
            metadata: Additional metadata to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding if not provided
            if embedding is None:
                embedding = await self.embedding_manager.generate_embedding(content)
            
            # Ensure embedding is the correct dimension
            if embedding.shape[0] != self.config.embedding_dimensions:
                logger.error(f"Embedding dimension mismatch: expected {self.config.embedding_dimensions}, got {embedding.shape[0]}")
                return False
            
            async with aiosqlite.connect(self.db_path) as conn:
                if self.vec_extension_available:
                    success = await self._store_embedding_vec(conn, content_id, content, embedding, metadata)
                else:
                    success = await self._store_embedding_fallback(conn, content_id, content, embedding, metadata)
                
                if success:
                    await conn.commit()
                    logger.debug(f"Stored embedding for content {content_id}")
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to store embedding for {content_id}: {e}")
            return False
    
    async def _store_embedding_vec(
        self,
        conn: aiosqlite.Connection,
        content_id: str,
        content: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any]
    ) -> bool:
        """Store embedding using sqlite-vec extension."""
        
        try:
            # Enable extension
            await conn.enable_load_extension(True)
            await conn.load_extension("vec0")
            await conn.enable_load_extension(False)
            
            # Generate embedding ID
            embedding_id = f"emb_{content_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Store embedding in vec table
            await conn.execute("""
                INSERT OR REPLACE INTO vec_embeddings (id, content_id, embedding_model, embedding)
                VALUES (?, ?, ?, ?)
            """, (embedding_id, content_id, self.config.embedding_model, embedding.tobytes()))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store embedding with vec extension: {e}")
            return False
    
    async def _store_embedding_fallback(
        self,
        conn: aiosqlite.Connection,
        content_id: str,
        content: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any]
    ) -> bool:
        """Store embedding using fallback method."""
        
        try:
            # Generate embedding ID
            embedding_id = f"emb_{content_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Store embedding as blob
            await conn.execute("""
                INSERT OR REPLACE INTO embeddings 
                (id, content_id, embedding_model, embedding_data, dimensions)
                VALUES (?, ?, ?, ?, ?)
            """, (
                embedding_id,
                content_id,
                self.config.embedding_model,
                embedding.tobytes(),
                len(embedding)
            ))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store embedding with fallback method: {e}")
            return False
    
    async def semantic_search(
        self,
        query: str,
        k: int = None,
        threshold: float = None,
        filter_metadata: Dict[str, Any] = None
    ) -> List[VectorSearchResult]:
        """
        Perform semantic search for similar content.
        
        Args:
            query: Search query text
            k: Number of results to return
            threshold: Similarity threshold (0-1)
            filter_metadata: Additional filtering criteria
            
        Returns:
            List of search results ordered by similarity
        """
        import time
        
        k = k or self.config.max_search_results
        threshold = threshold or self.config.similarity_threshold
        
        start_time = time.perf_counter()
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_manager.generate_embedding(query)
            
            # Perform search
            if self.vec_extension_available:
                results = await self._semantic_search_vec(query_embedding, k, threshold, filter_metadata)
            else:
                results = await self._semantic_search_fallback(query_embedding, k, threshold, filter_metadata)
            
            # Record search statistics
            search_time_ms = (time.perf_counter() - start_time) * 1000
            await self._record_search_stats(query, len(results), search_time_ms, threshold)
            
            logger.debug(f"Semantic search completed: {len(results)} results in {search_time_ms:.2f}ms")
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def _semantic_search_vec(
        self,
        query_embedding: np.ndarray,
        k: int,
        threshold: float,
        filter_metadata: Dict[str, Any]
    ) -> List[VectorSearchResult]:
        """Perform search using sqlite-vec extension."""
        
        async with aiosqlite.connect(self.db_path) as conn:
            # Enable extension
            await conn.enable_load_extension(True)
            await conn.load_extension("vec0")
            await conn.enable_load_extension(False)
            
            # Perform vector search
            query = """
                SELECT 
                    v.content_id,
                    c.content,
                    c.summary,
                    c.topic,
                    c.metadata,
                    (1.0 - distance) as similarity_score
                FROM vec_embeddings v
                JOIN context_registry c ON v.content_id = c.id
                WHERE v.embedding MATCH ? AND (1.0 - distance) >= ?
                ORDER BY distance ASC
                LIMIT ?
            """
            
            async with conn.execute(query, (query_embedding.tobytes(), threshold, k)) as cursor:
                rows = await cursor.fetchall()
            
            results = []
            for row in rows:
                content_id, content, summary, topic, metadata_json, similarity = row
                
                # Parse metadata
                try:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                except json.JSONDecodeError:
                    metadata = {}
                
                # Add context information
                context = f"Topic: {topic}" if topic else None
                if summary:
                    context = f"{context}\nSummary: {summary}" if context else f"Summary: {summary}"
                
                results.append(VectorSearchResult(
                    content_id=content_id,
                    content=content,
                    similarity_score=similarity,
                    metadata=metadata,
                    context=context
                ))
            
            return results
    
    async def _semantic_search_fallback(
        self,
        query_embedding: np.ndarray,
        k: int,
        threshold: float,
        filter_metadata: Dict[str, Any]
    ) -> List[VectorSearchResult]:
        """Perform search using cosine similarity fallback."""
        
        async with aiosqlite.connect(self.db_path) as conn:
            # Get all embeddings
            query = """
                SELECT 
                    e.content_id,
                    e.embedding_data,
                    c.content,
                    c.summary,
                    c.topic,
                    c.metadata
                FROM embeddings e
                JOIN context_registry c ON e.content_id = c.id
                WHERE e.embedding_model = ?
            """
            
            async with conn.execute(query, (self.config.embedding_model,)) as cursor:
                rows = await cursor.fetchall()
            
            # Calculate similarities
            similarities = []
            query_norm = np.linalg.norm(query_embedding)
            
            for row in rows:
                content_id, embedding_blob, content, summary, topic, metadata_json = row
                
                # Deserialize embedding
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                
                # Calculate cosine similarity
                if embedding.shape[0] == query_embedding.shape[0]:
                    embedding_norm = np.linalg.norm(embedding)
                    if embedding_norm > 0 and query_norm > 0:
                        similarity = np.dot(query_embedding, embedding) / (query_norm * embedding_norm)
                        
                        if similarity >= threshold:
                            # Parse metadata
                            try:
                                metadata = json.loads(metadata_json) if metadata_json else {}
                            except json.JSONDecodeError:
                                metadata = {}
                            
                            # Add context information  
                            context = f"Topic: {topic}" if topic else None
                            if summary:
                                context = f"{context}\nSummary: {summary}" if context else f"Summary: {summary}"
                            
                            similarities.append(VectorSearchResult(
                                content_id=content_id,
                                content=content,
                                similarity_score=float(similarity),
                                metadata=metadata,
                                context=context
                            ))
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x.similarity_score, reverse=True)
            return similarities[:k]
    
    async def get_similar_content(
        self,
        content_id: str,
        k: int = 5,
        threshold: float = None
    ) -> List[VectorSearchResult]:
        """
        Find content similar to an existing content item.
        
        Args:
            content_id: ID of the reference content
            k: Number of similar items to return
            threshold: Similarity threshold
            
        Returns:
            List of similar content items
        """
        threshold = threshold or self.config.similarity_threshold
        
        try:
            # Get the reference content
            async with aiosqlite.connect(self.db_path) as conn:
                async with conn.execute(
                    "SELECT content FROM context_registry WHERE id = ?",
                    (content_id,)
                ) as cursor:
                    row = await cursor.fetchone()
                    
                    if not row:
                        logger.warning(f"Content {content_id} not found")
                        return []
                    
                    reference_content = row[0]
            
            # Perform semantic search using the reference content
            results = await self.semantic_search(reference_content, k + 1, threshold)
            
            # Filter out the reference item itself
            filtered_results = [r for r in results if r.content_id != content_id]
            
            return filtered_results[:k]
            
        except Exception as e:
            logger.error(f"Failed to find similar content for {content_id}: {e}")
            return []
    
    async def store_content_with_embedding(
        self,
        content_id: str,
        content: str,
        summary: str = None,
        topic: str = None,
        metadata: Dict[str, Any] = None,
        priority: int = 0,
        pinned: bool = False
    ) -> bool:
        """
        Store content in context registry with automatic embedding generation.
        
        Args:
            content_id: Unique identifier
            content: Text content
            summary: Optional summary
            topic: Content topic/category
            metadata: Additional metadata
            priority: Content priority (higher = more important)
            pinned: Whether content is pinned (always included)
            
        Returns:
            True if successful
        """
        try:
            # Generate content hash for deduplication
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Estimate token count (rough approximation)
            token_count = len(content.split()) * 1.3  # Average tokens per word
            
            async with aiosqlite.connect(self.db_path) as conn:
                # Store in context registry
                await conn.execute("""
                    INSERT OR REPLACE INTO context_registry 
                    (id, content, summary, topic, priority, pinned, 
                     created_at, updated_at, token_count, content_hash, 
                     source_type, metadata, embedding_model, similarity_threshold, 
                     access_count, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 
                            ?, ?, 'vector_store', ?, ?, ?, 0, CURRENT_TIMESTAMP)
                """, (
                    content_id,
                    content,
                    summary,
                    topic,
                    priority,
                    pinned,
                    int(token_count),
                    content_hash,
                    json.dumps(metadata or {}),
                    self.config.embedding_model,
                    self.config.similarity_threshold
                ))
                
                await conn.commit()
            
            # Generate and store embedding
            embedding_success = await self.store_embedding(content_id, content)
            
            if embedding_success:
                logger.debug(f"Stored content and embedding for {content_id}")
                return True
            else:
                logger.error(f"Failed to store embedding for {content_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to store content with embedding: {e}")
            return False
    
    async def chunk_and_store_content(
        self,
        content_id: str,
        content: str,
        chunk_size: int = None,
        overlap_size: int = None,
        metadata: Dict[str, Any] = None
    ) -> List[str]:
        """
        Chunk large content and store with embeddings.
        
        Args:
            content_id: Base ID for the content
            content: Large text content to chunk
            chunk_size: Size of each chunk in characters
            overlap_size: Overlap between chunks in characters
            metadata: Metadata to attach to chunks
            
        Returns:
            List of chunk IDs created
        """
        chunk_size = chunk_size or self.config.chunk_size
        overlap_size = overlap_size or self.config.overlap_size
        
        chunk_ids = []
        
        try:
            # Split content into chunks
            chunks = self._create_chunks(content, chunk_size, overlap_size)
            
            # Store each chunk
            for i, chunk_content in enumerate(chunks):
                chunk_id = f"{content_id}_chunk_{i:03d}"
                
                # Store chunk in chunks table
                async with aiosqlite.connect(self.db_path) as conn:
                    await conn.execute("""
                        INSERT OR REPLACE INTO content_chunks 
                        (id, parent_id, chunk_index, content, token_count)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        chunk_id,
                        content_id,
                        i,
                        chunk_content,
                        len(chunk_content.split()) * 1.3  # Rough token estimate
                    ))
                    
                    await conn.commit()
                
                # Store chunk with embedding
                chunk_metadata = (metadata or {}).copy()
                chunk_metadata.update({
                    'parent_id': content_id,
                    'chunk_index': i,
                    'chunk_total': len(chunks)
                })
                
                success = await self.store_content_with_embedding(
                    chunk_id,
                    chunk_content,
                    summary=f"Chunk {i+1}/{len(chunks)} of {content_id}",
                    topic=metadata.get('topic') if metadata else None,
                    metadata=chunk_metadata
                )
                
                if success:
                    chunk_ids.append(chunk_id)
                else:
                    logger.warning(f"Failed to store chunk {chunk_id}")
            
            logger.info(f"Created {len(chunk_ids)} chunks for {content_id}")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Failed to chunk and store content: {e}")
            return []
    
    def _create_chunks(self, content: str, chunk_size: int, overlap_size: int) -> List[str]:
        """Create overlapping chunks from content."""
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]
            
            # Try to break at word boundaries
            if end < len(content):
                last_space = chunk.rfind(' ')
                if last_space > chunk_size * 0.8:  # Don't break too early
                    chunk = chunk[:last_space]
                    end = start + last_space
            
            chunks.append(chunk.strip())
            
            # Move start position with overlap
            if end >= len(content):
                break
            
            start = end - overlap_size
            
            # Ensure we make progress
            if start <= chunks.__len__() * chunk_size - overlap_size:
                start = end
        
        return chunks
    
    async def rebuild_embeddings(
        self,
        model_name: str = None,
        batch_size: int = 10
    ) -> int:
        """
        Rebuild all embeddings with a specified model.
        
        Args:
            model_name: New embedding model to use
            batch_size: Number of items to process in each batch
            
        Returns:
            Number of embeddings successfully rebuilt
        """
        if model_name and model_name != self.config.embedding_model:
            # Update embedding manager
            self.embedding_manager = LocalEmbedder(model_name)
            self.config.embedding_model = model_name
        
        rebuilt_count = 0
        
        try:
            # Get all content items
            async with aiosqlite.connect(self.db_path) as conn:
                async with conn.execute(
                    "SELECT id, content FROM context_registry ORDER BY created_at"
                ) as cursor:
                    rows = await cursor.fetchall()
            
            # Process in batches
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                
                batch_content = [row[1] for row in batch]
                batch_ids = [row[0] for row in batch]
                
                # Generate embeddings for batch
                embeddings = await self.embedding_manager.generate_embeddings_batch(batch_content)
                
                # Store embeddings
                for content_id, embedding in zip(batch_ids, embeddings):
                    if await self.store_embedding(content_id, "", embedding):
                        rebuilt_count += 1
                
                logger.info(f"Rebuilt embeddings: {rebuilt_count}/{len(rows)}")
                
                # Small delay to avoid overwhelming the system
                await asyncio.sleep(0.1)
            
            # Update configuration
            await self._update_embedding_config(model_name)
            
            logger.info(f"Successfully rebuilt {rebuilt_count} embeddings")
            return rebuilt_count
            
        except Exception as e:
            logger.error(f"Failed to rebuild embeddings: {e}")
            return rebuilt_count
    
    async def _update_embedding_config(self, model_name: str) -> None:
        """Update embedding configuration in database."""
        
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("""
                UPDATE vector_config 
                SET value = ?, updated_at = CURRENT_TIMESTAMP 
                WHERE key = 'default_embedding_model'
            """, (model_name,))
            
            await conn.commit()
    
    async def _record_search_stats(
        self,
        query: str,
        results_count: int,
        search_time_ms: float,
        threshold: float
    ) -> None:
        """Record search statistics for performance monitoring."""
        
        try:
            # Generate query hash for privacy
            query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
            
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("""
                    INSERT INTO vector_search_stats 
                    (query_hash, search_type, results_count, search_time_ms, threshold_used, model_used)
                    VALUES (?, 'semantic', ?, ?, ?, ?)
                """, (
                    query_hash,
                    results_count,
                    search_time_ms,
                    threshold,
                    self.config.embedding_model
                ))
                
                await conn.commit()
            
            # Update running averages
            self.search_times.append(search_time_ms)
            self.total_searches += 1
            
            # Keep only recent search times (for performance)
            if len(self.search_times) > 1000:
                self.search_times = self.search_times[-1000:]
                
        except Exception as e:
            logger.error(f"Failed to record search stats: {e}")
    
    async def get_stats(self) -> VectorSearchStats:
        """Get comprehensive vector search statistics."""
        
        stats = VectorSearchStats()
        
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                # Get total embeddings count
                if self.vec_extension_available:
                    async with conn.execute("SELECT COUNT(*) FROM vec_embeddings") as cursor:
                        row = await cursor.fetchone()
                        stats.total_embeddings = row[0] if row else 0
                else:
                    async with conn.execute("SELECT COUNT(*) FROM embeddings") as cursor:
                        row = await cursor.fetchone()
                        stats.total_embeddings = row[0] if row else 0
                
                # Get search statistics
                async with conn.execute("""
                    SELECT COUNT(*), AVG(search_time_ms) 
                    FROM vector_search_stats 
                    WHERE timestamp >= datetime('now', '-7 days')
                """) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        stats.total_searches = row[0] or 0
                        stats.average_search_time_ms = row[1] or 0.0
                
                # Get configuration
                async with conn.execute("SELECT key, value FROM vector_config") as cursor:
                    config_rows = await cursor.fetchall()
                    
                    for key, value in config_rows:
                        if key == 'default_embedding_model':
                            stats.embedding_model = value
                        elif key == 'embedding_dimension':
                            stats.embedding_dimensions = int(value)
                        elif key == 'vec_extension_available':
                            stats.vec_extension_available = value.lower() == 'true'
            
            # Add current configuration
            stats.vec_extension_available = self.vec_extension_available
            stats.embedding_model = self.config.embedding_model
            stats.embedding_dimensions = self.config.embedding_dimensions
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get vector stats: {e}")
            return stats
    
    async def cleanup_old_records(self, days: int = None) -> int:
        """
        Clean up old search statistics and expired content.
        
        Args:
            days: Number of days to keep (uses config default if None)
            
        Returns:
            Number of records cleaned up
        """
        days = days or self.config.auto_cleanup_days
        cleaned_count = 0
        
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                # Clean up old search stats
                result = await conn.execute("""
                    DELETE FROM vector_search_stats 
                    WHERE timestamp < datetime('now', '-{} days')
                """.format(days))
                
                search_stats_cleaned = result.rowcount
                
                # Clean up expired content
                result = await conn.execute("""
                    DELETE FROM context_registry 
                    WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP
                """)
                
                content_cleaned = result.rowcount
                
                # Clean up orphaned embeddings
                if self.vec_extension_available:
                    result = await conn.execute("""
                        DELETE FROM vec_embeddings 
                        WHERE content_id NOT IN (SELECT id FROM context_registry)
                    """)
                else:
                    result = await conn.execute("""
                        DELETE FROM embeddings 
                        WHERE content_id NOT IN (SELECT id FROM context_registry)
                    """)
                
                embeddings_cleaned = result.rowcount
                
                # Clean up orphaned chunks
                result = await conn.execute("""
                    DELETE FROM content_chunks 
                    WHERE parent_id NOT IN (SELECT id FROM context_registry)
                """)
                
                chunks_cleaned = result.rowcount
                
                await conn.commit()
                
                cleaned_count = search_stats_cleaned + content_cleaned + embeddings_cleaned + chunks_cleaned
                
                logger.info(f"Cleanup completed: {cleaned_count} records removed "
                           f"(stats: {search_stats_cleaned}, content: {content_cleaned}, "
                           f"embeddings: {embeddings_cleaned}, chunks: {chunks_cleaned})")
                
                return cleaned_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old records: {e}")
            return 0
    
    async def export_embeddings(self, output_path: str, format: str = "json") -> bool:
        """
        Export embeddings for backup or analysis.
        
        Args:
            output_path: Path to save exported data
            format: Export format ('json' or 'csv')
            
        Returns:
            True if successful
        """
        try:
            import json
            import csv
            from pathlib import Path
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Get all embeddings with content
            async with aiosqlite.connect(self.db_path) as conn:
                if self.vec_extension_available:
                    query = """
                        SELECT v.content_id, c.content, c.topic, c.metadata, v.embedding_model
                        FROM vec_embeddings v
                        JOIN context_registry c ON v.content_id = c.id
                        ORDER BY c.created_at
                    """
                else:
                    query = """
                        SELECT e.content_id, c.content, c.topic, c.metadata, e.embedding_model
                        FROM embeddings e
                        JOIN context_registry c ON e.content_id = c.id
                        ORDER BY c.created_at
                    """
                
                async with conn.execute(query) as cursor:
                    rows = await cursor.fetchall()
            
            if format.lower() == 'json':
                # Export as JSON
                export_data = []
                for row in rows:
                    content_id, content, topic, metadata_json, model = row
                    
                    try:
                        metadata = json.loads(metadata_json) if metadata_json else {}
                    except json.JSONDecodeError:
                        metadata = {}
                    
                    export_data.append({
                        'content_id': content_id,
                        'content': content,
                        'topic': topic,
                        'metadata': metadata,
                        'embedding_model': model
                    })
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            elif format.lower() == 'csv':
                # Export as CSV
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['content_id', 'content', 'topic', 'metadata', 'embedding_model'])
                    
                    for row in rows:
                        writer.writerow(row)
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Exported {len(rows)} embeddings to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export embeddings: {e}")
            return False
    
    async def import_embeddings(self, input_path: str, regenerate_embeddings: bool = True) -> int:
        """
        Import embeddings from exported data.
        
        Args:
            input_path: Path to import data from
            regenerate_embeddings: Whether to regenerate embeddings or trust exported data
            
        Returns:
            Number of items imported
        """
        try:
            import json
            from pathlib import Path
            
            input_file = Path(input_path)
            if not input_file.exists():
                logger.error(f"Import file not found: {input_file}")
                return 0
            
            # Load data
            with open(input_file, 'r', encoding='utf-8') as f:
                if input_file.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    logger.error(f"Unsupported import format: {input_file.suffix}")
                    return 0
            
            imported_count = 0
            
            # Import each item
            for item in data:
                content_id = item.get('content_id')
                content = item.get('content')
                topic = item.get('topic')
                metadata = item.get('metadata', {})
                
                if not content_id or not content:
                    logger.warning(f"Skipping item with missing content_id or content")
                    continue
                
                success = await self.store_content_with_embedding(
                    content_id=content_id,
                    content=content,
                    topic=topic,
                    metadata=metadata
                )
                
                if success:
                    imported_count += 1
                else:
                    logger.warning(f"Failed to import item {content_id}")
            
            logger.info(f"Imported {imported_count} items from {input_file}")
            return imported_count
            
        except Exception as e:
            logger.error(f"Failed to import embeddings: {e}")
            return 0
    
    async def vacuum_database(self) -> bool:
        """
        Optimize database by running VACUUM and ANALYZE.
        
        Returns:
            True if successful
        """
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                # Run VACUUM to reclaim space
                await conn.execute("VACUUM")
                
                # Run ANALYZE to update query planner statistics
                await conn.execute("ANALYZE")
                
                await conn.commit()
            
            logger.info("Database vacuum and analyze completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to vacuum database: {e}")
            return False
    
    async def close(self) -> None:
        """Clean up resources."""
        
        # Clean up embedding manager
        if hasattr(self.embedding_manager, 'close'):
            await self.embedding_manager.close()
        
        logger.info("Vector manager closed")


# Utility functions for integration

async def create_vector_manager(
    db_path: str,
    embedding_model: str = "all-MiniLM-L6-v2",
    config: VectorSearchConfig = None
) -> SQLiteVectorManager:
    """
    Create and initialize a vector manager.
    
    Args:
        db_path: Path to SQLite database
        embedding_model: Model to use for embeddings
        config: Optional configuration
        
    Returns:
        Initialized vector manager
    """
    if config is None:
        config = VectorSearchConfig(embedding_model=embedding_model)
    
    manager = SQLiteVectorManager(db_path, config)
    await manager.initialize()
    return manager


def get_default_vector_config() -> VectorSearchConfig:
    """Get default vector search configuration."""
    return VectorSearchConfig(
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimensions=384,
        similarity_threshold=0.5,
        max_search_results=50,
        chunk_size=1000,
        overlap_size=100,
        auto_cleanup_days=30
    )