"""
SQLite Storage Manager with sqlite-vec support.

Provides durable storage with semantic search capabilities for Autom8.
"""

import sqlite3
import asyncio
import aiosqlite
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import time
from datetime import datetime
import struct
import uuid

import numpy as np

from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class SQLiteManager:
    """
    SQLite storage manager with semantic search capabilities.
    
    Provides durable storage for context, decisions, and other Autom8 data
    with sqlite-vec for semantic search functionality.
    """
    
    def __init__(self, db_path: str = "./autom8.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = None
        self._setup_complete = False
        self._vec_available = False
        self._embedding_dimension = 384  # Default for all-MiniLM-L6-v2
        
    async def initialize(self):
        """Initialize the database and create tables."""
        if self._setup_complete:
            return
            
        logger.info(f"Initializing SQLite database at {self.db_path}")
        
        # Create database and tables
        await self._create_tables()
        await self._setup_vector_search()
        
        self._setup_complete = True
        logger.info("SQLite database initialized successfully")
    
    async def _get_connection(self) -> aiosqlite.Connection:
        """Get database connection."""
        if self._connection is None:
            self._connection = await aiosqlite.connect(
                self.db_path,
                timeout=30.0
            )
            # Enable WAL mode for better concurrency
            await self._connection.execute("PRAGMA journal_mode=WAL")
            await self._connection.execute("PRAGMA synchronous=NORMAL")
            await self._connection.execute("PRAGMA cache_size=10000")
            await self._connection.execute("PRAGMA temp_store=memory")
            
            # Try to load sqlite-vec extension
            await self._load_vec_extension()
        
        return self._connection
    
    async def _create_tables(self):
        """Create database schema."""
        conn = await self._get_connection()
        
        # Context registry for reusable snippets
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS context_registry (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                summary TEXT,
                topic TEXT,
                priority INTEGER DEFAULT 0,
                pinned BOOLEAN DEFAULT FALSE,
                expires_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                token_count INTEGER DEFAULT 0,
                content_hash TEXT,
                source_type TEXT DEFAULT 'unknown',
                metadata TEXT DEFAULT '{}'
            )
        """)
        
        # Agent decisions and actions
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_decisions (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                decision_type TEXT NOT NULL,
                summary TEXT NOT NULL,
                full_content TEXT,
                complexity_score REAL,
                model_used TEXT,
                success BOOLEAN DEFAULT TRUE,
                cost REAL DEFAULT 0.0,
                latency_ms INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                affects TEXT DEFAULT '[]',
                tags TEXT DEFAULT '[]'
            )
        """)
        
        # Usage tracking for optimization
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS usage_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                model TEXT NOT NULL,
                query_hash TEXT,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                estimated_cost REAL DEFAULT 0.0,
                actual_cost REAL DEFAULT 0.0,
                complexity_score REAL,
                success BOOLEAN DEFAULT TRUE,
                latency_ms INTEGER DEFAULT 0,
                quality_score REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Model performance statistics
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                model_name TEXT PRIMARY KEY,
                total_requests INTEGER DEFAULT 0,
                successful_requests INTEGER DEFAULT 0,
                failed_requests INTEGER DEFAULT 0,
                avg_latency_ms REAL DEFAULT 0.0,
                avg_input_tokens REAL DEFAULT 0.0,
                avg_output_tokens REAL DEFAULT 0.0,
                total_cost REAL DEFAULT 0.0,
                avg_quality_score REAL DEFAULT 0.0,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Summaries for compressed knowledge
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                title TEXT,
                content TEXT NOT NULL,
                source_ids TEXT DEFAULT '[]',
                token_count INTEGER DEFAULT 0,
                compression_ratio REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
        """)
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_context_topic ON context_registry(topic)",
            "CREATE INDEX IF NOT EXISTS idx_context_priority ON context_registry(priority DESC)",
            "CREATE INDEX IF NOT EXISTS idx_context_created ON context_registry(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_decisions_agent ON agent_decisions(agent_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_decisions_type ON agent_decisions(decision_type)",
            "CREATE INDEX IF NOT EXISTS idx_usage_model ON usage_ledger(model, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_usage_agent ON usage_ledger(agent_id, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_summaries_scope ON summaries(scope, created_at DESC)"
        ]
        
        for index_sql in indexes:
            await conn.execute(index_sql)
        
        await conn.commit()
        logger.debug("Database schema created successfully")
    
    async def _load_vec_extension(self):
        """Load sqlite-vec extension if available."""
        conn = await self._get_connection()
        
        try:
            # Try to load sqlite-vec extension
            await conn.enable_load_extension(True)
            await conn.load_extension("sqlite_vec")
            await conn.enable_load_extension(False)
            
            # Test if vec0 virtual table is available
            await conn.execute("SELECT vec_version()")
            self._vec_available = True
            logger.info("sqlite-vec extension loaded successfully")
            
        except Exception as e:
            logger.warning(f"sqlite-vec extension not available: {e}. Using fallback mode.")
            self._vec_available = False
    
    async def _setup_vector_search(self):
        """Set up sqlite-vec for semantic search."""
        conn = await self._get_connection()
        
        try:
            if self._vec_available:
                # Create vector table using sqlite-vec
                await conn.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
                        id TEXT PRIMARY KEY,
                        content_id TEXT,
                        embedding_model TEXT,
                        embedding FLOAT[{self._embedding_dimension}]
                    )
                """)
                
                logger.info("Vector search tables created with sqlite-vec")
            else:
                # Fallback: Create regular embeddings table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id TEXT PRIMARY KEY,
                        content_id TEXT NOT NULL,
                        embedding_model TEXT DEFAULT 'unknown',
                        embedding_data BLOB,
                        dimensions INTEGER DEFAULT 384,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (content_id) REFERENCES context_registry (id)
                    )
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_embeddings_content 
                    ON embeddings(content_id)
                """)
                
                logger.info("Vector search tables created (fallback mode)")
            
            await conn.commit()
            
        except Exception as e:
            logger.error(f"Could not set up vector search: {e}")
            raise
    
    # Context Management
    
    async def store_context(
        self,
        context_id: str,
        content: str,
        summary: Optional[str] = None,
        topic: Optional[str] = None,
        priority: int = 0,
        pinned: bool = False,
        expires_at: Optional[datetime] = None,
        source_type: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store context item in the registry."""
        try:
            conn = await self._get_connection()
            
            # Calculate content hash for deduplication
            import hashlib
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Estimate token count (simple approximation)
            token_count = len(content.split()) + len(content) // 4
            
            await conn.execute("""
                INSERT OR REPLACE INTO context_registry 
                (id, content, summary, topic, priority, pinned, expires_at, 
                 token_count, content_hash, source_type, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                context_id, content, summary, topic, priority, pinned,
                expires_at.isoformat() if expires_at else None,
                token_count, content_hash, source_type,
                json.dumps(metadata or {})
            ))
            
            await conn.commit()
            logger.debug(f"Stored context item: {context_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store context {context_id}: {e}")
            return False
    
    async def get_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve context item by ID."""
        try:
            conn = await self._get_connection()
            
            async with conn.execute("""
                SELECT * FROM context_registry WHERE id = ?
            """, (context_id,)) as cursor:
                row = await cursor.fetchone()
                
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, row))
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get context {context_id}: {e}")
            return None
    
    async def search_context(
        self,
        query: str,
        topic: Optional[str] = None,
        limit: int = 5,
        min_priority: int = 0
    ) -> List[Dict[str, Any]]:
        """Search context items using text search."""
        try:
            conn = await self._get_connection()
            
            # Build search query
            conditions = ["(content LIKE ? OR summary LIKE ? OR topic LIKE ?)"]
            params = [f"%{query}%", f"%{query}%", f"%{query}%"]
            
            if topic:
                conditions.append("topic = ?")
                params.append(topic)
            
            if min_priority > 0:
                conditions.append("priority >= ?")
                params.append(min_priority)
            
            # Add expiration check
            conditions.append("(expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)")
            
            sql = f"""
                SELECT * FROM context_registry 
                WHERE {' AND '.join(conditions)}
                ORDER BY priority DESC, created_at DESC
                LIMIT ?
            """
            params.append(limit)
            
            async with conn.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to search context: {e}")
            return []
    
    async def get_pinned_context(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get pinned context items."""
        try:
            conn = await self._get_connection()
            
            async with conn.execute("""
                SELECT * FROM context_registry 
                WHERE pinned = TRUE 
                AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                ORDER BY priority DESC, created_at DESC
                LIMIT ?
            """, (limit,)) as cursor:
                rows = await cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get pinned context: {e}")
            return []
    
    # Decision Management
    
    async def store_decision(
        self,
        decision_id: str,
        agent_id: str,
        decision_type: str,
        summary: str,
        full_content: Optional[str] = None,
        complexity_score: Optional[float] = None,
        model_used: Optional[str] = None,
        success: bool = True,
        cost: float = 0.0,
        latency_ms: int = 0,
        affects: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Store agent decision."""
        try:
            conn = await self._get_connection()
            
            await conn.execute("""
                INSERT OR REPLACE INTO agent_decisions
                (id, agent_id, decision_type, summary, full_content, 
                 complexity_score, model_used, success, cost, latency_ms,
                 affects, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision_id, agent_id, decision_type, summary, full_content,
                complexity_score, model_used, success, cost, latency_ms,
                json.dumps(affects or []), json.dumps(tags or [])
            ))
            
            await conn.commit()
            logger.debug(f"Stored decision: {decision_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store decision {decision_id}: {e}")
            return False
    
    async def get_agent_decisions(
        self,
        agent_id: str,
        limit: int = 10,
        decision_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent decisions for an agent."""
        try:
            conn = await self._get_connection()
            
            conditions = ["agent_id = ?"]
            params = [agent_id]
            
            if decision_type:
                conditions.append("decision_type = ?")
                params.append(decision_type)
            
            sql = f"""
                SELECT * FROM agent_decisions
                WHERE {' AND '.join(conditions)}
                ORDER BY created_at DESC
                LIMIT ?
            """
            params.append(limit)
            
            async with conn.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get decisions for {agent_id}: {e}")
            return []
    
    # Usage Tracking
    
    async def record_usage(
        self,
        agent_id: str,
        model: str,
        query_hash: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        estimated_cost: float = 0.0,
        actual_cost: float = 0.0,
        complexity_score: Optional[float] = None,
        success: bool = True,
        latency_ms: int = 0,
        quality_score: Optional[float] = None
    ) -> bool:
        """Record usage statistics."""
        try:
            conn = await self._get_connection()
            
            await conn.execute("""
                INSERT INTO usage_ledger
                (agent_id, model, query_hash, input_tokens, output_tokens,
                 estimated_cost, actual_cost, complexity_score, success,
                 latency_ms, quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agent_id, model, query_hash, input_tokens, output_tokens,
                estimated_cost, actual_cost, complexity_score, success,
                latency_ms, quality_score
            ))
            
            # Update model performance statistics
            await self._update_model_performance(
                model, success, latency_ms, input_tokens, output_tokens,
                actual_cost, quality_score
            )
            
            await conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to record usage: {e}")
            return False
    
    async def _update_model_performance(
        self,
        model_name: str,
        success: bool,
        latency_ms: int,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        quality_score: Optional[float]
    ):
        """Update model performance statistics."""
        conn = await self._get_connection()
        
        # Get current stats
        async with conn.execute("""
            SELECT * FROM model_performance WHERE model_name = ?
        """, (model_name,)) as cursor:
            row = await cursor.fetchone()
        
        if row:
            # Update existing stats
            columns = [desc[0] for desc in cursor.description]
            stats = dict(zip(columns, row))
            
            new_total = stats['total_requests'] + 1
            new_successful = stats['successful_requests'] + (1 if success else 0)
            new_failed = stats['failed_requests'] + (0 if success else 1)
            
            # Calculate new averages
            alpha = 0.1  # Learning rate for exponential moving average
            new_avg_latency = stats['avg_latency_ms'] * (1 - alpha) + latency_ms * alpha
            new_avg_input = stats['avg_input_tokens'] * (1 - alpha) + input_tokens * alpha
            new_avg_output = stats['avg_output_tokens'] * (1 - alpha) + output_tokens * alpha
            new_total_cost = stats['total_cost'] + cost
            
            if quality_score is not None:
                new_avg_quality = stats['avg_quality_score'] * (1 - alpha) + quality_score * alpha
            else:
                new_avg_quality = stats['avg_quality_score']
            
            await conn.execute("""
                UPDATE model_performance SET
                    total_requests = ?,
                    successful_requests = ?,
                    failed_requests = ?,
                    avg_latency_ms = ?,
                    avg_input_tokens = ?,
                    avg_output_tokens = ?,
                    total_cost = ?,
                    avg_quality_score = ?,
                    last_used = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE model_name = ?
            """, (
                new_total, new_successful, new_failed, new_avg_latency,
                new_avg_input, new_avg_output, new_total_cost, new_avg_quality,
                model_name
            ))
        else:
            # Insert new stats
            await conn.execute("""
                INSERT INTO model_performance
                (model_name, total_requests, successful_requests, failed_requests,
                 avg_latency_ms, avg_input_tokens, avg_output_tokens, total_cost,
                 avg_quality_score)
                VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_name, 1 if success else 0, 0 if success else 1,
                latency_ms, input_tokens, output_tokens, cost,
                quality_score or 0.0
            ))
    
    async def get_model_performance(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get performance statistics for a model."""
        try:
            conn = await self._get_connection()
            
            async with conn.execute("""
                SELECT * FROM model_performance WHERE model_name = ?
            """, (model_name,)) as cursor:
                row = await cursor.fetchone()
                
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, row))
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get model performance for {model_name}: {e}")
            return None
    
    async def get_usage_stats(
        self,
        agent_id: Optional[str] = None,
        model: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get usage statistics."""
        try:
            conn = await self._get_connection()
            
            conditions = ["timestamp > datetime('now', '-{} days')".format(days)]
            params = []
            
            if agent_id:
                conditions.append("agent_id = ?")
                params.append(agent_id)
            
            if model:
                conditions.append("model = ?")
                params.append(model)
            
            where_clause = " AND ".join(conditions)
            
            # Get summary statistics
            async with conn.execute(f"""
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_requests,
                    AVG(latency_ms) as avg_latency,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens,
                    SUM(actual_cost) as total_cost,
                    AVG(quality_score) as avg_quality
                FROM usage_ledger
                WHERE {where_clause}
            """, params) as cursor:
                row = await cursor.fetchone()
                columns = [desc[0] for desc in cursor.description]
                stats = dict(zip(columns, row)) if row else {}
            
            # Get model distribution
            async with conn.execute(f"""
                SELECT model, COUNT(*) as requests
                FROM usage_ledger
                WHERE {where_clause}
                GROUP BY model
                ORDER BY requests DESC
            """, params) as cursor:
                model_dist = await cursor.fetchall()
            
            stats['model_distribution'] = {row[0]: row[1] for row in model_dist}
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get usage stats: {e}")
            return {}
    
    # Vector Storage and Semantic Search
    
    async def store_embedding(
        self,
        content_id: str,
        embedding: np.ndarray,
        model_name: str = "all-MiniLM-L6-v2"
    ) -> bool:
        """Store embedding vector for content."""
        try:
            conn = await self._get_connection()
            embedding_id = str(uuid.uuid4())
            
            if self._vec_available:
                # Use sqlite-vec virtual table
                # Convert numpy array to list for sqlite-vec
                embedding_list = embedding.tolist()
                
                await conn.execute("""
                    INSERT OR REPLACE INTO vec_embeddings 
                    (id, content_id, embedding_model, embedding)
                    VALUES (?, ?, ?, ?)
                """, (embedding_id, content_id, model_name, embedding_list))
            else:
                # Fallback: Store as binary blob
                embedding_bytes = embedding.astype(np.float32).tobytes()
                
                await conn.execute("""
                    INSERT OR REPLACE INTO embeddings 
                    (id, content_id, embedding_model, embedding_data, dimensions)
                    VALUES (?, ?, ?, ?, ?)
                """, (embedding_id, content_id, model_name, embedding_bytes, len(embedding)))
            
            await conn.commit()
            logger.debug(f"Stored embedding for content: {content_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store embedding for {content_id}: {e}")
            return False
    
    async def get_embedding(self, content_id: str) -> Optional[np.ndarray]:
        """Retrieve embedding vector for content."""
        try:
            conn = await self._get_connection()
            
            if self._vec_available:
                async with conn.execute("""
                    SELECT embedding FROM vec_embeddings WHERE content_id = ?
                """, (content_id,)) as cursor:
                    row = await cursor.fetchone()
                    
                    if row:
                        # Convert back to numpy array
                        return np.array(row[0], dtype=np.float32)
            else:
                async with conn.execute("""
                    SELECT embedding_data FROM embeddings WHERE content_id = ?
                """, (content_id,)) as cursor:
                    row = await cursor.fetchone()
                    
                    if row and row[0]:
                        # Convert binary blob back to numpy array
                        return np.frombuffer(row[0], dtype=np.float32)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get embedding for {content_id}: {e}")
            return None
    
    async def semantic_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: float = 0.0,
        topic: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using vector similarity."""
        try:
            conn = await self._get_connection()
            
            if self._vec_available:
                return await self._semantic_search_with_vec(
                    conn, query_embedding, k, threshold, topic
                )
            else:
                return await self._semantic_search_fallback(
                    conn, query_embedding, k, threshold, topic
                )
                
        except Exception as e:
            logger.error(f"Failed to perform semantic search: {e}")
            return []
    
    async def _semantic_search_with_vec(
        self,
        conn: aiosqlite.Connection,
        query_embedding: np.ndarray,
        k: int,
        threshold: float,
        topic: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Semantic search using sqlite-vec."""
        try:
            query_vector = query_embedding.tolist()
            
            # Build query with optional topic filter
            if topic:
                sql = """
                    SELECT 
                        cr.id,
                        cr.content,
                        cr.summary,
                        cr.topic,
                        cr.priority,
                        cr.token_count,
                        cr.metadata,
                        vec.distance
                    FROM vec_embeddings vec
                    JOIN context_registry cr ON vec.content_id = cr.id
                    WHERE vec.embedding MATCH ? 
                    AND cr.topic = ?
                    AND (cr.expires_at IS NULL OR cr.expires_at > CURRENT_TIMESTAMP)
                    AND vec.distance >= ?
                    ORDER BY vec.distance DESC
                    LIMIT ?
                """
                params = (query_vector, topic, threshold, k)
            else:
                sql = """
                    SELECT 
                        cr.id,
                        cr.content,
                        cr.summary,
                        cr.topic,
                        cr.priority,
                        cr.token_count,
                        cr.metadata,
                        vec.distance
                    FROM vec_embeddings vec
                    JOIN context_registry cr ON vec.content_id = cr.id
                    WHERE vec.embedding MATCH ?
                    AND (cr.expires_at IS NULL OR cr.expires_at > CURRENT_TIMESTAMP)
                    AND vec.distance >= ?
                    ORDER BY vec.distance DESC
                    LIMIT ?
                """
                params = (query_vector, threshold, k)
            
            async with conn.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                results = []
                for row in rows:
                    result = dict(zip(columns, row))
                    # Parse metadata JSON
                    if result.get('metadata'):
                        try:
                            result['metadata'] = json.loads(result['metadata'])
                        except:
                            result['metadata'] = {}
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"sqlite-vec search failed: {e}")
            return []
    
    async def _semantic_search_fallback(
        self,
        conn: aiosqlite.Connection,
        query_embedding: np.ndarray,
        k: int,
        threshold: float,
        topic: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Fallback semantic search using cosine similarity."""
        try:
            # Get all embeddings and compute similarities
            if topic:
                sql = """
                    SELECT 
                        e.content_id,
                        e.embedding_data,
                        cr.content,
                        cr.summary,
                        cr.topic,
                        cr.priority,
                        cr.token_count,
                        cr.metadata
                    FROM embeddings e
                    JOIN context_registry cr ON e.content_id = cr.id
                    WHERE cr.topic = ?
                    AND (cr.expires_at IS NULL OR cr.expires_at > CURRENT_TIMESTAMP)
                """
                params = (topic,)
            else:
                sql = """
                    SELECT 
                        e.content_id,
                        e.embedding_data,
                        cr.content,
                        cr.summary,
                        cr.topic,
                        cr.priority,
                        cr.token_count,
                        cr.metadata
                    FROM embeddings e
                    JOIN context_registry cr ON e.content_id = cr.id
                    WHERE (cr.expires_at IS NULL OR cr.expires_at > CURRENT_TIMESTAMP)
                """
                params = ()
            
            async with conn.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                similarities = []
                query_norm = np.linalg.norm(query_embedding)
                
                for row in rows:
                    if row[1]:  # embedding_data exists
                        stored_embedding = np.frombuffer(row[1], dtype=np.float32)
                        
                        # Compute cosine similarity
                        dot_product = np.dot(query_embedding, stored_embedding)
                        stored_norm = np.linalg.norm(stored_embedding)
                        
                        if stored_norm > 0 and query_norm > 0:
                            similarity = dot_product / (query_norm * stored_norm)
                            
                            if similarity >= threshold:
                                result = dict(zip(columns, row))
                                result['distance'] = similarity
                                
                                # Parse metadata JSON
                                if result.get('metadata'):
                                    try:
                                        result['metadata'] = json.loads(result['metadata'])
                                    except:
                                        result['metadata'] = {}
                                
                                similarities.append(result)
                
                # Sort by similarity and return top k
                similarities.sort(key=lambda x: x['distance'], reverse=True)
                return similarities[:k]
                
        except Exception as e:
            logger.error(f"Fallback semantic search failed: {e}")
            return []
    
    async def store_context_with_embedding(
        self,
        context_id: str,
        content: str,
        embedding: np.ndarray,
        summary: Optional[str] = None,
        topic: Optional[str] = None,
        priority: int = 0,
        pinned: bool = False,
        expires_at: Optional[datetime] = None,
        source_type: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
        model_name: str = "all-MiniLM-L6-v2"
    ) -> bool:
        """Store context with its embedding in a single transaction."""
        try:
            conn = await self._get_connection()
            
            # Start transaction
            await conn.execute("BEGIN")
            
            try:
                # Store context
                success = await self.store_context(
                    context_id, content, summary, topic, priority, 
                    pinned, expires_at, source_type, metadata
                )
                
                if success:
                    # Store embedding
                    success = await self.store_embedding(
                        context_id, embedding, model_name
                    )
                
                if success:
                    await conn.execute("COMMIT")
                    logger.debug(f"Stored context with embedding: {context_id}")
                    return True
                else:
                    await conn.execute("ROLLBACK")
                    return False
                    
            except Exception as e:
                await conn.execute("ROLLBACK")
                raise e
                
        except Exception as e:
            logger.error(f"Failed to store context with embedding {context_id}: {e}")
            return False
    
    async def get_similar_content(
        self,
        content_id: str,
        k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find content similar to a given content item."""
        try:
            # Get embedding for the reference content
            embedding = await self.get_embedding(content_id)
            if embedding is None:
                logger.warning(f"No embedding found for content: {content_id}")
                return []
            
            # Perform semantic search
            results = await self.semantic_search(embedding, k + 1, threshold)
            
            # Remove the reference content from results
            return [r for r in results if r['id'] != content_id]
            
        except Exception as e:
            logger.error(f"Failed to find similar content for {content_id}: {e}")
            return []
    
    async def rebuild_embeddings(self, model_name: str = "all-MiniLM-L6-v2") -> int:
        """Rebuild all embeddings for stored content."""
        try:
            from autom8.core.memory.embeddings import LocalEmbedder
            
            embedder = LocalEmbedder(model_name)
            await embedder.initialize()
            
            if not embedder.is_available():
                logger.error("Embedder not available for rebuilding embeddings")
                return 0
            
            conn = await self._get_connection()
            
            # Get all content that needs embeddings
            async with conn.execute("""
                SELECT id, content FROM context_registry
                WHERE (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """) as cursor:
                rows = await cursor.fetchall()
            
            rebuilt_count = 0
            
            for content_id, content in rows:
                try:
                    # Generate new embedding
                    embedding = await embedder.embed(content)
                    if embedding is not None:
                        # Store embedding
                        success = await self.store_embedding(content_id, embedding, model_name)
                        if success:
                            rebuilt_count += 1
                            logger.debug(f"Rebuilt embedding for: {content_id}")
                        
                except Exception as e:
                    logger.warning(f"Failed to rebuild embedding for {content_id}: {e}")
            
            logger.info(f"Rebuilt {rebuilt_count} embeddings")
            return rebuilt_count
            
        except Exception as e:
            logger.error(f"Failed to rebuild embeddings: {e}")
            return 0
    
    async def get_vector_stats(self) -> Dict[str, Any]:
        """Get vector database statistics."""
        try:
            conn = await self._get_connection()
            
            stats = {
                "vec_extension_available": self._vec_available,
                "embedding_dimension": self._embedding_dimension
            }
            
            if self._vec_available:
                # Get vec_embeddings stats
                async with conn.execute("""
                    SELECT COUNT(*) as total_vectors
                    FROM vec_embeddings
                """) as cursor:
                    row = await cursor.fetchone()
                    stats["total_vectors"] = row[0] if row else 0
            else:
                # Get embeddings stats
                async with conn.execute("""
                    SELECT COUNT(*) as total_vectors
                    FROM embeddings
                """) as cursor:
                    row = await cursor.fetchone()
                    stats["total_vectors"] = row[0] if row else 0
            
            # Get content vs embedding coverage
            async with conn.execute("""
                SELECT COUNT(*) as total_content
                FROM context_registry
                WHERE (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """) as cursor:
                row = await cursor.fetchone()
                total_content = row[0] if row else 0
                
                stats["total_content"] = total_content
                if total_content > 0:
                    stats["embedding_coverage"] = stats["total_vectors"] / total_content
                else:
                    stats["embedding_coverage"] = 0.0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get vector stats: {e}")
            return {}
    
    # Helper methods for ContextBroker compatibility
    
    async def get_context_by_id(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Get context by ID (alias for get_context)."""
        return await self.get_context(context_id)
    
    async def search_context_ids(
        self,
        query: str = "",
        topic: Optional[str] = None,
        limit: int = 5,
        min_priority: int = 0
    ) -> List[str]:
        """Search context and return list of content IDs."""
        try:
            # Call the original search_context method defined earlier
            conn = await self._get_connection()
            
            # Build search query
            conditions = ["(content LIKE ? OR summary LIKE ? OR topic LIKE ?)"]
            params = [f"%{query}%", f"%{query}%", f"%{query}%"]
            
            if topic:
                conditions.append("topic = ?")
                params.append(topic)
            
            if min_priority > 0:
                conditions.append("priority >= ?")
                params.append(min_priority)
            
            # Add expiration check
            conditions.append("(expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)")
            
            sql = f"""
                SELECT id FROM context_registry 
                WHERE {' AND '.join(conditions)}
                ORDER BY priority DESC, created_at DESC
                LIMIT ?
            """
            params.append(limit)
            
            async with conn.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
                return [row[0] for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to search context: {e}")
            return []
    
    async def get_pinned_context(
        self, 
        token_limit: Optional[int] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get pinned context items with optional token limit."""
        try:
            conn = await self._get_connection()
            
            if token_limit:
                # Get pinned items respecting token limit
                async with conn.execute("""
                    SELECT * FROM context_registry 
                    WHERE pinned = TRUE 
                    AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                    ORDER BY priority DESC, created_at DESC
                """) as cursor:
                    rows = await cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    
                    results = []
                    current_tokens = 0
                    
                    for row in rows:
                        item = dict(zip(columns, row))
                        item_tokens = item.get('token_count', 0)
                        
                        if current_tokens + item_tokens <= token_limit:
                            # Parse metadata JSON
                            if item.get('metadata'):
                                try:
                                    item['metadata'] = json.loads(item['metadata'])
                                except:
                                    item['metadata'] = {}
                            
                            # Format for ContextBroker compatibility
                            results.append({
                                'id': item['id'],
                                'content': item['content'],
                                'tokens': item_tokens
                            })
                            current_tokens += item_tokens
                        
                        if len(results) >= limit:
                            break
                    
                    return results
            else:
                # Standard pinned context retrieval
                return await self.get_pinned_context(limit)
            
        except Exception as e:
            logger.error(f"Failed to get pinned context: {e}")
            return []
    
    async def record_context_usage(
        self,
        agent_id: str,
        query: str,
        context_tokens: int,
        response_tokens: int,
        success: bool,
        sources_used: int,
        model: str = "unknown"
    ) -> bool:
        """Record context usage for optimization."""
        try:
            # Create a hash of the query for tracking
            import hashlib
            query_hash = hashlib.md5(query.encode()).hexdigest()
            
            return await self.record_usage(
                agent_id=agent_id,
                model=model,
                query_hash=query_hash,
                input_tokens=context_tokens,
                output_tokens=response_tokens,
                success=success
            )
            
        except Exception as e:
            logger.error(f"Failed to record context usage: {e}")
            return False
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for ContextBroker."""
        try:
            conn = await self._get_connection()
            
            stats = {}
            
            # Context registry stats
            async with conn.execute("""
                SELECT 
                    COUNT(*) as total_contexts,
                    SUM(token_count) as total_tokens,
                    COUNT(CASE WHEN pinned THEN 1 END) as pinned_count,
                    COUNT(CASE WHEN expires_at IS NOT NULL THEN 1 END) as expiring_count
                FROM context_registry
                WHERE (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """) as cursor:
                row = await cursor.fetchone()
                if row:
                    stats.update({
                        'total_contexts': row[0],
                        'total_tokens': row[1] or 0,
                        'pinned_count': row[2] or 0,
                        'expiring_count': row[3] or 0
                    })
            
            # Vector stats
            vector_stats = await self.get_vector_stats()
            stats.update(vector_stats)
            
            # Usage stats
            usage_stats = await self.get_usage_stats(days=7)
            stats['usage_last_7_days'] = usage_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}
    
    async def cleanup_old_records(self, days: int = 30) -> int:
        """Clean up old records for maintenance."""
        try:
            conn = await self._get_connection()
            
            # Clean up old usage records
            cursor = await conn.execute("""
                DELETE FROM usage_ledger 
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days))
            
            deleted_usage = cursor.rowcount
            
            # Clean up orphaned embeddings
            if self._vec_available:
                cursor = await conn.execute("""
                    DELETE FROM vec_embeddings 
                    WHERE content_id NOT IN (SELECT id FROM context_registry)
                """)
            else:
                cursor = await conn.execute("""
                    DELETE FROM embeddings 
                    WHERE content_id NOT IN (SELECT id FROM context_registry)
                """)
            
            deleted_embeddings = cursor.rowcount
            
            await conn.commit()
            
            total_deleted = deleted_usage + deleted_embeddings
            if total_deleted > 0:
                logger.info(f"Cleaned up {deleted_usage} usage records and {deleted_embeddings} orphaned embeddings")
            
            return total_deleted
            
        except Exception as e:
            logger.error(f"Failed to cleanup old records: {e}")
            return 0

    # Cleanup and Maintenance
    
    async def cleanup_expired(self) -> int:
        """Clean up expired context items."""
        try:
            conn = await self._get_connection()
            
            cursor = await conn.execute("""
                DELETE FROM context_registry 
                WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP
            """)
            
            deleted = cursor.rowcount
            await conn.commit()
            
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} expired context items")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired items: {e}")
            return 0
    
    async def vacuum_database(self):
        """Optimize database performance."""
        try:
            conn = await self._get_connection()
            await conn.execute("VACUUM")
            logger.info("Database vacuumed successfully")
        except Exception as e:
            logger.error(f"Failed to vacuum database: {e}")
    
    async def close(self):
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.debug("Database connection closed")


# Global instance
_sqlite_manager = None


async def get_sqlite_manager() -> SQLiteManager:
    """Get global SQLite manager instance."""
    global _sqlite_manager
    
    if _sqlite_manager is None:
        _sqlite_manager = SQLiteManager()
        await _sqlite_manager.initialize()
    
    return _sqlite_manager
