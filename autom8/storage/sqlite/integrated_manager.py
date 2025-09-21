"""
Integrated SQLite and Vector Manager for Autom8.

This module provides a unified interface that combines the SQLiteManager
and SQLiteVectorManager into a single, cohesive system. It simplifies
database operations by providing high-level methods that automatically
handle both structured data storage and vector embeddings.

Key Features:
- Unified API for storing content with automatic embedding generation
- Seamless integration between SQL and vector operations
- Transaction-safe operations across both managers
- Automatic cleanup and maintenance
- Performance optimization with connection pooling
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from autom8.storage.sqlite.manager import SQLiteManager
from autom8.storage.sqlite.vector_manager import SQLiteVectorManager, VectorSearchConfig, VectorSearchResult
from autom8.core.memory.embeddings import LocalEmbedder
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class IntegratedStorageResult:
    """Result from integrated storage operations."""

    def __init__(
        self,
        success: bool,
        content_id: str = None,
        error: str = None,
        embedding_stored: bool = False,
        tokens_used: int = 0,
        vector_dimension: int = 0
    ):
        self.success = success
        self.content_id = content_id
        self.error = error
        self.embedding_stored = embedding_stored
        self.tokens_used = tokens_used
        self.vector_dimension = vector_dimension

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'content_id': self.content_id,
            'error': self.error,
            'embedding_stored': self.embedding_stored,
            'tokens_used': self.tokens_used,
            'vector_dimension': self.vector_dimension
        }


class IntegratedSearchResult:
    """Enhanced search result with both SQL and vector data."""

    def __init__(
        self,
        content_id: str,
        content: str,
        summary: Optional[str] = None,
        topic: Optional[str] = None,
        metadata: Dict[str, Any] = None,
        similarity_score: float = 0.0,
        search_method: str = "hybrid",
        token_count: int = 0,
        priority: int = 0,
        created_at: str = None
    ):
        self.content_id = content_id
        self.content = content
        self.summary = summary
        self.topic = topic
        self.metadata = metadata or {}
        self.similarity_score = similarity_score
        self.search_method = search_method
        self.token_count = token_count
        self.priority = priority
        self.created_at = created_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'content_id': self.content_id,
            'content': self.content,
            'summary': self.summary,
            'topic': self.topic,
            'metadata': self.metadata,
            'similarity_score': self.similarity_score,
            'search_method': self.search_method,
            'token_count': self.token_count,
            'priority': self.priority,
            'created_at': self.created_at
        }

    def __repr__(self) -> str:
        return f"IntegratedSearchResult(id={self.content_id}, score={self.similarity_score:.3f}, method={self.search_method})"


class IntegratedSQLiteManager:
    """
    Integrated manager that combines SQLite and vector operations.

    This class provides a unified interface for all database operations,
    automatically handling both structured data and vector embeddings.
    It's designed to be the primary interface for Autom8's storage needs.
    """

    def __init__(
        self,
        db_path: str = "./autom8.db",
        vector_config: VectorSearchConfig = None,
        embedding_manager: LocalEmbedder = None,
        auto_embed: bool = True
    ):
        """
        Initialize the integrated manager.

        Args:
            db_path: Path to SQLite database
            vector_config: Vector search configuration
            embedding_manager: Custom embedding manager
            auto_embed: Automatically generate embeddings for new content
        """
        self.db_path = Path(db_path)
        self.auto_embed = auto_embed

        # Initialize managers
        self.sqlite_manager = None
        self.vector_manager = None
        self.embedding_manager = embedding_manager

        # Configuration
        self.vector_config = vector_config or VectorSearchConfig()

        # State tracking
        self._initialized = False
        self._stats = {
            'operations_count': 0,
            'embeddings_generated': 0,
            'searches_performed': 0,
            'errors_count': 0
        }

        logger.info(f"Integrated SQLite manager initialized for {self.db_path}")

    async def initialize(self) -> bool:
        """Initialize both managers."""
        if self._initialized:
            return True

        try:
            # Initialize SQLite manager
            self.sqlite_manager = SQLiteManager(str(self.db_path))
            await self.sqlite_manager.initialize()

            # Initialize vector manager
            self.vector_manager = SQLiteVectorManager(
                db_path=str(self.db_path),
                config=self.vector_config,
                embedding_manager=self.embedding_manager
            )
            await self.vector_manager.initialize()

            # Initialize embedding manager if not provided
            if self.embedding_manager is None:
                self.embedding_manager = LocalEmbedder(
                    model_name=self.vector_config.embedding_model
                )
                await self.embedding_manager.initialize()

            self._initialized = True
            logger.info("Integrated manager initialization completed")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize integrated manager: {e}")
            return False

    async def store_content(
        self,
        content_id: str,
        content: str,
        summary: Optional[str] = None,
        topic: Optional[str] = None,
        priority: int = 0,
        pinned: bool = False,
        expires_at: Optional[datetime] = None,
        source_type: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
        generate_embedding: Optional[bool] = None
    ) -> IntegratedStorageResult:
        """
        Store content with optional automatic embedding generation.

        Args:
            content_id: Unique identifier for the content
            content: Text content to store
            summary: Optional summary of the content
            topic: Topic/category for the content
            priority: Priority level (higher = more important)
            pinned: Whether content should be pinned (always included)
            expires_at: Optional expiration timestamp
            source_type: Type of content source
            metadata: Additional metadata
            generate_embedding: Override auto_embed setting

        Returns:
            IntegratedStorageResult with operation status
        """
        if not self._initialized:
            await self.initialize()

        generate_embedding = generate_embedding if generate_embedding is not None else self.auto_embed

        try:
            self._stats['operations_count'] += 1

            # Calculate token count
            token_count = len(content.split()) + len(content) // 4

            # Store in SQLite
            sqlite_success = await self.sqlite_manager.store_context(
                context_id=content_id,
                content=content,
                summary=summary,
                topic=topic,
                priority=priority,
                pinned=pinned,
                expires_at=expires_at,
                source_type=source_type,
                metadata=metadata
            )

            if not sqlite_success:
                self._stats['errors_count'] += 1
                return IntegratedStorageResult(
                    success=False,
                    content_id=content_id,
                    error="Failed to store content in SQLite",
                    tokens_used=token_count
                )

            # Generate and store embedding if requested
            embedding_stored = False
            vector_dimension = 0

            if generate_embedding:
                try:
                    embedding_success = await self.vector_manager.store_embedding(
                        content_id=content_id,
                        content=content
                    )

                    if embedding_success:
                        embedding_stored = True
                        vector_dimension = self.vector_config.embedding_dimensions
                        self._stats['embeddings_generated'] += 1
                        logger.debug(f"Stored embedding for content: {content_id}")
                    else:
                        logger.warning(f"Failed to store embedding for content: {content_id}")

                except Exception as e:
                    logger.warning(f"Embedding generation failed for {content_id}: {e}")

            return IntegratedStorageResult(
                success=True,
                content_id=content_id,
                embedding_stored=embedding_stored,
                tokens_used=token_count,
                vector_dimension=vector_dimension
            )

        except Exception as e:
            self._stats['errors_count'] += 1
            logger.error(f"Failed to store content {content_id}: {e}")
            return IntegratedStorageResult(
                success=False,
                content_id=content_id,
                error=str(e)
            )

    async def get_content(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve content by ID."""
        if not self._initialized:
            await self.initialize()

        return await self.sqlite_manager.get_context(content_id)

    async def search_content(
        self,
        query: str,
        method: str = "hybrid",
        k: int = 5,
        topic: Optional[str] = None,
        min_priority: int = 0,
        similarity_threshold: float = None
    ) -> List[IntegratedSearchResult]:
        """
        Search content using specified method.

        Args:
            query: Search query
            method: Search method ('text', 'semantic', 'hybrid')
            k: Number of results to return
            topic: Filter by topic
            min_priority: Minimum priority level
            similarity_threshold: Minimum similarity score for semantic search

        Returns:
            List of search results
        """
        if not self._initialized:
            await self.initialize()

        self._stats['searches_performed'] += 1
        similarity_threshold = similarity_threshold or self.vector_config.similarity_threshold

        try:
            if method == "text":
                return await self._search_text_only(query, k, topic, min_priority)
            elif method == "semantic":
                return await self._search_semantic_only(query, k, topic, similarity_threshold)
            elif method == "hybrid":
                return await self._search_hybrid(query, k, topic, min_priority, similarity_threshold)
            else:
                raise ValueError(f"Unknown search method: {method}")

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def _search_text_only(
        self,
        query: str,
        k: int,
        topic: Optional[str],
        min_priority: int
    ) -> List[IntegratedSearchResult]:
        """Perform text-only search."""
        results = await self.sqlite_manager.search_context(
            query=query,
            topic=topic,
            limit=k,
            min_priority=min_priority
        )

        return [
            IntegratedSearchResult(
                content_id=r['id'],
                content=r['content'],
                summary=r.get('summary'),
                topic=r.get('topic'),
                metadata=json.loads(r.get('metadata', '{}')) if r.get('metadata') else {},
                similarity_score=1.0,  # Full match for text search
                search_method="text",
                token_count=r.get('token_count', 0),
                priority=r.get('priority', 0),
                created_at=r.get('created_at')
            )
            for r in results
        ]

    async def _search_semantic_only(
        self,
        query: str,
        k: int,
        topic: Optional[str],
        similarity_threshold: float
    ) -> List[IntegratedSearchResult]:
        """Perform semantic-only search."""
        vector_results = await self.vector_manager.semantic_search(
            query=query,
            k=k,
            threshold=similarity_threshold,
            filter_metadata={'topic': topic} if topic else None
        )

        results = []
        for vr in vector_results:
            # Get additional metadata from SQLite
            context_data = await self.sqlite_manager.get_context(vr.content_id)
            if context_data:
                results.append(IntegratedSearchResult(
                    content_id=vr.content_id,
                    content=vr.content,
                    summary=context_data.get('summary'),
                    topic=context_data.get('topic'),
                    metadata=json.loads(context_data.get('metadata', '{}')) if context_data.get('metadata') else {},
                    similarity_score=vr.similarity_score,
                    search_method="semantic",
                    token_count=context_data.get('token_count', 0),
                    priority=context_data.get('priority', 0),
                    created_at=context_data.get('created_at')
                ))

        return results

    async def _search_hybrid(
        self,
        query: str,
        k: int,
        topic: Optional[str],
        min_priority: int,
        similarity_threshold: float
    ) -> List[IntegratedSearchResult]:
        """Perform hybrid search combining text and semantic results."""
        # Run both searches in parallel
        text_task = self._search_text_only(query, k // 2, topic, min_priority)
        semantic_task = self._search_semantic_only(query, k // 2, topic, similarity_threshold)

        text_results, semantic_results = await asyncio.gather(text_task, semantic_task)

        # Combine and deduplicate results
        combined_results = {}

        # Add text results with boosted scores
        for result in text_results:
            result.search_method = "hybrid_text"
            result.similarity_score = 1.0  # Full score for text matches
            combined_results[result.content_id] = result

        # Add semantic results, merging with text results if they exist
        for result in semantic_results:
            if result.content_id in combined_results:
                # Combine scores - favor text matches but boost with semantic similarity
                existing = combined_results[result.content_id]
                existing.similarity_score = min(1.0, existing.similarity_score + result.similarity_score * 0.3)
                existing.search_method = "hybrid_both"
            else:
                result.search_method = "hybrid_semantic"
                combined_results[result.content_id] = result

        # Sort by combined score and priority
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: (x.similarity_score * 0.7 + (x.priority / 100) * 0.3),
            reverse=True
        )

        return sorted_results[:k]

    async def get_similar_content(
        self,
        content_id: str,
        k: int = 5,
        threshold: float = None
    ) -> List[IntegratedSearchResult]:
        """Get content similar to a specific item."""
        if not self._initialized:
            await self.initialize()

        threshold = threshold or self.vector_config.similarity_threshold

        try:
            vector_results = await self.vector_manager.get_similar_content(
                content_id=content_id,
                k=k,
                threshold=threshold
            )

            results = []
            for vr in vector_results:
                context_data = await self.sqlite_manager.get_context(vr.content_id)
                if context_data:
                    results.append(IntegratedSearchResult(
                        content_id=vr.content_id,
                        content=vr.content,
                        summary=context_data.get('summary'),
                        topic=context_data.get('topic'),
                        metadata=json.loads(context_data.get('metadata', '{}')) if context_data.get('metadata') else {},
                        similarity_score=vr.similarity_score,
                        search_method="similarity",
                        token_count=context_data.get('token_count', 0),
                        priority=context_data.get('priority', 0),
                        created_at=context_data.get('created_at')
                    ))

            return results

        except Exception as e:
            logger.error(f"Failed to get similar content for {content_id}: {e}")
            return []

    async def delete_content(self, content_id: str) -> bool:
        """Delete content and its associated embedding."""
        if not self._initialized:
            await self.initialize()

        try:
            # Delete from both SQLite and vector storage
            # Note: The foreign key constraint should handle cleanup automatically
            success = True

            # Delete from SQLite (this should cascade to embeddings due to FK constraint)
            async with self.sqlite_manager._get_connection() as conn:
                cursor = await conn.execute(
                    "DELETE FROM context_registry WHERE id = ?",
                    (content_id,)
                )
                success = cursor.rowcount > 0
                await conn.commit()

            if success:
                logger.debug(f"Deleted content: {content_id}")
            else:
                logger.warning(f"Content not found for deletion: {content_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to delete content {content_id}: {e}")
            return False

    async def update_content(
        self,
        content_id: str,
        content: Optional[str] = None,
        summary: Optional[str] = None,
        topic: Optional[str] = None,
        priority: Optional[int] = None,
        pinned: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        regenerate_embedding: bool = False
    ) -> IntegratedStorageResult:
        """Update existing content and optionally regenerate embedding."""
        if not self._initialized:
            await self.initialize()

        try:
            # Get existing content
            existing = await self.sqlite_manager.get_context(content_id)
            if not existing:
                return IntegratedStorageResult(
                    success=False,
                    content_id=content_id,
                    error="Content not found"
                )

            # Update fields
            updated_content = content if content is not None else existing['content']
            updated_summary = summary if summary is not None else existing.get('summary')
            updated_topic = topic if topic is not None else existing.get('topic')
            updated_priority = priority if priority is not None else existing.get('priority', 0)
            updated_pinned = pinned if pinned is not None else existing.get('pinned', False)

            if metadata is not None:
                updated_metadata = metadata
            else:
                existing_metadata = existing.get('metadata', '{}')
                updated_metadata = json.loads(existing_metadata) if existing_metadata else {}

            # Store updated content
            sqlite_success = await self.sqlite_manager.store_context(
                context_id=content_id,
                content=updated_content,
                summary=updated_summary,
                topic=updated_topic,
                priority=updated_priority,
                pinned=updated_pinned,
                source_type=existing.get('source_type', 'unknown'),
                metadata=updated_metadata
            )

            if not sqlite_success:
                return IntegratedStorageResult(
                    success=False,
                    content_id=content_id,
                    error="Failed to update content in SQLite"
                )

            # Regenerate embedding if requested or if content changed
            embedding_stored = False
            if regenerate_embedding or (content is not None and content != existing['content']):
                try:
                    embedding_success = await self.vector_manager.store_embedding(
                        content_id=content_id,
                        content=updated_content
                    )
                    embedding_stored = embedding_success
                    if embedding_success:
                        self._stats['embeddings_generated'] += 1

                except Exception as e:
                    logger.warning(f"Failed to regenerate embedding for {content_id}: {e}")

            return IntegratedStorageResult(
                success=True,
                content_id=content_id,
                embedding_stored=embedding_stored,
                tokens_used=len(updated_content.split()) + len(updated_content) // 4,
                vector_dimension=self.vector_config.embedding_dimensions if embedding_stored else 0
            )

        except Exception as e:
            logger.error(f"Failed to update content {content_id}: {e}")
            return IntegratedStorageResult(
                success=False,
                content_id=content_id,
                error=str(e)
            )

    async def get_content_stats(self) -> Dict[str, Any]:
        """Get comprehensive content statistics."""
        if not self._initialized:
            await self.initialize()

        try:
            # Get SQLite stats
            sqlite_stats = await self.sqlite_manager.get_storage_stats()

            # Get vector stats
            vector_stats = await self.vector_manager.get_stats()

            # Combine with integration stats
            stats = {
                'total_content_items': sqlite_stats.get('total_contexts', 0),
                'total_embeddings': vector_stats.total_embeddings,
                'pinned_items': sqlite_stats.get('pinned_count', 0),
                'total_tokens': sqlite_stats.get('total_tokens', 0),
                'vec_extension_available': vector_stats.vec_extension_available,
                'embedding_model': vector_stats.embedding_model,
                'embedding_dimensions': vector_stats.embedding_dimensions,
                'integration_stats': self._stats.copy(),
                'search_performance': {
                    'total_searches': vector_stats.total_searches,
                    'avg_search_time_ms': vector_stats.average_search_time_ms
                }
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get content stats: {e}")
            return {}

    async def cleanup_old_content(self, days: int = 30) -> int:
        """Clean up old content and embeddings."""
        if not self._initialized:
            await self.initialize()

        try:
            # Clean up SQLite records
            sqlite_cleaned = await self.sqlite_manager.cleanup_old_records(days)

            # Clean up vector records
            vector_cleaned = await self.vector_manager.cleanup_old_records(days)

            total_cleaned = sqlite_cleaned + vector_cleaned
            logger.info(f"Cleaned up {total_cleaned} old records ({days} days)")

            return total_cleaned

        except Exception as e:
            logger.error(f"Failed to cleanup old content: {e}")
            return 0

    async def rebuild_all_embeddings(self, model_name: str = None) -> int:
        """Rebuild all embeddings with specified model."""
        if not self._initialized:
            await self.initialize()

        try:
            count = await self.vector_manager.rebuild_embeddings(
                model_name=model_name or self.vector_config.embedding_model
            )

            self._stats['embeddings_generated'] += count
            logger.info(f"Rebuilt {count} embeddings")

            return count

        except Exception as e:
            logger.error(f"Failed to rebuild embeddings: {e}")
            return 0

    async def close(self):
        """Close all managers and cleanup resources."""
        try:
            if self.sqlite_manager:
                await self.sqlite_manager.close()

            if self.vector_manager:
                await self.vector_manager.close()

            if self.embedding_manager and hasattr(self.embedding_manager, 'close'):
                await self.embedding_manager.close()

            logger.debug("Integrated manager closed successfully")

        except Exception as e:
            logger.error(f"Error closing integrated manager: {e}")


# Global instance for convenience
_integrated_manager = None


async def get_integrated_manager(
    db_path: str = "./autom8.db",
    config: VectorSearchConfig = None,
    reset_instance: bool = False
) -> IntegratedSQLiteManager:
    """Get global integrated manager instance."""
    global _integrated_manager

    if _integrated_manager is None or reset_instance:
        _integrated_manager = IntegratedSQLiteManager(
            db_path=db_path,
            vector_config=config
        )
        await _integrated_manager.initialize()

    return _integrated_manager