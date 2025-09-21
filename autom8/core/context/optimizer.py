"""
Context Optimizer - Intelligent context compression and optimization for Autom8.

Addresses the "Context Bloat" problem by implementing multiple optimization strategies:
- Semantic compression using embedding similarity
- Priority-based filtering
- Token budget optimization
- Summarization and deduplication
- Quality-preserving compression algorithms

This system dramatically reduces token usage while preserving essential information quality.
"""

import asyncio
import hashlib
import json
import re
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

from autom8.models.context import (
    ContextPreview,
    ContextSource,
    ContextSourceType,
    ContextWarning,
    ContextWarningType,
    ContextOptimization,
)
from autom8.core.memory.embeddings import LocalEmbedder
from autom8.utils.tokens import get_token_counter
from autom8.utils.logging import get_logger
from autom8.config.settings import get_settings

logger = get_logger(__name__)


class OptimizationStrategy(str, Enum):
    """Available optimization strategies"""
    SEMANTIC_COMPRESSION = "semantic_compression"
    PRIORITY_FILTERING = "priority_filtering"
    TOKEN_BUDGET = "token_budget"
    SUMMARIZATION = "summarization"
    DEDUPLICATION = "deduplication"
    STRUCTURAL_COMPRESSION = "structural_compression"


class OptimizationProfile(str, Enum):
    """Optimization aggressiveness profiles"""
    CONSERVATIVE = "conservative"    # Minimal compression, preserve quality
    BALANCED = "balanced"           # Moderate compression, good quality
    AGGRESSIVE = "aggressive"       # Maximum compression, accept quality loss


class CompressionQuality(str, Enum):
    """Quality levels for compression results"""
    EXCELLENT = "excellent"  # 0.9+ quality retention
    GOOD = "good"           # 0.7-0.9 quality retention
    ACCEPTABLE = "acceptable"  # 0.5-0.7 quality retention
    POOR = "poor"           # <0.5 quality retention


@dataclass
class OptimizationAnalytics:
    """Analytics data for optimization operations"""
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    quality_score: float
    strategies_used: List[str]
    execution_time_ms: float
    memory_saved_bytes: int


class OptimizationResult(BaseModel):
    """Result of a context optimization operation"""
    optimized_sources: List[ContextSource] = Field(description="Optimized context sources")
    removed_sources: List[ContextSource] = Field(description="Sources that were removed")
    original_tokens: int = Field(description="Original total token count")
    optimized_tokens: int = Field(description="Optimized total token count")
    compression_ratio: float = Field(description="Compression ratio (0.0-1.0)")
    quality_retention: float = Field(description="Estimated quality retention (0.0-1.0)")
    strategies_applied: List[OptimizationStrategy] = Field(description="Strategies that were applied")
    optimization_log: List[str] = Field(default_factory=list, description="Log of optimization steps")
    analytics: Optional[OptimizationAnalytics] = Field(default=None, description="Detailed analytics")
    
    @property
    def tokens_saved(self) -> int:
        """Calculate tokens saved by optimization"""
        return self.original_tokens - self.optimized_tokens
    
    @property
    def quality_category(self) -> CompressionQuality:
        """Categorize the quality retention"""
        if self.quality_retention >= 0.9:
            return CompressionQuality.EXCELLENT
        elif self.quality_retention >= 0.7:
            return CompressionQuality.GOOD
        elif self.quality_retention >= 0.5:
            return CompressionQuality.ACCEPTABLE
        else:
            return CompressionQuality.POOR


class SimilarityCluster(BaseModel):
    """A cluster of similar content sources"""
    sources: List[ContextSource] = Field(description="Sources in this cluster")
    representative: ContextSource = Field(description="Representative source for the cluster")
    similarity_threshold: float = Field(description="Similarity threshold used")
    total_tokens: int = Field(description="Total tokens in cluster")
    preserved_tokens: int = Field(description="Tokens preserved after compression")
    
    @property
    def compression_ratio(self) -> float:
        """Compression ratio for this cluster"""
        if self.total_tokens == 0:
            return 1.0
        return self.preserved_tokens / self.total_tokens


class ContextOptimizer:
    """
    Intelligent context optimization system that reduces token usage while preserving quality.
    
    Key features:
    - Multiple optimization strategies
    - Embedding-based semantic analysis
    - Quality preservation algorithms
    - Optimization history and learning
    - Configurable aggressiveness profiles
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.token_counter = get_token_counter()
        self.embedder = LocalEmbedder()
        self._initialized = False
        
        # Optimization configuration
        self.similarity_threshold = 0.85
        self.min_cluster_size = 2
        self.max_summary_ratio = 0.3  # Max 30% of original length for summaries
        
        # Quality thresholds for different profiles
        self.profile_thresholds = {
            OptimizationProfile.CONSERVATIVE: {
                "min_quality": 0.9,
                "max_compression": 0.3,  # Max 30% compression
                "similarity_threshold": 0.95
            },
            OptimizationProfile.BALANCED: {
                "min_quality": 0.7,
                "max_compression": 0.6,  # Max 60% compression
                "similarity_threshold": 0.85
            },
            OptimizationProfile.AGGRESSIVE: {
                "min_quality": 0.5,
                "max_compression": 0.8,  # Max 80% compression
                "similarity_threshold": 0.75
            }
        }
        
        # Redis connection for optimization history
        self.redis_client = None
        self.optimization_history: List[OptimizationResult] = []
    
    async def initialize(self) -> bool:
        """Initialize the context optimizer"""
        try:
            # Initialize embedder for semantic analysis
            embedder_initialized = await self.embedder.initialize()
            if not embedder_initialized:
                logger.warning("Embedder initialization failed, semantic compression disabled")
            
            # Initialize Redis for optimization history
            await self._initialize_redis()
            
            self._initialized = True
            logger.info("Context optimizer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize context optimizer: {e}")
            return False
    
    async def _initialize_redis(self):
        """Initialize Redis connection for optimization history"""
        try:
            import redis.asyncio as redis
            
            if self.settings.redis_url:
                self.redis_client = redis.from_url(self.settings.redis_url)
            else:
                self.redis_client = redis.Redis(
                    host=self.settings.redis_host,
                    port=self.settings.redis_port,
                    decode_responses=True
                )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis connection established for optimization history")
            
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}. History will be in-memory only.")
            self.redis_client = None
    
    async def optimize(
        self,
        preview: ContextPreview,
        profile: OptimizationProfile = OptimizationProfile.BALANCED,
        token_budget: Optional[int] = None,
        strategies: Optional[List[OptimizationStrategy]] = None,
        preserve_query: bool = True
    ) -> OptimizationResult:
        """
        Optimize context using specified profile and strategies.
        
        Args:
            preview: Context preview to optimize
            profile: Optimization aggressiveness profile
            token_budget: Strict token budget to fit within
            strategies: Specific strategies to use (if None, uses all available)
            preserve_query: Whether to preserve the original query unchanged
            
        Returns:
            OptimizationResult with optimized sources and analytics
        """
        start_time = datetime.utcnow()
        logger.info(f"Starting context optimization with {profile.value} profile")
        
        if not self._initialized:
            raise RuntimeError("ContextOptimizer not initialized")
        
        # Determine strategies to use
        if strategies is None:
            strategies = self._get_default_strategies(profile)
        
        # Get profile configuration
        profile_config = self.profile_thresholds[profile]
        
        # Separate query from other sources
        query_sources = [s for s in preview.sources if s.type == ContextSourceType.QUERY]
        other_sources = [s for s in preview.sources if s.type != ContextSourceType.QUERY]
        
        original_tokens = sum(s.tokens for s in preview.sources)
        optimization_log = []
        
        # Apply optimization strategies in order of priority
        optimized_sources = other_sources.copy()
        removed_sources = []
        
        for strategy in strategies:
            if strategy == OptimizationStrategy.DEDUPLICATION:
                optimized_sources, removed = await self._apply_deduplication(
                    optimized_sources, profile_config
                )
                removed_sources.extend(removed)
                optimization_log.append(f"Deduplication: removed {len(removed)} duplicate sources")
                
            elif strategy == OptimizationStrategy.SEMANTIC_COMPRESSION:
                optimized_sources, removed = await self._apply_semantic_compression(
                    optimized_sources, profile_config
                )
                removed_sources.extend(removed)
                optimization_log.append(f"Semantic compression: clustered and compressed similar content")
                
            elif strategy == OptimizationStrategy.PRIORITY_FILTERING:
                optimized_sources, removed = await self._apply_priority_filtering(
                    optimized_sources, profile_config, token_budget
                )
                removed_sources.extend(removed)
                optimization_log.append(f"Priority filtering: removed {len(removed)} low-priority sources")
                
            elif strategy == OptimizationStrategy.SUMMARIZATION:
                optimized_sources = await self._apply_summarization(
                    optimized_sources, profile_config
                )
                optimization_log.append("Summarization: compressed large content blocks")
                
            elif strategy == OptimizationStrategy.STRUCTURAL_COMPRESSION:
                optimized_sources = await self._apply_structural_compression(
                    optimized_sources, profile_config
                )
                optimization_log.append("Structural compression: removed formatting and redundancy")
        
        # Apply token budget if specified
        if token_budget:
            query_tokens = sum(s.tokens for s in query_sources)
            remaining_budget = token_budget - query_tokens
            
            if remaining_budget > 0:
                optimized_sources, budget_removed = await self._enforce_token_budget(
                    optimized_sources, remaining_budget
                )
                removed_sources.extend(budget_removed)
                optimization_log.append(f"Token budget: trimmed to fit {token_budget} tokens")
        
        # Add back query sources if preserving
        if preserve_query:
            final_sources = query_sources + optimized_sources
        else:
            final_sources = optimized_sources
        
        # Calculate results
        optimized_tokens = sum(s.tokens for s in final_sources)
        compression_ratio = 1.0 - (optimized_tokens / original_tokens) if original_tokens > 0 else 0.0
        quality_retention = self._estimate_quality_retention(
            preview.sources, final_sources, removed_sources, strategies
        )
        
        # Create analytics
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        analytics = OptimizationAnalytics(
            original_tokens=original_tokens,
            compressed_tokens=optimized_tokens,
            compression_ratio=compression_ratio,
            quality_score=quality_retention,
            strategies_used=[s.value for s in strategies],
            execution_time_ms=execution_time,
            memory_saved_bytes=(original_tokens - optimized_tokens) * 4  # Rough estimate
        )
        
        # Create result
        result = OptimizationResult(
            optimized_sources=final_sources,
            removed_sources=removed_sources,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            compression_ratio=compression_ratio,
            quality_retention=quality_retention,
            strategies_applied=strategies,
            optimization_log=optimization_log,
            analytics=analytics
        )
        
        # Store optimization history
        await self._store_optimization_history(result)
        
        logger.info(
            f"Optimization complete: {original_tokens} -> {optimized_tokens} tokens "
            f"({compression_ratio:.1%} compression, {quality_retention:.1%} quality retention)"
        )
        
        return result
    
    def _get_default_strategies(self, profile: OptimizationProfile) -> List[OptimizationStrategy]:
        """Get default optimization strategies for a profile"""
        if profile == OptimizationProfile.CONSERVATIVE:
            return [
                OptimizationStrategy.DEDUPLICATION,
                OptimizationStrategy.STRUCTURAL_COMPRESSION,
                OptimizationStrategy.PRIORITY_FILTERING
            ]
        elif profile == OptimizationProfile.BALANCED:
            return [
                OptimizationStrategy.DEDUPLICATION,
                OptimizationStrategy.SEMANTIC_COMPRESSION,
                OptimizationStrategy.PRIORITY_FILTERING,
                OptimizationStrategy.SUMMARIZATION,
                OptimizationStrategy.STRUCTURAL_COMPRESSION
            ]
        else:  # AGGRESSIVE
            return [
                OptimizationStrategy.DEDUPLICATION,
                OptimizationStrategy.SEMANTIC_COMPRESSION,
                OptimizationStrategy.PRIORITY_FILTERING,
                OptimizationStrategy.SUMMARIZATION,
                OptimizationStrategy.STRUCTURAL_COMPRESSION,
                OptimizationStrategy.TOKEN_BUDGET
            ]
    
    async def _apply_deduplication(
        self,
        sources: List[ContextSource],
        profile_config: Dict[str, Any]
    ) -> Tuple[List[ContextSource], List[ContextSource]]:
        """Remove exact and near-duplicate content"""
        if not sources:
            return sources, []
        
        # Track content hashes and similar content
        content_hashes = {}
        similar_groups = []
        removed_sources = []
        
        for source in sources:
            # Exact duplicate detection
            content_hash = hashlib.md5(source.content.encode()).hexdigest()
            if content_hash in content_hashes:
                # Duplicate found - keep the one with higher priority
                existing_source = content_hashes[content_hash]
                if source.priority > existing_source.priority:
                    # Replace existing with new higher priority source
                    removed_sources.append(existing_source)
                    content_hashes[content_hash] = source
                else:
                    # Keep existing, remove new one
                    removed_sources.append(source)
                continue
            
            content_hashes[content_hash] = source
        
        # Keep non-duplicate sources
        deduplicated_sources = [s for s in sources if s not in removed_sources]
        
        return deduplicated_sources, removed_sources
    
    async def _apply_semantic_compression(
        self,
        sources: List[ContextSource],
        profile_config: Dict[str, Any]
    ) -> Tuple[List[ContextSource], List[ContextSource]]:
        """Compress semantically similar content using embeddings"""
        if not sources or not self.embedder.is_available():
            return sources, []
        
        # Generate embeddings for all sources
        texts = [source.content for source in sources]
        embeddings = await self.embedder.embed_batch(texts)
        
        if embeddings is None:
            logger.warning("Failed to generate embeddings for semantic compression")
            return sources, []
        
        # Find similar content clusters
        similarity_threshold = profile_config["similarity_threshold"]
        clusters = self._find_similarity_clusters(sources, embeddings, similarity_threshold)
        
        # Compress each cluster
        compressed_sources = []
        removed_sources = []
        
        for cluster in clusters:
            if len(cluster.sources) >= self.min_cluster_size:
                # Compress cluster to representative source
                compressed_source = self._compress_cluster(cluster)
                compressed_sources.append(compressed_source)
                
                # Mark other sources as removed
                for source in cluster.sources:
                    if source != cluster.representative:
                        removed_sources.append(source)
            else:
                # Keep individual sources if cluster too small
                compressed_sources.extend(cluster.sources)
        
        return compressed_sources, removed_sources
    
    def _find_similarity_clusters(
        self,
        sources: List[ContextSource],
        embeddings: np.ndarray,
        threshold: float
    ) -> List[SimilarityCluster]:
        """Find clusters of similar content using embedding similarity"""
        if len(sources) != len(embeddings):
            raise ValueError("Number of sources and embeddings must match")
        
        clusters = []
        used_indices = set()
        
        for i, source in enumerate(sources):
            if i in used_indices:
                continue
            
            # Find similar sources
            cluster_sources = [source]
            cluster_indices = [i]
            
            for j, other_source in enumerate(sources):
                if j <= i or j in used_indices:
                    continue
                
                # Calculate cosine similarity
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                
                if similarity >= threshold:
                    cluster_sources.append(other_source)
                    cluster_indices.append(j)
            
            # Mark indices as used
            used_indices.update(cluster_indices)
            
            # Choose representative (highest priority source)
            representative = max(cluster_sources, key=lambda s: s.priority)
            
            # Calculate token counts
            total_tokens = sum(s.tokens for s in cluster_sources)
            preserved_tokens = representative.tokens
            
            cluster = SimilarityCluster(
                sources=cluster_sources,
                representative=representative,
                similarity_threshold=threshold,
                total_tokens=total_tokens,
                preserved_tokens=preserved_tokens
            )
            
            clusters.append(cluster)
        
        return clusters
    
    def _compress_cluster(self, cluster: SimilarityCluster) -> ContextSource:
        """Compress a similarity cluster to a single representative source"""
        if len(cluster.sources) == 1:
            return cluster.sources[0]
        
        # Create enhanced representative with information from all sources
        representative = cluster.representative
        
        # Combine unique information from all sources
        all_content = [s.content for s in cluster.sources]
        combined_summary = self._create_cluster_summary(all_content)
        
        # Create new compressed source
        compressed_source = ContextSource(
            type=representative.type,
            content=combined_summary,
            tokens=self.token_counter.count_tokens(combined_summary),
            source=f"compressed({len(cluster.sources)} sources)",
            priority=representative.priority,
            summary=f"Semantic compression of {len(cluster.sources)} similar sources",
            location=representative.location,
            timestamp=representative.timestamp,
            expandable=True
        )
        
        return compressed_source
    
    def _create_cluster_summary(self, contents: List[str]) -> str:
        """Create a summary that preserves key information from multiple similar contents"""
        if len(contents) == 1:
            return contents[0]
        
        # Simple strategy: take the longest content as base and add unique key phrases
        base_content = max(contents, key=len)
        
        # Extract key phrases from other contents that aren't in base
        key_phrases = set()
        for content in contents:
            if content != base_content:
                # Simple key phrase extraction: look for important terms
                phrases = self._extract_key_phrases(content, base_content)
                key_phrases.update(phrases)
        
        # Combine base content with unique key phrases
        if key_phrases:
            summary = base_content + "\n\nAdditional key points: " + "; ".join(key_phrases)
        else:
            summary = base_content
        
        return summary
    
    def _extract_key_phrases(self, content: str, base_content: str) -> Set[str]:
        """Extract key phrases from content that aren't in base content"""
        # Simple implementation: extract sentences that contain important terms
        # and aren't already covered in base content
        
        important_terms = [
            "function", "class", "method", "variable", "import", "error", "warning",
            "critical", "important", "note", "todo", "fixme", "bug", "issue"
        ]
        
        phrases = set()
        sentences = content.split('. ')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Check if sentence contains important terms and isn't in base content
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in important_terms):
                if sentence not in base_content:
                    phrases.add(sentence)
        
        return phrases
    
    async def _apply_priority_filtering(
        self,
        sources: List[ContextSource],
        profile_config: Dict[str, Any],
        token_budget: Optional[int] = None
    ) -> Tuple[List[ContextSource], List[ContextSource]]:
        """Filter sources based on priority and budget constraints"""
        if not sources:
            return sources, []
        
        # Sort sources by priority (highest first)
        sorted_sources = sorted(sources, key=lambda s: s.priority, reverse=True)
        
        # Determine filtering threshold based on profile
        min_quality = profile_config["min_quality"]
        max_compression = profile_config["max_compression"]
        
        # Calculate priority threshold
        priorities = [s.priority for s in sorted_sources]
        if priorities:
            max_priority = max(priorities)
            min_priority = min(priorities)
            priority_range = max_priority - min_priority
            
            # Conservative: keep high priority only
            # Balanced: keep medium to high priority
            # Aggressive: keep some low priority
            if min_quality >= 0.9:  # Conservative
                threshold_ratio = 0.7
            elif min_quality >= 0.7:  # Balanced
                threshold_ratio = 0.4
            else:  # Aggressive
                threshold_ratio = 0.2
            
            priority_threshold = min_priority + (priority_range * threshold_ratio)
        else:
            priority_threshold = 0
        
        # Filter by priority
        kept_sources = []
        removed_sources = []
        current_tokens = 0
        
        for source in sorted_sources:
            # Check priority threshold
            if source.priority < priority_threshold:
                removed_sources.append(source)
                continue
            
            # Check token budget if specified
            if token_budget and current_tokens + source.tokens > token_budget:
                removed_sources.append(source)
                continue
            
            kept_sources.append(source)
            current_tokens += source.tokens
        
        return kept_sources, removed_sources
    
    async def _apply_summarization(
        self,
        sources: List[ContextSource],
        profile_config: Dict[str, Any]
    ) -> List[ContextSource]:
        """Apply summarization to large content blocks"""
        summarized_sources = []
        
        for source in sources:
            # Only summarize large sources
            if source.tokens > 200:  # Threshold for summarization
                summarized_content = self._summarize_content(
                    source.content, self.max_summary_ratio
                )
                
                summarized_source = ContextSource(
                    type=ContextSourceType.SUMMARY,
                    content=summarized_content,
                    tokens=self.token_counter.count_tokens(summarized_content),
                    source=f"summary({source.source})",
                    priority=source.priority,
                    summary=f"Summarized from {source.tokens} tokens",
                    location=source.location,
                    timestamp=source.timestamp,
                    expandable=True
                )
                
                summarized_sources.append(summarized_source)
            else:
                # Keep small sources unchanged
                summarized_sources.append(source)
        
        return summarized_sources
    
    def _summarize_content(self, content: str, max_ratio: float) -> str:
        """Summarize content using extractive summarization"""
        sentences = content.split('. ')
        if len(sentences) <= 3:
            return content  # Too short to summarize
        
        # Target length
        target_sentences = max(2, int(len(sentences) * max_ratio))
        
        # Simple extractive summarization: score sentences and keep top ones
        sentence_scores = []
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            # Score based on:
            # 1. Position (first and last sentences often important)
            # 2. Length (medium length sentences often more informative)
            # 3. Important keywords
            
            position_score = 1.0 if i == 0 or i == len(sentences) - 1 else 0.5
            length_score = min(1.0, len(sentence) / 100)  # Normalize around 100 chars
            
            # Keyword scoring
            important_keywords = [
                'function', 'class', 'method', 'import', 'error', 'warning',
                'critical', 'important', 'main', 'key', 'primary', 'essential'
            ]
            keyword_score = sum(1 for keyword in important_keywords 
                              if keyword in sentence.lower()) / len(important_keywords)
            
            total_score = position_score + length_score + keyword_score
            sentence_scores.append((sentence, total_score, i))
        
        # Sort by score and take top sentences
        top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:target_sentences]
        
        # Sort by original order to maintain narrative flow
        top_sentences.sort(key=lambda x: x[2])
        
        # Reconstruct summary
        summary = '. '.join(sentence[0] for sentence in top_sentences)
        if not summary.endswith('.'):
            summary += '.'
        
        return summary
    
    async def _apply_structural_compression(
        self,
        sources: List[ContextSource],
        profile_config: Dict[str, Any]
    ) -> List[ContextSource]:
        """Apply structural compression to remove formatting and redundancy"""
        compressed_sources = []
        
        for source in sources:
            compressed_content = self._compress_structure(source.content)
            
            compressed_source = ContextSource(
                type=source.type,
                content=compressed_content,
                tokens=self.token_counter.count_tokens(compressed_content),
                source=source.source,
                priority=source.priority,
                summary=source.summary,
                location=source.location,
                timestamp=source.timestamp,
                expandable=source.expandable
            )
            
            compressed_sources.append(compressed_source)
        
        return compressed_sources
    
    def _compress_structure(self, content: str) -> str:
        """Remove unnecessary formatting and whitespace"""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Multiple newlines to double
        content = re.sub(r' +', ' ', content)  # Multiple spaces to single
        
        # Remove markdown formatting in aggressive mode
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # Bold
        content = re.sub(r'\*(.*?)\*', r'\1', content)  # Italic
        content = re.sub(r'`(.*?)`', r'\1', content)  # Inline code
        
        # Remove comment-style formatting
        content = re.sub(r'^\s*#\s*', '', content, flags=re.MULTILINE)  # Remove # prefixes
        content = re.sub(r'^\s*-\s*', '• ', content, flags=re.MULTILINE)  # Compact bullets
        
        # Remove redundant phrases
        redundant_phrases = [
            'As you can see,', 'It should be noted that', 'Please note that',
            'It is important to', 'You should know that', 'Keep in mind that'
        ]
        
        for phrase in redundant_phrases:
            content = content.replace(phrase, '')
        
        return content.strip()
    
    async def _enforce_token_budget(
        self,
        sources: List[ContextSource],
        budget: int
    ) -> Tuple[List[ContextSource], List[ContextSource]]:
        """Enforce strict token budget by trimming sources"""
        if budget <= 0:
            return [], sources
        
        # Sort by priority and fit within budget
        sorted_sources = sorted(sources, key=lambda s: s.priority, reverse=True)
        
        kept_sources = []
        removed_sources = []
        current_tokens = 0
        
        for source in sorted_sources:
            if current_tokens + source.tokens <= budget:
                kept_sources.append(source)
                current_tokens += source.tokens
            else:
                # Try to partially include source if there's remaining budget
                remaining_budget = budget - current_tokens
                if remaining_budget > 50:  # Only if meaningful amount remaining
                    trimmed_content = self._trim_content_to_budget(
                        source.content, remaining_budget
                    )
                    
                    trimmed_source = ContextSource(
                        type=source.type,
                        content=trimmed_content,
                        tokens=self.token_counter.count_tokens(trimmed_content),
                        source=f"trimmed({source.source})",
                        priority=source.priority,
                        summary=f"Trimmed from {source.tokens} tokens",
                        location=source.location,
                        timestamp=source.timestamp,
                        expandable=True
                    )
                    
                    kept_sources.append(trimmed_source)
                    current_tokens += trimmed_source.tokens
                
                removed_sources.append(source)
        
        return kept_sources, removed_sources
    
    def _trim_content_to_budget(self, content: str, token_budget: int) -> str:
        """Trim content to fit within token budget while preserving meaning"""
        current_tokens = self.token_counter.count_tokens(content)
        
        if current_tokens <= token_budget:
            return content
        
        # Calculate target character count (rough estimation)
        chars_per_token = len(content) / current_tokens
        target_chars = int(token_budget * chars_per_token * 0.9)  # 10% buffer
        
        if target_chars <= 100:
            return content[:target_chars] + "..."
        
        # Try to preserve sentence boundaries
        sentences = content.split('. ')
        trimmed_sentences = []
        current_chars = 0
        
        for sentence in sentences:
            if current_chars + len(sentence) + 2 <= target_chars:  # +2 for '. '
                trimmed_sentences.append(sentence)
                current_chars += len(sentence) + 2
            else:
                break
        
        trimmed_content = '. '.join(trimmed_sentences)
        if not trimmed_content.endswith('.'):
            trimmed_content += '.'
        
        # Add truncation indicator if significant content was removed
        if len(trimmed_content) < len(content) * 0.8:
            trimmed_content += " [content truncated for token budget]"
        
        return trimmed_content
    
    def _estimate_quality_retention(
        self,
        original_sources: List[ContextSource],
        optimized_sources: List[ContextSource],
        removed_sources: List[ContextSource],
        strategies: List[OptimizationStrategy]
    ) -> float:
        """Estimate quality retention after optimization"""
        if not original_sources:
            return 1.0
        
        # Base quality retention
        base_quality = 1.0
        
        # Penalty for removed sources based on their priority
        if removed_sources:
            total_original_priority = sum(s.priority for s in original_sources)
            removed_priority = sum(s.priority for s in removed_sources)
            
            if total_original_priority > 0:
                priority_loss = removed_priority / total_original_priority
                base_quality -= priority_loss * 0.5  # Max 50% penalty for priority loss
        
        # Strategy-specific quality impacts
        strategy_impacts = {
            OptimizationStrategy.DEDUPLICATION: 0.0,      # No quality loss
            OptimizationStrategy.STRUCTURAL_COMPRESSION: -0.05,  # Minor loss
            OptimizationStrategy.PRIORITY_FILTERING: -0.1,       # Low loss
            OptimizationStrategy.SEMANTIC_COMPRESSION: -0.15,    # Moderate loss
            OptimizationStrategy.SUMMARIZATION: -0.25,           # Higher loss
            OptimizationStrategy.TOKEN_BUDGET: -0.3              # Potentially high loss
        }
        
        for strategy in strategies:
            base_quality += strategy_impacts.get(strategy, 0.0)
        
        # Ensure quality is between 0 and 1
        return max(0.0, min(1.0, base_quality))
    
    async def _store_optimization_history(self, result: OptimizationResult):
        """Store optimization result in history for learning"""
        self.optimization_history.append(result)
        
        # Also store in Redis if available
        if self.redis_client:
            try:
                history_key = f"autom8:optimization:history:{datetime.utcnow().isoformat()}"
                await self.redis_client.setex(
                    history_key,
                    timedelta(days=30),  # Keep for 30 days
                    json.dumps(result.dict(), default=str)
                )
            except Exception as e:
                logger.warning(f"Failed to store optimization history in Redis: {e}")
    
    async def get_optimization_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get optimization analytics for the specified period"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Filter recent optimizations
        recent_optimizations = [
            opt for opt in self.optimization_history
            if opt.analytics and opt.analytics.execution_time_ms > 0
        ]
        
        if not recent_optimizations:
            return {
                "total_optimizations": 0,
                "average_compression_ratio": 0.0,
                "average_quality_retention": 0.0,
                "total_tokens_saved": 0,
                "strategy_effectiveness": {}
            }
        
        # Calculate analytics
        total_optimizations = len(recent_optimizations)
        avg_compression = sum(opt.compression_ratio for opt in recent_optimizations) / total_optimizations
        avg_quality = sum(opt.quality_retention for opt in recent_optimizations) / total_optimizations
        total_tokens_saved = sum(opt.tokens_saved for opt in recent_optimizations)
        
        # Strategy effectiveness
        strategy_stats = {}
        for opt in recent_optimizations:
            for strategy in opt.strategies_applied:
                if strategy.value not in strategy_stats:
                    strategy_stats[strategy.value] = {
                        "count": 0,
                        "total_compression": 0.0,
                        "total_quality": 0.0
                    }
                
                stats = strategy_stats[strategy.value]
                stats["count"] += 1
                stats["total_compression"] += opt.compression_ratio
                stats["total_quality"] += opt.quality_retention
        
        # Calculate averages for strategies
        strategy_effectiveness = {}
        for strategy, stats in strategy_stats.items():
            strategy_effectiveness[strategy] = {
                "usage_count": stats["count"],
                "avg_compression": stats["total_compression"] / stats["count"],
                "avg_quality_retention": stats["total_quality"] / stats["count"]
            }
        
        return {
            "total_optimizations": total_optimizations,
            "average_compression_ratio": avg_compression,
            "average_quality_retention": avg_quality,
            "total_tokens_saved": total_tokens_saved,
            "strategy_effectiveness": strategy_effectiveness,
            "period_days": days
        }
    
    async def suggest_optimization_profile(
        self,
        context_preview: ContextPreview,
        quality_requirements: float = 0.8,
        cost_constraints: Optional[float] = None
    ) -> OptimizationProfile:
        """
        Suggest the best optimization profile based on context and requirements.
        
        Args:
            context_preview: Context to analyze
            quality_requirements: Minimum acceptable quality (0.0-1.0)
            cost_constraints: Maximum acceptable cost in USD
            
        Returns:
            Recommended optimization profile
        """
        # Analyze context characteristics
        total_tokens = context_preview.total_tokens
        source_diversity = len(set(s.type for s in context_preview.sources))
        high_priority_sources = len([s for s in context_preview.sources if s.priority > 70])
        
        # Conservative conditions
        if (quality_requirements >= 0.9 or 
            high_priority_sources / len(context_preview.sources) > 0.7 or
            total_tokens < 1000):
            return OptimizationProfile.CONSERVATIVE
        
        # Aggressive conditions
        if (quality_requirements <= 0.6 or
            total_tokens > 4000 or
            (cost_constraints and context_preview.cost_estimate > cost_constraints)):
            return OptimizationProfile.AGGRESSIVE
        
        # Default to balanced
        return OptimizationProfile.BALANCED


# Global optimizer instance
_optimizer = None


async def get_context_optimizer() -> ContextOptimizer:
    """Get global context optimizer instance"""
    global _optimizer
    if _optimizer is None:
        _optimizer = ContextOptimizer()
        await _optimizer.initialize()
    return _optimizer