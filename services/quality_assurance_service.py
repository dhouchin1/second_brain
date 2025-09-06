# ──────────────────────────────────────────────────────────────────────────────
# File: services/quality_assurance_service.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Quality Assurance Service

Advanced content validation, duplicate prevention, and processing quality metrics
for the Second Brain capture system. This service provides Phase 2 Smart Automation
capabilities including:

- Content validation with configurable quality standards
- Advanced duplicate detection with intelligent merging strategies  
- Processing confidence metrics and quality scoring
- Content completeness validation and enhancement suggestions
- Smart merging of duplicate content with conflict resolution
- Quality trend analysis and improvement recommendations
"""

import logging
import hashlib
import json
import re
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union, NamedTuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import difflib

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import Levenshtein
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False

from config import settings

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Quality levels for content assessment."""
    POOR = "poor"           # 0.0 - 0.4
    FAIR = "fair"           # 0.4 - 0.6
    GOOD = "good"           # 0.6 - 0.8
    EXCELLENT = "excellent" # 0.8 - 1.0


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DuplicateStrategy(Enum):
    """Strategies for handling duplicates."""
    SKIP = "skip"           # Skip duplicate, keep original
    REPLACE = "replace"     # Replace original with new
    MERGE = "merge"         # Merge content intelligently
    FLAG = "flag"           # Flag as duplicate but save both


class ContentValidationRule(Enum):
    """Built-in content validation rules."""
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    LANGUAGE_DETECTION = "language_detection"
    ENCODING_VALIDATION = "encoding_validation"
    STRUCTURE_VALIDATION = "structure_validation"
    SENTIMENT_FILTER = "sentiment_filter"
    SPAM_DETECTION = "spam_detection"
    PROFANITY_FILTER = "profanity_filter"
    DUPLICATE_DETECTION = "duplicate_detection"
    COMPLETENESS_CHECK = "completeness_check"


@dataclass
class ValidationIssue:
    """A content validation issue."""
    rule: ContentValidationRule
    severity: ValidationSeverity
    message: str
    suggestion: str
    confidence: float
    affected_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityMetrics:
    """Content quality assessment metrics."""
    overall_score: float  # 0-1
    quality_level: QualityLevel
    
    # Component scores
    content_completeness: float
    structural_quality: float
    language_quality: float
    relevance_score: float
    uniqueness_score: float
    
    # Detailed metrics
    word_count: int
    sentence_count: int
    paragraph_count: int
    reading_level: str
    sentiment_score: float
    
    # Processing metrics
    processing_confidence: float
    extraction_success: bool
    ai_enhancement_success: bool
    
    # Issues and suggestions
    issues: List[ValidationIssue] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    
    @property
    def has_critical_issues(self) -> bool:
        return any(issue.severity == ValidationSeverity.CRITICAL for issue in self.issues)
    
    @property
    def has_errors(self) -> bool:
        return any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)


@dataclass
class DuplicateMatch:
    """A duplicate content match."""
    original_id: Optional[int]
    similarity_score: float
    match_type: str  # "exact", "near_exact", "similar", "fuzzy"
    
    # Content comparison
    content_similarity: float
    title_similarity: float
    metadata_similarity: float
    
    # Recommendation
    recommended_strategy: DuplicateStrategy
    merge_confidence: float
    
    # Details
    differences: List[str] = field(default_factory=list)
    common_elements: List[str] = field(default_factory=list)
    conflict_areas: List[str] = field(default_factory=list)


@dataclass
class ContentValidationResult:
    """Result of content validation process."""
    is_valid: bool
    quality_metrics: QualityMetrics
    duplicate_matches: List[DuplicateMatch] = field(default_factory=list)
    
    # Processing recommendations
    should_process: bool = True
    recommended_strategy: Optional[DuplicateStrategy] = None
    merge_target_id: Optional[int] = None
    
    # Enhancement suggestions
    content_enhancements: Dict[str, Any] = field(default_factory=dict)
    metadata_enhancements: Dict[str, Any] = field(default_factory=dict)
    
    # Performance
    validation_time: float = 0.0
    rules_evaluated: int = 0


class ContentValidator:
    """Core content validation engine."""
    
    def __init__(self):
        """Initialize content validator."""
        self.validation_rules = {
            ContentValidationRule.MIN_LENGTH: self._validate_min_length,
            ContentValidationRule.MAX_LENGTH: self._validate_max_length,
            ContentValidationRule.LANGUAGE_DETECTION: self._validate_language,
            ContentValidationRule.ENCODING_VALIDATION: self._validate_encoding,
            ContentValidationRule.STRUCTURE_VALIDATION: self._validate_structure,
            ContentValidationRule.SENTIMENT_FILTER: self._validate_sentiment,
            ContentValidationRule.SPAM_DETECTION: self._detect_spam,
            ContentValidationRule.PROFANITY_FILTER: self._filter_profanity,
            ContentValidationRule.COMPLETENESS_CHECK: self._check_completeness
        }
        
        # Validation thresholds (configurable)
        self.thresholds = {
            'min_word_count': 3,
            'max_word_count': 50000,
            'min_sentence_completeness': 0.5,
            'spam_keywords_threshold': 3,
            'profanity_tolerance': 0,
            'sentiment_filter_enabled': False,
            'min_sentiment_score': -0.8
        }
        
        # Spam detection keywords
        self.spam_indicators = [
            'click here', 'free money', 'limited time', 'act now',
            'guaranteed', 'risk free', '100% free', 'no obligation',
            'winner', 'congratulations', 'selected', 'urgent'
        ]
        
        # Basic profanity filter (minimal set)
        self.profanity_words = {
            # This would typically be loaded from an external file
            # Keeping minimal for this implementation
        }
    
    def validate_content(
        self,
        content: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        validation_config: Optional[Dict[str, Any]] = None
    ) -> List[ValidationIssue]:
        """Validate content against all enabled rules."""
        issues = []
        config = validation_config or {}
        
        # Apply validation rules
        for rule, validator_func in self.validation_rules.items():
            if config.get(f'enable_{rule.value}', True):
                try:
                    rule_issues = validator_func(content, title, metadata, config)
                    issues.extend(rule_issues)
                except Exception as e:
                    logger.warning(f"Validation rule {rule.value} failed: {e}")
                    issues.append(ValidationIssue(
                        rule=rule,
                        severity=ValidationSeverity.WARNING,
                        message=f"Validation rule failed: {str(e)}",
                        suggestion="Check validation rule configuration",
                        confidence=0.5
                    ))
        
        return issues
    
    def _validate_min_length(
        self,
        content: str,
        title: Optional[str],
        metadata: Optional[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate minimum content length."""
        issues = []
        word_count = len(content.split())
        min_words = config.get('min_word_count', self.thresholds['min_word_count'])
        
        if word_count < min_words:
            issues.append(ValidationIssue(
                rule=ContentValidationRule.MIN_LENGTH,
                severity=ValidationSeverity.WARNING if word_count > 0 else ValidationSeverity.ERROR,
                message=f"Content is too short: {word_count} words (minimum: {min_words})",
                suggestion=f"Consider expanding content to at least {min_words} words",
                confidence=0.9,
                metadata={'word_count': word_count, 'min_required': min_words}
            ))
        
        return issues
    
    def _validate_max_length(
        self,
        content: str,
        title: Optional[str],
        metadata: Optional[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate maximum content length."""
        issues = []
        word_count = len(content.split())
        max_words = config.get('max_word_count', self.thresholds['max_word_count'])
        
        if word_count > max_words:
            issues.append(ValidationIssue(
                rule=ContentValidationRule.MAX_LENGTH,
                severity=ValidationSeverity.WARNING,
                message=f"Content is very long: {word_count} words (maximum recommended: {max_words})",
                suggestion="Consider breaking into smaller sections or summarizing",
                confidence=0.8,
                metadata={'word_count': word_count, 'max_recommended': max_words}
            ))
        
        return issues
    
    def _validate_language(
        self,
        content: str,
        title: Optional[str],
        metadata: Optional[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Basic language validation."""
        issues = []
        
        # Simple heuristic: check for non-ASCII characters ratio
        ascii_chars = sum(1 for char in content if ord(char) < 128)
        total_chars = len(content)
        
        if total_chars > 0:
            ascii_ratio = ascii_chars / total_chars
            
            if ascii_ratio < 0.7:  # Less than 70% ASCII
                issues.append(ValidationIssue(
                    rule=ContentValidationRule.LANGUAGE_DETECTION,
                    severity=ValidationSeverity.INFO,
                    message=f"Content contains {(1-ascii_ratio)*100:.1f}% non-ASCII characters",
                    suggestion="Verify content encoding and language settings",
                    confidence=0.6,
                    metadata={'ascii_ratio': ascii_ratio}
                ))
        
        return issues
    
    def _validate_encoding(
        self,
        content: str,
        title: Optional[str],
        metadata: Optional[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate content encoding."""
        issues = []
        
        # Check for common encoding issues
        encoding_indicators = [
            ('â€™', 'Possible UTF-8 encoding issue (apostrophe)'),
            ('â€œ', 'Possible UTF-8 encoding issue (quote)'),
            ('â€', 'Possible UTF-8 encoding issue (dash)'),
            ('Ã¡', 'Possible UTF-8 encoding issue (accented character)')
        ]
        
        for indicator, message in encoding_indicators:
            if indicator in content:
                issues.append(ValidationIssue(
                    rule=ContentValidationRule.ENCODING_VALIDATION,
                    severity=ValidationSeverity.WARNING,
                    message=message,
                    suggestion="Check source encoding and re-import if necessary",
                    confidence=0.8,
                    affected_content=indicator
                ))
        
        return issues
    
    def _validate_structure(
        self,
        content: str,
        title: Optional[str],
        metadata: Optional[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate content structure."""
        issues = []
        
        # Check for sentence structure
        sentences = re.split(r'[.!?]+', content)
        complete_sentences = [s for s in sentences if len(s.strip()) > 10]
        
        if sentences:
            completeness_ratio = len(complete_sentences) / len(sentences)
            min_completeness = config.get('min_sentence_completeness', 
                                        self.thresholds['min_sentence_completeness'])
            
            if completeness_ratio < min_completeness:
                issues.append(ValidationIssue(
                    rule=ContentValidationRule.STRUCTURE_VALIDATION,
                    severity=ValidationSeverity.INFO,
                    message=f"Content may lack proper sentence structure ({completeness_ratio:.1%} complete sentences)",
                    suggestion="Consider reviewing and improving sentence structure",
                    confidence=0.7,
                    metadata={'completeness_ratio': completeness_ratio}
                ))
        
        # Check for excessive repetition
        words = content.lower().split()
        if words:
            word_freq = {}
            for word in words:
                if len(word) > 3:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Find most repeated word
            if word_freq:
                max_freq = max(word_freq.values())
                if max_freq > len(words) * 0.1:  # More than 10% repetition
                    repeated_word = max(word_freq, key=word_freq.get)
                    issues.append(ValidationIssue(
                        rule=ContentValidationRule.STRUCTURE_VALIDATION,
                        severity=ValidationSeverity.INFO,
                        message=f"Excessive repetition detected: '{repeated_word}' appears {max_freq} times",
                        suggestion="Consider varying vocabulary to improve readability",
                        confidence=0.8,
                        affected_content=repeated_word
                    ))
        
        return issues
    
    def _validate_sentiment(
        self,
        content: str,
        title: Optional[str],
        metadata: Optional[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Basic sentiment validation."""
        issues = []
        
        if not config.get('sentiment_filter_enabled', self.thresholds['sentiment_filter_enabled']):
            return issues
        
        # Simple sentiment analysis (could be enhanced with proper NLP)
        negative_words = ['hate', 'terrible', 'awful', 'horrible', 'worst', 'disgusting']
        positive_words = ['love', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        
        words = content.lower().split()
        negative_count = sum(1 for word in words if word in negative_words)
        positive_count = sum(1 for word in words if word in positive_words)
        
        if words:
            sentiment_score = (positive_count - negative_count) / len(words)
            min_sentiment = config.get('min_sentiment_score', self.thresholds['min_sentiment_score'])
            
            if sentiment_score < min_sentiment:
                issues.append(ValidationIssue(
                    rule=ContentValidationRule.SENTIMENT_FILTER,
                    severity=ValidationSeverity.WARNING,
                    message=f"Content has very negative sentiment (score: {sentiment_score:.2f})",
                    suggestion="Consider if this content should be processed or filtered",
                    confidence=0.6,
                    metadata={'sentiment_score': sentiment_score}
                ))
        
        return issues
    
    def _detect_spam(
        self,
        content: str,
        title: Optional[str],
        metadata: Optional[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Detect spam-like content."""
        issues = []
        
        content_lower = content.lower()
        spam_matches = [keyword for keyword in self.spam_indicators if keyword in content_lower]
        
        threshold = config.get('spam_keywords_threshold', self.thresholds['spam_keywords_threshold'])
        
        if len(spam_matches) >= threshold:
            issues.append(ValidationIssue(
                rule=ContentValidationRule.SPAM_DETECTION,
                severity=ValidationSeverity.ERROR,
                message=f"Content contains {len(spam_matches)} spam indicators",
                suggestion="Review content for spam-like characteristics",
                confidence=min(0.8, len(spam_matches) * 0.2),
                metadata={'spam_keywords': spam_matches}
            ))
        
        return issues
    
    def _filter_profanity(
        self,
        content: str,
        title: Optional[str],
        metadata: Optional[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Filter profanity (basic implementation)."""
        issues = []
        
        if not self.profanity_words:
            return issues  # No profanity filter configured
        
        content_lower = content.lower()
        found_profanity = [word for word in self.profanity_words if word in content_lower]
        
        tolerance = config.get('profanity_tolerance', self.thresholds['profanity_tolerance'])
        
        if len(found_profanity) > tolerance:
            issues.append(ValidationIssue(
                rule=ContentValidationRule.PROFANITY_FILTER,
                severity=ValidationSeverity.WARNING,
                message=f"Content contains {len(found_profanity)} flagged words",
                suggestion="Review content for inappropriate language",
                confidence=0.9,
                metadata={'flagged_words_count': len(found_profanity)}
            ))
        
        return issues
    
    def _check_completeness(
        self,
        content: str,
        title: Optional[str],
        metadata: Optional[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Check content completeness."""
        issues = []
        
        # Check for truncation indicators
        truncation_indicators = ['...', '[truncated]', '[cut off]', 'see more', 'read more']
        
        content_lower = content.lower()
        for indicator in truncation_indicators:
            if indicator in content_lower:
                issues.append(ValidationIssue(
                    rule=ContentValidationRule.COMPLETENESS_CHECK,
                    severity=ValidationSeverity.WARNING,
                    message=f"Content appears to be truncated (found: '{indicator}')",
                    suggestion="Verify content is complete before processing",
                    confidence=0.7,
                    affected_content=indicator
                ))
        
        # Check for incomplete sentences at the end
        if content and not content.rstrip().endswith(('.', '!', '?', '"', "'")):
            last_sentence = content.split('.')[-1].strip()
            if len(last_sentence) > 20:  # Likely incomplete sentence
                issues.append(ValidationIssue(
                    rule=ContentValidationRule.COMPLETENESS_CHECK,
                    severity=ValidationSeverity.INFO,
                    message="Content may end with incomplete sentence",
                    suggestion="Verify content completeness",
                    confidence=0.5
                ))
        
        return issues


class DuplicateDetector:
    """Advanced duplicate detection with multiple algorithms."""
    
    def __init__(self, get_conn_func):
        """Initialize duplicate detector."""
        self.get_conn = get_conn_func
        
        # TF-IDF vectorizer for semantic similarity
        self.tfidf_vectorizer = None
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
        
        # Similarity thresholds
        self.thresholds = {
            'exact_match': 1.0,
            'near_exact': 0.95,
            'high_similarity': 0.8,
            'medium_similarity': 0.6,
            'low_similarity': 0.4
        }
        
        # Content normalization cache
        self.normalization_cache: Dict[str, str] = {}
    
    async def detect_duplicates(
        self,
        content: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        search_window_days: int = 30
    ) -> List[DuplicateMatch]:
        """Detect duplicate content using multiple algorithms."""
        matches = []
        
        # Generate content signatures
        content_hash = self._generate_content_hash(content)
        fuzzy_hash = self._generate_fuzzy_hash(content)
        
        # Search for potential duplicates in database
        candidates = await self._get_duplicate_candidates(
            content, title, content_hash, search_window_days
        )
        
        if not candidates:
            return matches
        
        # Evaluate each candidate
        for candidate_id, candidate_data in candidates:
            match = await self._evaluate_duplicate_candidate(
                content, title, metadata,
                candidate_id, candidate_data,
                content_hash, fuzzy_hash
            )
            
            if match and match.similarity_score >= self.thresholds['low_similarity']:
                matches.append(match)
        
        # Sort by similarity score (highest first)
        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        
        return matches
    
    async def _get_duplicate_candidates(
        self,
        content: str,
        title: Optional[str],
        content_hash: str,
        search_window_days: int
    ) -> List[Tuple[int, Dict[str, Any]]]:
        """Get potential duplicate candidates from database."""
        conn = self.get_conn()
        cursor = conn.cursor()
        candidates = []
        
        try:
            # Calculate search window
            cutoff_date = (datetime.now() - timedelta(days=search_window_days)).isoformat()
            
            # Search by exact content hash first
            cursor.execute("""
                SELECT id, title, body, metadata, created_at
                FROM notes
                WHERE json_extract(metadata, '$.content_hash') = ?
                AND created_at >= ?
                ORDER BY created_at DESC
                LIMIT 10
            """, (content_hash, cutoff_date))
            
            for row in cursor.fetchall():
                candidates.append((row[0], {
                    'title': row[1],
                    'body': row[2],
                    'metadata': json.loads(row[3]) if row[3] else {},
                    'created_at': row[4]
                }))
            
            # If no exact matches and we have a title, search by title similarity
            if not candidates and title and len(title) > 5:
                cursor.execute("""
                    SELECT id, title, body, metadata, created_at
                    FROM notes
                    WHERE title IS NOT NULL
                    AND created_at >= ?
                    ORDER BY created_at DESC
                    LIMIT 50
                """, (cutoff_date,))
                
                for row in cursor.fetchall():
                    stored_title = row[1] or ""
                    if self._quick_title_similarity(title, stored_title) > 0.6:
                        candidates.append((row[0], {
                            'title': row[1],
                            'body': row[2],
                            'metadata': json.loads(row[3]) if row[3] else {},
                            'created_at': row[4]
                        }))
            
            # If still no candidates, do a broader content-based search
            if not candidates and len(content) > 50:
                # Extract key phrases for search
                key_phrases = self._extract_key_phrases(content)
                if key_phrases:
                    # Use FTS search if available
                    search_query = " OR ".join(key_phrases[:3])  # Top 3 phrases
                    
                    try:
                        cursor.execute("""
                            SELECT id, title, body, metadata, created_at, rank
                            FROM notes
                            JOIN notes_fts ON notes.id = notes_fts.rowid
                            WHERE notes_fts MATCH ?
                            AND created_at >= ?
                            ORDER BY rank
                            LIMIT 20
                        """, (search_query, cutoff_date))
                        
                        for row in cursor.fetchall():
                            candidates.append((row[0], {
                                'title': row[1],
                                'body': row[2],
                                'metadata': json.loads(row[3]) if row[3] else {},
                                'created_at': row[4]
                            }))
                    except Exception as e:
                        logger.debug(f"FTS search failed, using fallback: {e}")
                        # Fallback to basic text search
                        cursor.execute("""
                            SELECT id, title, body, metadata, created_at
                            FROM notes
                            WHERE (body LIKE ? OR title LIKE ?)
                            AND created_at >= ?
                            ORDER BY created_at DESC
                            LIMIT 20
                        """, (f"%{key_phrases[0]}%", f"%{key_phrases[0]}%", cutoff_date))
                        
                        for row in cursor.fetchall():
                            candidates.append((row[0], {
                                'title': row[1],
                                'body': row[2],
                                'metadata': json.loads(row[3]) if row[3] else {},
                                'created_at': row[4]
                            }))
        
        finally:
            conn.close()
        
        return candidates
    
    async def _evaluate_duplicate_candidate(
        self,
        content: str,
        title: Optional[str],
        metadata: Optional[Dict[str, Any]],
        candidate_id: int,
        candidate_data: Dict[str, Any],
        content_hash: str,
        fuzzy_hash: str
    ) -> Optional[DuplicateMatch]:
        """Evaluate a duplicate candidate."""
        candidate_content = candidate_data.get('body', '')
        candidate_title = candidate_data.get('title', '')
        candidate_metadata = candidate_data.get('metadata', {})
        
        # Calculate different types of similarity
        content_similarity = self._calculate_content_similarity(content, candidate_content)
        title_similarity = self._calculate_title_similarity(title, candidate_title)
        metadata_similarity = self._calculate_metadata_similarity(metadata, candidate_metadata)
        
        # Determine overall similarity and match type
        overall_similarity = (
            content_similarity * 0.7 +
            title_similarity * 0.2 +
            metadata_similarity * 0.1
        )
        
        # Determine match type
        match_type = "fuzzy"
        if content_similarity >= self.thresholds['exact_match']:
            match_type = "exact"
        elif content_similarity >= self.thresholds['near_exact']:
            match_type = "near_exact"
        elif content_similarity >= self.thresholds['high_similarity']:
            match_type = "similar"
        
        # Calculate merge confidence
        merge_confidence = self._calculate_merge_confidence(
            content, candidate_content, title, candidate_title
        )
        
        # Determine recommended strategy
        recommended_strategy = self._determine_duplicate_strategy(
            overall_similarity, match_type, merge_confidence
        )
        
        # Find differences and common elements
        differences, common_elements, conflicts = self._analyze_content_differences(
            content, candidate_content, title, candidate_title
        )
        
        return DuplicateMatch(
            original_id=candidate_id,
            similarity_score=overall_similarity,
            match_type=match_type,
            content_similarity=content_similarity,
            title_similarity=title_similarity,
            metadata_similarity=metadata_similarity,
            recommended_strategy=recommended_strategy,
            merge_confidence=merge_confidence,
            differences=differences,
            common_elements=common_elements,
            conflict_areas=conflicts
        )
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate content hash for exact duplicate detection."""
        # Normalize content for hashing
        normalized = self._normalize_content_for_hash(content)
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def _generate_fuzzy_hash(self, content: str) -> str:
        """Generate fuzzy hash for near-duplicate detection."""
        # Simple fuzzy hash based on word frequency
        words = re.findall(r'\b\w+\b', content.lower())
        word_freq = {}
        
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Take top 20 most frequent words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        fuzzy_signature = ''.join(word for word, _ in top_words)
        
        return hashlib.md5(fuzzy_signature.encode()).hexdigest()
    
    def _normalize_content_for_hash(self, content: str) -> str:
        """Normalize content for consistent hashing."""
        cache_key = hashlib.md5(content.encode()).hexdigest()[:16]
        
        if cache_key in self.normalization_cache:
            return self.normalization_cache[cache_key]
        
        # Normalize whitespace, punctuation, and case
        normalized = re.sub(r'\s+', ' ', content.strip().lower())
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        self.normalization_cache[cache_key] = normalized
        return normalized
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity using multiple methods."""
        if not content1 or not content2:
            return 0.0
        
        # Method 1: Exact match after normalization
        norm1 = self._normalize_content_for_hash(content1)
        norm2 = self._normalize_content_for_hash(content2)
        
        if norm1 == norm2:
            return 1.0
        
        # Method 2: Jaccard similarity (word-based)
        words1 = set(re.findall(r'\b\w+\b', content1.lower()))
        words2 = set(re.findall(r'\b\w+\b', content2.lower()))
        
        if words1 or words2:
            jaccard = len(words1.intersection(words2)) / len(words1.union(words2))
        else:
            jaccard = 0.0
        
        # Method 3: TF-IDF cosine similarity (if available)
        tfidf_similarity = 0.0
        if SKLEARN_AVAILABLE and self.tfidf_vectorizer:
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform([content1, content2])
                cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
                tfidf_similarity = cosine_sim[0][0]
            except Exception as e:
                logger.debug(f"TF-IDF similarity calculation failed: {e}")
        
        # Method 4: Levenshtein distance (if available)
        levenshtein_similarity = 0.0
        if LEVENSHTEIN_AVAILABLE and len(content1) < 1000 and len(content2) < 1000:
            try:
                distance = Levenshtein.distance(content1, content2)
                max_len = max(len(content1), len(content2))
                levenshtein_similarity = 1 - (distance / max_len) if max_len > 0 else 0
            except Exception as e:
                logger.debug(f"Levenshtein similarity calculation failed: {e}")
        
        # Combine similarities (weighted average)
        similarities = []
        if jaccard > 0:
            similarities.append((jaccard, 0.4))
        if tfidf_similarity > 0:
            similarities.append((tfidf_similarity, 0.4))
        if levenshtein_similarity > 0:
            similarities.append((levenshtein_similarity, 0.2))
        
        if similarities:
            weighted_sum = sum(sim * weight for sim, weight in similarities)
            total_weight = sum(weight for _, weight in similarities)
            return weighted_sum / total_weight
        
        return jaccard  # Fallback to Jaccard similarity
    
    def _calculate_title_similarity(self, title1: Optional[str], title2: Optional[str]) -> float:
        """Calculate title similarity."""
        if not title1 or not title2:
            return 0.5 if not title1 and not title2 else 0.0
        
        return self._quick_title_similarity(title1, title2)
    
    def _quick_title_similarity(self, title1: str, title2: str) -> float:
        """Quick title similarity calculation."""
        # Normalize titles
        norm1 = re.sub(r'[^\w\s]', '', title1.lower().strip())
        norm2 = re.sub(r'[^\w\s]', '', title2.lower().strip())
        
        if norm1 == norm2:
            return 1.0
        
        # Word-based Jaccard similarity
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if words1 or words2:
            return len(words1.intersection(words2)) / len(words1.union(words2))
        
        return 0.0
    
    def _calculate_metadata_similarity(
        self,
        metadata1: Optional[Dict[str, Any]],
        metadata2: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate metadata similarity."""
        if not metadata1 or not metadata2:
            return 0.5 if not metadata1 and not metadata2 else 0.0
        
        # Compare specific metadata fields
        similarity_scores = []
        
        # Source type
        source1 = metadata1.get('source', metadata1.get('content_type'))
        source2 = metadata2.get('source', metadata2.get('content_type'))
        if source1 and source2:
            similarity_scores.append(1.0 if source1 == source2 else 0.0)
        
        # Tags comparison
        tags1 = set(metadata1.get('tags', '').split(', ')) if metadata1.get('tags') else set()
        tags2 = set(metadata2.get('tags', '').split(', ')) if metadata2.get('tags') else set()
        
        if tags1 or tags2:
            tag_similarity = len(tags1.intersection(tags2)) / len(tags1.union(tags2))
            similarity_scores.append(tag_similarity)
        
        return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    
    def _calculate_merge_confidence(
        self,
        content1: str,
        content2: str,
        title1: Optional[str],
        title2: Optional[str]
    ) -> float:
        """Calculate confidence in merging two pieces of content."""
        confidence_factors = []
        
        # Length ratio factor
        len1, len2 = len(content1), len(content2)
        if len1 > 0 and len2 > 0:
            length_ratio = min(len1, len2) / max(len1, len2)
            confidence_factors.append(length_ratio * 0.3)
        
        # Title consistency factor
        if title1 and title2:
            title_sim = self._quick_title_similarity(title1, title2)
            confidence_factors.append(title_sim * 0.4)
        elif not title1 and not title2:
            confidence_factors.append(0.3)  # Both missing titles
        else:
            confidence_factors.append(0.1)  # One missing title
        
        # Content structure similarity
        lines1 = content1.count('\n')
        lines2 = content2.count('\n')
        if lines1 > 0 or lines2 > 0:
            structure_sim = 1 - abs(lines1 - lines2) / max(lines1, lines2, 1)
            confidence_factors.append(structure_sim * 0.3)
        
        return sum(confidence_factors) if confidence_factors else 0.0
    
    def _determine_duplicate_strategy(
        self,
        similarity_score: float,
        match_type: str,
        merge_confidence: float
    ) -> DuplicateStrategy:
        """Determine recommended strategy for handling duplicates."""
        if similarity_score >= self.thresholds['exact_match']:
            return DuplicateStrategy.SKIP
        elif similarity_score >= self.thresholds['near_exact']:
            if merge_confidence > 0.7:
                return DuplicateStrategy.MERGE
            else:
                return DuplicateStrategy.FLAG
        elif similarity_score >= self.thresholds['high_similarity']:
            if merge_confidence > 0.8:
                return DuplicateStrategy.MERGE
            else:
                return DuplicateStrategy.FLAG
        else:
            return DuplicateStrategy.FLAG
    
    def _analyze_content_differences(
        self,
        content1: str,
        content2: str,
        title1: Optional[str],
        title2: Optional[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Analyze differences between two pieces of content."""
        differences = []
        common_elements = []
        conflicts = []
        
        # Title differences
        if title1 and title2 and title1 != title2:
            differences.append(f"Title: '{title1}' vs '{title2}'")
            conflicts.append("title")
        elif title1 and not title2:
            differences.append(f"New content has title: '{title1}'")
        elif not title1 and title2:
            differences.append(f"Original has title: '{title2}'")
        else:
            common_elements.append("Both have matching titles")
        
        # Content length differences
        len_diff = abs(len(content1) - len(content2))
        if len_diff > 100:
            differences.append(f"Content length differs by {len_diff} characters")
        
        # Line-by-line comparison for detailed differences
        lines1 = content1.split('\n')
        lines2 = content2.split('\n')
        
        differ = difflib.unified_diff(lines1, lines2, n=0)
        diff_lines = list(differ)
        
        if diff_lines:
            added_lines = len([line for line in diff_lines if line.startswith('+')])
            removed_lines = len([line for line in diff_lines if line.startswith('-')])
            
            if added_lines > 0:
                differences.append(f"{added_lines} lines added")
            if removed_lines > 0:
                differences.append(f"{removed_lines} lines removed")
        
        # Word-level common elements
        words1 = set(re.findall(r'\b\w+\b', content1.lower()))
        words2 = set(re.findall(r'\b\w+\b', content2.lower()))
        common_words = words1.intersection(words2)
        
        if len(common_words) > 5:
            common_elements.append(f"{len(common_words)} common words")
        
        return differences, common_elements, conflicts
    
    def _extract_key_phrases(self, content: str) -> List[str]:
        """Extract key phrases from content for search."""
        # Simple approach: extract noun phrases and important words
        words = re.findall(r'\b[A-Za-z]{4,}\b', content)
        
        # Basic frequency analysis
        word_freq = {}
        for word in words:
            word_lower = word.lower()
            if word_lower not in {'this', 'that', 'with', 'from', 'they', 'have', 'been', 'were'}:
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        # Return top frequent words as key phrases
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10] if freq > 1]


class QualityAssuranceService:
    """Main quality assurance service coordinating all QA components."""
    
    def __init__(self, get_conn_func):
        """Initialize quality assurance service."""
        self.get_conn = get_conn_func
        self.content_validator = ContentValidator()
        self.duplicate_detector = DuplicateDetector(get_conn_func)
        
        # QA configuration
        self.config = {
            'validation_enabled': True,
            'duplicate_detection_enabled': True,
            'auto_merge_threshold': 0.95,
            'quality_threshold': 0.4,
            'search_window_days': 30
        }
        
        # Quality metrics tracking
        self.metrics = {
            'total_validations': 0,
            'passed_validations': 0,
            'duplicates_found': 0,
            'duplicates_merged': 0,
            'average_quality_score': 0.0
        }
        
        logger.info("Quality Assurance Service initialized")
    
    async def validate_content(
        self,
        content: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        validation_config: Optional[Dict[str, Any]] = None
    ) -> ContentValidationResult:
        """
        Perform comprehensive content validation.
        
        Args:
            content: Content to validate
            title: Optional title
            metadata: Optional metadata
            validation_config: Optional validation configuration overrides
            
        Returns:
            ContentValidationResult: Comprehensive validation results
        """
        start_time = datetime.now()
        config = {**self.config, **(validation_config or {})}
        
        logger.info(f"Validating content: {len(content)} characters")
        
        # Initialize result
        result = ContentValidationResult(
            is_valid=True,
            quality_metrics=QualityMetrics(
                overall_score=0.0,
                quality_level=QualityLevel.POOR,
                content_completeness=0.0,
                structural_quality=0.0,
                language_quality=0.0,
                relevance_score=0.0,
                uniqueness_score=1.0,  # Assume unique until proven otherwise
                word_count=len(content.split()),
                sentence_count=len(re.split(r'[.!?]+', content)),
                paragraph_count=len(content.split('\n\n')),
                reading_level="Unknown",
                sentiment_score=0.0,
                processing_confidence=1.0,
                extraction_success=True,
                ai_enhancement_success=True
            )
        )
        
        rules_evaluated = 0
        
        try:
            # 1. Content validation
            if config.get('validation_enabled', True):
                validation_issues = self.content_validator.validate_content(
                    content, title, metadata, config
                )
                result.quality_metrics.issues = validation_issues
                rules_evaluated += len(self.content_validator.validation_rules)
                
                # Determine if content is valid
                critical_issues = [i for i in validation_issues if i.severity == ValidationSeverity.CRITICAL]
                error_issues = [i for i in validation_issues if i.severity == ValidationSeverity.ERROR]
                
                result.is_valid = len(critical_issues) == 0
                result.should_process = len(critical_issues) == 0 and len(error_issues) == 0
            
            # 2. Duplicate detection
            if config.get('duplicate_detection_enabled', True):
                duplicate_matches = await self.duplicate_detector.detect_duplicates(
                    content, title, metadata, config.get('search_window_days', 30)
                )
                result.duplicate_matches = duplicate_matches
                
                # Update uniqueness score based on duplicates
                if duplicate_matches:
                    best_match = duplicate_matches[0]
                    result.quality_metrics.uniqueness_score = 1.0 - best_match.similarity_score
                    
                    # Set merge recommendations
                    if best_match.similarity_score >= config.get('auto_merge_threshold', 0.95):
                        result.recommended_strategy = best_match.recommended_strategy
                        result.merge_target_id = best_match.original_id
                        result.should_process = False  # Skip processing if high duplicate
            
            # 3. Calculate quality metrics
            result.quality_metrics = await self._calculate_comprehensive_quality_metrics(
                content, title, metadata, result.quality_metrics
            )
            
            # 4. Generate enhancement suggestions
            result.content_enhancements, result.metadata_enhancements = self._generate_enhancement_suggestions(
                content, title, metadata, result.quality_metrics
            )
            
            # Update metrics
            self.metrics['total_validations'] += 1
            if result.is_valid:
                self.metrics['passed_validations'] += 1
            if result.duplicate_matches:
                self.metrics['duplicates_found'] += 1
            
            # Update average quality score
            current_avg = self.metrics['average_quality_score']
            total_validations = self.metrics['total_validations']
            self.metrics['average_quality_score'] = (
                (current_avg * (total_validations - 1) + result.quality_metrics.overall_score) / 
                total_validations
            )
            
        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
            result.is_valid = False
            result.should_process = False
            result.quality_metrics.issues.append(ValidationIssue(
                rule=ContentValidationRule.DUPLICATE_DETECTION,
                severity=ValidationSeverity.ERROR,
                message=f"Validation process failed: {str(e)}",
                suggestion="Review content and try again",
                confidence=0.9
            ))
        
        # Finalize result
        result.validation_time = (datetime.now() - start_time).total_seconds()
        result.rules_evaluated = rules_evaluated
        
        logger.info(f"Content validation completed in {result.validation_time:.2f}s: "
                   f"Valid={result.is_valid}, Quality={result.quality_metrics.overall_score:.2f}, "
                   f"Duplicates={len(result.duplicate_matches)}")
        
        return result
    
    async def _calculate_comprehensive_quality_metrics(
        self,
        content: str,
        title: Optional[str],
        metadata: Optional[Dict[str, Any]],
        base_metrics: QualityMetrics
    ) -> QualityMetrics:
        """Calculate comprehensive quality metrics."""
        # Content completeness
        completeness_factors = []
        
        # Has title
        if title and title.strip():
            completeness_factors.append(0.2)
        
        # Has reasonable length
        word_count = base_metrics.word_count
        if word_count >= 10:
            completeness_factors.append(min(word_count / 100, 0.3))
        
        # Has sentence structure
        if base_metrics.sentence_count > 0:
            completeness_factors.append(0.2)
        
        # Has paragraph structure
        if base_metrics.paragraph_count > 1:
            completeness_factors.append(0.1)
        
        # No truncation indicators
        if not any('truncated' in issue.message.lower() for issue in base_metrics.issues):
            completeness_factors.append(0.2)
        
        base_metrics.content_completeness = sum(completeness_factors)
        
        # Structural quality
        structure_score = 0.5  # Base score
        
        # Bonus for good sentence/paragraph ratio
        if word_count > 0:
            avg_words_per_sentence = word_count / max(base_metrics.sentence_count, 1)
            if 8 <= avg_words_per_sentence <= 25:  # Good range
                structure_score += 0.2
        
        # Penalty for structure issues
        structure_issues = [i for i in base_metrics.issues 
                          if i.rule == ContentValidationRule.STRUCTURE_VALIDATION]
        structure_score -= len(structure_issues) * 0.1
        
        base_metrics.structural_quality = max(0.0, min(1.0, structure_score))
        
        # Language quality
        language_score = 0.7  # Base score
        
        # Penalty for language/encoding issues
        language_issues = [i for i in base_metrics.issues 
                         if i.rule in [ContentValidationRule.LANGUAGE_DETECTION, 
                                     ContentValidationRule.ENCODING_VALIDATION]]
        language_score -= len(language_issues) * 0.2
        
        base_metrics.language_quality = max(0.0, min(1.0, language_score))
        
        # Relevance score (simple heuristic)
        relevance_score = 0.8  # Default assumption
        
        # Check for spam indicators
        spam_issues = [i for i in base_metrics.issues 
                      if i.rule == ContentValidationRule.SPAM_DETECTION]
        if spam_issues:
            relevance_score = 0.2
        
        base_metrics.relevance_score = relevance_score
        
        # Calculate overall score
        component_scores = [
            base_metrics.content_completeness * 0.25,
            base_metrics.structural_quality * 0.20,
            base_metrics.language_quality * 0.20,
            base_metrics.relevance_score * 0.15,
            base_metrics.uniqueness_score * 0.20
        ]
        
        base_metrics.overall_score = sum(component_scores)
        
        # Determine quality level
        if base_metrics.overall_score >= 0.8:
            base_metrics.quality_level = QualityLevel.EXCELLENT
        elif base_metrics.overall_score >= 0.6:
            base_metrics.quality_level = QualityLevel.GOOD
        elif base_metrics.overall_score >= 0.4:
            base_metrics.quality_level = QualityLevel.FAIR
        else:
            base_metrics.quality_level = QualityLevel.POOR
        
        # Generate improvement suggestions based on metrics
        suggestions = []
        
        if base_metrics.content_completeness < 0.5:
            suggestions.append("Consider adding more detail and context to improve completeness")
        
        if base_metrics.structural_quality < 0.5:
            suggestions.append("Improve content structure with better paragraphs and sentences")
        
        if base_metrics.language_quality < 0.5:
            suggestions.append("Review content for language and encoding issues")
        
        if base_metrics.uniqueness_score < 0.5:
            suggestions.append("Content appears to be similar to existing content")
        
        base_metrics.improvement_suggestions = suggestions
        
        return base_metrics
    
    def _generate_enhancement_suggestions(
        self,
        content: str,
        title: Optional[str],
        metadata: Optional[Dict[str, Any]],
        quality_metrics: QualityMetrics
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate content and metadata enhancement suggestions."""
        content_enhancements = {}
        metadata_enhancements = {}
        
        # Title enhancement
        if not title or len(title.strip()) < 5:
            # Suggest generating title from first sentence or keywords
            sentences = content.split('.')[:2]
            if sentences and len(sentences[0].strip()) > 10:
                content_enhancements['suggested_title'] = sentences[0].strip()[:60]
        
        # Content structure enhancement
        if quality_metrics.paragraph_count == 1 and len(content) > 200:
            content_enhancements['structure_suggestion'] = "Consider breaking content into paragraphs"
        
        # Metadata enhancements
        if not metadata or not metadata.get('tags'):
            # Suggest tags based on content
            key_phrases = self.duplicate_detector._extract_key_phrases(content)
            if key_phrases:
                metadata_enhancements['suggested_tags'] = key_phrases[:5]
        
        # Quality-based enhancements
        if quality_metrics.overall_score < 0.6:
            metadata_enhancements['quality_flag'] = 'needs_review'
        
        # Processing recommendations
        if quality_metrics.uniqueness_score < 0.8:
            metadata_enhancements['duplicate_flag'] = True
        
        return content_enhancements, metadata_enhancements
    
    async def merge_duplicate_content(
        self,
        original_id: int,
        new_content: str,
        new_title: Optional[str] = None,
        new_metadata: Optional[Dict[str, Any]] = None,
        merge_strategy: str = "smart"
    ) -> Dict[str, Any]:
        """
        Merge duplicate content with intelligent conflict resolution.
        
        Args:
            original_id: ID of original content
            new_content: New content to merge
            new_title: New title to merge
            new_metadata: New metadata to merge
            merge_strategy: Merge strategy ("smart", "prefer_new", "prefer_original")
            
        Returns:
            Dict containing merge results and merged content
        """
        conn = self.get_conn()
        cursor = conn.cursor()
        
        try:
            # Get original content
            cursor.execute("""
                SELECT title, body, metadata, created_at, updated_at
                FROM notes
                WHERE id = ?
            """, (original_id,))
            
            row = cursor.fetchone()
            if not row:
                return {"success": False, "error": "Original content not found"}
            
            original_title, original_content, original_metadata_str, created_at, updated_at = row
            original_metadata = json.loads(original_metadata_str) if original_metadata_str else {}
            
            # Perform intelligent merge
            merged_result = self._perform_intelligent_merge(
                original_content, new_content,
                original_title, new_title,
                original_metadata, new_metadata or {},
                merge_strategy
            )
            
            # Update database with merged content
            cursor.execute("""
                UPDATE notes
                SET title = ?, body = ?, metadata = ?, updated_at = ?
                WHERE id = ?
            """, (
                merged_result['title'],
                merged_result['content'],
                json.dumps(merged_result['metadata']),
                datetime.now().isoformat(),
                original_id
            ))
            
            conn.commit()
            
            # Update metrics
            self.metrics['duplicates_merged'] += 1
            
            return {
                "success": True,
                "merged_id": original_id,
                "merge_summary": merged_result['summary'],
                "conflicts_resolved": merged_result['conflicts_resolved'],
                "content_added": merged_result['content_added']
            }
        
        except Exception as e:
            logger.error(f"Content merge failed: {e}")
            conn.rollback()
            return {"success": False, "error": str(e)}
        
        finally:
            conn.close()
    
    def _perform_intelligent_merge(
        self,
        original_content: str,
        new_content: str,
        original_title: Optional[str],
        new_title: Optional[str],
        original_metadata: Dict[str, Any],
        new_metadata: Dict[str, Any],
        strategy: str
    ) -> Dict[str, Any]:
        """Perform intelligent merge of content."""
        merged_result = {
            'title': original_title,
            'content': original_content,
            'metadata': original_metadata.copy(),
            'summary': [],
            'conflicts_resolved': 0,
            'content_added': False
        }
        
        # Title merge
        if new_title and new_title != original_title:
            if strategy == "prefer_new":
                merged_result['title'] = new_title
                merged_result['summary'].append(f"Updated title to: '{new_title}'")
            elif strategy == "smart":
                # Choose longer, more descriptive title
                if not original_title or len(new_title) > len(original_title):
                    merged_result['title'] = new_title
                    merged_result['summary'].append(f"Updated title to: '{new_title}'")
            
            merged_result['conflicts_resolved'] += 1
        
        # Content merge
        content_similarity = self.duplicate_detector._calculate_content_similarity(
            original_content, new_content
        )
        
        if content_similarity < 0.9:  # Not identical
            if strategy == "prefer_new":
                merged_result['content'] = new_content
                merged_result['summary'].append("Replaced content with new version")
            elif strategy == "smart":
                # Append new content if significantly different
                if len(new_content) > len(original_content) * 1.2:
                    merged_result['content'] = f"{original_content}\n\n--- Updated Content ---\n{new_content}"
                    merged_result['summary'].append("Appended additional content")
                    merged_result['content_added'] = True
                elif len(new_content) > len(original_content):
                    merged_result['content'] = new_content
                    merged_result['summary'].append("Replaced with longer version")
        
        # Metadata merge
        for key, value in new_metadata.items():
            if key not in original_metadata:
                merged_result['metadata'][key] = value
                merged_result['summary'].append(f"Added metadata: {key}")
            elif original_metadata[key] != value:
                if strategy == "prefer_new":
                    merged_result['metadata'][key] = value
                elif strategy == "smart":
                    # Merge lists, prefer newer for scalars
                    if isinstance(original_metadata[key], list) and isinstance(value, list):
                        merged_list = list(set(original_metadata[key] + value))
                        merged_result['metadata'][key] = merged_list
                    else:
                        merged_result['metadata'][key] = value
                
                merged_result['conflicts_resolved'] += 1
        
        # Add merge metadata
        merged_result['metadata']['merge_history'] = merged_result['metadata'].get('merge_history', [])
        merged_result['metadata']['merge_history'].append({
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'conflicts_resolved': merged_result['conflicts_resolved']
        })
        
        return merged_result
    
    def get_quality_statistics(self) -> Dict[str, Any]:
        """Get quality assurance statistics."""
        success_rate = (
            self.metrics['passed_validations'] / max(1, self.metrics['total_validations'])
        ) * 100
        
        return {
            **self.metrics,
            'success_rate': success_rate,
            'duplicate_rate': (
                self.metrics['duplicates_found'] / max(1, self.metrics['total_validations'])
            ) * 100,
            'merge_rate': (
                self.metrics['duplicates_merged'] / max(1, self.metrics['duplicates_found'])
            ) * 100 if self.metrics['duplicates_found'] > 0 else 0,
            'features': {
                'content_validation': True,
                'duplicate_detection': True,
                'intelligent_merging': True,
                'quality_metrics': True,
                'enhancement_suggestions': True,
                'advanced_similarity': SKLEARN_AVAILABLE,
                'fuzzy_matching': LEVENSHTEIN_AVAILABLE
            }
        }
    
    def update_configuration(self, config_updates: Dict[str, Any]) -> bool:
        """Update QA service configuration."""
        try:
            self.config.update(config_updates)
            
            # Update validator thresholds if provided
            if 'validator_thresholds' in config_updates:
                self.content_validator.thresholds.update(
                    config_updates['validator_thresholds']
                )
            
            # Update duplicate detector thresholds if provided
            if 'duplicate_thresholds' in config_updates:
                self.duplicate_detector.thresholds.update(
                    config_updates['duplicate_thresholds']
                )
            
            logger.info(f"Updated QA configuration: {list(config_updates.keys())}")
            return True
        except Exception as e:
            logger.error(f"Failed to update QA configuration: {e}")
            return False
    
    def clear_caches(self):
        """Clear all QA service caches."""
        self.duplicate_detector.normalization_cache.clear()
        logger.info("QA service caches cleared")


# Global service instance
_quality_assurance_service: Optional[QualityAssuranceService] = None


def get_quality_assurance_service(get_conn_func) -> QualityAssuranceService:
    """Get the global quality assurance service instance."""
    global _quality_assurance_service
    if _quality_assurance_service is None:
        _quality_assurance_service = QualityAssuranceService(get_conn_func)
    return _quality_assurance_service