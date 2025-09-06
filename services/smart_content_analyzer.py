# ──────────────────────────────────────────────────────────────────────────────
# File: services/smart_content_analyzer.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Smart Content Analyzer Service

Advanced content analysis using NLP and ML for intelligent auto-tagging,
context-aware title generation, and automatic action item extraction.

This service provides Phase 2 Smart Automation capabilities including:
- NLP-based auto-tagging with confidence scoring
- Context-aware title generation beyond just content analysis
- Intelligent action item extraction with priority classification
- Semantic content classification and topic modeling
- Content quality assessment and improvement suggestions
"""

import logging
import re
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json

# NLP and ML imports
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from services.embeddings import Embeddings
from llm_utils import ollama_generate_title, ollama_summarize
from config import settings

logger = logging.getLogger(__name__)


class ContentCategory(Enum):
    """Content category classifications."""
    MEETING_NOTES = "meeting_notes"
    TASK_LIST = "task_list"
    DOCUMENTATION = "documentation"
    RESEARCH = "research"
    PERSONAL_NOTE = "personal_note"
    CODE_SNIPPET = "code_snippet"
    WEB_ARTICLE = "web_article"
    EMAIL = "email"
    CONVERSATION = "conversation"
    BRAINSTORM = "brainstorm"
    UNKNOWN = "unknown"


class ActionPriority(Enum):
    """Action item priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


class ConfidenceLevel(Enum):
    """Confidence levels for analysis results."""
    LOW = "low"       # 0.0 - 0.4
    MEDIUM = "medium" # 0.4 - 0.7
    HIGH = "high"     # 0.7 - 1.0


@dataclass
class SmartTag:
    """A smart tag with confidence and metadata."""
    tag: str
    confidence: float
    source: str  # "nlp", "pattern", "ml", "context", "manual"
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        if self.confidence < 0.4:
            return ConfidenceLevel.LOW
        elif self.confidence < 0.7:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.HIGH


@dataclass
class ActionItem:
    """An extracted action item with metadata."""
    text: str
    priority: ActionPriority
    confidence: float
    due_date: Optional[datetime] = None
    assignee: Optional[str] = None
    context: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        if self.confidence < 0.4:
            return ConfidenceLevel.LOW
        elif self.confidence < 0.7:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.HIGH


@dataclass
class SmartTitle:
    """A smart-generated title with context awareness."""
    title: str
    confidence: float
    generation_method: str  # "pattern", "ai", "semantic", "hybrid"
    context_factors: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)


@dataclass
class ContentAnalysis:
    """Complete content analysis results."""
    category: ContentCategory
    confidence: float
    
    # Core analysis results
    smart_tags: List[SmartTag]
    action_items: List[ActionItem]
    smart_titles: List[SmartTitle]
    
    # Content characteristics
    word_count: int
    reading_time_minutes: float
    complexity_score: float  # 0-1, higher = more complex
    sentiment_score: float   # -1 to 1, negative to positive
    
    # Semantic analysis
    key_topics: List[Tuple[str, float]]  # (topic, relevance)
    named_entities: List[Tuple[str, str, float]]  # (entity, type, confidence)
    keywords: List[Tuple[str, float]]  # (keyword, importance)
    
    # Quality metrics
    quality_score: float  # 0-1, higher = better quality
    improvement_suggestions: List[str]
    
    # Metadata
    processing_time: float
    nlp_features_used: List[str]
    analysis_timestamp: datetime = field(default_factory=datetime.now)


class NLPProcessor:
    """Handles NLP-specific processing tasks."""
    
    def __init__(self):
        """Initialize NLP processor with available libraries."""
        self.stopwords_en = set()
        self.nltk_ready = False
        
        if NLTK_AVAILABLE:
            try:
                # Download required NLTK data if not present
                self._ensure_nltk_data()
                self.stopwords_en = set(stopwords.words('english'))
                self.nltk_ready = True
                logger.info("NLTK processor ready")
            except Exception as e:
                logger.warning(f"NLTK setup failed: {e}")
        
        # TF-IDF vectorizer for keyword extraction
        self.tfidf_vectorizer = None
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
    
    def _ensure_nltk_data(self):
        """Ensure required NLTK data is available."""
        required_data = ['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words', 'stopwords']
        
        for data_name in required_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}')
            except LookupError:
                try:
                    nltk.download(data_name, quiet=True)
                except Exception as e:
                    logger.warning(f"Failed to download NLTK data {data_name}: {e}")
    
    def extract_keywords_tfidf(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF."""
        if not self.tfidf_vectorizer or not SKLEARN_AVAILABLE:
            return self._fallback_keyword_extraction(text, top_k)
        
        try:
            # Fit and transform text
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords with scores
            keyword_scores = list(zip(feature_names, tfidf_scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return keyword_scores[:top_k]
        except Exception as e:
            logger.warning(f"TF-IDF keyword extraction failed: {e}")
            return self._fallback_keyword_extraction(text, top_k)
    
    def _fallback_keyword_extraction(self, text: str, top_k: int) -> List[Tuple[str, float]]:
        """Fallback keyword extraction using simple frequency analysis."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = {}
        
        for word in words:
            if word not in self.stopwords_en:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Simple scoring based on frequency and length
        scored_words = []
        max_freq = max(word_freq.values()) if word_freq else 1
        
        for word, freq in word_freq.items():
            score = (freq / max_freq) * (len(word) / 10)  # Longer words get slight boost
            scored_words.append((word, min(score, 1.0)))
        
        scored_words.sort(key=lambda x: x[1], reverse=True)
        return scored_words[:top_k]
    
    def extract_named_entities(self, text: str) -> List[Tuple[str, str, float]]:
        """Extract named entities using NLTK."""
        if not self.nltk_ready:
            return []
        
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            chunks = ne_chunk(pos_tags, binary=False)
            
            entities = []
            current_entity = []
            current_label = None
            
            for chunk in chunks:
                if hasattr(chunk, 'label'):  # It's a named entity
                    if current_label and current_label != chunk.label():
                        # Save previous entity
                        entity_text = ' '.join([token for token, pos in current_entity])
                        entities.append((entity_text, current_label, 0.8))  # Default confidence
                        current_entity = []
                    
                    current_label = chunk.label()
                    current_entity.extend(chunk)
                else:
                    if current_entity:
                        # Save current entity
                        entity_text = ' '.join([token for token, pos in current_entity])
                        entities.append((entity_text, current_label, 0.8))
                        current_entity = []
                        current_label = None
            
            # Save final entity if exists
            if current_entity:
                entity_text = ' '.join([token for token, pos in current_entity])
                entities.append((entity_text, current_label, 0.8))
            
            return entities
        except Exception as e:
            logger.warning(f"Named entity extraction failed: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> float:
        """Basic sentiment analysis (-1 to 1)."""
        # Simple lexicon-based approach
        positive_words = set(['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                             'awesome', 'brilliant', 'outstanding', 'perfect', 'love', 'like'])
        negative_words = set(['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 
                             'poor', 'wrong', 'failed', 'problem', 'issue', 'difficult'])
        
        words = re.findall(r'\b\w+\b', text.lower())
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_words = len(words)
        if total_words == 0:
            return 0.0
        
        # Normalized sentiment score
        sentiment = (positive_count - negative_count) / total_words
        return max(-1.0, min(1.0, sentiment * 10))  # Scale and clamp
    
    def calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score (0-1)."""
        if not text.strip():
            return 0.0
        
        sentences = sent_tokenize(text) if self.nltk_ready else text.split('.')
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        # Basic complexity metrics
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Count complex words (3+ syllables, rough estimate)
        complex_words = sum(1 for word in words if len(word) > 6)
        complex_word_ratio = complex_words / len(words)
        
        # Normalized complexity score
        sentence_complexity = min(avg_sentence_length / 30, 1.0)  # 30 words = max complexity
        word_complexity = min(avg_word_length / 8, 1.0)  # 8 chars = max complexity
        
        complexity = (sentence_complexity + word_complexity + complex_word_ratio) / 3
        return min(complexity, 1.0)


class PatternAnalyzer:
    """Analyzes content using pattern matching."""
    
    def __init__(self):
        """Initialize pattern analyzer with common patterns."""
        self.action_patterns = [
            r'\b(?:todo|task|action|do|complete|finish|implement|fix|update|create|build|test)\b[:\s]+(.+?)(?:\n|\.|\!|\?|$)',
            r'\[\s*\]\s*(.+?)(?:\n|$)',  # Checkbox patterns
            r'\b(?:need to|must|should|have to|will|going to)\s+([^.!?\n]+)',
            r'\b(?:follow up|reach out|contact|call|email|schedule|meeting)\b[:\s]*(.+?)(?:\n|\.|\!|\?|$)',
            r'(?:next steps?|action items?)[:\s]*\n?(.+?)(?:\n\n|$)',
        ]
        
        self.priority_indicators = {
            ActionPriority.URGENT: ['urgent', 'asap', 'immediately', 'critical', 'emergency'],
            ActionPriority.HIGH: ['important', 'priority', 'soon', 'deadline', 'today'],
            ActionPriority.MEDIUM: ['should', 'need', 'would like', 'when possible'],
            ActionPriority.LOW: ['eventually', 'someday', 'later', 'consider', 'maybe']
        }
        
        self.category_patterns = {
            ContentCategory.MEETING_NOTES: [
                r'\b(?:meeting|agenda|minutes|discussion|attendees|action items)\b',
                r'\b(?:discussed|agreed|decided|next meeting)\b'
            ],
            ContentCategory.TASK_LIST: [
                r'\[\s*[x\s]\s*\]',  # Checkboxes
                r'\b(?:todo|task|checklist|items to complete)\b'
            ],
            ContentCategory.CODE_SNIPPET: [
                r'```[\s\S]*```',  # Code blocks
                r'\b(?:function|class|import|def|var|let|const)\b'
            ],
            ContentCategory.EMAIL: [
                r'\b(?:from|to|subject|dear|sincerely|regards)\b',
                r'@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            ]
        }
    
    def extract_action_items(self, text: str) -> List[ActionItem]:
        """Extract action items using pattern matching."""
        actions = []
        
        for pattern in self.action_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                action_text = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
                
                if len(action_text) < 5 or len(action_text) > 200:
                    continue  # Skip very short or long matches
                
                # Determine priority based on keywords
                priority = self._determine_action_priority(action_text)
                confidence = self._calculate_action_confidence(action_text, pattern)
                
                # Extract potential due date
                due_date = self._extract_due_date(action_text)
                
                # Extract keywords
                keywords = re.findall(r'\b[a-zA-Z]{3,}\b', action_text.lower())[:5]
                
                action = ActionItem(
                    text=action_text,
                    priority=priority,
                    confidence=confidence,
                    due_date=due_date,
                    context=f"Pattern match: {pattern[:50]}...",
                    keywords=keywords
                )
                
                actions.append(action)
        
        return actions
    
    def _determine_action_priority(self, text: str) -> ActionPriority:
        """Determine action priority based on keywords."""
        text_lower = text.lower()
        
        for priority, indicators in self.priority_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                return priority
        
        return ActionPriority.MEDIUM  # Default priority
    
    def _calculate_action_confidence(self, text: str, pattern: str) -> float:
        """Calculate confidence score for action item extraction."""
        base_confidence = 0.6
        
        # Boost confidence for specific patterns
        if 'todo' in pattern.lower():
            base_confidence += 0.2
        if r'\[\s*\]' in pattern:  # Checkbox pattern
            base_confidence += 0.3
        
        # Boost for action verbs
        action_verbs = ['complete', 'finish', 'create', 'update', 'fix', 'implement']
        if any(verb in text.lower() for verb in action_verbs):
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _extract_due_date(self, text: str) -> Optional[datetime]:
        """Extract due date from action text."""
        date_patterns = [
            r'\b(?:by|due|before|until)\s+(\d{1,2}/\d{1,2}/\d{2,4})',
            r'\b(?:by|due|before|until)\s+(today|tomorrow|next week|next month)',
            r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_text = match.group(1)
                # Simple date parsing (could be enhanced)
                if date_text == 'today':
                    return datetime.now().replace(hour=23, minute=59, second=59)
                elif date_text == 'tomorrow':
                    return (datetime.now() + timedelta(days=1)).replace(hour=23, minute=59, second=59)
                elif date_text == 'next week':
                    return (datetime.now() + timedelta(days=7)).replace(hour=23, minute=59, second=59)
                # Could add more sophisticated date parsing here
        
        return None
    
    def classify_content_category(self, text: str) -> Tuple[ContentCategory, float]:
        """Classify content category using pattern matching."""
        category_scores = {}
        
        for category, patterns in self.category_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches * 0.2  # Weight each match
            
            if score > 0:
                category_scores[category] = min(score, 1.0)
        
        if not category_scores:
            return ContentCategory.UNKNOWN, 0.3
        
        best_category = max(category_scores, key=category_scores.get)
        confidence = category_scores[best_category]
        
        return best_category, confidence


class SmartContentAnalyzer:
    """Main smart content analyzer coordinating all analysis components."""
    
    def __init__(self):
        """Initialize the smart content analyzer."""
        self.nlp_processor = NLPProcessor()
        self.pattern_analyzer = PatternAnalyzer()
        self.embedder = Embeddings()
        
        # Analysis cache to avoid reprocessing identical content
        self.analysis_cache: Dict[str, ContentAnalysis] = {}
        self.cache_ttl = timedelta(hours=1)
        
        # User preference learning (could be persisted to database)
        self.user_tag_preferences: Dict[str, Dict[str, float]] = {}
        
        logger.info("Smart Content Analyzer initialized")
    
    async def analyze_content(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        content_metadata: Optional[Dict[str, Any]] = None
    ) -> ContentAnalysis:
        """
        Perform comprehensive smart content analysis.
        
        Args:
            content: Text content to analyze
            context: Optional context information (source, time, location, etc.)
            user_id: Optional user ID for personalized analysis
            content_metadata: Optional metadata about the content
            
        Returns:
            ContentAnalysis: Complete analysis results
        """
        start_time = datetime.now()
        
        # Check cache first
        cache_key = self._generate_cache_key(content, context, user_id)
        if cache_key in self.analysis_cache:
            cached_analysis = self.analysis_cache[cache_key]
            if datetime.now() - cached_analysis.analysis_timestamp < self.cache_ttl:
                logger.debug("Returning cached content analysis")
                return cached_analysis
        
        logger.info(f"Performing smart content analysis for {len(content)} characters")
        
        # Initialize analysis components
        analysis_tasks = []
        nlp_features_used = []
        
        # Core content metrics
        word_count = len(content.split())
        reading_time = max(1.0, word_count / 200)  # ~200 words per minute
        
        # Pattern-based analysis (synchronous)
        category, category_confidence = self.pattern_analyzer.classify_content_category(content)
        action_items = self.pattern_analyzer.extract_action_items(content)
        
        # NLP-based analysis (potentially asynchronous)
        keywords = self.nlp_processor.extract_keywords_tfidf(content, top_k=10)
        named_entities = self.nlp_processor.extract_named_entities(content)
        sentiment_score = self.nlp_processor.analyze_sentiment(content)
        complexity_score = self.nlp_processor.calculate_complexity(content)
        
        # Smart tag generation
        smart_tags = await self._generate_smart_tags(
            content, keywords, named_entities, context, user_id
        )
        
        # Smart title generation
        smart_titles = await self._generate_smart_titles(
            content, category, keywords, context
        )
        
        # Topic extraction (simplified approach)
        key_topics = self._extract_key_topics(content, keywords)
        
        # Quality assessment
        quality_score, improvement_suggestions = self._assess_content_quality(
            content, word_count, complexity_score, sentiment_score
        )
        
        # Create analysis result
        processing_time = (datetime.now() - start_time).total_seconds()
        
        analysis = ContentAnalysis(
            category=category,
            confidence=category_confidence,
            smart_tags=smart_tags,
            action_items=action_items,
            smart_titles=smart_titles,
            word_count=word_count,
            reading_time_minutes=reading_time,
            complexity_score=complexity_score,
            sentiment_score=sentiment_score,
            key_topics=key_topics,
            named_entities=named_entities,
            keywords=keywords,
            quality_score=quality_score,
            improvement_suggestions=improvement_suggestions,
            processing_time=processing_time,
            nlp_features_used=nlp_features_used
        )
        
        # Cache the result
        self.analysis_cache[cache_key] = analysis
        
        # Update user preferences if user_id provided
        if user_id:
            self._update_user_preferences(user_id, smart_tags)
        
        logger.info(f"Smart content analysis completed in {processing_time:.2f}s")
        return analysis
    
    async def _generate_smart_tags(
        self,
        content: str,
        keywords: List[Tuple[str, float]],
        entities: List[Tuple[str, str, float]],
        context: Optional[Dict[str, Any]],
        user_id: Optional[str]
    ) -> List[SmartTag]:
        """Generate smart tags using multiple approaches."""
        tags = []
        
        # 1. Keyword-based tags
        for keyword, score in keywords[:5]:  # Top 5 keywords
            if len(keyword) > 2 and score > 0.1:
                tag = SmartTag(
                    tag=keyword,
                    confidence=min(score, 0.9),  # Cap at 0.9 for keywords
                    source="nlp",
                    category="keyword",
                    metadata={"tfidf_score": score}
                )
                tags.append(tag)
        
        # 2. Named entity tags
        for entity, entity_type, confidence in entities:
            if len(entity) > 2:
                tag = SmartTag(
                    tag=entity.lower(),
                    confidence=confidence * 0.8,  # Slight discount for entities
                    source="nlp",
                    category="entity",
                    metadata={"entity_type": entity_type}
                )
                tags.append(tag)
        
        # 3. Context-based tags
        if context:
            context_tags = self._extract_context_tags(context)
            tags.extend(context_tags)
        
        # 4. User preference-based tags
        if user_id and user_id in self.user_tag_preferences:
            preference_tags = self._suggest_preference_tags(content, user_id)
            tags.extend(preference_tags)
        
        # 5. Pattern-based tags
        pattern_tags = self._extract_pattern_tags(content)
        tags.extend(pattern_tags)
        
        # Remove duplicates and sort by confidence
        unique_tags = {}
        for tag in tags:
            if tag.tag not in unique_tags or tag.confidence > unique_tags[tag.tag].confidence:
                unique_tags[tag.tag] = tag
        
        return sorted(unique_tags.values(), key=lambda x: x.confidence, reverse=True)[:10]
    
    def _extract_context_tags(self, context: Dict[str, Any]) -> List[SmartTag]:
        """Extract tags from context information."""
        tags = []
        
        # Source-based tags
        if 'source_type' in context:
            tags.append(SmartTag(
                tag=context['source_type'],
                confidence=1.0,
                source="context",
                category="source"
            ))
        
        # Time-based tags
        if 'timestamp' in context:
            timestamp = context['timestamp']
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except ValueError:
                    timestamp = datetime.now()
            elif not isinstance(timestamp, datetime):
                timestamp = datetime.now()
            
            hour = timestamp.hour
            if 6 <= hour < 12:
                tags.append(SmartTag(tag="morning", confidence=0.8, source="context", category="time"))
            elif 12 <= hour < 18:
                tags.append(SmartTag(tag="afternoon", confidence=0.8, source="context", category="time"))
            elif 18 <= hour < 24:
                tags.append(SmartTag(tag="evening", confidence=0.8, source="context", category="time"))
            else:
                tags.append(SmartTag(tag="night", confidence=0.8, source="context", category="time"))
        
        # Location-based tags
        if 'location' in context and context['location']:
            location = context['location']
            if isinstance(location, dict):
                if 'name' in location:
                    tags.append(SmartTag(
                        tag=location['name'].lower(),
                        confidence=0.9,
                        source="context",
                        category="location"
                    ))
        
        return tags
    
    def _suggest_preference_tags(self, content: str, user_id: str) -> List[SmartTag]:
        """Suggest tags based on user preferences."""
        if user_id not in self.user_tag_preferences:
            return []
        
        tags = []
        user_prefs = self.user_tag_preferences[user_id]
        
        # Simple approach: if content contains keywords the user often tags with specific tags
        content_lower = content.lower()
        
        for tag, score in user_prefs.items():
            if score > 0.5:  # Only consider frequently used tags
                # Check if tag or related words appear in content
                if tag in content_lower or any(
                    keyword in content_lower 
                    for keyword in tag.split('_')  # Handle multi-word tags
                ):
                    tags.append(SmartTag(
                        tag=tag,
                        confidence=score * 0.7,  # Discount for preference-based
                        source="preference",
                        category="user_pattern",
                        metadata={"user_frequency": score}
                    ))
        
        return tags
    
    def _extract_pattern_tags(self, content: str) -> List[SmartTag]:
        """Extract tags using pattern matching."""
        tags = []
        
        # Programming language detection
        code_patterns = {
            'python': [r'\bdef\s+\w+', r'\bimport\s+\w+', r'\.py\b'],
            'javascript': [r'\bfunction\s+\w+', r'\bvar\s+\w+', r'\.js\b'],
            'java': [r'\bpublic\s+class', r'\bpublic\s+static', r'\.java\b'],
            'sql': [r'\bSELECT\b', r'\bFROM\b', r'\bWHERE\b'],
            'html': [r'<html', r'<div', r'<script'],
            'css': [r'\{[^}]*\}', r'\.class', r'#id']
        }
        
        for language, patterns in code_patterns.items():
            matches = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in patterns)
            if matches > 0:
                confidence = min(matches * 0.2, 0.8)
                tags.append(SmartTag(
                    tag=language,
                    confidence=confidence,
                    source="pattern",
                    category="programming",
                    metadata={"pattern_matches": matches}
                ))
        
        # Content type patterns
        type_patterns = {
            'meeting': [r'\b(?:meeting|agenda|attendees)\b'],
            'tutorial': [r'\b(?:tutorial|guide|how to|step by step)\b'],
            'documentation': [r'\b(?:documentation|docs|readme|manual)\b'],
            'research': [r'\b(?:research|study|analysis|findings)\b'],
            'planning': [r'\b(?:plan|strategy|roadmap|timeline)\b']
        }
        
        for tag_name, patterns in type_patterns.items():
            matches = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in patterns)
            if matches > 0:
                confidence = min(matches * 0.3, 0.7)
                tags.append(SmartTag(
                    tag=tag_name,
                    confidence=confidence,
                    source="pattern",
                    category="content_type",
                    metadata={"pattern_matches": matches}
                ))
        
        return tags
    
    async def _generate_smart_titles(
        self,
        content: str,
        category: ContentCategory,
        keywords: List[Tuple[str, float]],
        context: Optional[Dict[str, Any]]
    ) -> List[SmartTitle]:
        """Generate smart titles using multiple approaches."""
        titles = []
        
        # 1. Pattern-based title generation
        pattern_title = self._generate_pattern_title(content, category, keywords)
        if pattern_title:
            titles.append(pattern_title)
        
        # 2. AI-based title generation (if available)
        try:
            ai_title_text = await asyncio.wait_for(
                asyncio.create_task(
                    asyncio.to_thread(ollama_generate_title, content[:1000])
                ),
                timeout=30.0
            )
            
            if ai_title_text and ai_title_text.strip():
                context_factors = []
                if context:
                    context_factors.extend([
                        f"source_{context.get('source_type', 'unknown')}",
                        f"category_{category.value}"
                    ])
                
                ai_title = SmartTitle(
                    title=ai_title_text.strip(),
                    confidence=0.8,
                    generation_method="ai",
                    context_factors=context_factors
                )
                titles.append(ai_title)
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"AI title generation failed: {e}")
        
        # 3. Semantic title generation
        semantic_title = self._generate_semantic_title(content, keywords)
        if semantic_title:
            titles.append(semantic_title)
        
        # 4. Context-aware title enhancement
        if context and titles:
            enhanced_titles = self._enhance_titles_with_context(titles, context)
            titles.extend(enhanced_titles)
        
        # Remove duplicates and sort by confidence
        unique_titles = {}
        for title in titles:
            key = title.title.lower().strip()
            if key not in unique_titles or title.confidence > unique_titles[key].confidence:
                unique_titles[key] = title
        
        return sorted(unique_titles.values(), key=lambda x: x.confidence, reverse=True)[:3]
    
    def _generate_pattern_title(
        self,
        content: str,
        category: ContentCategory,
        keywords: List[Tuple[str, float]]
    ) -> Optional[SmartTitle]:
        """Generate title using pattern matching."""
        # Look for existing title patterns
        title_patterns = [
            r'^#\s+(.+)$',  # Markdown headers
            r'^Title:\s*(.+)$',  # Explicit title
            r'^Subject:\s*(.+)$',  # Email subject
            r'^\*\*(.+)\*\*$'  # Bold text as title
        ]
        
        lines = content.split('\n')[:5]  # Check first 5 lines
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern in title_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    title_text = match.group(1).strip()
                    if 5 <= len(title_text) <= 100:  # Reasonable title length
                        return SmartTitle(
                            title=title_text,
                            confidence=0.9,
                            generation_method="pattern",
                            context_factors=[f"category_{category.value}"]
                        )
        
        # Generate from keywords if no pattern found
        if keywords:
            top_keywords = [kw[0] for kw in keywords[:3]]
            title_text = f"{category.value.replace('_', ' ').title()}: {', '.join(top_keywords)}"
            
            return SmartTitle(
                title=title_text,
                confidence=0.6,
                generation_method="pattern",
                context_factors=["keywords", f"category_{category.value}"]
            )
        
        return None
    
    def _generate_semantic_title(
        self,
        content: str,
        keywords: List[Tuple[str, float]]
    ) -> Optional[SmartTitle]:
        """Generate title using semantic analysis."""
        # Extract first meaningful sentence
        sentences = content.split('.')[:3]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if 10 <= len(sentence) <= 80 and not sentence.lower().startswith(('the', 'this', 'that', 'i', 'we')):
                # This could be a good title candidate
                return SmartTitle(
                    title=sentence,
                    confidence=0.7,
                    generation_method="semantic",
                    context_factors=["first_sentence"],
                    alternatives=[s.strip() for s in sentences[1:] if 10 <= len(s.strip()) <= 80]
                )
        
        return None
    
    def _enhance_titles_with_context(
        self,
        titles: List[SmartTitle],
        context: Dict[str, Any]
    ) -> List[SmartTitle]:
        """Enhance existing titles with context information."""
        enhanced = []
        
        for title in titles:
            if len(enhanced) >= 2:  # Limit enhanced titles
                break
                
            enhanced_text = title.title
            context_factors = list(title.context_factors)
            
            # Add timestamp context
            if 'timestamp' in context:
                try:
                    timestamp = context['timestamp']
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp)
                    
                    date_str = timestamp.strftime("%Y-%m-%d")
                    enhanced_text = f"{enhanced_text} ({date_str})"
                    context_factors.append("timestamp")
                except Exception:
                    pass
            
            # Add source context
            if 'source_type' in context and context['source_type'] not in enhanced_text.lower():
                source_labels = {
                    'discord': 'Discord',
                    'apple_shortcuts': 'iOS',
                    'web_ui': 'Web',
                    'api': 'API'
                }
                source_label = source_labels.get(context['source_type'], context['source_type'])
                enhanced_text = f"[{source_label}] {enhanced_text}"
                context_factors.append("source")
            
            if enhanced_text != title.title:
                enhanced_title = SmartTitle(
                    title=enhanced_text,
                    confidence=title.confidence * 0.9,  # Slight discount for enhancement
                    generation_method="hybrid",
                    context_factors=context_factors,
                    alternatives=[title.title]
                )
                enhanced.append(enhanced_title)
        
        return enhanced
    
    def _extract_key_topics(
        self,
        content: str,
        keywords: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """Extract key topics from content."""
        # Simple approach: use top keywords as topics
        topics = []
        
        for keyword, score in keywords[:5]:
            # Group related keywords into topics
            topic_score = score
            
            # Boost score for multi-word topics
            if '_' in keyword or ' ' in keyword:
                topic_score *= 1.2
            
            topics.append((keyword, min(topic_score, 1.0)))
        
        return topics
    
    def _assess_content_quality(
        self,
        content: str,
        word_count: int,
        complexity_score: float,
        sentiment_score: float
    ) -> Tuple[float, List[str]]:
        """Assess content quality and provide improvement suggestions."""
        quality_factors = []
        suggestions = []
        
        # Length factor
        if word_count < 10:
            quality_factors.append(0.3)
            suggestions.append("Content is very short. Consider adding more detail.")
        elif word_count < 50:
            quality_factors.append(0.6)
            suggestions.append("Content could benefit from more elaboration.")
        elif word_count > 1000:
            quality_factors.append(0.9)
        else:
            quality_factors.append(0.8)
        
        # Complexity factor
        if complexity_score < 0.2:
            quality_factors.append(0.7)
            suggestions.append("Content could use more varied vocabulary.")
        elif complexity_score > 0.8:
            quality_factors.append(0.7)
            suggestions.append("Content might be overly complex for some readers.")
        else:
            quality_factors.append(0.9)
        
        # Structure factor
        has_structure = bool(re.search(r'[.!?]', content))  # Has sentences
        has_paragraphs = len(content.split('\n\n')) > 1
        
        if has_structure and has_paragraphs:
            quality_factors.append(0.9)
        elif has_structure:
            quality_factors.append(0.7)
            suggestions.append("Consider breaking content into paragraphs for better readability.")
        else:
            quality_factors.append(0.5)
            suggestions.append("Content lacks proper sentence structure.")
        
        # Calculate overall quality
        quality_score = sum(quality_factors) / len(quality_factors)
        
        # Add general suggestions based on quality score
        if quality_score < 0.6:
            suggestions.append("Consider reviewing and expanding the content.")
        elif quality_score > 0.85:
            suggestions.append("Content quality is excellent!")
        
        return quality_score, suggestions
    
    def _generate_cache_key(
        self,
        content: str,
        context: Optional[Dict[str, Any]],
        user_id: Optional[str]
    ) -> str:
        """Generate cache key for analysis results."""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
        context_hash = hashlib.md5(str(context or {}).encode()).hexdigest()[:8]
        user_hash = hashlib.md5((user_id or "").encode()).hexdigest()[:8]
        
        return f"{content_hash}_{context_hash}_{user_hash}"
    
    def _update_user_preferences(self, user_id: str, tags: List[SmartTag]):
        """Update user tag preferences based on analysis results."""
        if user_id not in self.user_tag_preferences:
            self.user_tag_preferences[user_id] = {}
        
        user_prefs = self.user_tag_preferences[user_id]
        
        # Increment preference scores for high-confidence tags
        for tag in tags:
            if tag.confidence > 0.6:
                current_score = user_prefs.get(tag.tag, 0.0)
                # Exponential moving average
                user_prefs[tag.tag] = 0.8 * current_score + 0.2 * tag.confidence
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics and performance metrics."""
        return {
            "cache_size": len(self.analysis_cache),
            "nltk_available": NLTK_AVAILABLE,
            "sklearn_available": SKLEARN_AVAILABLE,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
            "nlp_processor_ready": self.nlp_processor.nltk_ready,
            "user_preferences_count": len(self.user_tag_preferences),
            "features": {
                "smart_tagging": True,
                "action_item_extraction": True,
                "content_classification": True,
                "quality_assessment": True,
                "context_awareness": True,
                "user_preference_learning": True,
                "title_generation": True,
                "sentiment_analysis": True,
                "named_entity_recognition": NLTK_AVAILABLE,
                "advanced_nlp": SKLEARN_AVAILABLE
            }
        }
    
    def clear_cache(self):
        """Clear analysis cache."""
        self.analysis_cache.clear()
        logger.info("Analysis cache cleared")
    
    def get_user_preferences(self, user_id: str) -> Dict[str, float]:
        """Get user tag preferences."""
        return self.user_tag_preferences.get(user_id, {})
    
    def set_user_preferences(self, user_id: str, preferences: Dict[str, float]):
        """Set user tag preferences."""
        self.user_tag_preferences[user_id] = preferences
        logger.info(f"Updated preferences for user {user_id}: {len(preferences)} tags")


# Global service instance
_smart_analyzer: Optional[SmartContentAnalyzer] = None


def get_smart_content_analyzer() -> SmartContentAnalyzer:
    """Get the global smart content analyzer instance."""
    global _smart_analyzer
    if _smart_analyzer is None:
        _smart_analyzer = SmartContentAnalyzer()
    return _smart_analyzer