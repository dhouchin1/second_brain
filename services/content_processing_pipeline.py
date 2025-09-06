# ──────────────────────────────────────────────────────────────────────────────
# File: services/content_processing_pipeline.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Advanced Content Processing Pipeline

Enhanced content processing with intelligent chunking, multi-modal coordination,
smart routing, and quality validation for the Second Brain capture system.
"""

import logging
import asyncio
import hashlib
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import json

from services.capture_error_handler import (
    get_capture_error_handler, 
    ErrorContext, 
    ErrorCategory, 
    ErrorSeverity,
    ProcessingProgress
)
from services.embeddings import Embeddings
from llm_utils import ollama_summarize, ollama_generate_title
from config import settings

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Enhanced content type classification."""
    TEXT = "text"
    RICH_TEXT = "rich_text" 
    MARKDOWN = "markdown"
    CODE = "code"
    AUDIO = "audio"
    IMAGE = "image"
    PDF = "pdf"
    VIDEO = "video"
    URL = "url"
    EMAIL = "email"
    DOCUMENT = "document"
    SPREADSHEET = "spreadsheet"
    PRESENTATION = "presentation"
    ARCHIVE = "archive"
    UNKNOWN = "unknown"


class ProcessingStage(Enum):
    """Stages of content processing pipeline."""
    INTAKE = "intake"
    CLASSIFICATION = "classification" 
    CHUNKING = "chunking"
    EXTRACTION = "extraction"
    AI_PROCESSING = "ai_processing"
    VALIDATION = "validation"
    INDEXING = "indexing"
    STORAGE = "storage"
    COMPLETION = "completion"


class ContentPriority(Enum):
    """Processing priority levels."""
    LOW = 1
    NORMAL = 2 
    HIGH = 3
    URGENT = 4


class ChunkingStrategy(Enum):
    """Strategies for content chunking."""
    NONE = "none"
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


@dataclass
class ContentChunk:
    """A chunk of processed content."""
    id: str
    content: str
    chunk_type: str = "text"
    sequence: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 1.0
    processing_notes: List[str] = field(default_factory=list)
    
    @property
    def word_count(self) -> int:
        return len(self.content.split())
    
    @property
    def char_count(self) -> int:
        return len(self.content)


@dataclass
class ProcessingContext:
    """Context for content processing operations."""
    source_type: str
    content_type: ContentType
    original_size: int
    priority: ContentPriority = ContentPriority.NORMAL
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    processing_options: Dict[str, Any] = field(default_factory=dict)
    quality_requirements: Dict[str, Any] = field(default_factory=dict)
    

@dataclass 
class ProcessingResult:
    """Result from content processing pipeline."""
    success: bool
    content_type: ContentType
    chunks: List[ContentChunk]
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def total_content_size(self) -> int:
        return sum(chunk.char_count for chunk in self.chunks)
    
    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


class ContentClassifier:
    """Intelligent content type classification."""
    
    def __init__(self):
        """Initialize the classifier with detection patterns."""
        self.patterns = {
            ContentType.CODE: [
                r'(function|class|import|def|var|let|const)\s+\w+',
                r'[{}();].*\n.*[{}();]',
                r'(#include|using namespace|package\s+\w+)',
            ],
            ContentType.MARKDOWN: [
                r'^#{1,6}\s+.+$',
                r'\*\*.*\*\*|\*.*\*',
                r'^\s*[-*+]\s+',
                r'\[.*\]\(.*\)',
            ],
            ContentType.EMAIL: [
                r'^(From|To|Subject|Date):\s+.+$',
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            ],
            ContentType.URL: [
                r'^https?://[^\s]+$',
                r'www\.[^\s]+\.[a-z]{2,}',
            ]
        }
    
    def classify_content(self, content: str, filename: Optional[str] = None) -> ContentType:
        """Classify content based on patterns and filename."""
        if not content or not content.strip():
            return ContentType.UNKNOWN
        
        # Check filename extension first
        if filename:
            ext = filename.lower().split('.')[-1] if '.' in filename else ''
            if ext in ['md', 'markdown']:
                return ContentType.MARKDOWN
            elif ext in ['py', 'js', 'java', 'cpp', 'c', 'html', 'css', 'sql']:
                return ContentType.CODE
            elif ext in ['pdf']:
                return ContentType.PDF
            elif ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
                return ContentType.IMAGE
            elif ext in ['mp3', 'wav', 'ogg', 'flac']:
                return ContentType.AUDIO
            elif ext in ['mp4', 'avi', 'mov', 'wmv']:
                return ContentType.VIDEO
        
        # Pattern-based classification
        content_sample = content[:2000]  # First 2KB for efficiency
        
        for content_type, patterns in self.patterns.items():
            matches = sum(1 for pattern in patterns if re.search(pattern, content_sample, re.MULTILINE | re.IGNORECASE))
            if matches >= 2:  # Require multiple pattern matches
                return content_type
        
        # Check for rich text indicators
        if any(tag in content_sample.lower() for tag in ['<html>', '<div>', '<span>', '<p>']):
            return ContentType.RICH_TEXT
        
        # Default to text
        return ContentType.TEXT


class ContentChunker:
    """Intelligent content chunking with multiple strategies."""
    
    def __init__(self):
        """Initialize chunker with configuration."""
        self.max_chunk_size = getattr(settings, 'content_max_chunk_size', 8000)
        self.min_chunk_size = getattr(settings, 'content_min_chunk_size', 500)
        self.chunk_overlap = getattr(settings, 'content_chunk_overlap', 200)
    
    def chunk_content(
        self,
        content: str,
        content_type: ContentType,
        strategy: ChunkingStrategy = ChunkingStrategy.ADAPTIVE
    ) -> List[ContentChunk]:
        """Chunk content using the specified strategy."""
        if len(content) <= self.max_chunk_size:
            # Content is small enough, no chunking needed
            return [ContentChunk(
                id="chunk_0",
                content=content,
                chunk_type="full",
                sequence=0,
                metadata={"chunking_strategy": "none"}
            )]
        
        if strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic_chunking(content, content_type)
        elif strategy == ChunkingStrategy.HIERARCHICAL:
            return self._hierarchical_chunking(content, content_type)
        elif strategy == ChunkingStrategy.FIXED_SIZE:
            return self._fixed_size_chunking(content)
        elif strategy == ChunkingStrategy.ADAPTIVE:
            return self._adaptive_chunking(content, content_type)
        else:
            return self._fixed_size_chunking(content)
    
    def _semantic_chunking(self, content: str, content_type: ContentType) -> List[ContentChunk]:
        """Chunk based on semantic boundaries."""
        chunks = []
        
        # Different semantic boundaries based on content type
        if content_type == ContentType.MARKDOWN:
            # Split on headers
            sections = re.split(r'\n(#{1,6}\s+.+)', content)
            current_chunk = ""
            current_header = ""
            
            for i, section in enumerate(sections):
                if re.match(r'^#{1,6}\s+.+', section):
                    if current_chunk and len(current_chunk) > self.min_chunk_size:
                        chunks.append(self._create_chunk(
                            content=current_chunk,
                            chunk_id=f"semantic_{len(chunks)}",
                            sequence=len(chunks),
                            metadata={"header": current_header, "type": "markdown_section"}
                        ))
                        current_chunk = ""
                    current_header = section
                    current_chunk = section + "\n"
                else:
                    current_chunk += section
                    
                    # Check if chunk is getting too large
                    if len(current_chunk) > self.max_chunk_size:
                        chunks.append(self._create_chunk(
                            content=current_chunk,
                            chunk_id=f"semantic_{len(chunks)}",
                            sequence=len(chunks),
                            metadata={"header": current_header, "type": "markdown_section"}
                        ))
                        current_chunk = ""
            
            if current_chunk:
                chunks.append(self._create_chunk(
                    content=current_chunk,
                    chunk_id=f"semantic_{len(chunks)}",
                    sequence=len(chunks),
                    metadata={"header": current_header, "type": "markdown_section"}
                ))
        
        elif content_type == ContentType.CODE:
            # Split on function/class boundaries
            return self._code_aware_chunking(content)
        
        else:
            # Generic semantic chunking on paragraphs
            paragraphs = re.split(r'\n\s*\n', content)
            current_chunk = ""
            
            for paragraph in paragraphs:
                if len(current_chunk + paragraph) > self.max_chunk_size and current_chunk:
                    chunks.append(self._create_chunk(
                        content=current_chunk.strip(),
                        chunk_id=f"semantic_{len(chunks)}",
                        sequence=len(chunks),
                        metadata={"type": "paragraph_group"}
                    ))
                    current_chunk = paragraph + "\n\n"
                else:
                    current_chunk += paragraph + "\n\n"
            
            if current_chunk:
                chunks.append(self._create_chunk(
                    content=current_chunk.strip(),
                    chunk_id=f"semantic_{len(chunks)}",
                    sequence=len(chunks),
                    metadata={"type": "paragraph_group"}
                ))
        
        return chunks if chunks else [self._create_chunk(content, "semantic_0", 0)]
    
    def _hierarchical_chunking(self, content: str, content_type: ContentType) -> List[ContentChunk]:
        """Create hierarchical chunks (summaries + details)."""
        chunks = []
        
        # First, create semantic chunks
        semantic_chunks = self._semantic_chunking(content, content_type)
        
        # If we have multiple semantic chunks, create a summary chunk
        if len(semantic_chunks) > 1:
            # Create a summary of all chunks
            summaries = []
            for i, chunk in enumerate(semantic_chunks[:5]):  # Limit to first 5 for summary
                summary = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                summaries.append(f"Section {i+1}: {summary}")
            
            summary_content = "Document Overview:\n\n" + "\n\n".join(summaries)
            chunks.append(self._create_chunk(
                content=summary_content,
                chunk_id="hierarchical_summary",
                sequence=0,
                metadata={"type": "summary", "covers_chunks": len(semantic_chunks)}
            ))
        
        # Add all semantic chunks with updated sequence numbers
        for i, chunk in enumerate(semantic_chunks):
            chunk.sequence = i + (1 if len(semantic_chunks) > 1 else 0)
            chunk.id = f"hierarchical_detail_{i}"
            chunk.metadata["type"] = "detail"
            chunks.append(chunk)
        
        return chunks
    
    def _fixed_size_chunking(self, content: str) -> List[ContentChunk]:
        """Simple fixed-size chunking with overlap."""
        chunks = []
        start = 0
        sequence = 0
        
        while start < len(content):
            end = min(start + self.max_chunk_size, len(content))
            chunk_content = content[start:end]
            
            chunks.append(self._create_chunk(
                content=chunk_content,
                chunk_id=f"fixed_{sequence}",
                sequence=sequence,
                metadata={"type": "fixed_size", "start": start, "end": end}
            ))
            
            # Apply overlap for next chunk
            start = max(start + self.max_chunk_size - self.chunk_overlap, end)
            sequence += 1
        
        return chunks
    
    def _adaptive_chunking(self, content: str, content_type: ContentType) -> List[ContentChunk]:
        """Adaptive chunking that chooses the best strategy based on content."""
        content_length = len(content)
        
        # For very large content, use hierarchical
        if content_length > 50000:
            return self._hierarchical_chunking(content, content_type)
        
        # For structured content, use semantic
        if content_type in [ContentType.MARKDOWN, ContentType.CODE, ContentType.RICH_TEXT]:
            return self._semantic_chunking(content, content_type)
        
        # For medium-sized content, try semantic first
        if content_length > 10000:
            semantic_chunks = self._semantic_chunking(content, content_type)
            if len(semantic_chunks) > 1:
                return semantic_chunks
        
        # Fall back to fixed size
        return self._fixed_size_chunking(content)
    
    def _code_aware_chunking(self, content: str) -> List[ContentChunk]:
        """Specialized chunking for code content."""
        chunks = []
        
        # Patterns for different code structures
        function_pattern = r'^\s*(def|function|async\s+def|class)\s+\w+[^{]*[{:]'
        
        lines = content.split('\n')
        current_chunk = []
        current_function = ""
        brace_count = 0
        
        for line in lines:
            current_chunk.append(line)
            
            # Track braces/indentation to detect function boundaries
            brace_count += line.count('{') - line.count('}')
            
            # Check for new function/class
            if re.match(function_pattern, line, re.MULTILINE):
                current_function = line.strip()
            
            # Check if we should split (function ended or chunk too large)
            chunk_content = '\n'.join(current_chunk)
            if ((brace_count == 0 and current_function and len(current_chunk) > 5) or 
                len(chunk_content) > self.max_chunk_size):
                
                if len(chunk_content) > self.min_chunk_size:
                    chunks.append(self._create_chunk(
                        content=chunk_content,
                        chunk_id=f"code_{len(chunks)}",
                        sequence=len(chunks),
                        metadata={
                            "type": "code_block",
                            "function": current_function,
                            "lines": len(current_chunk)
                        }
                    ))
                    current_chunk = []
                    current_function = ""
        
        # Add remaining content
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append(self._create_chunk(
                content=chunk_content,
                chunk_id=f"code_{len(chunks)}",
                sequence=len(chunks),
                metadata={
                    "type": "code_block",
                    "function": current_function,
                    "lines": len(current_chunk)
                }
            ))
        
        return chunks if chunks else [self._create_chunk(content, "code_0", 0)]
    
    def _create_chunk(
        self,
        content: str,
        chunk_id: str,
        sequence: int,
        metadata: Dict[str, Any] = None
    ) -> ContentChunk:
        """Helper to create a ContentChunk with quality scoring."""
        chunk = ContentChunk(
            id=chunk_id,
            content=content,
            sequence=sequence,
            metadata=metadata or {}
        )
        
        # Calculate quality score based on various factors
        quality_score = 1.0
        
        # Penalize very short chunks
        if chunk.word_count < 10:
            quality_score *= 0.7
        
        # Penalize chunks that are mostly whitespace
        content_density = len(content.strip()) / max(len(content), 1)
        quality_score *= content_density
        
        # Reward chunks with good structure (sentences, etc.)
        sentence_count = len(re.findall(r'[.!?]+', content))
        if sentence_count > 0 and chunk.word_count > 0:
            avg_sentence_length = chunk.word_count / sentence_count
            if 10 <= avg_sentence_length <= 30:  # Good sentence length range
                quality_score *= 1.1
        
        chunk.quality_score = min(quality_score, 1.0)
        return chunk


class MultiModalProcessor:
    """Coordinates processing across different content modalities."""
    
    def __init__(self):
        """Initialize multi-modal processor."""
        self.processor_map = {
            ContentType.TEXT: self._process_text,
            ContentType.RICH_TEXT: self._process_rich_text,
            ContentType.MARKDOWN: self._process_markdown,
            ContentType.CODE: self._process_code,
            ContentType.PDF: self._process_pdf,
            ContentType.IMAGE: self._process_image,
            ContentType.AUDIO: self._process_audio,
            ContentType.URL: self._process_url,
            ContentType.EMAIL: self._process_email,
        }
    
    async def process_content(
        self,
        content: str,
        content_type: ContentType,
        context: ProcessingContext,
        metadata: Dict[str, Any] = None
    ) -> ProcessingResult:
        """Process content based on its type."""
        processor = self.processor_map.get(content_type, self._process_generic)
        return await processor(content, context, metadata or {})
    
    async def _process_text(
        self,
        content: str,
        context: ProcessingContext,
        metadata: Dict[str, Any]
    ) -> ProcessingResult:
        """Process plain text content."""
        chunker = ContentChunker()
        chunks = chunker.chunk_content(content, ContentType.TEXT)
        
        # Text-specific processing
        for chunk in chunks:
            # Extract key phrases or entities if content is substantial
            if chunk.word_count > 50:
                # Simple key phrase extraction (could be enhanced with NLP)
                words = chunk.content.lower().split()
                word_freq = {}
                for word in words:
                    if len(word) > 3 and word.isalpha():
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                # Top 5 frequent words as basic key phrases
                key_phrases = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                chunk.metadata["key_phrases"] = [phrase[0] for phrase in key_phrases]
        
        return ProcessingResult(
            success=True,
            content_type=ContentType.TEXT,
            chunks=chunks,
            metadata={"processing_type": "text", "language": "auto-detected"},
            quality_metrics={"avg_chunk_quality": sum(c.quality_score for c in chunks) / len(chunks)}
        )
    
    async def _process_rich_text(
        self,
        content: str,
        context: ProcessingContext,
        metadata: Dict[str, Any]
    ) -> ProcessingResult:
        """Process rich text/HTML content."""
        # Strip HTML tags for plain text version
        import re
        plain_text = re.sub(r'<[^>]+>', '', content)
        
        # Process as text but preserve HTML structure info
        result = await self._process_text(plain_text, context, metadata)
        result.content_type = ContentType.RICH_TEXT
        result.metadata["original_format"] = "html"
        result.metadata["has_html_tags"] = True
        
        return result
    
    async def _process_markdown(
        self,
        content: str,
        context: ProcessingContext,
        metadata: Dict[str, Any]
    ) -> ProcessingResult:
        """Process Markdown content with structure preservation."""
        chunker = ContentChunker()
        chunks = chunker.chunk_content(content, ContentType.MARKDOWN)
        
        # Extract markdown structure
        headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        
        return ProcessingResult(
            success=True,
            content_type=ContentType.MARKDOWN,
            chunks=chunks,
            metadata={
                "processing_type": "markdown",
                "structure": {
                    "headers": [(len(h[0]), h[1]) for h in headers],
                    "links": links,
                    "has_code_blocks": "```" in content,
                    "has_tables": "|" in content and "---" in content
                }
            },
            quality_metrics={"avg_chunk_quality": sum(c.quality_score for c in chunks) / len(chunks)}
        )
    
    async def _process_code(
        self,
        content: str,
        context: ProcessingContext,
        metadata: Dict[str, Any]
    ) -> ProcessingResult:
        """Process code content with syntax awareness."""
        chunker = ContentChunker()
        chunks = chunker.chunk_content(content, ContentType.CODE)
        
        # Extract code metrics
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        comment_lines = [line for line in non_empty_lines if line.strip().startswith(('#', '//', '/*', '*'))]
        
        # Detect programming language (basic heuristics)
        language = "unknown"
        if "def " in content and "import " in content:
            language = "python"
        elif "function " in content and "{" in content:
            language = "javascript"
        elif "#include" in content and "int main" in content:
            language = "c/c++"
        elif "class " in content and "public " in content:
            language = "java"
        
        return ProcessingResult(
            success=True,
            content_type=ContentType.CODE,
            chunks=chunks,
            metadata={
                "processing_type": "code",
                "language": language,
                "metrics": {
                    "total_lines": len(lines),
                    "code_lines": len(non_empty_lines),
                    "comment_lines": len(comment_lines),
                    "comment_ratio": len(comment_lines) / max(len(non_empty_lines), 1)
                }
            },
            quality_metrics={"avg_chunk_quality": sum(c.quality_score for c in chunks) / len(chunks)}
        )
    
    async def _process_pdf(
        self,
        content: str,
        context: ProcessingContext,
        metadata: Dict[str, Any]
    ) -> ProcessingResult:
        """Process PDF content (already extracted text)."""
        # PDF content has likely been extracted already
        result = await self._process_text(content, context, metadata)
        result.content_type = ContentType.PDF
        result.metadata["original_format"] = "pdf"
        result.metadata["extraction_method"] = metadata.get("extraction_method", "unknown")
        
        return result
    
    async def _process_image(
        self,
        content: str,
        context: ProcessingContext,
        metadata: Dict[str, Any]
    ) -> ProcessingResult:
        """Process image content (OCR text)."""
        # Image content should be OCR-extracted text
        result = await self._process_text(content, context, metadata)
        result.content_type = ContentType.IMAGE
        result.metadata["original_format"] = "image"
        result.metadata["ocr_confidence"] = metadata.get("ocr_confidence", 0.0)
        
        return result
    
    async def _process_audio(
        self,
        content: str,
        context: ProcessingContext,
        metadata: Dict[str, Any]
    ) -> ProcessingResult:
        """Process audio content (transcribed text)."""
        result = await self._process_text(content, context, metadata)
        result.content_type = ContentType.AUDIO
        result.metadata["original_format"] = "audio"
        result.metadata["transcription_confidence"] = metadata.get("transcription_confidence", 0.0)
        
        return result
    
    async def _process_url(
        self,
        content: str,
        context: ProcessingContext,
        metadata: Dict[str, Any]
    ) -> ProcessingResult:
        """Process URL content (web-scraped text)."""
        result = await self._process_text(content, context, metadata)
        result.content_type = ContentType.URL
        result.metadata["original_format"] = "webpage"
        result.metadata["source_url"] = metadata.get("url", "")
        
        return result
    
    async def _process_email(
        self,
        content: str,
        context: ProcessingContext,
        metadata: Dict[str, Any]
    ) -> ProcessingResult:
        """Process email content with header extraction."""
        # Extract email headers
        headers = {}
        lines = content.split('\n')
        in_headers = True
        body_start = 0
        
        for i, line in enumerate(lines):
            if in_headers and ':' in line and not line.startswith(' '):
                key, value = line.split(':', 1)
                headers[key.strip().lower()] = value.strip()
            elif in_headers and not line.strip():
                in_headers = False
                body_start = i + 1
                break
        
        # Process the body
        body = '\n'.join(lines[body_start:]) if body_start < len(lines) else content
        result = await self._process_text(body, context, metadata)
        result.content_type = ContentType.EMAIL
        result.metadata["email_headers"] = headers
        result.metadata["original_format"] = "email"
        
        return result
    
    async def _process_generic(
        self,
        content: str,
        context: ProcessingContext,
        metadata: Dict[str, Any]
    ) -> ProcessingResult:
        """Generic processing for unknown content types."""
        return await self._process_text(content, context, metadata)


class ContentQualityValidator:
    """Validates processing quality and content integrity."""
    
    def __init__(self):
        """Initialize quality validator."""
        self.quality_thresholds = {
            "min_content_ratio": 0.1,  # Minimum content to whitespace ratio
            "min_word_count": 3,       # Minimum words per chunk
            "max_duplicate_ratio": 0.3,  # Maximum duplicate content ratio
            "min_quality_score": 0.3   # Minimum chunk quality score
        }
    
    def validate_processing_result(self, result: ProcessingResult) -> Tuple[bool, List[str]]:
        """Validate a processing result and return issues found."""
        issues = []
        
        # Check if any chunks were produced
        if not result.chunks:
            issues.append("No content chunks were produced")
            return False, issues
        
        # Check individual chunk quality
        low_quality_chunks = [
            chunk for chunk in result.chunks 
            if chunk.quality_score < self.quality_thresholds["min_quality_score"]
        ]
        
        if len(low_quality_chunks) > len(result.chunks) * 0.5:  # More than 50% low quality
            issues.append(f"High proportion of low-quality chunks: {len(low_quality_chunks)}/{len(result.chunks)}")
        
        # Check for content loss
        original_size = result.metadata.get("original_size", 0)
        processed_size = result.total_content_size
        
        if original_size > 0:
            content_ratio = processed_size / original_size
            if content_ratio < 0.1:  # Lost more than 90% of content
                issues.append(f"Significant content loss detected: {content_ratio:.1%} retained")
        
        # Check for duplicate content
        chunk_contents = [chunk.content for chunk in result.chunks]
        unique_contents = set(chunk_contents)
        
        if len(unique_contents) < len(chunk_contents) * 0.7:  # More than 30% duplicates
            duplicate_ratio = 1 - (len(unique_contents) / len(chunk_contents))
            issues.append(f"High duplicate content ratio: {duplicate_ratio:.1%}")
        
        # Validate content structure based on type
        if result.content_type == ContentType.MARKDOWN:
            if not any("header" in chunk.metadata for chunk in result.chunks):
                issues.append("Markdown content lacks proper structure (no headers detected)")
        
        elif result.content_type == ContentType.CODE:
            total_chars = sum(len(chunk.content) for chunk in result.chunks)
            if total_chars > 0:
                code_density = sum(
                    len(re.findall(r'[{}();=]', chunk.content)) for chunk in result.chunks
                ) / total_chars
                if code_density < 0.01:  # Very low code density
                    issues.append("Code content appears to lack proper syntax elements")
        
        return len(issues) == 0, issues
    
    def calculate_overall_quality_score(self, result: ProcessingResult) -> float:
        """Calculate an overall quality score for the processing result."""
        if not result.chunks:
            return 0.0
        
        # Base score from chunk quality
        avg_chunk_quality = sum(chunk.quality_score for chunk in result.chunks) / len(result.chunks)
        
        # Adjust for processing completeness
        completeness_score = 1.0
        if result.errors:
            completeness_score *= 0.7
        if result.warnings:
            completeness_score *= 0.9
        
        # Adjust for content type appropriateness
        type_score = 1.0
        if result.content_type == ContentType.UNKNOWN:
            type_score = 0.8
        
        # Adjust for chunk distribution
        distribution_score = 1.0
        if len(result.chunks) == 1 and result.total_content_size > 10000:
            distribution_score = 0.9  # Large content should probably be chunked
        
        return avg_chunk_quality * completeness_score * type_score * distribution_score


class ContentProcessingPipeline:
    """Main pipeline orchestrating all content processing stages."""
    
    def __init__(self):
        """Initialize the processing pipeline."""
        self.classifier = ContentClassifier()
        self.chunker = ContentChunker()
        self.multimodal_processor = MultiModalProcessor()
        self.quality_validator = ContentQualityValidator()
        self.error_handler = get_capture_error_handler()
        self.embedder = Embeddings()
        
        # Processing statistics
        self.stats = {
            "total_processed": 0,
            "by_content_type": {},
            "by_stage": {},
            "avg_processing_time": 0.0,
            "quality_scores": []
        }
    
    async def process_content(
        self,
        content: str,
        context: ProcessingContext,
        metadata: Dict[str, Any] = None,
        operation_id: Optional[str] = None
    ) -> ProcessingResult:
        """
        Main processing method that orchestrates the entire pipeline.
        
        Args:
            content: Raw content to process
            context: Processing context with options and preferences
            metadata: Additional metadata about the content
            operation_id: Optional operation ID for progress tracking
            
        Returns:
            ProcessingResult: Complete processing results
        """
        metadata = metadata or {}
        start_time = time.time()
        
        # Create error context for unified error handling
        error_context = ErrorContext(
            operation="content_processing",
            source_type=context.source_type,
            content_type=context.content_type.value,
            additional_context={
                "content_size": len(content),
                "priority": context.priority.value
            }
        )
        
        # Use error handler's operation context for progress tracking
        async with self.error_handler.capture_operation(
            operation_name="content_processing_pipeline",
            context=error_context,
            total_steps=8,
            enable_progress=bool(operation_id)
        ) as progress:
            
            try:
                # Stage 1: Content Classification
                await progress.update_step("Classifying content type", 1)
                classified_type = self.classifier.classify_content(
                    content, 
                    metadata.get("filename")
                )
                
                # Update context with classified type if not specified
                if context.content_type == ContentType.UNKNOWN:
                    context.content_type = classified_type
                
                # Stage 2: Multi-modal Processing
                await progress.update_step("Processing content by type", 2)
                processing_result = await self.multimodal_processor.process_content(
                    content, context.content_type, context, metadata
                )
                
                # Stage 3: AI Enhancement (if enabled)
                if context.processing_options.get("enable_ai", True):
                    await progress.update_step("Enhancing with AI", 3)
                    processing_result = await self._enhance_with_ai(processing_result, context)
                
                # Stage 4: Quality Validation
                await progress.update_step("Validating quality", 4)
                is_valid, validation_issues = self.quality_validator.validate_processing_result(processing_result)
                
                if validation_issues:
                    processing_result.warnings.extend(validation_issues)
                    await progress.add_warning(f"Quality issues found: {len(validation_issues)}")
                
                # Stage 5: Embedding Generation (if enabled)
                if context.processing_options.get("enable_embeddings", True):
                    await progress.update_step("Generating embeddings", 5)
                    processing_result = await self._generate_embeddings(processing_result)
                
                # Stage 6: Content Optimization
                await progress.update_step("Optimizing content", 6)
                processing_result = await self._optimize_content(processing_result, context)
                
                # Stage 7: Final Quality Assessment
                await progress.update_step("Final quality assessment", 7)
                quality_score = self.quality_validator.calculate_overall_quality_score(processing_result)
                processing_result.quality_metrics["overall_score"] = quality_score
                
                # Stage 8: Finalization
                await progress.update_step("Finalizing processing", 8)
                processing_result.processing_time = time.time() - start_time
                
                # Update statistics
                self._update_statistics(processing_result, context)
                
                await progress.complete("Content processing completed successfully")
                
                return processing_result
                
            except Exception as e:
                # Error will be handled by the context manager
                logger.error(f"Content processing pipeline failed: {e}")
                
                # Return a partial result with error information
                return ProcessingResult(
                    success=False,
                    content_type=context.content_type,
                    chunks=[],
                    processing_time=time.time() - start_time,
                    errors=[str(e)]
                )
    
    async def _enhance_with_ai(
        self,
        result: ProcessingResult,
        context: ProcessingContext
    ) -> ProcessingResult:
        """Enhance processing result with AI-generated metadata."""
        try:
            for chunk in result.chunks:
                if chunk.word_count < 20:  # Skip very short chunks
                    continue
                
                # Generate title if it's a significant chunk
                if chunk.sequence == 0 or chunk.chunk_type == "summary":
                    try:
                        title = await asyncio.wait_for(
                            asyncio.create_task(
                                asyncio.to_thread(ollama_generate_title, chunk.content)
                            ),
                            timeout=30.0
                        )
                        if title:
                            chunk.metadata["ai_generated_title"] = title
                    except asyncio.TimeoutError:
                        chunk.processing_notes.append("AI title generation timed out")
                    except Exception as e:
                        chunk.processing_notes.append(f"AI title generation failed: {e}")
                
                # Generate summary for longer chunks
                if chunk.word_count > 100:
                    try:
                        ai_result = await asyncio.wait_for(
                            asyncio.create_task(
                                asyncio.to_thread(ollama_summarize, chunk.content)
                            ),
                            timeout=60.0
                        )
                        
                        if ai_result and isinstance(ai_result, dict):
                            if ai_result.get("summary"):
                                chunk.metadata["ai_summary"] = ai_result["summary"]
                            if ai_result.get("tags"):
                                chunk.metadata["ai_tags"] = ai_result["tags"][:5]  # Limit tags
                            if ai_result.get("actions"):
                                chunk.metadata["ai_actions"] = ai_result["actions"][:3]  # Limit actions
                                
                    except asyncio.TimeoutError:
                        chunk.processing_notes.append("AI summarization timed out")
                    except Exception as e:
                        chunk.processing_notes.append(f"AI summarization failed: {e}")
            
            result.metadata["ai_enhanced"] = True
            
        except Exception as e:
            result.warnings.append(f"AI enhancement failed: {e}")
            result.metadata["ai_enhanced"] = False
        
        return result
    
    async def _generate_embeddings(self, result: ProcessingResult) -> ProcessingResult:
        """Generate embeddings for content chunks."""
        try:
            for chunk in result.chunks:
                if chunk.word_count < 5:  # Skip very short chunks
                    continue
                
                try:
                    # Combine content with title/summary for better embeddings
                    embedding_text = chunk.content
                    if chunk.metadata.get("ai_generated_title"):
                        embedding_text = chunk.metadata["ai_generated_title"] + "\n\n" + embedding_text
                    elif chunk.metadata.get("ai_summary"):
                        embedding_text = chunk.metadata["ai_summary"] + "\n\n" + embedding_text
                    
                    # Generate embedding
                    embedding = self.embedder.embed(embedding_text[:8000])  # Limit size
                    chunk.metadata["embedding"] = embedding
                    chunk.metadata["embedding_model"] = self.embedder.model_name
                    
                except Exception as e:
                    chunk.processing_notes.append(f"Embedding generation failed: {e}")
            
            result.metadata["embeddings_generated"] = True
            
        except Exception as e:
            result.warnings.append(f"Embedding generation failed: {e}")
            result.metadata["embeddings_generated"] = False
        
        return result
    
    async def _optimize_content(
        self,
        result: ProcessingResult,
        context: ProcessingContext
    ) -> ProcessingResult:
        """Optimize content based on processing context and requirements."""
        try:
            # Sort chunks by quality score if needed
            if len(result.chunks) > 1 and context.priority == ContentPriority.HIGH:
                result.chunks.sort(key=lambda x: x.quality_score, reverse=True)
                # Update sequence numbers
                for i, chunk in enumerate(result.chunks):
                    chunk.sequence = i
            
            # Remove very low quality chunks if we have better ones
            if len(result.chunks) > 1:
                high_quality_chunks = [c for c in result.chunks if c.quality_score >= 0.5]
                if high_quality_chunks and len(high_quality_chunks) >= len(result.chunks) * 0.7:
                    removed_count = len(result.chunks) - len(high_quality_chunks)
                    result.chunks = high_quality_chunks
                    if removed_count > 0:
                        result.warnings.append(f"Removed {removed_count} low-quality chunks")
            
            # Add cross-references between related chunks
            if len(result.chunks) > 1:
                for i, chunk in enumerate(result.chunks):
                    related_chunks = []
                    for j, other_chunk in enumerate(result.chunks):
                        if i != j:
                            # Simple content similarity (could be enhanced)
                            common_words = set(chunk.content.lower().split()) & set(other_chunk.content.lower().split())
                            if len(common_words) >= 5:  # At least 5 common words
                                related_chunks.append(j)
                    
                    if related_chunks:
                        chunk.metadata["related_chunks"] = related_chunks[:3]  # Limit to top 3
            
            result.metadata["content_optimized"] = True
            
        except Exception as e:
            result.warnings.append(f"Content optimization failed: {e}")
            result.metadata["content_optimized"] = False
        
        return result
    
    def _update_statistics(self, result: ProcessingResult, context: ProcessingContext):
        """Update processing statistics."""
        self.stats["total_processed"] += 1
        
        # By content type
        content_type_key = context.content_type.value
        self.stats["by_content_type"][content_type_key] = \
            self.stats["by_content_type"].get(content_type_key, 0) + 1
        
        # Processing time
        total_time = self.stats["avg_processing_time"] * (self.stats["total_processed"] - 1)
        self.stats["avg_processing_time"] = (total_time + result.processing_time) / self.stats["total_processed"]
        
        # Quality scores
        if result.quality_metrics.get("overall_score"):
            self.stats["quality_scores"].append(result.quality_metrics["overall_score"])
            # Keep only last 100 scores
            if len(self.stats["quality_scores"]) > 100:
                self.stats["quality_scores"] = self.stats["quality_scores"][-100:]
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        stats = self.stats.copy()
        
        if self.stats["quality_scores"]:
            stats["avg_quality_score"] = sum(self.stats["quality_scores"]) / len(self.stats["quality_scores"])
            stats["min_quality_score"] = min(self.stats["quality_scores"])
            stats["max_quality_score"] = max(self.stats["quality_scores"])
        
        return stats


# Global pipeline instance
_pipeline: Optional[ContentProcessingPipeline] = None


def get_content_processing_pipeline() -> ContentProcessingPipeline:
    """Get the global content processing pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = ContentProcessingPipeline()
    return _pipeline