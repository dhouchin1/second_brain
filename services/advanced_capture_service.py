# ──────────────────────────────────────────────────────────────────────────────
# File: services/advanced_capture_service.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Advanced Capture Service for Second Brain

Handles complex capture scenarios including OCR, PDF processing, image analysis,
video extraction, and multi-modal content processing.
"""

import base64
import io
import json
import logging
import hashlib
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import requests

# Optional dependencies for advanced features
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    YOUTUBE_API_AVAILABLE = False

from config import settings
from services.embeddings import Embeddings
from llm_utils import ollama_summarize, ollama_generate_title

logger = logging.getLogger(__name__)

@dataclass
class CaptureResult:
    """Result from capture operation."""
    success: bool
    note_id: Optional[int] = None
    title: str = ""
    content: str = ""
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    error: Optional[str] = None
    processing_time: float = 0.0

@dataclass
class CaptureOptions:
    """Options for capture processing."""
    enable_ocr: bool = True
    enable_ai_processing: bool = True
    enable_embeddings: bool = True
    custom_tags: List[str] = None
    take_screenshot: bool = True
    extract_images: bool = True
    quality: str = "high"  # low, medium, high
    language: str = "en"

class AdvancedCaptureService:
    """Service for advanced content capture and processing."""
    
    def __init__(self, get_conn_func):
        """Initialize with database connection function."""
        self.get_conn = get_conn_func
        self.embedder = Embeddings()
        
        # Check feature availability
        self.features = {
            "ocr": TESSERACT_AVAILABLE,
            "pdf": PYMUPDF_AVAILABLE,
            "image_processing": PIL_AVAILABLE,
            "youtube": YOUTUBE_API_AVAILABLE
        }
        
        logger.info(f"Advanced capture features: {self.features}")
    
    def get_feature_availability(self) -> Dict[str, bool]:
        """Get availability of advanced features."""
        return self.features.copy()
    
    async def capture_screenshot_with_ocr(
        self, 
        image_data: str, 
        options: CaptureOptions = None
    ) -> CaptureResult:
        """Process screenshot with OCR extraction."""
        start_time = datetime.now()
        
        if not options:
            options = CaptureOptions()
        
        try:
            if not self.features["ocr"] or not self.features["image_processing"]:
                return CaptureResult(
                    success=False,
                    error="OCR not available. Install: pip install pytesseract pillow"
                )
            
            # Decode base64 image
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Perform OCR
            logger.info("Performing OCR on screenshot")
            extracted_text = pytesseract.image_to_string(
                image, 
                lang=options.language,
                config='--oem 3 --psm 6'  # Optimized for general text
            ).strip()
            
            if not extracted_text:
                extracted_text = "Screenshot captured (no text detected)"
            
            # Generate title and process with AI
            title = "Screenshot with OCR"
            tags = ["screenshot", "ocr"] + (options.custom_tags or [])
            
            if options.enable_ai_processing and extracted_text != "Screenshot captured (no text detected)":
                try:
                    title = ollama_generate_title(extracted_text) or title
                    ai_result = ollama_summarize(extracted_text)
                    if ai_result.get("tags"):
                        tags.extend(ai_result["tags"])
                except Exception as e:
                    logger.warning(f"AI processing failed: {e}")
            
            # Save to database
            note_id = await self._save_note(
                title=title,
                content=extracted_text,
                tags=tags,
                metadata={
                    "content_type": "screenshot_ocr",
                    "ocr_language": options.language,
                    "image_format": "png",
                    "processing_method": "pytesseract",
                    "has_image_data": True
                },
                image_data=image_data if len(image_data) < 1000000 else None,  # Store if < 1MB
                options=options
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return CaptureResult(
                success=True,
                note_id=note_id,
                title=title,
                content=extracted_text,
                tags=tags,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Screenshot OCR failed: {e}")
            return CaptureResult(
                success=False,
                error=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def capture_pdf_content(
        self, 
        pdf_url: str = None, 
        pdf_data: bytes = None,
        options: CaptureOptions = None
    ) -> CaptureResult:
        """Extract content from PDF file."""
        start_time = datetime.now()
        
        if not options:
            options = CaptureOptions()
        
        try:
            if not self.features["pdf"]:
                return CaptureResult(
                    success=False,
                    error="PDF processing not available. Install: pip install PyMuPDF"
                )
            
            # Get PDF data
            if pdf_url:
                logger.info(f"Downloading PDF from: {pdf_url}")
                response = requests.get(pdf_url, timeout=30)
                response.raise_for_status()
                pdf_data = response.content
            
            if not pdf_data:
                return CaptureResult(
                    success=False,
                    error="No PDF data provided"
                )
            
            # Process PDF
            logger.info("Extracting PDF content")
            with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_file:
                temp_file.write(pdf_data)
                temp_file.flush()
                
                doc = fitz.open(temp_file.name)
                
                # Extract text and metadata
                text_content = ""
                page_count = len(doc)
                images_found = 0
                
                for page_num in range(page_count):
                    page = doc[page_num]
                    
                    # Extract text
                    page_text = page.get_text().strip()
                    if page_text:
                        text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    
                    # Count images
                    images_found += len(page.get_images())
                
                # Get PDF metadata
                metadata = doc.metadata
                doc.close()
                
                if not text_content.strip():
                    text_content = f"PDF document with {page_count} pages (no extractable text found)"
                
                # Generate title
                title = metadata.get('title', '') or ollama_generate_title(text_content[:500]) or "PDF Document"
                
                # Process with AI
                tags = ["pdf", "document"] + (options.custom_tags or [])
                summary = ""
                
                if options.enable_ai_processing and len(text_content.strip()) > 50:
                    try:
                        ai_result = ollama_summarize(text_content[:5000])  # Limit for AI processing
                        if ai_result.get("summary"):
                            summary = ai_result["summary"]
                        if ai_result.get("tags"):
                            tags.extend(ai_result["tags"])
                    except Exception as e:
                        logger.warning(f"AI processing failed: {e}")
                
                # Combine content
                final_content = text_content
                if summary:
                    final_content = f"**Summary:** {summary}\n\n{text_content}"
                
                # Save to database
                note_id = await self._save_note(
                    title=title,
                    content=final_content,
                    tags=tags,
                    metadata={
                        "content_type": "pdf",
                        "page_count": page_count,
                        "images_found": images_found,
                        "pdf_title": metadata.get('title', ''),
                        "pdf_author": metadata.get('author', ''),
                        "pdf_creator": metadata.get('creator', ''),
                        "pdf_subject": metadata.get('subject', ''),
                        "file_size_bytes": len(pdf_data),
                        "source_url": pdf_url
                    },
                    options=options
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return CaptureResult(
                    success=True,
                    note_id=note_id,
                    title=title,
                    content=final_content,
                    tags=tags,
                    metadata={"page_count": page_count, "summary": summary},
                    processing_time=processing_time
                )
                
        except Exception as e:
            logger.error(f"PDF capture failed: {e}")
            return CaptureResult(
                success=False,
                error=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def capture_youtube_transcript(
        self, 
        video_url: str, 
        options: CaptureOptions = None
    ) -> CaptureResult:
        """Extract transcript from YouTube video."""
        start_time = datetime.now()
        
        if not options:
            options = CaptureOptions()
        
        try:
            if not self.features["youtube"]:
                return CaptureResult(
                    success=False,
                    error="YouTube API not available. Install: pip install youtube-transcript-api"
                )
            
            # Extract video ID
            video_id = self._extract_youtube_id(video_url)
            if not video_id:
                return CaptureResult(
                    success=False,
                    error="Invalid YouTube URL"
                )
            
            logger.info(f"Extracting transcript for video: {video_id}")
            
            # Get transcript
            try:
                transcript = YouTubeTranscriptApi.get_transcript(
                    video_id, 
                    languages=[options.language, 'en']
                )
            except Exception as e:
                # Try auto-generated transcript
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Format transcript
            transcript_text = "\n".join([
                f"[{self._format_timestamp(entry['start'])}] {entry['text']}"
                for entry in transcript
            ])
            
            # Get video info (basic)
            video_info = await self._get_youtube_info(video_url)
            title = video_info.get('title', f'YouTube Video: {video_id}')
            
            # Process with AI
            tags = ["youtube", "video", "transcript"] + (options.custom_tags or [])
            summary = ""
            
            if options.enable_ai_processing:
                try:
                    # Use first part of transcript for AI processing
                    ai_input = transcript_text[:4000]
                    ai_result = ollama_summarize(ai_input, 
                        "Summarize this video transcript and extract key topics and insights.")
                    
                    if ai_result.get("summary"):
                        summary = ai_result["summary"]
                    if ai_result.get("tags"):
                        tags.extend(ai_result["tags"])
                except Exception as e:
                    logger.warning(f"AI processing failed: {e}")
            
            # Format final content
            content = f"**Video:** {title}\n**URL:** {video_url}\n"
            if summary:
                content += f"**Summary:** {summary}\n"
            content += f"\n**Transcript:**\n{transcript_text}"
            
            # Save to database
            note_id = await self._save_note(
                title=f"Transcript: {title}",
                content=content,
                tags=tags,
                metadata={
                    "content_type": "youtube_transcript",
                    "video_id": video_id,
                    "video_url": video_url,
                    "transcript_length": len(transcript),
                    "duration_seconds": transcript[-1]['start'] + transcript[-1].get('duration', 0) if transcript else 0,
                    "language": options.language
                },
                options=options
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return CaptureResult(
                success=True,
                note_id=note_id,
                title=f"Transcript: {title}",
                content=content,
                tags=tags,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"YouTube transcript capture failed: {e}")
            return CaptureResult(
                success=False,
                error=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def capture_bulk_urls(
        self, 
        urls: List[str], 
        options: CaptureOptions = None
    ) -> List[CaptureResult]:
        """Process multiple URLs in batch."""
        if not options:
            options = CaptureOptions()
        
        results = []
        total_urls = len(urls)
        
        logger.info(f"Processing {total_urls} URLs in batch")
        
        for i, url in enumerate(urls):
            try:
                logger.info(f"Processing URL {i+1}/{total_urls}: {url}")
                
                # Determine URL type and process accordingly
                if 'youtube.com/watch' in url or 'youtu.be/' in url:
                    result = await self.capture_youtube_transcript(url, options)
                elif url.lower().endswith('.pdf'):
                    result = await self.capture_pdf_content(pdf_url=url, options=options)
                else:
                    # Use web ingestion service
                    from services.web_ingestion_service import WebIngestionService
                    web_service = WebIngestionService()
                    
                    web_result = await web_service.extract_and_process_url(
                        url, 
                        take_screenshot=options.take_screenshot,
                        ai_processing=options.enable_ai_processing
                    )
                    
                    if web_result:
                        result = CaptureResult(
                            success=True,
                            title=web_result.title,
                            content=web_result.content,
                            tags=["web-content"] + (options.custom_tags or [])
                        )
                    else:
                        result = CaptureResult(success=False, error="Web extraction failed")
                
                results.append(result)
                
                # Small delay to avoid overwhelming servers
                if i < total_urls - 1:
                    import asyncio
                    await asyncio.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Failed to process URL {url}: {e}")
                results.append(CaptureResult(
                    success=False,
                    error=f"Failed to process {url}: {str(e)}"
                ))
        
        successful = sum(1 for r in results if r.success)
        logger.info(f"Bulk URL processing complete: {successful}/{total_urls} successful")
        
        return results
    
    async def _save_note(
        self,
        title: str,
        content: str,
        tags: List[str],
        metadata: Dict[str, Any],
        image_data: str = None,
        options: CaptureOptions = None
    ) -> int:
        """Save processed content to database."""
        conn = self.get_conn()
        cursor = conn.cursor()
        
        try:
            # Clean and prepare data
            tags_str = ", ".join(set(tag.strip() for tag in tags if tag.strip()))
            
            # Add image data to metadata if provided
            if image_data:
                metadata["has_screenshot"] = True
                metadata["screenshot_data"] = image_data[:1000000]  # Limit size
            
            # Insert note
            cursor.execute("""
                INSERT INTO notes (title, body, tags, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                title,
                content,
                tags_str,
                json.dumps(metadata),
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            
            note_id = cursor.lastrowid
            
            # Generate embeddings if enabled
            if options and options.enable_embeddings:
                try:
                    embedding_text = f"{title}\n\n{content}"
                    embedding = self.embedder.embed(embedding_text)
                    
                    # Store in vector table if available
                    try:
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='note_vecs'")
                        if cursor.fetchone():
                            cursor.execute(
                                "INSERT OR REPLACE INTO note_vecs(note_id, embedding) VALUES (?, ?)",
                                (note_id, json.dumps(embedding))
                            )
                    except Exception as e:
                        logger.debug(f"Vector storage not available: {e}")
                        
                except Exception as e:
                    logger.warning(f"Failed to generate embedding: {e}")
            
            conn.commit()
            return note_id
            
        finally:
            conn.close()
    
    def _extract_youtube_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL."""
        import re
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)',
            r'youtube\.com\/embed\/([a-zA-Z0-9_-]+)',
            r'youtube\.com\/v\/([a-zA-Z0-9_-]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as MM:SS timestamp."""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    async def _get_youtube_info(self, url: str) -> Dict[str, Any]:
        """Get basic YouTube video info (title, description)."""
        # This is a simplified implementation
        # In production, you might want to use the YouTube API
        return {
            "title": f"YouTube Video",
            "description": "",
            "channel": ""
        }


def get_advanced_capture_service(get_conn_func):
    """Factory function to get advanced capture service."""
    return AdvancedCaptureService(get_conn_func)