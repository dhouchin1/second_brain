#!/usr/bin/env python3
"""
URL Ingestion Service for Second Brain

Comprehensive service for ingesting web content, PDFs, and other online resources.
Integrates with existing capture system and embedding generation.
"""

import re
import asyncio
import logging
import hashlib
import json
import tempfile
import os
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from urllib.parse import urlparse, urljoin, unquote
from pathlib import Path

import requests
import aiohttp
from bs4 import BeautifulSoup

# Import existing models and services
from web_content_models import WebExtractionResult, WebMetadata, WebContentType, ExtractionStatus
from web_extractor import WebContentExtractor
from services.unified_capture_service import UnifiedCaptureService, UnifiedCaptureRequest, CaptureContentType, CaptureSourceType
from config import settings

logger = logging.getLogger(__name__)

class URLIngestionService:
    """Comprehensive URL ingestion service"""

    def __init__(self, capture_service: UnifiedCaptureService):
        self.capture_service = capture_service
        self.web_extractor = WebContentExtractor()
        self.session_timeout = aiohttp.ClientTimeout(total=30)

        # Supported content types for different processing
        self.pdf_extensions = {'.pdf'}
        self.document_extensions = {'.doc', '.docx', '.txt', '.md', '.rst'}
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg'}

        # Cache directory for downloaded files
        self.cache_dir = Path(settings.base_dir) / "web_cache"
        self.cache_dir.mkdir(exist_ok=True)

    async def ingest_url(self, url: str, user_id: int, source_text: str = "") -> Dict[str, Any]:
        """
        Main method to ingest content from a URL

        Args:
            url: The URL to ingest
            user_id: User ID for the capture
            source_text: Original text containing the URL

        Returns:
            Dictionary with ingestion results
        """
        try:
            logger.info(f"Starting URL ingestion for: {url}")

            # Validate and normalize URL
            normalized_url = self._normalize_url(url)
            if not normalized_url:
                return {"success": False, "error": "Invalid URL"}

            # Determine content type from URL
            content_type = self._detect_content_type(normalized_url)

            # Route to appropriate handler
            if content_type == "pdf":
                result = await self._ingest_pdf(normalized_url, user_id, source_text)
            elif content_type == "image":
                result = await self._ingest_image(normalized_url, user_id, source_text)
            elif content_type == "document":
                result = await self._ingest_document(normalized_url, user_id, source_text)
            else:
                result = await self._ingest_webpage(normalized_url, user_id, source_text)

            logger.info(f"URL ingestion completed for {url}: {result['success']}")
            return result

        except Exception as e:
            logger.error(f"Error ingesting URL {url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }

    def detect_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        # Improved URL regex pattern
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )

        urls = url_pattern.findall(text)

        # Also look for www. patterns without protocol
        www_pattern = re.compile(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        www_urls = www_pattern.findall(text)

        # Add https:// to www URLs
        for www_url in www_urls:
            if www_url not in text.replace("https://", "").replace("http://", ""):
                urls.append(f"https://{www_url}")

        return list(set(urls))  # Remove duplicates

    def _normalize_url(self, url: str) -> Optional[str]:
        """Normalize and validate URL"""
        try:
            # Add protocol if missing
            if not url.startswith(('http://', 'https://')):
                if url.startswith('www.'):
                    url = f"https://{url}"
                else:
                    url = f"https://{url}"

            # Parse and validate
            parsed = urlparse(url)
            if not parsed.netloc:
                return None

            # Reconstruct clean URL
            normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if parsed.query:
                normalized += f"?{parsed.query}"
            if parsed.fragment:
                normalized += f"#{parsed.fragment}"

            return normalized

        except Exception as e:
            logger.error(f"Error normalizing URL {url}: {e}")
            return None

    def _detect_content_type(self, url: str) -> str:
        """Detect content type from URL"""
        parsed = urlparse(url)
        path = parsed.path.lower()

        # Get file extension
        if '.' in path:
            ext = Path(path).suffix.lower()

            if ext in self.pdf_extensions:
                return "pdf"
            elif ext in self.image_extensions:
                return "image"
            elif ext in self.document_extensions:
                return "document"

        # Check for known patterns
        if any(pattern in url.lower() for pattern in ['/pdf/', '.pdf?', '.pdf#']):
            return "pdf"

        return "webpage"

    async def _ingest_webpage(self, url: str, user_id: int, source_text: str) -> Dict[str, Any]:
        """Ingest a web page using Playwright"""
        try:
            # Extract content using web extractor
            extraction_result = await self.web_extractor.extract_content(url)

            if not extraction_result.success:
                return {
                    "success": False,
                    "error": extraction_result.error_message or "Web extraction failed",
                    "url": url
                }

            # Create enhanced note content
            content = self._create_web_note_content(extraction_result, source_text)

            # Prepare metadata
            metadata = {
                "source_url": url,
                "content_type": "web_page",
                "extraction_method": "playwright",
                "web_metadata": extraction_result.metadata if extraction_result.metadata else {},
                "screenshot_path": extraction_result.screenshot_path
            }

            # Create unified capture request (use API source type to avoid circular call)
            capture_request = UnifiedCaptureRequest(
                content_type=CaptureContentType.TEXT,
                source_type=CaptureSourceType.API,
                primary_content=content,
                metadata=metadata
            )

            # Use unified capture service to save
            capture_result = await self.capture_service.unified_capture(capture_request, str(user_id))

            return {
                "success": capture_result.success,
                "note_id": capture_result.note_id,
                "title": capture_result.title or extraction_result.title,
                "content_type": "webpage",
                "url": url,
                "screenshot_path": extraction_result.screenshot_path,
                "word_count": len(extraction_result.text_content.split()) if extraction_result.text_content else 0
            }

        except Exception as e:
            logger.error(f"Error ingesting webpage {url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }

    async def _ingest_pdf(self, url: str, user_id: int, source_text: str) -> Dict[str, Any]:
        """Ingest a PDF file from URL"""
        try:
            # Download PDF to temp file
            temp_path = await self._download_file(url, '.pdf')
            if not temp_path:
                return {"success": False, "error": "Failed to download PDF", "url": url}

            try:
                # Use existing advanced capture service for PDF processing
                with open(temp_path, 'rb') as pdf_file:
                    files = [('files', ('document.pdf', pdf_file, 'application/pdf'))]

                    # Create form data
                    form_data = {
                        'text': source_text or f"PDF document from {url}",
                        'user_id': str(user_id),
                        'source_url': url,
                        'content_type': 'pdf_url'
                    }

                    # Process through existing capture system
                    capture_result = await self.capture_service.capture_file(
                        content=form_data['text'],
                        files=files,
                        user_id=user_id,
                        metadata={
                            "source_url": url,
                            "content_type": "pdf_url",
                            "original_filename": Path(urlparse(url).path).name or "document.pdf"
                        }
                    )

                    return {
                        "success": True,
                        "note_id": capture_result.get("note_id"),
                        "title": f"PDF: {Path(urlparse(url).path).name or 'Document'}",
                        "content_type": "pdf",
                        "url": url,
                        "file_path": capture_result.get("file_path")
                    }

            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            logger.error(f"Error ingesting PDF {url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }

    async def _ingest_image(self, url: str, user_id: int, source_text: str) -> Dict[str, Any]:
        """Ingest an image from URL"""
        try:
            # Download image to temp file
            parsed_url = urlparse(url)
            extension = Path(parsed_url.path).suffix or '.jpg'
            temp_path = await self._download_file(url, extension)

            if not temp_path:
                return {"success": False, "error": "Failed to download image", "url": url}

            try:
                # Use existing capture service for image processing (OCR, etc.)
                with open(temp_path, 'rb') as img_file:
                    files = [('files', (f'image{extension}', img_file, f'image/{extension[1:]}'))]

                    capture_result = await self.capture_service.capture_file(
                        content=source_text or f"Image from {url}",
                        files=files,
                        user_id=user_id,
                        metadata={
                            "source_url": url,
                            "content_type": "image_url",
                            "original_filename": Path(parsed_url.path).name or f"image{extension}"
                        }
                    )

                    return {
                        "success": True,
                        "note_id": capture_result.get("note_id"),
                        "title": f"Image: {Path(parsed_url.path).name or 'Image'}",
                        "content_type": "image",
                        "url": url,
                        "file_path": capture_result.get("file_path")
                    }

            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            logger.error(f"Error ingesting image {url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }

    async def _ingest_document(self, url: str, user_id: int, source_text: str) -> Dict[str, Any]:
        """Ingest a document file from URL"""
        try:
            # Download document
            parsed_url = urlparse(url)
            extension = Path(parsed_url.path).suffix or '.txt'
            temp_path = await self._download_file(url, extension)

            if not temp_path:
                return {"success": False, "error": "Failed to download document", "url": url}

            try:
                # Read content based on file type
                if extension == '.txt':
                    with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                else:
                    # For other document types, read as binary and let capture service handle
                    with open(temp_path, 'rb') as doc_file:
                        files = [('files', (f'document{extension}', doc_file, 'application/octet-stream'))]

                        capture_result = await self.capture_service.capture_file(
                            content=source_text or f"Document from {url}",
                            files=files,
                            user_id=user_id,
                            metadata={
                                "source_url": url,
                                "content_type": "document_url",
                                "original_filename": Path(parsed_url.path).name or f"document{extension}"
                            }
                        )

                        return {
                            "success": True,
                            "note_id": capture_result.get("note_id"),
                            "title": f"Document: {Path(parsed_url.path).name or 'Document'}",
                            "content_type": "document",
                            "url": url,
                            "file_path": capture_result.get("file_path")
                        }

                # For text files, process content directly
                enhanced_content = f"{source_text}\n\n---\n\n{content}" if source_text else content

                capture_result = await self.capture_service.capture_text(
                    content=enhanced_content,
                    user_id=user_id,
                    metadata={
                        "source_url": url,
                        "content_type": "text_document_url",
                        "original_filename": Path(parsed_url.path).name or "document.txt"
                    },
                    auto_title=True,
                    auto_tags=True,
                    auto_summary=True
                )

                return {
                    "success": True,
                    "note_id": capture_result.get("note_id"),
                    "title": f"Document: {Path(parsed_url.path).name or 'Text Document'}",
                    "content_type": "text_document",
                    "url": url,
                    "word_count": len(content.split())
                }

            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            logger.error(f"Error ingesting document {url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }

    async def _download_file(self, url: str, extension: str) -> Optional[str]:
        """Download file from URL to temporary location"""
        try:
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        # Create temp file
                        temp_fd, temp_path = tempfile.mkstemp(suffix=extension, dir=self.cache_dir)

                        try:
                            with os.fdopen(temp_fd, 'wb') as temp_file:
                                async for chunk in response.content.iter_chunked(8192):
                                    temp_file.write(chunk)

                            return temp_path

                        except Exception as e:
                            os.unlink(temp_path)
                            raise e
                    else:
                        logger.error(f"Failed to download {url}: HTTP {response.status}")
                        return None

        except Exception as e:
            logger.error(f"Error downloading file from {url}: {e}")
            return None

    def _create_web_note_content(self, extraction_result, source_text: str) -> str:
        """Create formatted note content from web extraction result"""
        content_parts = []

        # Add source context if provided
        if source_text and source_text.strip() != extraction_result.url:
            content_parts.append(f"**Source Context:**\n{source_text}\n")

        # Add URL
        content_parts.append(f"**URL:** {extraction_result.url}\n")

        # Add title if different from what will be auto-generated
        title = extraction_result.title
        if title and title != extraction_result.url:
            content_parts.append(f"**Title:** {title}\n")

        # Add metadata if substantial
        if extraction_result.metadata:
            metadata_parts = []
            if extraction_result.metadata.get('author'):
                metadata_parts.append(f"Author: {extraction_result.metadata['author']}")
            if extraction_result.metadata.get('publish_date'):
                metadata_parts.append(f"Published: {extraction_result.metadata['publish_date']}")
            if extraction_result.metadata.get('og_site_name'):
                metadata_parts.append(f"Site: {extraction_result.metadata['og_site_name']}")

            if metadata_parts:
                content_parts.append(f"**Metadata:** {' | '.join(metadata_parts)}\n")

        # Add main content
        if extraction_result.content:
            content_parts.append("---\n")
            content_parts.append(extraction_result.content)
        elif extraction_result.text_content:
            content_parts.append("---\n")
            content_parts.append(extraction_result.text_content)

        return "\n".join(content_parts)

# Singleton instance
_url_ingestion_service = None

def get_url_ingestion_service(capture_service: UnifiedCaptureService = None) -> URLIngestionService:
    """Get or create URL ingestion service instance"""
    global _url_ingestion_service

    if _url_ingestion_service is None:
        if capture_service is None:
            from services.unified_capture_service import get_unified_capture_service
            from database import get_db_connection
            capture_service = get_unified_capture_service(get_db_connection)

        _url_ingestion_service = URLIngestionService(capture_service)

    return _url_ingestion_service