#!/usr/bin/env python3
"""
URL Ingestion Router for Second Brain

FastAPI router for web content ingestion endpoints.
Provides REST API for ingesting URLs, PDFs, and web content.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator

# Import services and dependencies
from services.url_ingestion_service import get_url_ingestion_service, URLIngestionService
from services.unified_capture_service import get_unified_capture_service
from services.auth_service import User
from database import get_db_connection

logger = logging.getLogger(__name__)

# Global variables to hold functions from app.py context
get_current_user = None

router = APIRouter(prefix="/api/ingest", tags=["url-ingestion"])

def init_url_ingestion_router(get_current_user_func):
    """Initialize the URL ingestion router with required dependencies"""
    global get_current_user
    get_current_user = get_current_user_func

# Pydantic models for API
class URLIngestRequest(BaseModel):
    """Request model for URL ingestion"""
    url: str = Field(..., description="URL to ingest")
    context: Optional[str] = Field(None, description="Additional context or source text")
    auto_title: bool = Field(True, description="Auto-generate title from content")
    auto_tags: bool = Field(True, description="Auto-generate tags from content")
    auto_summary: bool = Field(True, description="Auto-generate summary from content")

    @validator('url')
    def validate_url(cls, v):
        """Validate URL format"""
        if not v or not v.strip():
            raise ValueError("URL cannot be empty")

        v = v.strip()
        if not (v.startswith('http://') or v.startswith('https://') or v.startswith('www.')):
            raise ValueError("URL must start with http://, https://, or www.")

        return v

class URLDetectRequest(BaseModel):
    """Request model for URL detection in text"""
    text: str = Field(..., description="Text to scan for URLs")

class URLIngestResponse(BaseModel):
    """Response model for URL ingestion"""
    success: bool
    message: str
    url: str
    note_id: Optional[int] = None
    title: Optional[str] = None
    content_type: Optional[str] = None
    screenshot_path: Optional[str] = None
    file_path: Optional[str] = None
    word_count: Optional[int] = None
    processing_time_seconds: Optional[float] = None
    error: Optional[str] = None

class URLDetectResponse(BaseModel):
    """Response model for URL detection"""
    urls_found: List[str]
    count: int
    text_with_urls: str

class BatchURLIngestRequest(BaseModel):
    """Request model for batch URL ingestion"""
    urls: List[str] = Field(..., description="List of URLs to ingest")
    context: Optional[str] = Field(None, description="Context for all URLs")
    auto_title: bool = Field(True, description="Auto-generate titles")
    auto_tags: bool = Field(True, description="Auto-generate tags")
    auto_summary: bool = Field(True, description="Auto-generate summaries")

    @validator('urls')
    def validate_urls(cls, v):
        """Validate URL list"""
        if not v or len(v) == 0:
            raise ValueError("URLs list cannot be empty")
        if len(v) > 10:  # Limit batch size
            raise ValueError("Maximum 10 URLs allowed per batch")
        return v

class BatchURLIngestResponse(BaseModel):
    """Response model for batch URL ingestion"""
    total_urls: int
    successful: int
    failed: int
    results: List[URLIngestResponse]
    processing_time_seconds: float

@router.post("/url", response_model=URLIngestResponse)
async def ingest_url(
    request: URLIngestRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Ingest content from a single URL

    Supports:
    - Web pages (HTML content extraction)
    - PDF files (download and process)
    - Images (download and OCR if applicable)
    - Documents (various formats)
    """
    import time
    start_time = time.time()

    try:
        logger.info(f"URL ingestion request from user {current_user.id}: {request.url}")

        # Get URL ingestion service
        capture_service = get_unified_capture_service(get_db_connection)
        url_service = get_url_ingestion_service(capture_service)

        # Ingest the URL
        result = await url_service.ingest_url(
            url=request.url,
            user_id=current_user.id,
            source_text=request.context or ""
        )

        processing_time = time.time() - start_time

        if result["success"]:
            return URLIngestResponse(
                success=True,
                message="URL content ingested successfully",
                url=request.url,
                note_id=result.get("note_id"),
                title=result.get("title"),
                content_type=result.get("content_type"),
                screenshot_path=result.get("screenshot_path"),
                file_path=result.get("file_path"),
                word_count=result.get("word_count"),
                processing_time_seconds=round(processing_time, 2)
            )
        else:
            return URLIngestResponse(
                success=False,
                message="Failed to ingest URL content",
                url=request.url,
                error=result.get("error"),
                processing_time_seconds=round(processing_time, 2)
            )

    except Exception as e:
        logger.error(f"Error in URL ingestion endpoint: {e}")
        processing_time = time.time() - start_time

        return URLIngestResponse(
            success=False,
            message="URL ingestion failed",
            url=request.url,
            error=str(e),
            processing_time_seconds=round(processing_time, 2)
        )

@router.post("/detect-urls", response_model=URLDetectResponse)
async def detect_urls(
    request: URLDetectRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Detect URLs in provided text

    Useful for:
    - Previewing URLs before ingestion
    - Validating text contains URLs
    - Extracting URLs from larger text blocks
    """
    try:
        # Get URL ingestion service
        capture_service = get_unified_capture_service(get_db_connection)
        url_service = get_url_ingestion_service(capture_service)

        # Detect URLs
        urls = url_service.detect_urls(request.text)

        return URLDetectResponse(
            urls_found=urls,
            count=len(urls),
            text_with_urls=request.text
        )

    except Exception as e:
        logger.error(f"Error in URL detection endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=BatchURLIngestResponse)
async def ingest_batch_urls(
    request: BatchURLIngestRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Ingest multiple URLs in batch

    Processes URLs concurrently for better performance.
    Limited to 10 URLs per batch to prevent resource exhaustion.
    """
    import time
    start_time = time.time()

    try:
        logger.info(f"Batch URL ingestion request from user {current_user.id}: {len(request.urls)} URLs")

        # Get URL ingestion service
        capture_service = get_unified_capture_service(get_db_connection)
        url_service = get_url_ingestion_service(capture_service)

        # Process URLs concurrently
        tasks = []
        for url in request.urls:
            task = url_service.ingest_url(
                url=url,
                user_id=current_user.id,
                source_text=request.context or ""
            )
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        response_results = []
        successful = 0
        failed = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                response_results.append(URLIngestResponse(
                    success=False,
                    message="URL ingestion failed",
                    url=request.urls[i],
                    error=str(result)
                ))
                failed += 1
            elif result.get("success"):
                response_results.append(URLIngestResponse(
                    success=True,
                    message="URL content ingested successfully",
                    url=request.urls[i],
                    note_id=result.get("note_id"),
                    title=result.get("title"),
                    content_type=result.get("content_type"),
                    screenshot_path=result.get("screenshot_path"),
                    file_path=result.get("file_path"),
                    word_count=result.get("word_count")
                ))
                successful += 1
            else:
                response_results.append(URLIngestResponse(
                    success=False,
                    message="Failed to ingest URL content",
                    url=request.urls[i],
                    error=result.get("error")
                ))
                failed += 1

        processing_time = time.time() - start_time

        return BatchURLIngestResponse(
            total_urls=len(request.urls),
            successful=successful,
            failed=failed,
            results=response_results,
            processing_time_seconds=round(processing_time, 2)
        )

    except Exception as e:
        logger.error(f"Error in batch URL ingestion endpoint: {e}")
        processing_time = time.time() - start_time

        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "processing_time_seconds": round(processing_time, 2)
            }
        )

@router.get("/status")
async def get_ingestion_status(
    current_user: User = Depends(get_current_user)
):
    """
    Get URL ingestion service status and capabilities

    Returns information about:
    - Service availability
    - Supported content types
    - Processing capabilities
    """
    try:
        # Check if Playwright is available
        from web_extractor import PLAYWRIGHT_AVAILABLE

        # Get basic stats
        capture_service = get_unified_capture_service(get_db_connection)
        url_service = get_url_ingestion_service(capture_service)

        return {
            "status": "available",
            "playwright_available": PLAYWRIGHT_AVAILABLE,
            "supported_content_types": [
                "web_pages",
                "pdf_files",
                "images",
                "documents"
            ],
            "capabilities": {
                "web_scraping": PLAYWRIGHT_AVAILABLE,
                "pdf_processing": True,
                "image_ocr": True,
                "screenshot_capture": PLAYWRIGHT_AVAILABLE,
                "batch_processing": True,
                "url_detection": True
            },
            "limits": {
                "max_batch_size": 10,
                "timeout_seconds": 30,
                "max_file_size_mb": 50
            }
        }

    except Exception as e:
        logger.error(f"Error checking ingestion status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "playwright_available": False
        }

@router.post("/preview")
async def preview_url_content(
    request: URLIngestRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Preview URL content without saving to database

    Useful for:
    - Checking if URL can be processed
    - Previewing extracted content
    - Validating URLs before batch ingestion
    """
    try:
        logger.info(f"URL preview request from user {current_user.id}: {request.url}")

        # Get URL ingestion service
        capture_service = get_unified_capture_service(get_db_connection)
        url_service = get_url_ingestion_service(capture_service)

        # Detect content type
        content_type = url_service._detect_content_type(request.url)

        # For web pages, we can extract content for preview
        if content_type == "webpage":
            # Extract content using web extractor only
            extraction_result = await url_service.web_extractor.extract_content(request.url)

            if extraction_result.success:
                return {
                    "success": True,
                    "url": request.url,
                    "content_type": "webpage",
                    "title": extraction_result.get_best_title(),
                    "description": extraction_result.get_best_description(),
                    "word_count": len(extraction_result.text_content.split()) if extraction_result.text_content else 0,
                    "has_screenshot": bool(extraction_result.screenshot_path),
                    "preview_content": extraction_result.text_content[:500] + "..." if extraction_result.text_content and len(extraction_result.text_content) > 500 else extraction_result.text_content,
                    "metadata": extraction_result.metadata.to_dict() if extraction_result.metadata else {}
                }
            else:
                return {
                    "success": False,
                    "url": request.url,
                    "error": extraction_result.error_message
                }
        else:
            # For other content types, return basic info
            return {
                "success": True,
                "url": request.url,
                "content_type": content_type,
                "title": f"{content_type.title()} from URL",
                "description": f"Detected {content_type} content that can be downloaded and processed",
                "preview_content": f"This URL appears to contain a {content_type} file that will be downloaded and processed when ingested."
            }

    except Exception as e:
        logger.error(f"Error in URL preview endpoint: {e}")
        return {
            "success": False,
            "url": request.url,
            "error": str(e)
        }