# ──────────────────────────────────────────────────────────────────────────────
# File: services/unified_capture_router.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Unified Capture API Router

REST endpoints for the unified capture system that orchestrates all content
ingestion sources through a single, consistent API.
"""

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import logging
from datetime import datetime

from services.unified_capture_service import (
    get_unified_capture_service, 
    UnifiedCaptureRequest,
    UnifiedCaptureResponse,
    CaptureSourceType,
    CaptureContentType
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/unified-capture", tags=["unified-capture"])

# Request models for the API
class TextCaptureRequest(BaseModel):
    """Text content capture request."""
    content: str = Field(..., description="Text content to capture")
    source: str = Field(default="api", description="Source of the capture")
    title: Optional[str] = Field(None, description="Custom title for the note")
    tags: List[str] = Field(default_factory=list, description="Custom tags")
    note_type: str = Field(default="text", description="Type of note")
    auto_tag: bool = Field(default=True, description="Auto-generate tags")
    generate_summary: bool = Field(default=True, description="Generate AI summary")
    extract_actions: bool = Field(default=True, description="Extract action items")
    priority: int = Field(default=1, description="Processing priority (1-3)")
    user_context: Optional[Dict[str, Any]] = Field(None, description="User context data")
    location_data: Optional[Dict[str, Any]] = Field(None, description="Location information")

class AudioCaptureRequest(BaseModel):
    """Audio content capture request."""
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio data")
    transcription: Optional[str] = Field(None, description="Pre-existing transcription")
    source: str = Field(default="api", description="Source of the capture")
    title: Optional[str] = Field(None, description="Custom title")
    tags: List[str] = Field(default_factory=list, description="Custom tags")
    auto_tag: bool = Field(default=True, description="Auto-generate tags")
    generate_summary: bool = Field(default=True, description="Generate AI summary")
    user_context: Optional[Dict[str, Any]] = Field(None, description="User context data")
    location_data: Optional[Dict[str, Any]] = Field(None, description="Location information")

class ImageCaptureRequest(BaseModel):
    """Image/OCR capture request."""
    image_data: str = Field(..., description="Base64 encoded image data")
    source: str = Field(default="api", description="Source of the capture")
    title: Optional[str] = Field(None, description="Custom title")
    tags: List[str] = Field(default_factory=list, description="Custom tags")
    auto_tag: bool = Field(default=True, description="Auto-generate tags")
    generate_summary: bool = Field(default=True, description="Generate AI summary")
    user_context: Optional[Dict[str, Any]] = Field(None, description="User context data")
    location_data: Optional[Dict[str, Any]] = Field(None, description="Location information")

class URLCaptureRequest(BaseModel):
    """URL/web content capture request."""
    url: str = Field(..., description="URL to capture")
    selected_text: Optional[str] = Field(None, description="Selected text from page")
    page_title: Optional[str] = Field(None, description="Page title")
    source: str = Field(default="api", description="Source of the capture")
    tags: List[str] = Field(default_factory=list, description="Custom tags")
    auto_tag: bool = Field(default=True, description="Auto-generate tags")
    generate_summary: bool = Field(default=True, description="Generate AI summary")
    user_context: Optional[Dict[str, Any]] = Field(None, description="User context data")

class PDFCaptureRequest(BaseModel):
    """PDF content capture request."""
    file_data: str = Field(..., description="Base64 encoded PDF data")
    filename: str = Field(..., description="PDF filename")
    source: str = Field(default="api", description="Source of the capture")
    title: Optional[str] = Field(None, description="Custom title")
    tags: List[str] = Field(default_factory=list, description="Custom tags")
    auto_tag: bool = Field(default=True, description="Auto-generate tags")
    generate_summary: bool = Field(default=True, description="Generate AI summary")
    user_context: Optional[Dict[str, Any]] = Field(None, description="User context data")

class BatchCaptureRequest(BaseModel):
    """Batch capture request."""
    requests: List[Dict[str, Any]] = Field(..., description="List of capture requests")
    max_concurrency: int = Field(default=5, description="Maximum concurrent processing")

class CaptureResponse(BaseModel):
    """Unified capture response."""
    success: bool
    note_id: Optional[int] = None
    title: Optional[str] = None
    content_preview: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    action_items: List[str] = Field(default_factory=list)
    processing_time: float = 0.0
    source_service: Optional[str] = None
    error: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

# Global service instance
_service = None

def init_unified_capture_router(get_conn_func):
    """Initialize the router with database connection."""
    global _service
    _service = get_unified_capture_service(get_conn_func)

@router.get("/integrations")
async def get_supported_integrations():
    """Get information about supported capture integrations."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        integrations = _service.get_supported_integrations()
        return JSONResponse(content={
            "success": True,
            "data": integrations
        })
    except Exception as e:
        logger.error(f"Failed to get integrations info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_processing_stats():
    """Get unified capture processing statistics."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        stats = _service.get_processing_stats()
        return JSONResponse(content={
            "success": True,
            "data": stats
        })
    except Exception as e:
        logger.error(f"Failed to get processing stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/text")
async def capture_text(request: TextCaptureRequest) -> CaptureResponse:
    """Capture text content through unified API."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Convert to unified request
        unified_request = UnifiedCaptureRequest(
            content_type=CaptureContentType.TEXT,
            source_type=CaptureSourceType(request.source) if request.source in [s.value for s in CaptureSourceType] else CaptureSourceType.API,
            primary_content=request.content,
            metadata={"note_type": request.note_type},
            auto_tag=request.auto_tag,
            generate_summary=request.generate_summary,
            extract_actions=request.extract_actions,
            user_context=request.user_context,
            location_data=request.location_data,
            custom_title=request.title,
            custom_tags=request.tags,
            processing_priority=request.priority
        )
        
        result = await _service.unified_capture(unified_request)
        
        return CaptureResponse(
            success=result.success,
            note_id=result.note_id,
            title=result.title,
            content_preview=result.content_preview,
            tags=result.tags or [],
            summary=result.summary,
            action_items=result.action_items or [],
            processing_time=result.processing_time,
            source_service=result.source_service,
            error=result.error,
            warnings=result.warnings or []
        )
        
    except Exception as e:
        logger.error(f"Text capture failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/audio")
async def capture_audio(request: AudioCaptureRequest) -> CaptureResponse:
    """Capture audio content through unified API."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        unified_request = UnifiedCaptureRequest(
            content_type=CaptureContentType.AUDIO,
            source_type=CaptureSourceType(request.source) if request.source in [s.value for s in CaptureSourceType] else CaptureSourceType.API,
            primary_content=request.transcription or "",
            metadata={},
            audio_data=request.audio_data,
            auto_tag=request.auto_tag,
            generate_summary=request.generate_summary,
            user_context=request.user_context,
            location_data=request.location_data,
            custom_title=request.title,
            custom_tags=request.tags
        )
        
        result = await _service.unified_capture(unified_request)
        
        return CaptureResponse(
            success=result.success,
            note_id=result.note_id,
            title=result.title,
            content_preview=result.content_preview,
            tags=result.tags or [],
            summary=result.summary,
            action_items=result.action_items or [],
            processing_time=result.processing_time,
            source_service=result.source_service,
            error=result.error,
            warnings=result.warnings or []
        )
        
    except Exception as e:
        logger.error(f"Audio capture failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/image")
async def capture_image(request: ImageCaptureRequest) -> CaptureResponse:
    """Capture image/OCR content through unified API."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        unified_request = UnifiedCaptureRequest(
            content_type=CaptureContentType.PHOTO_OCR,
            source_type=CaptureSourceType(request.source) if request.source in [s.value for s in CaptureSourceType] else CaptureSourceType.API,
            primary_content="",
            metadata={},
            image_data=request.image_data,
            auto_tag=request.auto_tag,
            generate_summary=request.generate_summary,
            user_context=request.user_context,
            location_data=request.location_data,
            custom_title=request.title,
            custom_tags=request.tags
        )
        
        result = await _service.unified_capture(unified_request)
        
        return CaptureResponse(
            success=result.success,
            note_id=result.note_id,
            title=result.title,
            content_preview=result.content_preview,
            tags=result.tags or [],
            summary=result.summary,
            action_items=result.action_items or [],
            processing_time=result.processing_time,
            source_service=result.source_service,
            error=result.error,
            warnings=result.warnings or []
        )
        
    except Exception as e:
        logger.error(f"Image capture failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/url")
async def capture_url(request: URLCaptureRequest) -> CaptureResponse:
    """Capture URL/web content through unified API."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        unified_request = UnifiedCaptureRequest(
            content_type=CaptureContentType.URL,
            source_type=CaptureSourceType(request.source) if request.source in [s.value for s in CaptureSourceType] else CaptureSourceType.API,
            primary_content="",
            metadata={
                "selected_text": request.selected_text,
                "page_title": request.page_title
            },
            url=request.url,
            auto_tag=request.auto_tag,
            generate_summary=request.generate_summary,
            user_context=request.user_context,
            custom_tags=request.tags
        )
        
        result = await _service.unified_capture(unified_request)
        
        return CaptureResponse(
            success=result.success,
            note_id=result.note_id,
            title=result.title,
            content_preview=result.content_preview,
            tags=result.tags or [],
            summary=result.summary,
            action_items=result.action_items or [],
            processing_time=result.processing_time,
            source_service=result.source_service,
            error=result.error,
            warnings=result.warnings or []
        )
        
    except Exception as e:
        logger.error(f"URL capture failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pdf")
async def capture_pdf(request: PDFCaptureRequest) -> CaptureResponse:
    """Capture PDF content through unified API."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        unified_request = UnifiedCaptureRequest(
            content_type=CaptureContentType.PDF,
            source_type=CaptureSourceType(request.source) if request.source in [s.value for s in CaptureSourceType] else CaptureSourceType.API,
            primary_content="",
            metadata={"filename": request.filename},
            file_data=request.file_data,
            auto_tag=request.auto_tag,
            generate_summary=request.generate_summary,
            user_context=request.user_context,
            custom_title=request.title,
            custom_tags=request.tags
        )
        
        result = await _service.unified_capture(unified_request)
        
        return CaptureResponse(
            success=result.success,
            note_id=result.note_id,
            title=result.title,
            content_preview=result.content_preview,
            tags=result.tags or [],
            summary=result.summary,
            action_items=result.action_items or [],
            processing_time=result.processing_time,
            source_service=result.source_service,
            error=result.error,
            warnings=result.warnings or []
        )
        
    except Exception as e:
        logger.error(f"PDF capture failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def capture_batch(request: BatchCaptureRequest):
    """Process multiple capture requests in batch."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        if len(request.requests) > 50:
            raise HTTPException(status_code=422, detail="Maximum 50 requests per batch")
        
        # Convert request data to unified requests
        unified_requests = []
        for req_data in request.requests:
            try:
                # Determine content and source type from request
                content_type = CaptureContentType(req_data.get("content_type", "text"))
                source_type = CaptureSourceType(req_data.get("source", "api"))
                
                unified_req = UnifiedCaptureRequest(
                    content_type=content_type,
                    source_type=source_type,
                    primary_content=req_data.get("content", ""),
                    metadata=req_data.get("metadata", {}),
                    audio_data=req_data.get("audio_data"),
                    image_data=req_data.get("image_data"),
                    file_data=req_data.get("file_data"),
                    url=req_data.get("url"),
                    auto_tag=req_data.get("auto_tag", True),
                    generate_summary=req_data.get("generate_summary", True),
                    extract_actions=req_data.get("extract_actions", True),
                    user_context=req_data.get("user_context"),
                    location_data=req_data.get("location_data"),
                    custom_title=req_data.get("title"),
                    custom_tags=req_data.get("tags", []),
                    processing_priority=req_data.get("priority", 1)
                )
                unified_requests.append(unified_req)
                
            except Exception as e:
                logger.error(f"Invalid request in batch: {e}")
                # Add error placeholder
                unified_requests.append(None)
        
        # Process valid requests
        valid_requests = [req for req in unified_requests if req is not None]
        results = await _service.batch_capture(valid_requests)
        
        # Convert results to API response format
        response_results = []
        result_idx = 0
        
        for i, unified_req in enumerate(unified_requests):
            if unified_req is None:
                response_results.append({
                    "index": i,
                    "success": False,
                    "error": "Invalid request format"
                })
            else:
                result = results[result_idx] if result_idx < len(results) else None
                response_results.append({
                    "index": i,
                    "success": result.success if result else False,
                    "note_id": result.note_id if result else None,
                    "title": result.title if result else None,
                    "error": result.error if result else "Processing failed",
                    "processing_time": result.processing_time if result else 0.0
                })
                result_idx += 1
        
        successful = sum(1 for r in response_results if r["success"])
        
        return JSONResponse(content={
            "success": True,
            "total_requests": len(request.requests),
            "successful": successful,
            "failed": len(request.requests) - successful,
            "results": response_results,
            "message": f"Batch processed: {successful}/{len(request.requests)} successful"
        })
        
    except Exception as e:
        logger.error(f"Batch capture failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class QuickNoteRequest(BaseModel):
    """Quick note capture request supporting both JSON and form submissions.

    Accepts either `content` or `text` for the note body to be flexible
    with different clients (web UI uses `content`, some clients use `text`).
    """
    content: Optional[str] = None
    text: Optional[str] = None
    source: str = "api"
    note_type: str = "thought"

@router.post("/quick-note")
async def capture_quick_note(request: Request) -> JSONResponse:
    """Quick note capture endpoint for simple text notes.

    Supports content sent as:
    - JSON body: {"content": "..."} or {"text": "..."}
    - Form-encoded body (application/x-www-form-urlencoded): content=...
    """
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Try to parse incoming data flexibly
        payload: Dict[str, Any] = {}
        ct = (request.headers.get("content-type") or "").lower()
        try:
            if "application/json" in ct:
                payload = await request.json()
            elif "application/x-www-form-urlencoded" in ct or "multipart/form-data" in ct:
                form = await request.form()
                payload = dict(form)
            else:
                # Fallback to attempt JSON first, then form, without relying on header
                try:
                    payload = await request.json()
                except Exception:
                    try:
                        form = await request.form()
                        payload = dict(form)
                    except Exception:
                        payload = {}
        except Exception:
            payload = {}

        # Validate into model (tolerant to extra keys)
        qn = QuickNoteRequest(**{k: v for k, v in payload.items() if k in {"content", "text", "source", "note_type"}})

        note_content = (qn.content or qn.text or "").strip()
        if not note_content:
            return JSONResponse(
                status_code=422,
                content={"success": False, "error": "No content provided"}
            )

        source = qn.source
        note_type = qn.note_type

        unified_request = UnifiedCaptureRequest(
            content_type=CaptureContentType.QUICK_NOTE,
            source_type=CaptureSourceType(source) if source in [s.value for s in CaptureSourceType] else CaptureSourceType.API,
            primary_content=note_content,
            metadata={"note_type": note_type},
            auto_tag=True,
            generate_summary=True,
            extract_actions=True
        )

        # Use default user_id=1 for quick notes when no authentication is provided
        # This ensures quick notes appear in the main dashboard
        result = await _service.unified_capture(unified_request, user_id="1")

        if result.success:
            return JSONResponse(content={
                "success": True,
                "note_id": result.note_id,
                "title": result.title,
                "message": "Quick note captured successfully"
            })
        else:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": result.error
                }
            )

    except Exception as e:
        logger.error(f"Quick note capture failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check for unified capture service."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    stats = _service.get_processing_stats()
    integrations = _service.get_supported_integrations()
    
    return JSONResponse(content={
        "success": True,
        "service": "unified_capture",
        "status": "healthy",
        "stats": {
            "total_requests": stats["total_requests"],
            "success_rate": f"{stats['success_rate']:.1f}%",
            "avg_processing_time": f"{stats['average_processing_time']:.2f}s"
        },
        "supported_features": integrations["features"],
        "supported_sources": integrations["sources"],
        "supported_content_types": integrations["content_types"]
    })

# Webhook endpoint for external integrations
@router.post("/webhook/{source}")
async def webhook_capture(source: str, request: Request):
    """Generic webhook endpoint for external capture sources."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Get raw request data
        raw_data = await request.json()
        
        # Simple webhook processing - convert to unified request
        content = raw_data.get("content", raw_data.get("text", ""))
        if not content:
            raise HTTPException(status_code=422, detail="No content provided")
        
        unified_request = UnifiedCaptureRequest(
            content_type=CaptureContentType.TEXT,
            source_type=CaptureSourceType.WEBHOOK,
            primary_content=content,
            metadata={
                "webhook_source": source,
                "original_data": raw_data
            },
            custom_title=raw_data.get("title"),
            custom_tags=raw_data.get("tags", [])
        )
        
        result = await _service.unified_capture(unified_request)
        
        return JSONResponse(content={
            "success": result.success,
            "note_id": result.note_id,
            "title": result.title,
            "error": result.error,
            "processing_time": result.processing_time
        })
        
    except Exception as e:
        logger.error(f"Webhook capture failed for {source}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
