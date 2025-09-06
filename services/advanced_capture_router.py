# ──────────────────────────────────────────────────────────────────────────────
# File: services/advanced_capture_router.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Advanced Capture API Router

REST endpoints for advanced capture capabilities including OCR, PDF processing,
video transcripts, and bulk operations.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import json

from services.advanced_capture_service import get_advanced_capture_service, CaptureOptions, CaptureResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/capture/advanced", tags=["advanced-capture"])

class ScreenshotCaptureRequest(BaseModel):
    """Request for screenshot with OCR."""
    image_data: str
    enable_ocr: bool = True
    enable_ai_processing: bool = True
    custom_tags: Optional[List[str]] = []
    language: str = "en"

class PDFCaptureRequest(BaseModel):
    """Request for PDF content extraction."""
    pdf_url: Optional[str] = None
    enable_ai_processing: bool = True
    custom_tags: Optional[List[str]] = []
    quality: str = "high"

class YouTubeCaptureRequest(BaseModel):
    """Request for YouTube transcript extraction."""
    video_url: str
    enable_ai_processing: bool = True
    custom_tags: Optional[List[str]] = []
    language: str = "en"

class BulkUrlCaptureRequest(BaseModel):
    """Request for bulk URL processing."""
    urls: List[str]
    enable_ai_processing: bool = True
    take_screenshot: bool = False
    custom_tags: Optional[List[str]] = []

class FeatureCheckResponse(BaseModel):
    """Response for feature availability check."""
    ocr: bool
    pdf: bool
    image_processing: bool
    youtube: bool
    message: str

# Global service instance
_service = None

def init_advanced_capture_router(get_conn_func):
    """Initialize the router with database connection."""
    global _service
    _service = get_advanced_capture_service(get_conn_func)

@router.get("/features", response_model=FeatureCheckResponse)
async def check_feature_availability():
    """Check which advanced capture features are available."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    features = _service.get_feature_availability()
    
    missing_features = [name for name, available in features.items() if not available]
    message = "All features available"
    
    if missing_features:
        message = f"Missing features: {', '.join(missing_features)}. Install required packages."
    
    return FeatureCheckResponse(
        **features,
        message=message
    )

@router.post("/screenshot-ocr")
async def capture_screenshot_with_ocr(request: ScreenshotCaptureRequest):
    """Process screenshot with OCR extraction."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        options = CaptureOptions(
            enable_ocr=request.enable_ocr,
            enable_ai_processing=request.enable_ai_processing,
            custom_tags=request.custom_tags or [],
            language=request.language
        )
        
        result = await _service.capture_screenshot_with_ocr(request.image_data, options)
        
        if result.success:
            return JSONResponse(content={
                "success": True,
                "note_id": result.note_id,
                "title": result.title,
                "content": result.content[:500] + "..." if len(result.content) > 500 else result.content,
                "tags": result.tags,
                "processing_time": result.processing_time,
                "message": "Screenshot processed with OCR successfully"
            })
        else:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": result.error,
                    "processing_time": result.processing_time
                }
            )
            
    except Exception as e:
        logger.error(f"Screenshot OCR failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pdf")
async def capture_pdf_content(request: PDFCaptureRequest):
    """Extract and process content from PDF."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        options = CaptureOptions(
            enable_ai_processing=request.enable_ai_processing,
            custom_tags=request.custom_tags or [],
            quality=request.quality
        )
        
        result = await _service.capture_pdf_content(
            pdf_url=request.pdf_url,
            options=options
        )
        
        if result.success:
            return JSONResponse(content={
                "success": True,
                "note_id": result.note_id,
                "title": result.title,
                "content_length": len(result.content),
                "content_preview": result.content[:500] + "..." if len(result.content) > 500 else result.content,
                "tags": result.tags,
                "metadata": result.metadata,
                "processing_time": result.processing_time,
                "message": "PDF processed successfully"
            })
        else:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": result.error,
                    "processing_time": result.processing_time
                }
            )
            
    except Exception as e:
        logger.error(f"PDF capture failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pdf-upload")
async def capture_pdf_upload(
    file: UploadFile = File(...),
    enable_ai_processing: bool = Form(True),
    custom_tags: str = Form(""),
    quality: str = Form("high")
):
    """Process uploaded PDF file."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=422, detail="Only PDF files are supported")
        
        # Read file data
        pdf_data = await file.read()
        
        # Parse custom tags
        tags_list = [tag.strip() for tag in custom_tags.split(",") if tag.strip()]
        
        options = CaptureOptions(
            enable_ai_processing=enable_ai_processing,
            custom_tags=tags_list,
            quality=quality
        )
        
        result = await _service.capture_pdf_content(
            pdf_data=pdf_data,
            options=options
        )
        
        if result.success:
            return JSONResponse(content={
                "success": True,
                "note_id": result.note_id,
                "title": result.title,
                "content_length": len(result.content),
                "tags": result.tags,
                "processing_time": result.processing_time,
                "message": f"PDF '{file.filename}' processed successfully"
            })
        else:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": result.error,
                    "processing_time": result.processing_time
                }
            )
            
    except Exception as e:
        logger.error(f"PDF upload processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/youtube")
async def capture_youtube_transcript(request: YouTubeCaptureRequest):
    """Extract transcript from YouTube video."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        options = CaptureOptions(
            enable_ai_processing=request.enable_ai_processing,
            custom_tags=request.custom_tags or [],
            language=request.language
        )
        
        result = await _service.capture_youtube_transcript(request.video_url, options)
        
        if result.success:
            return JSONResponse(content={
                "success": True,
                "note_id": result.note_id,
                "title": result.title,
                "content_length": len(result.content),
                "tags": result.tags,
                "processing_time": result.processing_time,
                "message": "YouTube transcript extracted successfully"
            })
        else:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": result.error,
                    "processing_time": result.processing_time
                }
            )
            
    except Exception as e:
        logger.error(f"YouTube capture failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bulk-urls")
async def capture_bulk_urls(request: BulkUrlCaptureRequest, background_tasks: BackgroundTasks):
    """Process multiple URLs in batch."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        if len(request.urls) > 50:
            raise HTTPException(status_code=422, detail="Maximum 50 URLs allowed per batch")
        
        options = CaptureOptions(
            enable_ai_processing=request.enable_ai_processing,
            take_screenshot=request.take_screenshot,
            custom_tags=request.custom_tags or []
        )
        
        # Process URLs
        results = await _service.capture_bulk_urls(request.urls, options)
        
        # Summarize results
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        return JSONResponse(content={
            "success": True,
            "total_urls": len(request.urls),
            "successful": len(successful),
            "failed": len(failed),
            "results": [
                {
                    "success": r.success,
                    "note_id": r.note_id,
                    "title": r.title,
                    "error": r.error,
                    "processing_time": r.processing_time
                } for r in results
            ],
            "message": f"Processed {len(request.urls)} URLs: {len(successful)} successful, {len(failed)} failed"
        })
        
    except Exception as e:
        logger.error(f"Bulk URL capture failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_capture_stats():
    """Get statistics about captured content types."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        conn = _service.get_conn()
        cursor = conn.cursor()
        
        # Get content type statistics
        cursor.execute("""
            SELECT 
                json_extract(metadata, '$.content_type') as content_type,
                COUNT(*) as count
            FROM notes 
            WHERE metadata IS NOT NULL
            AND json_extract(metadata, '$.content_type') IS NOT NULL
            GROUP BY content_type
            ORDER BY count DESC
        """)
        
        content_types = dict(cursor.fetchall())
        
        # Get recent advanced captures
        cursor.execute("""
            SELECT title, created_at, json_extract(metadata, '$.content_type') as content_type
            FROM notes 
            WHERE metadata IS NOT NULL
            AND json_extract(metadata, '$.content_type') IN ('screenshot_ocr', 'pdf', 'youtube_transcript')
            ORDER BY created_at DESC
            LIMIT 10
        """)
        
        recent_captures = [
            {
                "title": row[0],
                "created_at": row[1],
                "content_type": row[2]
            }
            for row in cursor.fetchall()
        ]
        
        conn.close()
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "content_types": content_types,
                "recent_advanced_captures": recent_captures,
                "features_available": _service.get_feature_availability()
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get capture stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))