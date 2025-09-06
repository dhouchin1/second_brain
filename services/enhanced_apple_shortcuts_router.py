# ──────────────────────────────────────────────────────────────────────────────
# File: services/enhanced_apple_shortcuts_router.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Enhanced Apple Shortcuts API Router

Advanced REST endpoints for iOS/macOS Shortcuts integration including voice memos,
photo OCR, location-based notes, and deep iOS integration.
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging

from services.enhanced_apple_shortcuts_service import get_enhanced_apple_shortcuts_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/shortcuts", tags=["apple-shortcuts-enhanced"])

class VoiceMemoRequest(BaseModel):
    """Voice memo capture request."""
    audio_data: Optional[str] = None
    audio_url: Optional[str] = None  
    transcription: str
    location_data: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None

class PhotoOCRRequest(BaseModel):
    """Photo OCR processing request."""
    image_data: str
    location_data: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None

class QuickNoteRequest(BaseModel):
    """Quick note capture request."""
    text: str
    note_type: str = "thought"
    location_data: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    auto_tag: bool = True

class WebClipRequest(BaseModel):
    """Web clip capture request."""
    url: str
    selected_text: Optional[str] = None
    page_title: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

# Global service instance
_service = None

def init_enhanced_apple_shortcuts_router(get_conn_func):
    """Initialize the router with database connection."""
    global _service
    _service = get_enhanced_apple_shortcuts_service(get_conn_func)

@router.get("/templates")
async def get_shortcut_templates():
    """Get pre-built iOS Shortcuts templates and examples."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        templates = _service.get_shortcut_templates()
        
        return JSONResponse(content={
            "success": True,
            "templates": templates,
            "total": len(templates),
            "message": "Shortcut templates retrieved successfully"
        })
        
    except Exception as e:
        logger.error(f"Failed to get shortcut templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/voice-memo")
async def process_voice_memo(request: VoiceMemoRequest):
    """
    Process voice memo from iOS Shortcuts.
    
    Supports:
    - Pre-transcribed text from iOS
    - Audio data for server-side transcription
    - Location and context information
    - AI-powered summarization and tagging
    """
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        result = await _service.process_voice_memo(
            audio_data=request.audio_data,
            audio_url=request.audio_url,
            transcription=request.transcription,
            location_data=request.location_data,
            context=request.context
        )
        
        if result["success"]:
            return JSONResponse(content={
                "success": True,
                "note_id": result["note_id"],
                "title": result["title"],
                "summary": result.get("summary"),
                "action_items": result.get("action_items", []),
                "tags": result["tags"],
                "message": result["message"]
            })
        else:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": result["error"]
                }
            )
            
    except Exception as e:
        logger.error(f"Voice memo processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/photo-ocr")
async def process_photo_ocr(request: PhotoOCRRequest):
    """
    Process photo with OCR from iOS Shortcuts.
    
    Features:
    - OCR text extraction from photos
    - Location tagging
    - Context preservation
    - AI-powered content processing
    """
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        result = await _service.process_photo_ocr(
            image_data=request.image_data,
            location_data=request.location_data,
            context=request.context
        )
        
        if result["success"]:
            return JSONResponse(content={
                "success": True,
                "note_id": result["note_id"],
                "title": result["title"],
                "extracted_text": result["extracted_text"],
                "tags": result["tags"],
                "message": result["message"]
            })
        else:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": result["error"]
                }
            )
            
    except Exception as e:
        logger.error(f"Photo OCR processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/quick-note")
async def process_quick_note(request: QuickNoteRequest):
    """
    Process quick note from iOS Shortcuts.
    
    Note types supported:
    - thought: Random thoughts and ideas
    - task: Action items and todos
    - meeting: Meeting-related notes
    - idea: Creative ideas and insights
    - reminder: Things to remember
    """
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        result = await _service.process_quick_note(
            text=request.text,
            note_type=request.note_type,
            location_data=request.location_data,
            context=request.context,
            auto_tag=request.auto_tag
        )
        
        if result["success"]:
            return JSONResponse(content={
                "success": True,
                "note_id": result["note_id"],
                "title": result["title"],
                "tags": result["tags"],
                "message": result["message"]
            })
        else:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": result["error"]
                }
            )
            
    except Exception as e:
        logger.error(f"Quick note processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/web-clip")
async def process_web_clip(request: WebClipRequest):
    """
    Process web clip from iOS Safari Share Sheet.
    
    Features:
    - Full page content extraction
    - Selected text capture
    - Context preservation
    - AI-powered summarization
    """
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        result = await _service.process_web_clip(
            url=request.url,
            selected_text=request.selected_text,
            page_title=request.page_title,
            context=request.context
        )
        
        if result["success"]:
            return JSONResponse(content={
                "success": True,
                "note_id": result["note_id"],
                "title": result["title"],
                "content_type": result["content_type"],
                "message": result["message"]
            })
        else:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": result["error"]
                }
            )
            
    except Exception as e:
        logger.error(f"Web clip processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_shortcuts_stats():
    """Get statistics about iOS Shortcuts usage."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        conn = _service.get_conn()
        cursor = conn.cursor()
        
        # Get shortcuts content statistics
        cursor.execute("""
            SELECT 
                json_extract(metadata, '$.source') as source,
                json_extract(metadata, '$.content_type') as content_type,
                COUNT(*) as count
            FROM notes 
            WHERE json_extract(metadata, '$.source') = 'ios_shortcuts'
            GROUP BY content_type
            ORDER BY count DESC
        """)
        
        content_types = {}
        for row in cursor.fetchall():
            content_types[row[1]] = row[2]
        
        # Get recent shortcuts captures
        cursor.execute("""
            SELECT title, created_at, json_extract(metadata, '$.content_type') as content_type
            FROM notes 
            WHERE json_extract(metadata, '$.source') = 'ios_shortcuts'
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
        
        # Get usage by day (last 7 days)
        cursor.execute("""
            SELECT DATE(created_at) as date, COUNT(*) as count
            FROM notes 
            WHERE json_extract(metadata, '$.source') = 'ios_shortcuts'
            AND DATE(created_at) >= DATE('now', '-7 days')
            GROUP BY DATE(created_at)
            ORDER BY date DESC
        """)
        
        daily_usage = dict(cursor.fetchall())
        
        conn.close()
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "content_types": content_types,
                "recent_captures": recent_captures,
                "daily_usage_7_days": daily_usage,
                "total_shortcuts_notes": sum(content_types.values())
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get shortcuts stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check for iOS Shortcuts integration."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return JSONResponse(content={
        "success": True,
        "service": "enhanced_apple_shortcuts",
        "status": "healthy",
        "features": {
            "voice_memos": True,
            "photo_ocr": True,
            "quick_notes": True,
            "web_clips": True,
            "location_support": True,
            "ai_processing": True
        }
    })

@router.post("/batch")
async def process_batch_shortcuts(requests: List[Dict[str, Any]]):
    """
    Process multiple shortcuts requests in batch.
    
    Useful for offline sync or bulk operations.
    """
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        if len(requests) > 20:
            raise HTTPException(status_code=422, detail="Maximum 20 requests per batch")
        
        results = []
        
        for i, req_data in enumerate(requests):
            try:
                req_type = req_data.get("type")
                
                if req_type == "voice_memo":
                    result = await _service.process_voice_memo(**req_data.get("data", {}))
                elif req_type == "photo_ocr":
                    result = await _service.process_photo_ocr(**req_data.get("data", {}))
                elif req_type == "quick_note":
                    result = await _service.process_quick_note(**req_data.get("data", {}))
                elif req_type == "web_clip":
                    result = await _service.process_web_clip(**req_data.get("data", {}))
                else:
                    result = {"success": False, "error": f"Unknown request type: {req_type}"}
                
                results.append({
                    "index": i,
                    "type": req_type,
                    "success": result["success"],
                    "note_id": result.get("note_id"),
                    "error": result.get("error")
                })
                
            except Exception as e:
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(e)
                })
        
        successful = sum(1 for r in results if r["success"])
        
        return JSONResponse(content={
            "success": True,
            "total_requests": len(requests),
            "successful": successful,
            "failed": len(requests) - successful,
            "results": results,
            "message": f"Batch processed: {successful}/{len(requests)} successful"
        })
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))