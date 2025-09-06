# ──────────────────────────────────────────────────────────────────────────────
# File: services/demo_data_router.py  
# ──────────────────────────────────────────────────────────────────────────────
"""
Demo Data API Router

Provides REST endpoints for managing demo/stock content including seeding,
removal, statistics, and visibility controls.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging

from database import get_db_connection
from services.demo_data_service import get_demo_data_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/demo", tags=["demo-data"])

class DemoDataToggleRequest(BaseModel):
    """Request model for toggling demo data visibility."""
    visible: bool
    
class DemoDataSeedRequest(BaseModel):
    """Request model for seeding demo data."""
    force_reseed: Optional[bool] = False
    sets: Optional[list[str]] = None  # Specific sets to seed, None for all

@router.get("/stats")
async def get_demo_stats():
    """Get statistics about demo data in the system."""
    try:
        service = get_demo_data_service(get_db_connection)
        stats = service.get_demo_data_stats()
        
        return JSONResponse(content={
            "success": True,
            "data": stats
        })
        
    except Exception as e:
        logger.error(f"Failed to get demo stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/seed")
async def seed_demo_data(request: DemoDataSeedRequest):
    """Seed the database with demo content."""
    try:
        service = get_demo_data_service(get_db_connection)
        result = service.seed_demo_data(force_reseed=request.force_reseed)
        
        return JSONResponse(content={
            "success": True,
            "data": result
        })
        
    except Exception as e:
        logger.error(f"Failed to seed demo data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/remove")
async def remove_demo_data():
    """Remove all demo data from the system."""
    try:
        service = get_demo_data_service(get_db_connection)
        result = service.remove_demo_data()
        
        return JSONResponse(content={
            "success": True,
            "data": result
        })
        
    except Exception as e:
        logger.error(f"Failed to remove demo data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/toggle-visibility") 
async def toggle_demo_visibility(request: DemoDataToggleRequest):
    """Toggle demo data visibility in search and UI."""
    try:
        service = get_demo_data_service(get_db_connection)
        result = service.toggle_demo_data_visibility(request.visible)
        
        # This could also update user preferences in database
        return JSONResponse(content={
            "success": True,
            "data": result,
            "demo_visible": request.visible
        })
        
    except Exception as e:
        logger.error(f"Failed to toggle demo visibility: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/content-sets")
async def get_available_content_sets():
    """Get list of available demo content sets."""
    try:
        service = get_demo_data_service(get_db_connection)
        
        sets_info = {}
        for set_name, content_items in service.DEMO_CONTENT_SETS.items():
            sets_info[set_name] = {
                "name": set_name.replace("_", " ").title(),
                "count": len(content_items),
                "description": f"{len(content_items)} items covering {set_name.replace('_', ' ')} topics",
                "content_types": list(set(item.content_type for item in content_items)),
                "sample_tags": list(set(tag for item in content_items for tag in item.tags[:3]))[:10]
            }
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "total_sets": len(sets_info),
                "total_items": sum(info["count"] for info in sets_info.values()),
                "sets": sets_info
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get content sets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset")
async def reset_demo_data():
    """Remove all demo data and re-seed with fresh content."""
    try:
        service = get_demo_data_service(get_db_connection)
        
        # Remove existing
        remove_result = service.remove_demo_data()
        
        # Re-seed
        seed_result = service.seed_demo_data(force_reseed=True)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "removed": remove_result,
                "seeded": seed_result,
                "message": "Demo data reset successfully"
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to reset demo data: {e}")
        raise HTTPException(status_code=500, detail=str(e))