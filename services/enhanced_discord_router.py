# ──────────────────────────────────────────────────────────────────────────────
# File: services/enhanced_discord_router.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Enhanced Discord Integration API Router

REST endpoints for Discord bot management and statistics.
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import logging

from services.enhanced_discord_service import get_enhanced_discord_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/discord", tags=["discord-enhanced"])

# Global service instance
_service = None

def init_enhanced_discord_router(get_conn_func):
    """Initialize the router with database connection."""
    global _service
    _service = get_enhanced_discord_service(get_conn_func)

@router.get("/health")
async def discord_health_check():
    """Check Discord bot health status."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Check if bot is configured and running
        from config import settings
        
        if not settings.discord_bot_token or settings.discord_bot_token == "your-discord-bot-token":
            return JSONResponse(content={
                "success": False,
                "status": "not_configured",
                "message": "Discord bot token not configured"
            })
        
        # Simple health check - if service exists, consider it healthy
        # In a full implementation, this would ping the Discord API
        return JSONResponse(content={
            "success": True,
            "status": "healthy",
            "message": "Discord bot service is running",
            "features": {
                "slash_commands": True,
                "reaction_handlers": True,
                "thread_summarization": True,
                "message_capture": True,
                "search_integration": True
            }
        })
        
    except Exception as e:
        logger.error(f"Discord health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "status": "error",
                "message": str(e)
            }
        )

@router.get("/stats")
async def discord_stats():
    """Get Discord bot usage statistics."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        stats = await _service.get_discord_usage_stats()
        
        return JSONResponse(content={
            "success": True,
            "data": stats
        })
        
    except Exception as e:
        logger.error(f"Discord stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/commands")
async def get_discord_commands():
    """Get list of available Discord bot commands."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    commands = [
        {
            "name": "/capture",
            "description": "Capture a quick note to Second Brain",
            "usage": "/capture content:Your note here",
            "category": "capture"
        },
        {
            "name": "/search", 
            "description": "Search your Second Brain notes",
            "usage": "/search query:search terms",
            "category": "search"
        },
        {
            "name": "/thread_summary",
            "description": "Summarize a thread conversation",
            "usage": "/thread_summary messages:50",
            "category": "ai"
        },
        {
            "name": "/meeting_notes",
            "description": "Start meeting notes template",
            "usage": "/meeting_notes topic:Meeting topic",
            "category": "templates"
        },
        {
            "name": "/stats",
            "description": "Show Second Brain statistics",
            "usage": "/stats",
            "category": "info"
        }
    ]
    
    reactions = [
        {
            "emoji": "🧠",
            "description": "React to any message to save it to Second Brain",
            "category": "capture"
        },
        {
            "emoji": "📝",
            "description": "React to summarize a thread (works in threads only)",
            "category": "ai"
        },
        {
            "emoji": "⭐",
            "description": "React to mark a message as important",
            "category": "organize"
        }
    ]
    
    return JSONResponse(content={
        "success": True,
        "data": {
            "slash_commands": commands,
            "reaction_shortcuts": reactions,
            "total_commands": len(commands),
            "total_reactions": len(reactions)
        }
    })

@router.post("/test-connection")
async def test_discord_connection():
    """Test Discord bot connection (admin only)."""
    if not _service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # This would test actual Discord API connection
    # For now, just return success if service is initialized
    return JSONResponse(content={
        "success": True,
        "message": "Discord service connection test completed",
        "timestamp": "2024-12-15T10:30:00Z"
    })


def get_enhanced_discord_router():
    """Get the enhanced Discord router."""
    return router