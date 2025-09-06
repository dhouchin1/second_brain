"""
Smart Templates FastAPI Router

Provides REST API endpoints for AI-powered context-aware note templates 
that adapt based on content analysis, time context, and user patterns.
"""

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from services.smart_templates_service import SmartTemplatesService
from services.auth_service import User

# Global service instances and functions (initialized by app.py)
templates_service: Optional[SmartTemplatesService] = None
get_conn = None
get_current_user = None

# FastAPI router
router = APIRouter(prefix="/api/templates", tags=["smart-templates"])


def init_smart_templates_router(get_conn_func, get_current_user_func):
    """Initialize Smart Templates router with dependencies"""
    global templates_service, get_conn, get_current_user
    get_conn = get_conn_func
    get_current_user = get_current_user_func
    templates_service = SmartTemplatesService(get_conn_func)


# ─── Pydantic Models ───

class TemplateSuggestionsRequest(BaseModel):
    """Request model for template suggestions"""
    content: str = ""
    context: Dict[str, Any] = {}

class ApplyTemplateRequest(BaseModel):
    """Request model for applying templates"""
    template_id: str
    variables: Dict[str, str] = {}
    user_customizations: Dict[str, Any] = {}

class CreateCustomTemplateRequest(BaseModel):
    """Request model for creating custom templates"""
    name: str
    type: str
    content: str
    description: str = ""
    keywords: List[str] = []
    time_contexts: List[str] = ["morning", "afternoon", "evening"]

class PublishTemplateRequest(BaseModel):
    """Request model for publishing templates"""
    template_id: str
    is_public: bool = True
    category: str = "general"
    tags: List[str] = []

class TemplateDiscoveryRequest(BaseModel):
    """Request model for template discovery"""
    category: Optional[str] = None
    template_type: Optional[str] = None
    keywords: List[str] = []
    sort_by: str = "rating"  # rating, downloads, newest
    limit: int = 20

class RateTemplateRequest(BaseModel):
    """Request model for rating templates"""
    public_id: str
    rating: float
    review: Optional[str] = None


# ─── Core Template Endpoints ───

@router.post("/suggestions")
async def get_template_suggestions(
    request_data: TemplateSuggestionsRequest,
    fastapi_request: Request
):
    """Get template suggestions based on content and context"""
    if not templates_service:
        raise HTTPException(status_code=500, detail="Templates service not initialized")
    
    # Get current user
    current_user = await get_current_user(fastapi_request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        context = request_data.context.copy()
        context["user_id"] = current_user.id
        
        suggestions = await templates_service.suggest_templates(request_data.content, context)
        
        return JSONResponse(content={
            "suggestions": suggestions,
            "content_analyzed": bool(request_data.content),
            "context_provided": context,
            "suggestion_count": len(suggestions)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get template suggestions: {str(e)}")


@router.get("/suggestions")  
async def get_template_suggestions_simple(
    fastapi_request: Request,
    content: str = "",
    limit: int = 5
):
    """Simple GET endpoint for template suggestions (fallback)"""
    if not templates_service:
        raise HTTPException(status_code=500, detail="Templates service not initialized")
    
    # Get current user
    current_user = await get_current_user(fastapi_request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        context = {"user_id": current_user.id, "has_calendar_event": False}
        suggestions = await templates_service.suggest_templates(content, context)
        
        return JSONResponse(content={
            "suggestions": suggestions[:limit],
            "content_analyzed": bool(content),
            "suggestion_count": len(suggestions)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get template suggestions: {str(e)}")


@router.get("/{template_id}")
async def get_template(
    template_id: str,
    fastapi_request: Request,
    variables: Optional[str] = None  # JSON string of variables
):
    """Get specific template with optional variables"""
    if not templates_service:
        raise HTTPException(status_code=500, detail="Templates service not initialized")
    
    # Get current user
    current_user = await get_current_user(fastapi_request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        # Parse variables if provided
        template_variables = {}
        if variables:
            import json
            try:
                template_variables = json.loads(variables)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid variables JSON format")
        
        template = await templates_service.get_template(template_id, template_variables)
        
        return JSONResponse(content=template)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get template: {str(e)}")


@router.post("/apply")
async def apply_template(
    request_data: ApplyTemplateRequest,
    fastapi_request: Request
):
    """Apply template with user customizations"""
    if not templates_service:
        raise HTTPException(status_code=500, detail="Templates service not initialized")
    
    # Get current user
    current_user = await get_current_user(fastapi_request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        # Get the template with variables applied
        template = await templates_service.get_template(
            request_data.template_id, 
            request_data.variables
        )
        
        # Apply any user customizations
        final_content = template["content"]
        if request_data.user_customizations:
            # Apply customizations like style preferences, additional sections, etc.
            for key, value in request_data.user_customizations.items():
                if key == "additional_sections":
                    final_content += f"\n\n## {value.get('title', 'Additional Notes')}\n{value.get('content', '')}"
                elif key == "style_preferences":
                    # Could modify formatting, emoji usage, etc.
                    pass
        
        result = {
            **template,
            "final_content": final_content,
            "customizations_applied": request_data.user_customizations,
            "applied_by": current_user.email,
            "applied_at": templates_service.templates[request_data.template_id].last_used.strftime("%Y-%m-%d %H:%M:%S") if templates_service.templates[request_data.template_id].last_used else None
        }
        
        return JSONResponse(content=result)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply template: {str(e)}")


@router.get("/library")
async def get_template_library(fastapi_request: Request):
    """Get complete template library"""
    if not templates_service:
        raise HTTPException(status_code=500, detail="Templates service not initialized")
    
    # Get current user
    current_user = await get_current_user(fastapi_request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        library = templates_service.get_template_library()
        
        from datetime import datetime
        return JSONResponse(content={
            **library,
            "user_id": current_user.id,
            "retrieved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get template library: {str(e)}")


@router.post("/custom")
async def create_custom_template(
    request_data: CreateCustomTemplateRequest,
    fastapi_request: Request
):
    """Create custom template"""
    if not templates_service:
        raise HTTPException(status_code=500, detail="Templates service not initialized")
    
    # Get current user
    current_user = await get_current_user(fastapi_request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        template_data = {
            "name": request_data.name,
            "type": request_data.type,
            "content": request_data.content,
            "description": request_data.description,
            "keywords": request_data.keywords,
            "time_contexts": request_data.time_contexts,
            "ai_generated": False
        }
        
        template_id = await templates_service.create_custom_template(
            current_user.id, 
            template_data
        )
        
        return JSONResponse(content={
            "template_id": template_id,
            "name": request_data.name,
            "type": request_data.type,
            "created_by": current_user.email,
            "status": "created",
            "message": "Custom template created successfully"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create custom template: {str(e)}")


@router.get("/analytics")
async def get_template_analytics(fastapi_request: Request):
    """Template usage analytics"""
    if not templates_service:
        raise HTTPException(status_code=500, detail="Templates service not initialized")
    
    # Get current user
    current_user = await get_current_user(fastapi_request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        analytics = await templates_service.get_template_analytics(current_user.id)
        
        return JSONResponse(content={
            **analytics,
            "user_id": current_user.id,
            "analytics_generated_at": templates_service.datetime.now().strftime("%Y-%m-%d %H:%M:%S") if hasattr(templates_service, 'datetime') else None
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get template analytics: {str(e)}")


@router.get("/status")
async def templates_status():
    """Service status/health check"""
    if not templates_service:
        return JSONResponse(
            content={"status": "unavailable", "error": "Service not initialized"},
            status_code=503
        )
    
    try:
        library = templates_service.get_template_library()
        
        return JSONResponse(content={
            "status": "healthy",
            "service": "Smart Templates",
            "templates_loaded": library["total_templates"],
            "template_types_available": len(library["template_types"]),
            "context_triggers": len(library["context_triggers"]),
            "features": {
                "ai_suggestions": True,
                "context_awareness": True,
                "custom_templates": True,
                "usage_analytics": True,
                "variable_substitution": True,
                "time_based_suggestions": True,
                "keyword_matching": True
            },
            "version": "1.0.0"
        })
        
    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "service": "Smart Templates",
                "error": str(e)
            },
            status_code=500
        )


# ─── Utility Endpoints ───

@router.post("/test")
async def test_templates_integration(fastapi_request: Request):
    """Test endpoint for Smart Templates integration"""
    if not templates_service:
        raise HTTPException(status_code=500, detail="Templates service not initialized")
    
    # Get current user
    current_user = await get_current_user(fastapi_request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        # Test template suggestions
        test_content = "I need to plan a team meeting for next week to discuss project progress"
        test_context = {
            "has_calendar_event": True,
            "user_id": current_user.id
        }
        
        suggestions = await templates_service.suggest_templates(test_content, test_context)
        
        # Test getting a template
        if suggestions:
            template_id = suggestions[0]["template_id"]
            template = await templates_service.get_template(template_id, {
                "meeting_title": "Test Meeting",
                "date": "2024-01-15"
            })
        else:
            template = None
        
        return JSONResponse(content={
            "test": True,
            "status": "success",
            "message": "Smart Templates integration test successful!",
            "test_results": {
                "suggestions_found": len(suggestions),
                "template_retrieved": template is not None,
                "service_responsive": True
            },
            "sample_suggestions": suggestions[:2] if suggestions else [],
            "tested_by": current_user.email
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")


@router.get("/health")
async def templates_health():
    """Health check for Smart Templates service (public endpoint)"""
    return JSONResponse(content={
        "status": "healthy" if templates_service else "unavailable",
        "service": "Smart Templates Integration",
        "endpoints_active": templates_service is not None,
        "features": {
            "ai_suggestions": True,
            "context_awareness": True,
            "custom_templates": True,
            "usage_analytics": True,
            "variable_substitution": True,
            "template_library": True,
            "smart_matching": True
        }
    })

@router.get("/status/public")
async def templates_status_public():
    """Public service status/health check (no authentication)"""
    if not templates_service:
        return JSONResponse(
            content={"status": "unavailable", "error": "Service not initialized"},
            status_code=503
        )
    
    try:
        library = templates_service.get_template_library()
        
        return JSONResponse(content={
            "status": "healthy",
            "service": "Smart Templates",
            "templates_loaded": library["total_templates"],
            "template_types_available": len(library["template_types"]),
            "context_triggers": len(library["context_triggers"]),
            "features": {
                "ai_suggestions": True,
                "context_awareness": True,
                "custom_templates": True,
                "usage_analytics": True,
                "variable_substitution": True,
                "time_based_suggestions": True,
                "keyword_matching": True,
                "template_sharing": True,
                "community_discovery": True,
                "template_variations": True
            },
            "version": "1.1.0"
        })
        
    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "service": "Smart Templates",
                "error": str(e)
            },
            status_code=500
        )


# ─── Template Sharing & Discovery Endpoints ───

@router.post("/publish")
async def publish_template(
    request_data: PublishTemplateRequest,
    fastapi_request: Request
):
    """Publish a template to the community library"""
    if not templates_service:
        raise HTTPException(status_code=500, detail="Templates service not initialized")
    
    # Get current user
    current_user = await get_current_user(fastapi_request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        result = await templates_service.publish_template(
            current_user.id,
            request_data.template_id,
            request_data.is_public
        )
        
        return JSONResponse(content={
            **result,
            "published_by": current_user.email
        })
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to publish template: {str(e)}")


@router.post("/discover")
async def discover_templates(
    request_data: TemplateDiscoveryRequest,
    fastapi_request: Request
):
    """Discover public templates from the community library"""
    if not templates_service:
        raise HTTPException(status_code=500, detail="Templates service not initialized")
    
    # Get current user
    current_user = await get_current_user(fastapi_request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        templates = await templates_service.discover_templates(
            category=request_data.category,
            template_type=request_data.template_type,
            keywords=request_data.keywords,
            sort_by=request_data.sort_by,
            limit=request_data.limit
        )
        
        return JSONResponse(content={
            "templates": templates,
            "total_found": len(templates),
            "search_criteria": {
                "category": request_data.category,
                "template_type": request_data.template_type,
                "keywords": request_data.keywords,
                "sort_by": request_data.sort_by,
                "limit": request_data.limit
            },
            "searched_by": current_user.email
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to discover templates: {str(e)}")


@router.get("/discover")
async def discover_templates_simple(
    fastapi_request: Request,
    category: Optional[str] = None,
    template_type: Optional[str] = None,
    sort_by: str = "rating",
    limit: int = 20
):
    """Simple GET endpoint for template discovery"""
    if not templates_service:
        raise HTTPException(status_code=500, detail="Templates service not initialized")
    
    # Get current user
    current_user = await get_current_user(fastapi_request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        templates = await templates_service.discover_templates(
            category=category,
            template_type=template_type,
            keywords=[],
            sort_by=sort_by,
            limit=limit
        )
        
        return JSONResponse(content={
            "templates": templates,
            "total_found": len(templates),
            "search_criteria": {
                "category": category,
                "template_type": template_type,
                "sort_by": sort_by,
                "limit": limit
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to discover templates: {str(e)}")


@router.post("/install/{public_id}")
async def install_public_template(
    public_id: str,
    fastapi_request: Request
):
    """Install a public template for the user"""
    if not templates_service:
        raise HTTPException(status_code=500, detail="Templates service not initialized")
    
    # Get current user
    current_user = await get_current_user(fastapi_request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        template_id = await templates_service.install_public_template(
            current_user.id,
            public_id
        )
        
        return JSONResponse(content={
            "template_id": template_id,
            "public_id": public_id,
            "installed_by": current_user.email,
            "status": "installed",
            "message": "Template installed successfully"
        })
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to install template: {str(e)}")


@router.post("/rate")
async def rate_template(
    request_data: RateTemplateRequest,
    fastapi_request: Request
):
    """Rate a public template"""
    if not templates_service:
        raise HTTPException(status_code=500, detail="Templates service not initialized")
    
    # Get current user
    current_user = await get_current_user(fastapi_request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        result = await templates_service.rate_template(
            current_user.id,
            request_data.public_id,
            request_data.rating
        )
        
        return JSONResponse(content={
            **result,
            "rated_by": current_user.email
        })
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rate template: {str(e)}")


@router.get("/categories")
async def get_template_categories(fastapi_request: Request):
    """Get available template categories with counts"""
    if not templates_service:
        raise HTTPException(status_code=500, detail="Templates service not initialized")
    
    # Get current user
    current_user = await get_current_user(fastapi_request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        categories = await templates_service.get_template_categories()
        
        return JSONResponse(content={
            "categories": categories,
            "total_categories": len(categories),
            "retrieved_by": current_user.email
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {str(e)}")


@router.get("/variations/{template_id}")
async def get_template_variations(
    template_id: str,
    fastapi_request: Request,
    mobile: bool = False
):
    """Get template variations based on user context"""
    if not templates_service:
        raise HTTPException(status_code=500, detail="Templates service not initialized")
    
    # Get current user
    current_user = await get_current_user(fastapi_request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        user_context = {
            "mobile": mobile,
            "user_id": current_user.id
        }
        
        variations = await templates_service.get_template_variations(
            template_id,
            user_context
        )
        
        return JSONResponse(content={
            "template_id": template_id,
            "variations": variations,
            "context": user_context,
            "total_variations": len(variations)
        })
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get template variations: {str(e)}")


print("[Smart Templates Router] Enhanced with sharing & discovery features - Loaded successfully")