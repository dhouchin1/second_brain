"""
ArchiveBox Integration Router

FastAPI router for ArchiveBox web archival functionality.
Provides endpoints for URL archiving, status checking, and content retrieval.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
import asyncio
import logging

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, HttpUrl, validator

from services.archivebox_service import (
    ArchiveBoxService, ArchiveRequest, ArchiveResult, ArchiveBoxConfig,
    get_archivebox_service
)
from services.ingestion_queue import (
    IngestionJobType, ArchiveWebCapturePayload, create_archive_job, ingestion_queue
)
from services.auth_service import User

logger = logging.getLogger(__name__)

# Global service instances and functions (initialized by app.py)
archivebox_service: Optional[ArchiveBoxService] = None
get_conn = None
get_current_user = None

# FastAPI router
router = APIRouter(prefix="/api/archivebox", tags=["archivebox"])


def init_archivebox_router(get_conn_func, get_current_user_func):
    """Initialize ArchiveBox services"""
    global archivebox_service, get_conn, get_current_user
    get_conn = get_conn_func
    get_current_user = get_current_user_func
    archivebox_service = get_archivebox_service()


# Request/Response models
class ArchiveUrlRequest(BaseModel):
    """Request to archive a URL."""

    url: HttpUrl
    extract_types: Optional[List[str]] = None
    timeout: Optional[int] = None
    only_new: bool = True
    overwrite: bool = False
    priority: int = 0
    async_processing: bool = True
    storage_strategy: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @validator('extract_types')
    def validate_extract_types(cls, v):
        if v is None:
            return v
        valid_types = {
            'title', 'favicon', 'wget', 'pdf', 'screenshot', 'dom',
            'singlefile', 'readability', 'mercury', 'htmltotext', 'git',
            'media', 'archive_dot_org'
        }
        for extract_type in v:
            if extract_type.lower() not in valid_types:
                raise ValueError(f"Invalid extract type: {extract_type}")
        return [t.lower() for t in v]

    @validator('storage_strategy')
    def validate_storage_strategy(cls, v):
        if v is not None and v not in ['symlink', 'copy']:
            raise ValueError("Storage strategy must be 'symlink' or 'copy'")
        return v


class ArchiveUrlResponse(BaseModel):
    """Response from archive URL request."""

    success: bool
    message: str
    job_key: Optional[str] = None
    archive_result: Optional[Dict[str, Any]] = None
    estimated_duration: Optional[int] = None


class ArchiveStatusResponse(BaseModel):
    """Archive status response."""

    url: str
    archived: bool
    archive_result: Optional[Dict[str, Any]] = None
    last_updated: Optional[str] = None


class ArchiveListResponse(BaseModel):
    """List of archived URLs response."""

    total: int
    archives: List[Dict[str, Any]]
    page: int
    limit: int


class ArchiveContentResponse(BaseModel):
    """Archive content response."""

    snapshot_id: str
    content: Dict[str, Any]
    available_formats: List[str]


class ConfigResponse(BaseModel):
    """ArchiveBox configuration response."""

    enabled: bool
    available: bool
    feature_flag: bool
    storage_strategy: str
    supported_extract_types: List[str]
    docker_available: bool
    cli_available: bool


# API endpoints
@router.get("/status", response_model=ConfigResponse)
async def get_archivebox_status():
    """Get ArchiveBox service status and configuration."""
    if not archivebox_service:
        raise HTTPException(status_code=503, detail="ArchiveBox service not initialized")

    config = archivebox_service.config
    is_available = await archivebox_service.is_available()
    docker_available = await archivebox_service._check_docker()
    cli_available = await archivebox_service._check_archivebox_cli()

    return ConfigResponse(
        enabled=config.enabled,
        available=is_available,
        feature_flag=config.feature_flag,
        storage_strategy=config.storage_strategy,
        supported_extract_types=config.extract_types_list,
        docker_available=docker_available,
        cli_available=cli_available
    )


@router.post("/archive", response_model=ArchiveUrlResponse)
async def archive_url(
    request: ArchiveUrlRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Archive a URL using ArchiveBox."""
    if not archivebox_service:
        raise HTTPException(status_code=503, detail="ArchiveBox service not initialized")

    if not await archivebox_service.is_available():
        raise HTTPException(status_code=503, detail="ArchiveBox is not available")

    try:
        url_str = str(request.url)
        user_id = current_user.id if current_user else None

        if request.async_processing:
            # Queue for background processing
            job = create_archive_job(
                url=url_str,
                user_id=user_id,
                extract_types=request.extract_types,
                timeout=request.timeout,
                only_new=request.only_new,
                overwrite=request.overwrite,
                storage_strategy=request.storage_strategy,
                priority=request.priority,
                integration_metadata=request.metadata
            )

            return ArchiveUrlResponse(
                success=True,
                message=f"Archive job queued for {url_str}",
                job_key=job.job_key,
                estimated_duration=archivebox_service.config.timeout
            )
        else:
            # Process immediately
            archive_request = ArchiveRequest(
                url=url_str,
                user_id=user_id,
                extract_types=request.extract_types,
                timeout=request.timeout,
                only_new=request.only_new,
                overwrite=request.overwrite,
                metadata=request.metadata or {}
            )

            result = await archivebox_service.archive_url(archive_request)

            return ArchiveUrlResponse(
                success=result.success,
                message=f"Archive {'completed' if result.success else 'failed'} for {url_str}",
                archive_result=result.to_dict()
            )

    except Exception as e:
        logger.error(f"Failed to archive URL {request.url}: {e}")
        raise HTTPException(status_code=500, detail=f"Archive failed: {str(e)}")


@router.get("/archive/{url:path}", response_model=ArchiveStatusResponse)
async def get_archive_status(url: str):
    """Get archive status for a specific URL."""
    if not archivebox_service:
        raise HTTPException(status_code=503, detail="ArchiveBox service not initialized")

    try:
        result = await archivebox_service.get_archive_status(url)

        return ArchiveStatusResponse(
            url=url,
            archived=result is not None and result.success,
            archive_result=result.to_dict() if result else None,
            last_updated=result.timestamp if result else None
        )

    except Exception as e:
        logger.error(f"Failed to get archive status for {url}: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.get("/archives", response_model=ArchiveListResponse)
async def list_archives(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user)
):
    """List archived URLs with pagination."""
    if not archivebox_service:
        raise HTTPException(status_code=503, detail="ArchiveBox service not initialized")

    # TODO: Implement listing from ArchiveBox index
    # For now, return placeholder response
    return ArchiveListResponse(
        total=0,
        archives=[],
        page=page,
        limit=limit
    )


@router.get("/content/{snapshot_id}", response_model=ArchiveContentResponse)
async def get_archive_content(snapshot_id: str):
    """Get content from an archived snapshot."""
    if not archivebox_service:
        raise HTTPException(status_code=503, detail="ArchiveBox service not initialized")

    try:
        content = await archivebox_service.get_archive_content(snapshot_id)

        if not content:
            raise HTTPException(status_code=404, detail="Archive content not found")

        available_formats = list(content.keys())

        return ArchiveContentResponse(
            snapshot_id=snapshot_id,
            content=content,
            available_formats=available_formats
        )

    except Exception as e:
        logger.error(f"Failed to get archive content for {snapshot_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Content retrieval failed: {str(e)}")


@router.get("/content/{snapshot_id}/file/{file_type}")
async def get_archive_file(snapshot_id: str, file_type: str):
    """Get a specific file from an archived snapshot."""
    if not archivebox_service:
        raise HTTPException(status_code=503, detail="ArchiveBox service not initialized")

    try:
        # Map file types to filenames
        file_mapping = {
            'html': 'output.html',
            'pdf': 'output.pdf',
            'screenshot': 'screenshot.png',
            'text': 'output.txt',
            'title': 'title.txt'
        }

        if file_type not in file_mapping:
            raise HTTPException(status_code=400, detail=f"Invalid file type: {file_type}")

        archive_path = archivebox_service.config.data_dir / "archive" / snapshot_id
        file_path = archive_path / file_mapping[file_type]

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found in archive")

        # Determine media type
        media_types = {
            'html': 'text/html',
            'pdf': 'application/pdf',
            'screenshot': 'image/png',
            'text': 'text/plain',
            'title': 'text/plain'
        }

        return FileResponse(
            path=str(file_path),
            media_type=media_types[file_type],
            filename=f"{snapshot_id}_{file_mapping[file_type]}"
        )

    except Exception as e:
        logger.error(f"Failed to get archive file {file_type} for {snapshot_id}: {e}")
        raise HTTPException(status_code=500, detail=f"File retrieval failed: {str(e)}")


@router.post("/cleanup")
async def cleanup_old_archives(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Clean up old archives based on configuration."""
    if not archivebox_service:
        raise HTTPException(status_code=503, detail="ArchiveBox service not initialized")

    # Require authentication for cleanup operations
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required for cleanup")

    background_tasks.add_task(archivebox_service.cleanup_old_archives)

    return JSONResponse(
        content={"message": "Archive cleanup started in background"},
        status_code=202
    )


@router.get("/jobs/{job_key}")
async def get_archive_job_status(job_key: str):
    """Get status of an archive job."""
    job = ingestion_queue.get_job_by_key(job_key)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.job_type != IngestionJobType.ARCHIVE_WEB_CAPTURE.value:
        raise HTTPException(status_code=400, detail="Not an archive job")

    return JSONResponse(content=job.to_api())


@router.post("/bulk-archive")
async def bulk_archive_urls(
    urls: List[HttpUrl],
    extract_types: Optional[List[str]] = None,
    priority: int = 0,
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_user)
):
    """Archive multiple URLs in bulk."""
    if not archivebox_service:
        raise HTTPException(status_code=503, detail="ArchiveBox service not initialized")

    if not await archivebox_service.is_available():
        raise HTTPException(status_code=503, detail="ArchiveBox is not available")

    if len(urls) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 URLs allowed per bulk request")

    try:
        user_id = current_user.id if current_user else None
        job_keys = []

        for url in urls:
            job = create_archive_job(
                url=str(url),
                user_id=user_id,
                extract_types=extract_types,
                priority=priority
            )
            job_keys.append(job.job_key)

        return JSONResponse(
            content={
                "success": True,
                "message": f"Queued {len(urls)} URLs for archiving",
                "job_keys": job_keys
            }
        )

    except Exception as e:
        logger.error(f"Failed to bulk archive URLs: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk archive failed: {str(e)}")


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint for ArchiveBox service."""
    if not archivebox_service:
        return JSONResponse(
            content={"status": "unhealthy", "reason": "Service not initialized"},
            status_code=503
        )

    is_available = await archivebox_service.is_available()

    return JSONResponse(
        content={
            "status": "healthy" if is_available else "degraded",
            "enabled": archivebox_service.config.enabled,
            "feature_flag": archivebox_service.config.feature_flag,
            "available": is_available
        },
        status_code=200 if is_available else 503
    )