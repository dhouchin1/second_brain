"""
Notification Router for Second Brain API.

This module provides REST endpoints for notification management including:
- Retrieving user notifications
- Managing notification preferences
- Marking notifications as read/unread
- Real-time notification statistics
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from pydantic import BaseModel, Field

from services.auth_service import User
from services.notification_service import (
    get_notification_service,
    NotificationService,
    Notification,
    NotificationPreferences,
    NotificationType,
    NotificationPriority
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/notifications", tags=["notifications"])

# This router follows the pattern used by other service routers:
# `get_current_user` is injected from app.py to avoid circular imports.
get_current_user = None

def init_notification_router(get_current_user_func):
    global get_current_user
    get_current_user = get_current_user_func

# Pydantic models for API requests/responses

class NotificationResponse(BaseModel):
    """API response model for notifications."""
    id: str
    user_id: int
    type: str
    title: str
    message: str
    priority: str
    data: Optional[Dict[str, Any]] = None
    created_at: str
    read_at: Optional[str] = None
    expires_at: Optional[str] = None
    persistent: bool = True

class CreateNotificationRequest(BaseModel):
    """API request model for creating notifications."""
    type: str = Field(..., description="Notification type")
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")
    priority: str = Field(default="normal", description="Notification priority")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional notification data")
    expires_at: Optional[str] = Field(None, description="Expiration timestamp")
    persistent: bool = Field(True, description="Whether notification persists in database")

class NotificationPreferencesResponse(BaseModel):
    """API response model for notification preferences."""
    user_id: int
    enabled_types: List[str]
    min_priority: str
    sound_enabled: bool
    desktop_enabled: bool
    email_enabled: bool
    auto_dismiss_seconds: Optional[int]
    max_notifications: int

class UpdateNotificationPreferencesRequest(BaseModel):
    """API request model for updating notification preferences."""
    enabled_types: Optional[List[str]] = Field(None, description="List of enabled notification types")
    min_priority: Optional[str] = Field(None, description="Minimum priority level")
    sound_enabled: Optional[bool] = Field(None, description="Enable sound notifications")
    desktop_enabled: Optional[bool] = Field(None, description="Enable desktop notifications")
    email_enabled: Optional[bool] = Field(None, description="Enable email notifications")
    auto_dismiss_seconds: Optional[int] = Field(None, description="Auto-dismiss timeout in seconds")
    max_notifications: Optional[int] = Field(None, description="Maximum notifications to keep")

class NotificationStatsResponse(BaseModel):
    """API response model for notification statistics."""
    total_count: int
    unread_count: int
    type_counts: Dict[str, int]
    priority_counts: Dict[str, int]
    has_unread: bool

class BulkMarkReadRequest(BaseModel):
    """API request model for bulk marking notifications as read."""
    notification_ids: Optional[List[str]] = Field(None, description="Specific notification IDs to mark as read")
    mark_all: bool = Field(False, description="Mark all notifications as read")

@router.get("/", response_model=List[NotificationResponse])
async def get_notifications(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of notifications to return"),
    unread_only: bool = Query(False, description="Return only unread notifications"),
    type_filter: Optional[str] = Query(None, description="Filter by notification type"),
    priority_filter: Optional[str] = Query(None, description="Filter by minimum priority"),
    current_user: User = Depends(get_current_user)
):
    """Get notifications for the current user."""
    try:
        service = get_notification_service()
        notifications = await service.get_user_notifications(
            user_id=current_user.id,
            limit=limit,
            unread_only=unread_only
        )
        
        # Apply additional filters
        if type_filter:
            notifications = [n for n in notifications if n.type.value == type_filter]
        
        if priority_filter:
            priority_order = {"low": 0, "normal": 1, "high": 2, "urgent": 3}
            min_priority_level = priority_order.get(priority_filter, 0)
            notifications = [
                n for n in notifications 
                if priority_order.get(n.priority.value, 0) >= min_priority_level
            ]
        
        # Convert to response models
        response_notifications = []
        for notification in notifications:
            response_notifications.append(NotificationResponse(
                id=notification.id,
                user_id=notification.user_id,
                type=notification.type.value,
                title=notification.title,
                message=notification.message,
                priority=notification.priority.value,
                data=notification.data,
                created_at=notification.created_at.isoformat() if notification.created_at else "",
                read_at=notification.read_at.isoformat() if notification.read_at else None,
                expires_at=notification.expires_at.isoformat() if notification.expires_at else None,
                persistent=notification.persistent
            ))
        
        return response_notifications
        
    except Exception as e:
        logger.error(f"Error getting notifications for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve notifications")

@router.post("/", response_model=Dict[str, str])
async def create_notification(
    request: CreateNotificationRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new notification (for testing or system notifications)."""
    try:
        # Validate enum values
        try:
            notification_type = NotificationType(request.type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid notification type: {request.type}")
        
        try:
            notification_priority = NotificationPriority(request.priority)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid notification priority: {request.priority}")
        
        # Parse expires_at if provided
        expires_at = None
        if request.expires_at:
            try:
                expires_at = datetime.fromisoformat(request.expires_at)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid expires_at timestamp format")
        
        # Create notification
        service = get_notification_service()
        notification = Notification(
            user_id=current_user.id,
            type=notification_type,
            title=request.title,
            message=request.message,
            priority=notification_priority,
            data=request.data,
            expires_at=expires_at,
            persistent=request.persistent
        )
        
        notification_id = await service.create_notification(notification)
        
        if notification_id:
            return {"notification_id": notification_id, "status": "created"}
        else:
            return {"status": "filtered_out", "message": "Notification was filtered by user preferences"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating notification for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create notification")

@router.get("/preferences", response_model=NotificationPreferencesResponse)
async def get_notification_preferences(current_user: User = Depends(get_current_user)):
    """Get notification preferences for the current user."""
    try:
        service = get_notification_service()
        preferences = await service.get_user_preferences(current_user.id)
        
        return NotificationPreferencesResponse(
            user_id=preferences.user_id,
            enabled_types=[t.value for t in preferences.enabled_types],
            min_priority=preferences.min_priority.value,
            sound_enabled=preferences.sound_enabled,
            desktop_enabled=preferences.desktop_enabled,
            email_enabled=preferences.email_enabled,
            auto_dismiss_seconds=preferences.auto_dismiss_seconds,
            max_notifications=preferences.max_notifications
        )
        
    except Exception as e:
        logger.error(f"Error getting notification preferences for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve notification preferences")

@router.put("/preferences", response_model=Dict[str, str])
async def update_notification_preferences(
    request: UpdateNotificationPreferencesRequest,
    current_user: User = Depends(get_current_user)
):
    """Update notification preferences for the current user."""
    try:
        service = get_notification_service()
        current_preferences = await service.get_user_preferences(current_user.id)
        
        # Update only provided fields
        if request.enabled_types is not None:
            # Validate notification types
            try:
                enabled_types = {NotificationType(t) for t in request.enabled_types}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid notification type: {e}")
            current_preferences.enabled_types = enabled_types
        
        if request.min_priority is not None:
            try:
                current_preferences.min_priority = NotificationPriority(request.min_priority)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid priority: {request.min_priority}")
        
        if request.sound_enabled is not None:
            current_preferences.sound_enabled = request.sound_enabled
        
        if request.desktop_enabled is not None:
            current_preferences.desktop_enabled = request.desktop_enabled
        
        if request.email_enabled is not None:
            current_preferences.email_enabled = request.email_enabled
        
        if request.auto_dismiss_seconds is not None:
            current_preferences.auto_dismiss_seconds = request.auto_dismiss_seconds
        
        if request.max_notifications is not None:
            if request.max_notifications < 1 or request.max_notifications > 1000:
                raise HTTPException(status_code=400, detail="max_notifications must be between 1 and 1000")
            current_preferences.max_notifications = request.max_notifications
        
        # Save updated preferences
        await service.update_user_preferences(current_preferences)
        
        return {"status": "updated"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating notification preferences for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update notification preferences")

@router.post("/{notification_id}/read", response_model=Dict[str, str])
async def mark_notification_read(
    notification_id: str,
    current_user: User = Depends(get_current_user)
):
    """Mark a specific notification as read."""
    try:
        service = get_notification_service()
        await service.mark_notification_read(notification_id, current_user.id)
        return {"status": "marked_read"}
        
    except Exception as e:
        logger.error(f"Error marking notification {notification_id} as read for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to mark notification as read")

@router.post("/read", response_model=Dict[str, str])
async def bulk_mark_read(
    request: BulkMarkReadRequest,
    current_user: User = Depends(get_current_user)
):
    """Mark multiple notifications or all notifications as read."""
    try:
        service = get_notification_service()
        
        if request.mark_all:
            await service.mark_all_notifications_read(current_user.id)
            return {"status": "all_marked_read"}
        elif request.notification_ids:
            for notification_id in request.notification_ids:
                await service.mark_notification_read(notification_id, current_user.id)
            return {"status": "marked_read", "count": len(request.notification_ids)}
        else:
            raise HTTPException(status_code=400, detail="Either mark_all or notification_ids must be provided")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error bulk marking notifications as read for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to mark notifications as read")

@router.delete("/{notification_id}", response_model=Dict[str, str])
async def delete_notification(
    notification_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a specific notification."""
    try:
        service = get_notification_service()
        await service.delete_notification(notification_id, current_user.id)
        return {"status": "deleted"}
        
    except Exception as e:
        logger.error(f"Error deleting notification {notification_id} for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete notification")

@router.get("/stats", response_model=NotificationStatsResponse)
async def get_notification_stats(current_user: User = Depends(get_current_user)):
    """Get notification statistics for the current user."""
    try:
        service = get_notification_service()
        stats = await service.get_notification_stats(current_user.id)
        
        return NotificationStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error getting notification stats for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve notification statistics")

@router.post("/cleanup", response_model=Dict[str, str])
async def cleanup_expired_notifications(current_user: User = Depends(get_current_user)):
    """Manually trigger cleanup of expired notifications."""
    try:
        service = get_notification_service()
        await service.cleanup_expired_notifications()
        return {"status": "cleanup_completed"}
        
    except Exception as e:
        logger.error(f"Error cleaning up expired notifications: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup expired notifications")

@router.get("/types", response_model=List[str])
async def get_notification_types():
    """Get all available notification types."""
    return [t.value for t in NotificationType]

@router.get("/priorities", response_model=List[str])
async def get_notification_priorities():
    """Get all available notification priorities."""
    return [p.value for p in NotificationPriority]
