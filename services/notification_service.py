"""
Real-time Notification Service for Second Brain.

This service provides comprehensive notification management including:
- Real-time WebSocket notifications
- User preference management  
- Notification persistence and history
- Type-based notification categorization
- Processing status notifications
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from contextlib import contextmanager

from config import settings
from database import get_db_connection

logger = logging.getLogger(__name__)

class NotificationType(Enum):
    """Enumeration of notification types."""
    PROCESSING_STARTED = "processing_started"
    PROCESSING_COMPLETED = "processing_completed" 
    PROCESSING_FAILED = "processing_failed"
    NOTE_CREATED = "note_created"
    NOTE_UPDATED = "note_updated"
    NOTE_DELETED = "note_deleted"
    SEARCH_INDEXED = "search_indexed"
    AUDIO_TRANSCRIBED = "audio_transcribed"
    FILE_UPLOADED = "file_uploaded"
    SYSTEM_STATUS = "system_status"
    SYNC_COMPLETED = "sync_completed"
    SYNC_FAILED = "sync_failed"

class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

@dataclass
class Notification:
    """Notification data structure."""
    id: Optional[str] = None
    user_id: int = 0
    type: NotificationType = NotificationType.SYSTEM_STATUS
    title: str = ""
    message: str = ""
    priority: NotificationPriority = NotificationPriority.NORMAL
    data: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    persistent: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert notification to dictionary for JSON serialization."""
        result = asdict(self)
        result['type'] = self.type.value
        result['priority'] = self.priority.value
        if self.created_at:
            result['created_at'] = self.created_at.isoformat()
        if self.read_at:
            result['read_at'] = self.read_at.isoformat()
        if self.expires_at:
            result['expires_at'] = self.expires_at.isoformat()
        return result

@dataclass
class NotificationPreferences:
    """User notification preferences."""
    user_id: int
    enabled_types: Set[NotificationType]
    min_priority: NotificationPriority = NotificationPriority.NORMAL
    sound_enabled: bool = True
    desktop_enabled: bool = True
    email_enabled: bool = False
    auto_dismiss_seconds: Optional[int] = None
    max_notifications: int = 50

class NotificationService:
    """Service for managing real-time notifications."""
    
    def __init__(self):
        self._user_preferences: Dict[int, NotificationPreferences] = {}
        self._notification_history: Dict[int, List[Notification]] = {}
        self._init_database()
        
    def _init_database(self):
        """Initialize notification database tables."""
        try:
            conn = get_db_connection()
            with conn:
                # Create notifications table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS notifications (
                        id TEXT PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        message TEXT NOT NULL,
                        priority TEXT NOT NULL DEFAULT 'normal',
                        data TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        read_at TIMESTAMP,
                        expires_at TIMESTAMP,
                        persistent BOOLEAN DEFAULT 1,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                # Create notification_preferences table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS notification_preferences (
                        user_id INTEGER PRIMARY KEY,
                        enabled_types TEXT NOT NULL DEFAULT '{}',
                        min_priority TEXT NOT NULL DEFAULT 'normal',
                        sound_enabled BOOLEAN DEFAULT 1,
                        desktop_enabled BOOLEAN DEFAULT 1,
                        email_enabled BOOLEAN DEFAULT 0,
                        auto_dismiss_seconds INTEGER,
                        max_notifications INTEGER DEFAULT 50,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                # Create indexes for performance
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_notifications_user_created 
                    ON notifications (user_id, created_at DESC)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_notifications_type 
                    ON notifications (type)
                """)
                
        except Exception as e:
            logger.error(f"Failed to initialize notification database: {e}")
    
    async def create_notification(self, notification: Notification) -> Optional[str]:
        """Create and store a new notification."""
        try:
            # Generate ID if not provided
            if not notification.id:
                notification.id = f"notif_{notification.user_id}_{int(datetime.now().timestamp() * 1000)}"
            
            # Set created_at if not provided
            if not notification.created_at:
                notification.created_at = datetime.now()
            
            # Check user preferences
            preferences = await self.get_user_preferences(notification.user_id)
            if not self._should_send_notification(notification, preferences):
                return None
            
            # Store persistent notifications in database
            if notification.persistent:
                await self._store_notification(notification)
            
            # Add to in-memory history
            if notification.user_id not in self._notification_history:
                self._notification_history[notification.user_id] = []
            
            user_notifications = self._notification_history[notification.user_id]
            user_notifications.append(notification)
            
            # Trim history to max_notifications
            max_notifications = preferences.max_notifications
            if len(user_notifications) > max_notifications:
                self._notification_history[notification.user_id] = user_notifications[-max_notifications:]
            
            logger.info(f"Created notification {notification.id} for user {notification.user_id}")
            return notification.id
            
        except Exception as e:
            logger.error(f"Failed to create notification: {e}")
            return None
    
    async def _store_notification(self, notification: Notification):
        """Store notification in database."""
        try:
            conn = get_db_connection()
            with conn:
                conn.execute("""
                    INSERT INTO notifications 
                    (id, user_id, type, title, message, priority, data, created_at, expires_at, persistent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    notification.id,
                    notification.user_id,
                    notification.type.value,
                    notification.title,
                    notification.message,
                    notification.priority.value,
                    json.dumps(notification.data) if notification.data else None,
                    notification.created_at,
                    notification.expires_at,
                    notification.persistent
                ))
        except Exception as e:
            logger.error(f"Failed to store notification in database: {e}")
    
    def _should_send_notification(self, notification: Notification, preferences: NotificationPreferences) -> bool:
        """Check if notification should be sent based on user preferences."""
        # Check if notification type is enabled
        if notification.type not in preferences.enabled_types:
            return False
        
        # Check priority threshold
        priority_order = {
            NotificationPriority.LOW: 0,
            NotificationPriority.NORMAL: 1,
            NotificationPriority.HIGH: 2,
            NotificationPriority.URGENT: 3
        }
        
        if priority_order[notification.priority] < priority_order[preferences.min_priority]:
            return False
        
        # Check expiration
        if notification.expires_at and notification.expires_at < datetime.now():
            return False
        
        return True
    
    async def get_user_preferences(self, user_id: int) -> NotificationPreferences:
        """Get user notification preferences."""
        if user_id in self._user_preferences:
            return self._user_preferences[user_id]
        
        try:
            conn = get_db_connection()
            cursor = conn.execute("""
                SELECT enabled_types, min_priority, sound_enabled, desktop_enabled, 
                       email_enabled, auto_dismiss_seconds, max_notifications
                FROM notification_preferences 
                WHERE user_id = ?
            """, (user_id,))
            
            row = cursor.fetchone()
            if row:
                enabled_types_json, min_priority, sound_enabled, desktop_enabled, \
                email_enabled, auto_dismiss_seconds, max_notifications = row
                
                try:
                    enabled_types_list = json.loads(enabled_types_json or '[]')
                    enabled_types = {NotificationType(t) for t in enabled_types_list}
                except (json.JSONDecodeError, ValueError):
                    enabled_types = set(NotificationType)  # Default to all types
                
                preferences = NotificationPreferences(
                    user_id=user_id,
                    enabled_types=enabled_types,
                    min_priority=NotificationPriority(min_priority),
                    sound_enabled=bool(sound_enabled),
                    desktop_enabled=bool(desktop_enabled),
                    email_enabled=bool(email_enabled),
                    auto_dismiss_seconds=auto_dismiss_seconds,
                    max_notifications=max_notifications or 50
                )
            else:
                # Default preferences for new user
                preferences = NotificationPreferences(
                    user_id=user_id,
                    enabled_types=set(NotificationType)  # All types enabled by default
                )
                await self.update_user_preferences(preferences)
            
            self._user_preferences[user_id] = preferences
            return preferences
            
        except Exception as e:
            logger.error(f"Failed to get user preferences: {e}")
            # Return default preferences
            return NotificationPreferences(
                user_id=user_id,
                enabled_types=set(NotificationType)
            )
    
    async def update_user_preferences(self, preferences: NotificationPreferences):
        """Update user notification preferences."""
        try:
            enabled_types_json = json.dumps([t.value for t in preferences.enabled_types])
            
            conn = get_db_connection()
            with conn:
                conn.execute("""
                    INSERT OR REPLACE INTO notification_preferences 
                    (user_id, enabled_types, min_priority, sound_enabled, desktop_enabled,
                     email_enabled, auto_dismiss_seconds, max_notifications, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    preferences.user_id,
                    enabled_types_json,
                    preferences.min_priority.value,
                    preferences.sound_enabled,
                    preferences.desktop_enabled,
                    preferences.email_enabled,
                    preferences.auto_dismiss_seconds,
                    preferences.max_notifications
                ))
            
            # Update cache
            self._user_preferences[preferences.user_id] = preferences
            logger.info(f"Updated notification preferences for user {preferences.user_id}")
            
        except Exception as e:
            logger.error(f"Failed to update user preferences: {e}")
    
    async def get_user_notifications(self, user_id: int, limit: int = 50, 
                                   unread_only: bool = False) -> List[Notification]:
        """Get notifications for a user."""
        try:
            conn = get_db_connection()
            
            query = """
                SELECT id, user_id, type, title, message, priority, data, 
                       created_at, read_at, expires_at, persistent
                FROM notifications 
                WHERE user_id = ?
            """
            params = [user_id]
            
            if unread_only:
                query += " AND read_at IS NULL"
            
            # Filter out expired notifications
            query += " AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)"
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            notifications = []
            
            for row in cursor.fetchall():
                notification = Notification(
                    id=row[0],
                    user_id=row[1],
                    type=NotificationType(row[2]),
                    title=row[3],
                    message=row[4],
                    priority=NotificationPriority(row[5]),
                    data=json.loads(row[6]) if row[6] else None,
                    created_at=datetime.fromisoformat(row[7]) if row[7] else None,
                    read_at=datetime.fromisoformat(row[8]) if row[8] else None,
                    expires_at=datetime.fromisoformat(row[9]) if row[9] else None,
                    persistent=bool(row[10])
                )
                notifications.append(notification)
            
            return notifications
            
        except Exception as e:
            logger.error(f"Failed to get user notifications: {e}")
            return []
    
    async def mark_notification_read(self, notification_id: str, user_id: int):
        """Mark a notification as read."""
        try:
            conn = get_db_connection()
            with conn:
                conn.execute("""
                    UPDATE notifications 
                    SET read_at = CURRENT_TIMESTAMP 
                    WHERE id = ? AND user_id = ?
                """, (notification_id, user_id))
                
            # Update in-memory history
            if user_id in self._notification_history:
                for notification in self._notification_history[user_id]:
                    if notification.id == notification_id:
                        notification.read_at = datetime.now()
                        break
                        
            logger.info(f"Marked notification {notification_id} as read for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to mark notification as read: {e}")
    
    async def mark_all_notifications_read(self, user_id: int):
        """Mark all notifications as read for a user."""
        try:
            conn = get_db_connection()
            with conn:
                conn.execute("""
                    UPDATE notifications 
                    SET read_at = CURRENT_TIMESTAMP 
                    WHERE user_id = ? AND read_at IS NULL
                """, (user_id,))
            
            # Update in-memory history
            if user_id in self._notification_history:
                for notification in self._notification_history[user_id]:
                    if not notification.read_at:
                        notification.read_at = datetime.now()
                        
            logger.info(f"Marked all notifications as read for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to mark all notifications as read: {e}")
    
    async def delete_notification(self, notification_id: str, user_id: int):
        """Delete a notification."""
        try:
            conn = get_db_connection()
            with conn:
                conn.execute("""
                    DELETE FROM notifications 
                    WHERE id = ? AND user_id = ?
                """, (notification_id, user_id))
            
            # Remove from in-memory history
            if user_id in self._notification_history:
                self._notification_history[user_id] = [
                    n for n in self._notification_history[user_id] 
                    if n.id != notification_id
                ]
                
            logger.info(f"Deleted notification {notification_id} for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete notification: {e}")
    
    async def cleanup_expired_notifications(self):
        """Clean up expired notifications from database and memory."""
        try:
            conn = get_db_connection()
            with conn:
                cursor = conn.execute("""
                    DELETE FROM notifications 
                    WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP
                """)
                deleted_count = cursor.rowcount
            
            # Clean up in-memory history
            current_time = datetime.now()
            for user_id in list(self._notification_history.keys()):
                self._notification_history[user_id] = [
                    n for n in self._notification_history[user_id]
                    if not n.expires_at or n.expires_at > current_time
                ]
                
                if not self._notification_history[user_id]:
                    del self._notification_history[user_id]
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired notifications")
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired notifications: {e}")
    
    async def get_notification_stats(self, user_id: int) -> Dict[str, Any]:
        """Get notification statistics for a user."""
        try:
            conn = get_db_connection()
            
            # Get total count
            cursor = conn.execute("""
                SELECT COUNT(*) FROM notifications WHERE user_id = ?
            """, (user_id,))
            total_count = cursor.fetchone()[0]
            
            # Get unread count
            cursor = conn.execute("""
                SELECT COUNT(*) FROM notifications 
                WHERE user_id = ? AND read_at IS NULL
                AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """, (user_id,))
            unread_count = cursor.fetchone()[0]
            
            # Get counts by type
            cursor = conn.execute("""
                SELECT type, COUNT(*) FROM notifications 
                WHERE user_id = ? 
                AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                GROUP BY type
            """, (user_id,))
            type_counts = dict(cursor.fetchall())
            
            # Get counts by priority
            cursor = conn.execute("""
                SELECT priority, COUNT(*) FROM notifications 
                WHERE user_id = ? 
                AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                GROUP BY priority
            """, (user_id,))
            priority_counts = dict(cursor.fetchall())
            
            return {
                'total_count': total_count,
                'unread_count': unread_count,
                'type_counts': type_counts,
                'priority_counts': priority_counts,
                'has_unread': unread_count > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get notification stats: {e}")
            return {
                'total_count': 0,
                'unread_count': 0,
                'type_counts': {},
                'priority_counts': {},
                'has_unread': False
            }

# Global notification service instance
_notification_service: Optional[NotificationService] = None

def get_notification_service() -> NotificationService:
    """Get the global notification service instance."""
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service

# Convenience functions for common notification types
async def notify_processing_started(user_id: int, task_type: str, task_id: str):
    """Send processing started notification."""
    service = get_notification_service()
    notification = Notification(
        user_id=user_id,
        type=NotificationType.PROCESSING_STARTED,
        title="Processing Started",
        message=f"Started processing {task_type}",
        priority=NotificationPriority.NORMAL,
        data={"task_type": task_type, "task_id": task_id},
        persistent=False
    )
    return await service.create_notification(notification)

async def notify_processing_completed(user_id: int, task_type: str, task_id: str, result_data: Dict[str, Any] = None):
    """Send processing completed notification."""
    service = get_notification_service()
    notification = Notification(
        user_id=user_id,
        type=NotificationType.PROCESSING_COMPLETED,
        title="Processing Completed",
        message=f"Successfully processed {task_type}",
        priority=NotificationPriority.NORMAL,
        data={"task_type": task_type, "task_id": task_id, "result": result_data},
        persistent=True
    )
    return await service.create_notification(notification)

async def notify_processing_failed(user_id: int, task_type: str, task_id: str, error: str):
    """Send processing failed notification."""
    service = get_notification_service()
    notification = Notification(
        user_id=user_id,
        type=NotificationType.PROCESSING_FAILED,
        title="Processing Failed",
        message=f"Failed to process {task_type}: {error}",
        priority=NotificationPriority.HIGH,
        data={"task_type": task_type, "task_id": task_id, "error": error},
        persistent=True
    )
    return await service.create_notification(notification)

async def notify_note_created(user_id: int, note_id: str, title: str):
    """Send note created notification."""
    service = get_notification_service()
    notification = Notification(
        user_id=user_id,
        type=NotificationType.NOTE_CREATED,
        title="Note Created",
        message=f"Created note: {title}",
        priority=NotificationPriority.LOW,
        data={"note_id": note_id, "note_title": title},
        persistent=False,
        expires_at=datetime.now() + timedelta(hours=1)
    )
    return await service.create_notification(notification)