"""
Enhanced WebSocket Manager for Second Brain.

This module provides comprehensive WebSocket connection management with:
- Real-time notification broadcasting
- Connection health monitoring
- User presence tracking
- Message queuing for offline users
- Performance metrics
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, asdict

from fastapi import WebSocket, WebSocketDisconnect
from services.notification_service import get_notification_service, Notification

logger = logging.getLogger(__name__)

@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection."""
    websocket: WebSocket
    user_id: int
    connected_at: datetime
    last_ping: datetime
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'user_id': self.user_id,
            'connected_at': self.connected_at.isoformat(),
            'last_ping': self.last_ping.isoformat(),
            'user_agent': self.user_agent,
            'ip_address': self.ip_address
        }

@dataclass
class UserPresence:
    """User presence information."""
    user_id: int
    is_online: bool
    last_seen: datetime
    active_connections: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'user_id': self.user_id,
            'is_online': self.is_online,
            'last_seen': self.last_seen.isoformat(),
            'active_connections': self.active_connections
        }

class EnhancedConnectionManager:
    """Enhanced WebSocket connection manager with notification broadcasting."""
    
    def __init__(self):
        self.active_connections: Dict[str, ConnectionInfo] = {}  # connection_id -> ConnectionInfo
        self.user_connections: Dict[int, Set[str]] = defaultdict(set)  # user_id -> set of connection_ids
        self.connection_counter = 0
        self.message_queue: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))  # user_id -> message queue
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'messages_sent': 0,
            'messages_queued': 0,
            'connection_errors': 0,
            'last_cleanup': datetime.now()
        }
        self._cleanup_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None

    def _ensure_background_tasks(self) -> None:
        """Start long-running background tasks once an event loop is active."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop yet; tasks will be started on first async entry.
            return

        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = loop.create_task(self._periodic_cleanup())
        if self._ping_task is None or self._ping_task.done():
            self._ping_task = loop.create_task(self._periodic_ping())
    
    def _generate_connection_id(self) -> str:
        """Generate a unique connection ID."""
        self.connection_counter += 1
        return f"conn_{int(time.time())}_{self.connection_counter}"
    
    async def connect(self, websocket: WebSocket, user_id: int, 
                     user_agent: Optional[str] = None, 
                     ip_address: Optional[str] = None) -> str:
        """Connect a new WebSocket with enhanced tracking."""
        try:
            self._ensure_background_tasks()
            await websocket.accept()
            
            connection_id = self._generate_connection_id()
            now = datetime.now()
            
            # Create connection info
            connection_info = ConnectionInfo(
                websocket=websocket,
                user_id=user_id,
                connected_at=now,
                last_ping=now,
                user_agent=user_agent,
                ip_address=ip_address
            )
            
            # Store connection
            self.active_connections[connection_id] = connection_info
            self.user_connections[user_id].add(connection_id)
            
            # Update stats
            self.stats['total_connections'] += 1
            self.stats['active_connections'] = len(self.active_connections)
            
            # Send queued messages
            await self._send_queued_messages(user_id)
            
            # Send welcome message
            await self._send_to_connection(connection_id, {
                'type': 'connection_established',
                'connection_id': connection_id,
                'timestamp': now.isoformat(),
                'user_id': user_id
            })
            
            # Broadcast user presence update
            await self._broadcast_presence_update(user_id)
            
            logger.info(f"User {user_id} connected with connection {connection_id}. "
                       f"Active connections: {len(self.user_connections[user_id])}")
            
            return connection_id
            
        except Exception as e:
            logger.error(f"Error connecting user {user_id}: {e}")
            self.stats['connection_errors'] += 1
            raise
    
    async def disconnect(self, connection_id: str, user_id: int):
        """Disconnect a WebSocket connection."""
        try:
            if connection_id in self.active_connections:
                # Remove connection
                del self.active_connections[connection_id]
                self.user_connections[user_id].discard(connection_id)
                
                # Clean up empty user entries
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
                
                # Update stats
                self.stats['active_connections'] = len(self.active_connections)
                
                # Broadcast user presence update
                await self._broadcast_presence_update(user_id)
                
                logger.info(f"User {user_id} disconnected from {connection_id}. "
                           f"Remaining connections: {len(self.user_connections.get(user_id, []))}")
                
        except Exception as e:
            logger.error(f"Error disconnecting {connection_id}: {e}")
    
    async def _send_to_connection(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """Send message to a specific connection."""
        try:
            if connection_id in self.active_connections:
                connection_info = self.active_connections[connection_id]
                await connection_info.websocket.send_json(message)
                self.stats['messages_sent'] += 1
                return True
        except Exception as e:
            logger.warning(f"Failed to send message to connection {connection_id}: {e}")
            # Connection is likely dead, remove it
            if connection_id in self.active_connections:
                user_id = self.active_connections[connection_id].user_id
                await self.disconnect(connection_id, user_id)
        return False
    
    async def send_to_user(self, user_id: int, message: Dict[str, Any]) -> int:
        """Send message to all connections of a user."""
        if user_id not in self.user_connections:
            # User is offline, queue the message
            self.message_queue[user_id].append({
                **message,
                'queued_at': datetime.now().isoformat()
            })
            self.stats['messages_queued'] += 1
            logger.debug(f"Queued message for offline user {user_id}")
            return 0
        
        sent_count = 0
        connection_ids = list(self.user_connections[user_id])  # Create copy to avoid modification during iteration
        
        for connection_id in connection_ids:
            if await self._send_to_connection(connection_id, message):
                sent_count += 1
        
        return sent_count
    
    async def broadcast_to_all(self, message: Dict[str, Any]) -> int:
        """Broadcast message to all connected users."""
        sent_count = 0
        for user_id in list(self.user_connections.keys()):
            sent_count += await self.send_to_user(user_id, message)
        return sent_count
    
    async def send_notification(self, notification: Notification) -> int:
        """Send a notification via WebSocket."""
        message = {
            'type': 'notification',
            'notification': notification.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        return await self.send_to_user(notification.user_id, message)
    
    async def _send_queued_messages(self, user_id: int):
        """Send queued messages when user comes online."""
        if user_id in self.message_queue and self.message_queue[user_id]:
            queued_messages = list(self.message_queue[user_id])
            self.message_queue[user_id].clear()
            
            for message in queued_messages:
                await self.send_to_user(user_id, message)
            
            logger.info(f"Sent {len(queued_messages)} queued messages to user {user_id}")
    
    async def _broadcast_presence_update(self, user_id: int):
        """Broadcast user presence update to relevant connections."""
        presence = self.get_user_presence(user_id)
        message = {
            'type': 'presence_update',
            'presence': presence.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Broadcast to other users (if implementing collaborative features)
        # For now, just send to the user themselves
        await self.send_to_user(user_id, message)
    
    async def handle_message(self, connection_id: str, data: str):
        """Handle incoming WebSocket message."""
        try:
            if connection_id not in self.active_connections:
                logger.warning(f"Received message from unknown connection {connection_id}")
                return
            
            connection_info = self.active_connections[connection_id]
            message = json.loads(data)
            message_type = message.get('type')
            
            # Update last ping time
            connection_info.last_ping = datetime.now()
            
            if message_type == 'ping':
                # Respond with pong
                await self._send_to_connection(connection_id, {
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                })
                
            elif message_type == 'subscribe_notifications':
                # Send recent notifications
                await self._send_recent_notifications(connection_info.user_id)
                
            elif message_type == 'mark_notification_read':
                # Mark notification as read
                notification_id = message.get('notification_id')
                if notification_id:
                    service = get_notification_service()
                    await service.mark_notification_read(notification_id, connection_info.user_id)
                    
            elif message_type == 'typing':
                # Handle typing indicators (for future collaborative features)
                pass
                
            else:
                logger.debug(f"Unknown message type '{message_type}' from {connection_id}")
                
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from connection {connection_id}: {data}")
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")
    
    async def _send_recent_notifications(self, user_id: int, limit: int = 10):
        """Send recent unread notifications to user."""
        try:
            service = get_notification_service()
            notifications = await service.get_user_notifications(
                user_id=user_id,
                limit=limit,
                unread_only=True
            )
            
            for notification in notifications:
                await self.send_notification(notification)
                
        except Exception as e:
            logger.error(f"Error sending recent notifications to user {user_id}: {e}")
    
    def get_user_presence(self, user_id: int) -> UserPresence:
        """Get user presence information."""
        is_online = user_id in self.user_connections
        active_connections = len(self.user_connections.get(user_id, []))
        
        # Calculate last_seen
        last_seen = datetime.now()
        if is_online:
            # Find most recent ping time
            for connection_id in self.user_connections.get(user_id, []):
                if connection_id in self.active_connections:
                    connection_last_ping = self.active_connections[connection_id].last_ping
                    if connection_last_ping > last_seen:
                        last_seen = connection_last_ping
        else:
            # User is offline, use a default or lookup from database
            last_seen = datetime.now() - timedelta(minutes=5)  # Placeholder
        
        return UserPresence(
            user_id=user_id,
            is_online=is_online,
            last_seen=last_seen,
            active_connections=active_connections
        )
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            **self.stats,
            'active_users': len(self.user_connections),
            'queued_messages': sum(len(queue) for queue in self.message_queue.values()),
            'avg_connections_per_user': (
                len(self.active_connections) / len(self.user_connections)
                if self.user_connections else 0
            )
        }
    
    def get_all_connections(self) -> List[Dict[str, Any]]:
        """Get information about all active connections."""
        connections = []
        for connection_id, connection_info in self.active_connections.items():
            connections.append({
                'connection_id': connection_id,
                **connection_info.to_dict()
            })
        return connections
    
    async def _periodic_cleanup(self):
        """Periodically clean up dead connections and expired messages."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                now = datetime.now()
                dead_connections = []
                
                # Find connections that haven't pinged recently
                for connection_id, connection_info in self.active_connections.items():
                    if now - connection_info.last_ping > timedelta(minutes=5):
                        dead_connections.append((connection_id, connection_info.user_id))
                
                # Remove dead connections
                for connection_id, user_id in dead_connections:
                    await self.disconnect(connection_id, user_id)
                    logger.info(f"Cleaned up dead connection {connection_id}")
                
                # Clean up old queued messages
                for user_id, queue in list(self.message_queue.items()):
                    if queue:
                        # Remove messages older than 24 hours
                        cutoff = now - timedelta(hours=24)
                        while queue and 'queued_at' in queue[0]:
                            try:
                                queued_at = datetime.fromisoformat(queue[0]['queued_at'])
                                if queued_at < cutoff:
                                    queue.popleft()
                                else:
                                    break
                            except (ValueError, KeyError):
                                queue.popleft()
                    
                    # Remove empty queues
                    if not queue:
                        del self.message_queue[user_id]
                
                self.stats['last_cleanup'] = now
                
                if dead_connections:
                    logger.info(f"Cleanup completed: removed {len(dead_connections)} dead connections")
                    
            except asyncio.CancelledError:
                logger.info("Periodic cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error during periodic cleanup: {e}")
            
    async def _periodic_ping(self):
        """Periodically ping all connections to detect dead ones."""
        while True:
            try:
                await asyncio.sleep(30)  # Ping every 30 seconds
                
                ping_message = {
                    'type': 'server_ping',
                    'timestamp': datetime.now().isoformat()
                }
                
                await self.broadcast_to_all(ping_message)
                
            except asyncio.CancelledError:
                logger.info("Periodic ping task cancelled")
                break
            except Exception as e:
                logger.error(f"Error during periodic ping: {e}")

# Global connection manager instance
_connection_manager: Optional[EnhancedConnectionManager] = None

def get_connection_manager() -> EnhancedConnectionManager:
    """Get the global connection manager instance."""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = EnhancedConnectionManager()
    return _connection_manager
