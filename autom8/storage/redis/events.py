"""
Redis EventBus for Agent Coordination

Provides a high-performance event system using Redis Streams for reliable
message delivery between agents in the Autom8 system. Supports both broadcast
and directed messaging with event filtering, persistence, and replay capabilities.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field

# Enhanced event models with backward compatibility
from autom8.models.events import (
    AgentEvent, EventType, EventPriority, EventCategory, EventSeverity,
    BaseEventPayload, TaskEventPayload, AgentEventPayload, DecisionEventPayload,
    EventBuilder, EventValidator, EventValidationError
)
from autom8.models.memory import AgentEvent as LegacyAgentEvent, EventType as LegacyEventType, Priority
from autom8.storage.serialization import EventSerializer, SerializationFormat, CompressionType, event_to_json, event_from_json
from autom8.storage.redis.client import RedisClient, get_redis_client

logger = logging.getLogger(__name__)


class EventFilter(BaseModel):
    """Event filtering configuration for subscriptions."""
    
    # Event type filtering
    event_types: Optional[Set[str]] = Field(default=None, description="Event types to include")
    exclude_types: Optional[Set[str]] = Field(default=None, description="Event types to exclude")
    
    # Agent filtering
    source_agents: Optional[Set[str]] = Field(default=None, description="Source agents to include")
    exclude_sources: Optional[Set[str]] = Field(default=None, description="Source agents to exclude")
    target_agents: Optional[Set[str]] = Field(default=None, description="Target agents to include")
    
    # Priority filtering
    min_priority: Optional[int] = Field(default=None, description="Minimum priority level")
    max_priority: Optional[int] = Field(default=None, description="Maximum priority level")
    
    # Content filtering
    required_fields: Optional[Set[str]] = Field(default=None, description="Required fields in event data")
    content_patterns: Optional[List[str]] = Field(default=None, description="Regex patterns for content matching")
    
    def matches(self, event: Union[AgentEvent, LegacyAgentEvent]) -> bool:
        """Check if an event matches this filter."""
        # Handle both enhanced and legacy events
        event_type = event.type.value if hasattr(event.type, 'value') else str(event.type)
        
        # Event type filtering
        if self.event_types and event_type not in self.event_types:
            return False
        if self.exclude_types and event_type in self.exclude_types:
            return False
            
        # Agent filtering
        if self.source_agents and event.source_agent not in self.source_agents:
            return False
        if self.exclude_sources and event.source_agent in self.exclude_sources:
            return False
        if self.target_agents and event.target_agent and event.target_agent not in self.target_agents:
            return False
            
        # Priority filtering - handle both enhanced and legacy priority
        priority_value = event.priority.value if hasattr(event.priority, 'value') else int(event.priority)
        if self.min_priority is not None and priority_value < self.min_priority:
            return False
        if self.max_priority is not None and priority_value > self.max_priority:
            return False
            
        # Content filtering - handle both payload and data fields
        if self.required_fields:
            if hasattr(event, 'payload') and hasattr(event.payload, 'to_dict'):
                event_fields = set(event.payload.to_dict().keys())
            elif hasattr(event, 'data') and isinstance(event.data, dict):
                event_fields = set(event.data.keys())
            else:
                event_fields = set()
            
            if not self.required_fields.issubset(event_fields):
                return False
                
        # Content pattern matching would require regex implementation
        # Skipping for now as it's not in core requirements
        
        return True


class EventSubscription(BaseModel):
    """Represents an active event subscription."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = Field(description="Subscribing agent ID")
    stream_name: str = Field(description="Redis stream name")
    consumer_group: str = Field(description="Redis consumer group")
    filter: Optional[EventFilter] = Field(default=None, description="Event filter")
    last_id: str = Field(default="$", description="Last processed message ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    active: bool = Field(default=True)


class EventBusConfig(BaseModel):
    """Configuration for EventBus."""
    
    # Stream settings
    max_stream_length: int = Field(default=10000, description="Maximum entries per stream")
    default_block_timeout: int = Field(default=1000, description="Default block timeout in ms")
    batch_size: int = Field(default=100, description="Default batch size for reading")
    
    # Consumer group settings
    consumer_group_prefix: str = Field(default="autom8", description="Prefix for consumer groups")
    consumer_idle_timeout: int = Field(default=300000, description="Consumer idle timeout in ms")
    
    # Event retention
    event_ttl_hours: int = Field(default=24, description="Event TTL in hours")
    cleanup_interval_minutes: int = Field(default=60, description="Cleanup interval in minutes")


class EventBus:
    """
    Enhanced Redis-based EventBus for agent coordination.
    
    Provides publish/subscribe functionality using Redis Streams for reliable
    message delivery with event filtering, persistence, and replay capabilities.
    Now supports enhanced event models with type safety, validation, and serialization.
    """
    
    def __init__(self, config: Optional[EventBusConfig] = None, redis_client: Optional[RedisClient] = None):
        """Initialize EventBus."""
        self.config = config or EventBusConfig()
        self._redis_client = redis_client
        self._subscriptions: Dict[str, EventSubscription] = {}
        self._running_consumers: Dict[str, asyncio.Task] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._shutdown_event = asyncio.Event()
        
        # Enhanced event handling
        self._event_serializer = EventSerializer(
            default_format=SerializationFormat.JSON,
            default_compression=CompressionType.NONE
        )
        self._event_validator = EventValidator()
        
        # Stream names
        self.main_stream = "autom8:events:main"
        self.agent_streams_prefix = "autom8:events:agent:"
        self.broadcast_stream = "autom8:events:broadcast"
        
        # Event statistics
        self._stats = {
            'events_published': 0,
            'events_consumed': 0,
            'validation_errors': 0,
            'serialization_errors': 0
        }
        
    async def _get_redis(self) -> RedisClient:
        """Get Redis client instance."""
        if self._redis_client is None:
            self._redis_client = await get_redis_client()
        return self._redis_client
    
    async def start(self) -> bool:
        """Start the EventBus system."""
        try:
            redis = await self._get_redis()
            if not redis.is_connected:
                logger.error("Redis not connected, cannot start EventBus")
                return False
                
            # Initialize streams and consumer groups
            await self._initialize_streams()
            
            logger.info("EventBus started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start EventBus: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop the EventBus and cleanup resources."""
        try:
            # Signal shutdown
            self._shutdown_event.set()
            
            # Cancel running consumers
            for task in self._running_consumers.values():
                task.cancel()
                
            # Wait for consumers to finish
            if self._running_consumers:
                await asyncio.gather(*self._running_consumers.values(), return_exceptions=True)
            
            self._running_consumers.clear()
            self._subscriptions.clear()
            
            logger.info("EventBus stopped")
            
        except Exception as e:
            logger.error(f"Error stopping EventBus: {e}")
    
    async def _initialize_streams(self) -> None:
        """Initialize Redis streams and consumer groups."""
        redis = await self._get_redis()
        
        # Create main streams if they don't exist
        streams = [self.main_stream, self.broadcast_stream]
        
        for stream in streams:
            # Create consumer group (will create stream if it doesn't exist)
            await redis.xgroup_create(stream, self.config.consumer_group_prefix, id="0", mkstream=True)
    
    async def publish(
        self, 
        event_type: Union[str, EventType], 
        data: Optional[Dict[str, Any]] = None,
        source_agent: str = None,
        target_agent: Optional[str] = None,
        priority: Union[int, EventPriority] = EventPriority.MEDIUM,
        expires_in_hours: Optional[int] = None,
        event: Optional[AgentEvent] = None,
        payload: Optional[BaseEventPayload] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Publish an event to the event bus.
        
        Args:
            event_type: Type of event (EventType enum or string)
            data: Legacy event data (for backward compatibility)
            source_agent: Agent publishing the event
            target_agent: Target agent (None for broadcast)
            priority: Event priority (EventPriority enum or int)
            expires_in_hours: Event expiration in hours
            event: Pre-constructed AgentEvent (takes precedence)
            payload: Event payload (BaseEventPayload)
            **kwargs: Additional event parameters
            
        Returns:
            Event ID if successful, None otherwise
        """
        try:
            redis = await self._get_redis()
            
            # Use provided event or create new one
            if event is None:
                # Handle backward compatibility for legacy calls
                if data is not None and payload is None:
                    # Legacy call with data dict
                    payload = BaseEventPayload(**data)
                elif payload is None:
                    payload = BaseEventPayload()
                
                # Convert string event type to enum if needed
                if isinstance(event_type, str):
                    try:
                        event_type = EventType(event_type)
                    except ValueError:
                        # Handle legacy event types or custom types
                        event_type = EventType.CUSTOM
                
                # Convert priority to enum if needed
                if isinstance(priority, int):
                    try:
                        priority = EventPriority(priority)
                    except ValueError:
                        priority = EventPriority.MEDIUM
                
                # Set expiration
                expires_at = None
                ttl_seconds = None
                if expires_in_hours:
                    expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
                    ttl_seconds = expires_in_hours * 3600
                elif self.config.event_ttl_hours > 0:
                    expires_at = datetime.utcnow() + timedelta(hours=self.config.event_ttl_hours)
                    ttl_seconds = self.config.event_ttl_hours * 3600
                
                # Create enhanced event using EventBuilder
                builder = EventBuilder(event_type, source_agent)\
                    .with_payload(payload)\
                    .with_priority(priority)
                
                if target_agent:
                    builder.with_target(target_agent)
                
                if ttl_seconds:
                    builder.with_ttl(ttl_seconds)
                
                # Add any additional kwargs as metadata
                for key, value in kwargs.items():
                    builder.with_metadata(key, value)
                
                event = builder.build()
            
            # Validate event
            try:
                self._event_validator.validate_event(event)
            except EventValidationError as e:
                logger.error(f"Event validation failed: {e}")
                self._stats['validation_errors'] += 1
                return None
            
            # Determine target streams
            streams_to_publish = []
            
            if target_agent:
                # Directed event - publish to specific agent stream
                agent_stream = f"{self.agent_streams_prefix}{target_agent}"
                streams_to_publish.append(agent_stream)
            else:
                # Broadcast event
                streams_to_publish.append(self.broadcast_stream)
            
            # Always publish to main stream for persistence
            streams_to_publish.append(self.main_stream)
            
            # Serialize event using enhanced serializer
            try:
                event_json = event_to_json(event)
            except Exception as e:
                logger.error(f"Event serialization failed: {e}")
                self._stats['serialization_errors'] += 1
                return None
            
            event_data = {
                "event": event_json,
                "routing": {
                    "source": event.source_agent,
                    "target": event.target_agent,
                    "broadcast": event.is_broadcast,
                    "directed": event.is_directed
                },
                "metadata": {
                    "schema_version": event.schema_version.value,
                    "category": event.category.value,
                    "priority": event.priority.value,
                    "severity": event.severity.value
                }
            }
            
            # Publish to all target streams
            published_to = []
            for stream in streams_to_publish:
                message_id = await redis.xadd(
                    stream, 
                    event_data, 
                    maxlen=self.config.max_stream_length
                )
                if message_id:
                    published_to.append(f"{stream}:{message_id}")
            
            if published_to:
                logger.debug(f"Published event {event.id} to streams: {published_to}")
                self._stats['events_published'] += 1
                return event.id
            else:
                logger.error(f"Failed to publish event {event.id} to any stream")
                return None
                
        except Exception as e:
            logger.error(f"Error publishing event: {e}")
            return None
    
    async def subscribe(
        self, 
        agent_id: str,
        event_filter: Optional[EventFilter] = None,
        include_broadcasts: bool = True
    ) -> str:
        """
        Subscribe to events for an agent.
        
        Args:
            agent_id: ID of the subscribing agent
            event_filter: Optional filter for events
            include_broadcasts: Whether to include broadcast events
            
        Returns:
            Subscription ID
        """
        try:
            redis = await self._get_redis()
            
            # Create subscription
            subscription_id = str(uuid.uuid4())
            agent_stream = f"{self.agent_streams_prefix}{agent_id}"
            consumer_group = f"{self.config.consumer_group_prefix}:{agent_id}"
            
            # Initialize agent stream and consumer group
            await redis.xgroup_create(agent_stream, consumer_group, id="0", mkstream=True)
            
            # If including broadcasts, also set up broadcast subscription
            if include_broadcasts:
                await redis.xgroup_create(self.broadcast_stream, consumer_group, id="0", mkstream=True)
            
            subscription = EventSubscription(
                id=subscription_id,
                agent_id=agent_id,
                stream_name=agent_stream,
                consumer_group=consumer_group,
                filter=event_filter
            )
            
            self._subscriptions[subscription_id] = subscription
            
            logger.info(f"Created subscription {subscription_id} for agent {agent_id}")
            return subscription_id
            
        except Exception as e:
            logger.error(f"Error creating subscription for agent {agent_id}: {e}")
            raise
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        try:
            if subscription_id in self._subscriptions:
                # Cancel consumer task if running
                if subscription_id in self._running_consumers:
                    self._running_consumers[subscription_id].cancel()
                    del self._running_consumers[subscription_id]
                
                # Remove subscription
                subscription = self._subscriptions.pop(subscription_id)
                logger.info(f"Removed subscription {subscription_id} for agent {subscription.agent_id}")
                return True
            else:
                logger.warning(f"Subscription {subscription_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error unsubscribing {subscription_id}: {e}")
            return False
    
    async def listen(
        self, 
        subscription_id: str,
        handler: Optional[Callable[[AgentEvent], None]] = None
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        Listen for events on a subscription.
        
        Args:
            subscription_id: Subscription to listen on
            handler: Optional event handler function
            
        Yields:
            AgentEvent objects as they arrive
        """
        if subscription_id not in self._subscriptions:
            logger.error(f"Subscription {subscription_id} not found")
            return
        
        subscription = self._subscriptions[subscription_id]
        redis = await self._get_redis()
        
        consumer_name = f"consumer:{subscription.agent_id}:{subscription_id[:8]}"
        
        logger.info(f"Starting event listener for subscription {subscription_id}")
        
        try:
            while not self._shutdown_event.is_set() and subscription.active:
                try:
                    # Read from agent-specific stream
                    streams_to_read = {subscription.stream_name: ">"}
                    
                    # Also read from broadcast stream if no specific target
                    if subscription.filter is None or subscription.filter.target_agents is None:
                        streams_to_read[self.broadcast_stream] = ">"
                    
                    # Read messages using XREADGROUP for consumer groups
                    messages = []
                    for stream_name, stream_id in streams_to_read.items():
                        try:
                            stream_messages = await redis.xreadgroup(
                                subscription.consumer_group,
                                consumer_name,
                                {stream_name: stream_id},
                                count=self.config.batch_size,
                                block=self.config.default_block_timeout,
                                noack=False
                            )
                            if stream_messages:
                                messages.extend(stream_messages)
                        except Exception as e:
                            logger.error(f"Error reading from stream {stream_name}: {e}")
                            # If consumer group doesn't exist, try to create it
                            await redis.xgroup_create(stream_name, subscription.consumer_group, id="0", mkstream=True)
                    
                    for stream_name, stream_messages in messages:
                        for message_id, fields in stream_messages:
                            try:
                                # Parse event using enhanced deserializer
                                event_json = fields.get("event")
                                if not event_json:
                                    continue
                                
                                # Use enhanced event deserialization
                                try:
                                    event = event_from_json(event_json)
                                except Exception as e:
                                    # Fallback to legacy deserialization
                                    logger.debug(f"Enhanced deserialization failed, trying legacy: {e}")
                                    event = AgentEvent.model_validate_json(event_json)
                                
                                # Update statistics
                                self._stats['events_consumed'] += 1
                                
                                # Apply filtering
                                if subscription.filter and not subscription.filter.matches(event):
                                    continue
                                
                                # Skip if event is expired
                                if event.is_expired:
                                    continue
                                
                                # Update last processed ID
                                subscription.last_id = message_id
                                
                                # Call handler if provided
                                if handler:
                                    try:
                                        if asyncio.iscoroutinefunction(handler):
                                            await handler(event)
                                        else:
                                            handler(event)
                                    except Exception as e:
                                        logger.error(f"Error in event handler: {e}")
                                
                                # Yield event
                                yield event
                                
                                # Acknowledge message
                                await redis.xack(
                                    stream_name,
                                    subscription.consumer_group,
                                    message_id
                                )
                                
                            except Exception as e:
                                logger.error(f"Error processing message {message_id}: {e}")
                                
                except asyncio.TimeoutError:
                    # Timeout is expected, continue listening
                    continue
                except Exception as e:
                    logger.error(f"Error in event listener: {e}")
                    await asyncio.sleep(1)  # Brief pause before retrying
                    
        except asyncio.CancelledError:
            logger.info(f"Event listener for subscription {subscription_id} cancelled")
        except Exception as e:
            logger.error(f"Event listener error: {e}")
        finally:
            logger.info(f"Event listener for subscription {subscription_id} stopped")
    
    async def get_events(
        self, 
        agent_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[AgentEvent]:
        """
        Retrieve historical events with filtering.
        
        Args:
            agent_id: Filter by target agent
            start_time: Start time for events
            end_time: End time for events
            event_types: Filter by event types
            limit: Maximum number of events to return
            
        Returns:
            List of matching events
        """
        try:
            redis = await self._get_redis()
            
            # Determine which stream to read from
            if agent_id:
                stream_name = f"{self.agent_streams_prefix}{agent_id}"
            else:
                stream_name = self.main_stream
            
            # Convert times to Redis stream IDs if provided
            start_id = "0"
            end_id = "+"
            
            if start_time:
                # Convert to millisecond timestamp
                timestamp_ms = int(start_time.timestamp() * 1000)
                start_id = f"{timestamp_ms}-0"
                
            if end_time:
                timestamp_ms = int(end_time.timestamp() * 1000)
                end_id = f"{timestamp_ms}-0"
            
            # Read messages from stream
            messages = await redis.xrange(
                stream_name,
                min=start_id,
                max=end_id,
                count=limit
            )
            
            events = []
            for message_id, fields in messages:
                try:
                    event_json = fields.get("event")
                    if not event_json:
                        continue
                    
                    event = AgentEvent.model_validate_json(event_json)
                    
                    # Apply additional filtering
                    if event_types and event.type not in event_types:
                        continue
                    
                    events.append(event)
                    
                except Exception as e:
                    logger.error(f"Error parsing event from message {message_id}: {e}")
            
            return events
            
        except Exception as e:
            logger.error(f"Error retrieving events: {e}")
            return []
    
    async def replay_events(
        self,
        subscription_id: str,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
        handler: Optional[Callable[[AgentEvent], None]] = None
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        Replay historical events for a subscription.
        
        Args:
            subscription_id: Subscription ID
            from_time: Start time for replay
            to_time: End time for replay
            handler: Optional event handler
            
        Yields:
            Historical events in chronological order
        """
        if subscription_id not in self._subscriptions:
            logger.error(f"Subscription {subscription_id} not found")
            return
        
        subscription = self._subscriptions[subscription_id]
        
        # Get historical events
        events = await self.get_events(
            agent_id=subscription.agent_id,
            start_time=from_time,
            end_time=to_time,
            limit=10000  # Large limit for replay
        )
        
        # Filter events using subscription filter
        for event in events:
            if subscription.filter and not subscription.filter.matches(event):
                continue
            
            # Call handler if provided
            if handler:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Error in replay handler: {e}")
            
            yield event
    
    async def get_subscription_info(self, subscription_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a subscription."""
        if subscription_id not in self._subscriptions:
            return None
        
        subscription = self._subscriptions[subscription_id]
        redis = await self._get_redis()
        
        # Get consumer group info
        try:
            group_info = await redis.client.xinfo_groups(subscription.stream_name)
            consumer_info = await redis.client.xinfo_consumers(
                subscription.stream_name,
                subscription.consumer_group
            )
        except Exception:
            group_info = []
            consumer_info = []
        
        return {
            "subscription": subscription.model_dump(),
            "consumer_group_info": group_info,
            "consumer_info": consumer_info,
            "is_running": subscription_id in self._running_consumers
        }
    
    async def cleanup_expired_events(self) -> int:
        """Clean up expired events from streams."""
        try:
            redis = await self._get_redis()
            cleaned_count = 0
            
            # Get all streams
            streams = await redis.keys(f"{self.agent_streams_prefix}*")
            streams.extend([self.main_stream, self.broadcast_stream])
            
            cutoff_time = datetime.utcnow() - timedelta(hours=self.config.event_ttl_hours)
            cutoff_timestamp_ms = int(cutoff_time.timestamp() * 1000)
            cutoff_id = f"{cutoff_timestamp_ms}-0"
            
            for stream in streams:
                try:
                    # Delete messages older than cutoff
                    deleted = await redis.xtrim(stream, minid=cutoff_id)
                    cleaned_count += deleted
                except Exception as e:
                    logger.error(f"Error cleaning stream {stream}: {e}")
            
            logger.info(f"Cleaned {cleaned_count} expired events")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0
    
    # Enhanced EventBus methods
    
    async def publish_event(self, event: AgentEvent) -> Optional[str]:
        """Publish a pre-constructed AgentEvent."""
        return await self.publish(event.type, event=event)
    
    async def publish_task_event(
        self,
        event_type: EventType,
        source_agent: str,
        task_id: str,
        task_name: Optional[str] = None,
        progress: Optional[float] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        target_agent: Optional[str] = None,
        priority: EventPriority = EventPriority.MEDIUM
    ) -> Optional[str]:
        """Publish a task-related event."""
        from autom8.models.events import create_task_event
        
        event = create_task_event(
            event_type=event_type,
            source_agent=source_agent,
            task_id=task_id,
            task_name=task_name,
            progress=progress,
            result=result,
            error=error
        )
        
        if target_agent:
            event.target_agent = target_agent
        
        event.priority = priority
        
        return await self.publish_event(event)
    
    async def publish_error_event(
        self,
        source_agent: str,
        error_id: str,
        error_type: str,
        error_message: str,
        error_code: Optional[str] = None,
        stack_trace: Optional[str] = None,
        priority: EventPriority = EventPriority.HIGH
    ) -> Optional[str]:
        """Publish an error event."""
        from autom8.models.events import create_error_event
        
        event = create_error_event(
            source_agent=source_agent,
            error_id=error_id,
            error_type=error_type,
            error_message=error_message,
            error_code=error_code,
            stack_trace=stack_trace
        )
        
        event.priority = priority
        
        return await self.publish_event(event)
    
    async def get_event_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        redis = await self._get_redis()
        
        # Get stream info
        stream_stats = {}
        for stream in [self.main_stream, self.broadcast_stream]:
            try:
                info = await redis.client.xinfo_stream(stream)
                stream_stats[stream] = {
                    'length': info.get('length', 0),
                    'groups': info.get('groups', 0),
                    'last_generated_id': info.get('last-generated-id', '0-0')
                }
            except Exception as e:
                logger.debug(f"Could not get stats for stream {stream}: {e}")
                stream_stats[stream] = {'error': str(e)}
        
        return {
            **self._stats,
            'active_subscriptions': len(self._subscriptions),
            'running_consumers': len(self._running_consumers),
            'stream_stats': stream_stats,
            'uptime_seconds': (datetime.utcnow() - datetime.utcnow()).total_seconds()  # Will be updated with actual start time
        }
    
    def get_serializer(self) -> EventSerializer:
        """Get the event serializer instance."""
        return self._event_serializer
    
    def get_validator(self) -> EventValidator:
        """Get the event validator instance."""
        return self._event_validator
    
    async def validate_event(self, event: AgentEvent) -> bool:
        """Validate an event."""
        try:
            self._event_validator.validate_event(event)
            return True
        except EventValidationError:
            return False
    
    async def serialize_event(self, event: AgentEvent) -> str:
        """Serialize an event to JSON."""
        return event_to_json(event)
    
    async def deserialize_event(self, event_json: str) -> AgentEvent:
        """Deserialize an event from JSON."""
        return event_from_json(event_json)


# Global EventBus instance
_event_bus: Optional[EventBus] = None


async def get_event_bus(config: Optional[EventBusConfig] = None) -> EventBus:
    """Get global EventBus instance."""
    global _event_bus
    
    if _event_bus is None:
        _event_bus = EventBus(config)
        await _event_bus.start()
    
    return _event_bus


async def close_event_bus() -> None:
    """Close global EventBus instance."""
    global _event_bus
    
    if _event_bus:
        await _event_bus.stop()
        _event_bus = None


# Convenience functions for enhanced event publishing

async def publish_task_started(
    source_agent: str,
    task_id: str,
    task_name: str,
    target_agent: Optional[str] = None
) -> Optional[str]:
    """Publish a task started event."""
    bus = await get_event_bus()
    return await bus.publish_task_event(
        EventType.TASK_START,
        source_agent,
        task_id,
        task_name,
        target_agent=target_agent
    )


async def publish_task_completed(
    source_agent: str,
    task_id: str,
    result: Dict[str, Any],
    target_agent: Optional[str] = None
) -> Optional[str]:
    """Publish a task completed event."""
    bus = await get_event_bus()
    return await bus.publish_task_event(
        EventType.TASK_COMPLETE,
        source_agent,
        task_id,
        progress=1.0,
        result=result,
        target_agent=target_agent
    )


async def publish_error(
    source_agent: str,
    error_type: str,
    error_message: str,
    error_code: Optional[str] = None
) -> Optional[str]:
    """Publish an error event."""
    bus = await get_event_bus()
    return await bus.publish_error_event(
        source_agent=source_agent,
        error_id=str(uuid.uuid4()),
        error_type=error_type,
        error_message=error_message,
        error_code=error_code
    )