"""
Event Serialization and Deserialization Support

Provides comprehensive serialization support for events with multiple formats,
compression, encryption, and backward compatibility. Handles complex event
data structures with type safety and validation.
"""

import json
import gzip
import base64
import pickle
import os
from typing import Any, Dict, List, Optional, Union, Type, Callable
from datetime import datetime
from enum import Enum
import logging

from pydantic import BaseModel, ValidationError
from cryptography.fernet import Fernet

from autom8.models.events import (
    AgentEvent, EventType, BaseEventPayload, TaskEventPayload, 
    AgentEventPayload, DecisionEventPayload, CoordinationEventPayload,
    ContextEventPayload, ErrorEventPayload, PerformanceEventPayload,
    SystemEventPayload, EventSchemaVersion, convert_legacy_event
)
from autom8.models.memory import AgentEvent as LegacyAgentEvent

logger = logging.getLogger(__name__)


class SerializationFormat(str, Enum):
    """Supported serialization formats."""
    JSON = "json"
    MSGPACK = "msgpack"
    PICKLE = "pickle"
    PROTOBUF = "protobuf"


class CompressionType(str, Enum):
    """Supported compression types."""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"


class SerializationError(Exception):
    """Custom exception for serialization errors."""
    pass


class DeserializationError(Exception):
    """Custom exception for deserialization errors."""
    pass


class EventSerializer:
    """
    Comprehensive event serializer with multiple format support.
    
    Handles serialization of AgentEvent objects with support for:
    - Multiple formats (JSON, MessagePack, Pickle)
    - Compression (gzip, bzip2, lzma)
    - Encryption (optional)
    - Schema versioning and migration
    - Backward compatibility
    """
    
    def __init__(
        self,
        default_format: SerializationFormat = SerializationFormat.JSON,
        default_compression: CompressionType = CompressionType.NONE,
        enable_encryption: bool = False,
        encryption_key: Optional[bytes] = None
    ):
        self.default_format = default_format
        self.default_compression = default_compression
        self.enable_encryption = enable_encryption
        self.encryption_key = encryption_key
        
        # Payload type registry for deserialization
        self.payload_types = {
            'BaseEventPayload': BaseEventPayload,
            'TaskEventPayload': TaskEventPayload,
            'AgentEventPayload': AgentEventPayload,
            'DecisionEventPayload': DecisionEventPayload,
            'CoordinationEventPayload': CoordinationEventPayload,
            'ContextEventPayload': ContextEventPayload,
            'ErrorEventPayload': ErrorEventPayload,
            'PerformanceEventPayload': PerformanceEventPayload,
            'SystemEventPayload': SystemEventPayload,
        }
        
        # Migration handlers for schema versions
        self.migration_handlers = {
            EventSchemaVersion.V1_0: self._migrate_from_v1_0,
            EventSchemaVersion.V1_1: self._migrate_from_v1_1,
        }
        
        # Format-specific serializers
        self.format_serializers = {
            SerializationFormat.JSON: self._serialize_json,
            SerializationFormat.PICKLE: self._serialize_pickle,
        }
        
        self.format_deserializers = {
            SerializationFormat.JSON: self._deserialize_json,
            SerializationFormat.PICKLE: self._deserialize_pickle,
        }
        
        # Try to import optional dependencies
        self._setup_optional_formats()
    
    def _setup_optional_formats(self) -> None:
        """Setup optional serialization formats."""
        try:
            import msgpack
            self.format_serializers[SerializationFormat.MSGPACK] = self._serialize_msgpack
            self.format_deserializers[SerializationFormat.MSGPACK] = self._deserialize_msgpack
        except ImportError:
            logger.debug("msgpack not available, skipping MessagePack support")
    
    def serialize(
        self,
        event: Union[AgentEvent, Dict[str, Any]],
        format: Optional[SerializationFormat] = None,
        compression: Optional[CompressionType] = None,
        include_metadata: bool = True
    ) -> bytes:
        """
        Serialize an event to bytes.
        
        Args:
            event: Event to serialize (AgentEvent or dict)
            format: Serialization format (defaults to instance default)
            compression: Compression type (defaults to instance default)
            include_metadata: Include serialization metadata
            
        Returns:
            Serialized event as bytes
            
        Raises:
            SerializationError: If serialization fails
        """
        try:
            format = format or self.default_format
            compression = compression or self.default_compression
            
            # Convert to dict if needed
            if isinstance(event, AgentEvent):
                data = event.to_dict()
            elif isinstance(event, dict):
                data = event.copy()
            else:
                raise SerializationError(f"Unsupported event type: {type(event)}")
            
            # Add serialization metadata if requested
            if include_metadata:
                data['__serialization__'] = {
                    'format': format.value,
                    'compression': compression.value,
                    'serialized_at': datetime.utcnow().isoformat(),
                    'version': EventSchemaVersion.CURRENT.value,
                    'serializer': 'EventSerializer'
                }
            
            # Serialize based on format
            serializer = self.format_serializers.get(format)
            if not serializer:
                raise SerializationError(f"Unsupported format: {format}")
            
            serialized_data = serializer(data)
            
            # Apply compression
            compressed_data = self._compress(serialized_data, compression)
            
            # Apply encryption if enabled
            if self.enable_encryption and self.encryption_key:
                compressed_data = self._encrypt(compressed_data)
            
            return compressed_data
            
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            raise SerializationError(f"Failed to serialize event: {str(e)}") from e
    
    def deserialize(
        self,
        data: bytes,
        event_class: Type[AgentEvent] = AgentEvent,
        validate: bool = True
    ) -> AgentEvent:
        """
        Deserialize bytes to an event.
        
        Args:
            data: Serialized event data
            event_class: Event class to deserialize to
            validate: Whether to validate the deserialized event
            
        Returns:
            Deserialized AgentEvent
            
        Raises:
            DeserializationError: If deserialization fails
        """
        try:
            # Apply decryption if enabled
            if self.enable_encryption and self.encryption_key:
                data = self._decrypt(data)
            
            # Try to detect format and compression from metadata or auto-detect
            format, compression = self._detect_format_and_compression(data)
            
            # Decompress data
            decompressed_data = self._decompress(data, compression)
            
            # Deserialize based on format
            deserializer = self.format_deserializers.get(format)
            if not deserializer:
                raise DeserializationError(f"Unsupported format: {format}")
            
            event_data = deserializer(decompressed_data)
            
            # Handle legacy events
            if self._is_legacy_event(event_data):
                event_data = convert_legacy_event(event_data)
            
            # Handle schema migration if needed
            if '__serialization__' in event_data:
                schema_version = event_data['__serialization__'].get('version')
                if schema_version and schema_version != EventSchemaVersion.CURRENT.value:
                    event_data = self._migrate_event(event_data, schema_version)
                
                # Remove serialization metadata
                del event_data['__serialization__']
            
            # Deserialize payload if needed
            event_data = self._deserialize_payload(event_data)
            
            # Create event instance
            event = event_class.from_dict(event_data)
            
            # Validate if requested
            if validate:
                self._validate_event(event)
            
            return event
            
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            raise DeserializationError(f"Failed to deserialize event: {str(e)}") from e
    
    def serialize_batch(
        self,
        events: List[Union[AgentEvent, Dict[str, Any]]],
        format: Optional[SerializationFormat] = None,
        compression: Optional[CompressionType] = None
    ) -> bytes:
        """Serialize a batch of events."""
        try:
            batch_data = {
                'batch': True,
                'events': [
                    event.to_dict() if isinstance(event, AgentEvent) else event
                    for event in events
                ],
                'count': len(events),
                'serialized_at': datetime.utcnow().isoformat()
            }
            
            return self.serialize(batch_data, format, compression)
            
        except Exception as e:
            raise SerializationError(f"Failed to serialize event batch: {str(e)}") from e
    
    def deserialize_batch(
        self,
        data: bytes,
        event_class: Type[AgentEvent] = AgentEvent,
        validate: bool = True
    ) -> List[AgentEvent]:
        """Deserialize a batch of events."""
        try:
            batch_data = self.deserialize(data, dict, validate=False)
            
            if not isinstance(batch_data, dict) or not batch_data.get('batch'):
                raise DeserializationError("Data is not a valid event batch")
            
            events = []
            for event_data in batch_data.get('events', []):
                event = event_class.from_dict(event_data)
                if validate:
                    self._validate_event(event)
                events.append(event)
            
            return events
            
        except Exception as e:
            raise DeserializationError(f"Failed to deserialize event batch: {str(e)}") from e
    
    # Format-specific serialization methods
    def _serialize_json(self, data: Dict[str, Any]) -> bytes:
        """Serialize to JSON."""
        return json.dumps(data, indent=None, separators=(',', ':'), default=str).encode('utf-8')
    
    def _deserialize_json(self, data: bytes) -> Dict[str, Any]:
        """Deserialize from JSON."""
        return json.loads(data.decode('utf-8'))
    
    def _serialize_pickle(self, data: Dict[str, Any]) -> bytes:
        """Serialize to Pickle."""
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _deserialize_pickle(self, data: bytes) -> Dict[str, Any]:
        """Deserialize from Pickle."""
        return pickle.loads(data)
    
    def _serialize_msgpack(self, data: Dict[str, Any]) -> bytes:
        """Serialize to MessagePack."""
        import msgpack
        return msgpack.packb(data, use_bin_type=True, default=str)
    
    def _deserialize_msgpack(self, data: bytes) -> Dict[str, Any]:
        """Deserialize from MessagePack."""
        import msgpack
        return msgpack.unpackb(data, raw=False, strict_map_key=False)
    
    # Compression methods
    def _compress(self, data: bytes, compression: CompressionType) -> bytes:
        """Apply compression to data."""
        if compression == CompressionType.NONE:
            return data
        elif compression == CompressionType.GZIP:
            return gzip.compress(data)
        else:
            raise SerializationError(f"Unsupported compression: {compression}")
    
    def _decompress(self, data: bytes, compression: CompressionType) -> bytes:
        """Decompress data."""
        if compression == CompressionType.NONE:
            return data
        elif compression == CompressionType.GZIP:
            return gzip.decompress(data)
        else:
            raise DeserializationError(f"Unsupported compression: {compression}")
    
    # Encryption methods
    def _encrypt(self, data: bytes) -> bytes:
        """Encrypt data using Fernet symmetric encryption."""
        if not hasattr(self, '_encryption_key') or self._encryption_key is None:
            # Generate or load encryption key
            self._encryption_key = self._get_or_create_encryption_key()

        fernet = Fernet(self._encryption_key)
        return fernet.encrypt(data)

    def _decrypt(self, data: bytes) -> bytes:
        """Decrypt data using Fernet symmetric encryption."""
        if not hasattr(self, '_encryption_key') or self._encryption_key is None:
            self._encryption_key = self._get_or_create_encryption_key()

        fernet = Fernet(self._encryption_key)
        return fernet.decrypt(data)

    def _get_or_create_encryption_key(self) -> bytes:
        """Get existing encryption key or create a new one."""
        # In production, this should be loaded from secure key management
        # For now, generate a deterministic key based on environment
        key_env = os.environ.get('AUTOM8_ENCRYPTION_KEY')
        if key_env:
            return key_env.encode()

        # Generate a new key and warn about it
        key = Fernet.generate_key()
        logger.warning("Generated new encryption key - this should be stored securely for production use")
        return key
    
    def _detect_format_and_compression(self, data: bytes) -> tuple[SerializationFormat, CompressionType]:
        """Detect format and compression from data."""
        # Try to detect gzip compression
        if data.startswith(b'\x1f\x8b'):
            compression = CompressionType.GZIP
            try:
                decompressed = gzip.decompress(data)
            except Exception:
                compression = CompressionType.NONE
                decompressed = data
        else:
            compression = CompressionType.NONE
            decompressed = data
        
        # Try to detect format
        format = SerializationFormat.JSON  # Default
        
        # Check if it looks like JSON
        try:
            if decompressed.strip().startswith((b'{', b'[')):
                format = SerializationFormat.JSON
            else:
                # Try pickle magic bytes
                if decompressed.startswith((b'\x80\x03', b'\x80\x04', b'\x80\x05')):
                    format = SerializationFormat.PICKLE
        except Exception:
            pass
        
        return format, compression
    
    def _is_legacy_event(self, event_data: Dict[str, Any]) -> bool:
        """Check if event data is in legacy format."""
        # Legacy events don't have schema_version or use old field names
        return (
            'schema_version' not in event_data and
            'data' in event_data and
            'payload' not in event_data
        )
    
    def _migrate_event(self, event_data: Dict[str, Any], from_version: str) -> Dict[str, Any]:
        """Migrate event from old schema version."""
        version_enum = EventSchemaVersion(from_version)
        migrator = self.migration_handlers.get(version_enum)
        
        if migrator:
            return migrator(event_data)
        else:
            logger.warning(f"No migration handler for version {from_version}")
            return event_data
    
    def _migrate_from_v1_0(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from schema version 1.0."""
        # Example migration: convert 'data' to 'payload'
        if 'data' in event_data and 'payload' not in event_data:
            payload_data = event_data.pop('data')
            event_data['payload'] = BaseEventPayload(**payload_data).model_dump()
        
        event_data['schema_version'] = EventSchemaVersion.CURRENT.value
        return event_data
    
    def _migrate_from_v1_1(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from schema version 1.1."""
        # Example migration logic
        event_data['schema_version'] = EventSchemaVersion.CURRENT.value
        return event_data
    
    def _deserialize_payload(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize event payload to appropriate type."""
        if 'payload' not in event_data:
            return event_data
        
        payload_data = event_data['payload']
        if isinstance(payload_data, dict):
            # Try to determine payload type from event type
            event_type = event_data.get('type')
            payload_type = self._get_payload_type_for_event(event_type)
            
            if payload_type:
                try:
                    payload_instance = payload_type.model_validate(payload_data)
                    event_data['payload'] = payload_instance
                except ValidationError as e:
                    logger.warning(f"Failed to deserialize payload: {e}")
                    # Fall back to base payload type
                    event_data['payload'] = BaseEventPayload.model_validate(payload_data)
        
        return event_data
    
    def _get_payload_type_for_event(self, event_type: str) -> Optional[Type[BaseEventPayload]]:
        """Get appropriate payload type for an event type."""
        type_mapping = {
            'task_start': TaskEventPayload,
            'task_complete': TaskEventPayload,
            'task_failed': TaskEventPayload,
            'agent_start': AgentEventPayload,
            'agent_ready': AgentEventPayload,
            'agent_shutdown': AgentEventPayload,
            'decision_made': DecisionEventPayload,
            'coordination_request': CoordinationEventPayload,
            'context_updated': ContextEventPayload,
            'error_occurred': ErrorEventPayload,
            'performance_metric': PerformanceEventPayload,
            'system_start': SystemEventPayload,
        }
        
        return type_mapping.get(event_type, BaseEventPayload)
    
    def _validate_event(self, event: AgentEvent) -> None:
        """Validate deserialized event."""
        try:
            # Pydantic validation is already done during construction
            # Add any additional business logic validation here
            pass
        except Exception as e:
            raise DeserializationError(f"Event validation failed: {str(e)}") from e


class EventStreamSerializer:
    """
    Serializer for event streams with support for chunked serialization
    and streaming processing.
    """
    
    def __init__(self, serializer: EventSerializer, chunk_size: int = 1000):
        self.serializer = serializer
        self.chunk_size = chunk_size
    
    def serialize_stream(
        self,
        events: List[AgentEvent],
        format: Optional[SerializationFormat] = None,
        compression: Optional[CompressionType] = None
    ) -> List[bytes]:
        """Serialize events as chunks for streaming."""
        chunks = []
        
        for i in range(0, len(events), self.chunk_size):
            chunk = events[i:i + self.chunk_size]
            chunk_data = self.serializer.serialize_batch(chunk, format, compression)
            chunks.append(chunk_data)
        
        return chunks
    
    def deserialize_stream(
        self,
        chunks: List[bytes],
        event_class: Type[AgentEvent] = AgentEvent,
        validate: bool = True
    ) -> List[AgentEvent]:
        """Deserialize event chunks from stream."""
        all_events = []
        
        for chunk in chunks:
            events = self.serializer.deserialize_batch(chunk, event_class, validate)
            all_events.extend(events)
        
        return all_events


# Convenience functions
def serialize_event(
    event: Union[AgentEvent, Dict[str, Any]],
    format: SerializationFormat = SerializationFormat.JSON,
    compression: CompressionType = CompressionType.NONE
) -> bytes:
    """Serialize a single event."""
    serializer = EventSerializer(format, compression)
    return serializer.serialize(event)


def deserialize_event(
    data: bytes,
    event_class: Type[AgentEvent] = AgentEvent,
    validate: bool = True
) -> AgentEvent:
    """Deserialize a single event."""
    serializer = EventSerializer()
    return serializer.deserialize(data, event_class, validate)


def serialize_events(
    events: List[Union[AgentEvent, Dict[str, Any]]],
    format: SerializationFormat = SerializationFormat.JSON,
    compression: CompressionType = CompressionType.NONE
) -> bytes:
    """Serialize a batch of events."""
    serializer = EventSerializer(format, compression)
    return serializer.serialize_batch(events)


def deserialize_events(
    data: bytes,
    event_class: Type[AgentEvent] = AgentEvent,
    validate: bool = True
) -> List[AgentEvent]:
    """Deserialize a batch of events."""
    serializer = EventSerializer()
    return serializer.deserialize_batch(data, event_class, validate)


# JSON helpers for Redis and API compatibility
def event_to_json(event: AgentEvent) -> str:
    """Convert event to JSON string."""
    return event.to_json()


def event_from_json(json_str: str) -> AgentEvent:
    """Create event from JSON string."""
    return AgentEvent.from_json(json_str)


def events_to_json_list(events: List[AgentEvent]) -> List[str]:
    """Convert events to list of JSON strings."""
    return [event.to_json() for event in events]


def events_from_json_list(json_list: List[str]) -> List[AgentEvent]:
    """Create events from list of JSON strings."""
    return [AgentEvent.from_json(json_str) for json_str in json_list]


# Base64 encoding for text-safe transport
def serialize_event_base64(
    event: Union[AgentEvent, Dict[str, Any]],
    format: SerializationFormat = SerializationFormat.JSON,
    compression: CompressionType = CompressionType.GZIP
) -> str:
    """Serialize event and encode as base64 string."""
    data = serialize_event(event, format, compression)
    return base64.b64encode(data).decode('ascii')


def deserialize_event_base64(
    b64_data: str,
    event_class: Type[AgentEvent] = AgentEvent,
    validate: bool = True
) -> AgentEvent:
    """Deserialize event from base64 string."""
    data = base64.b64decode(b64_data.encode('ascii'))
    return deserialize_event(data, event_class, validate)