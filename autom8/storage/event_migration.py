"""
Event Migration and Backward Compatibility Support

Provides migration utilities for converting between different event formats
and versions, ensuring backward compatibility while enabling gradual
migration to enhanced event models.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union, Type
from datetime import datetime

from pydantic import ValidationError

from autom8.models.events import (
    AgentEvent, EventType, EventPriority, EventCategory, EventSeverity,
    BaseEventPayload, TaskEventPayload, AgentEventPayload, DecisionEventPayload,
    EventSchemaVersion
)
from autom8.models.memory import AgentEvent as LegacyAgentEvent, EventType as LegacyEventType, Priority

logger = logging.getLogger(__name__)


class EventMigrationError(Exception):
    """Exception raised during event migration."""
    pass


class EventMigrator:
    """
    Handles migration between different event formats and versions.
    
    Provides bidirectional conversion between legacy and enhanced events,
    schema version migration, and data structure transformation.
    """
    
    def __init__(self):
        # Map legacy event types to enhanced event types
        self.legacy_type_mapping = {
            'task_start': EventType.TASK_START,
            'task_complete': EventType.TASK_COMPLETE,
            'decision_made': EventType.DECISION_MADE,
            'context_updated': EventType.CONTEXT_UPDATED,
            'error_occurred': EventType.ERROR_OCCURRED,
            'agent_ready': EventType.AGENT_READY,
            'agent_shutdown': EventType.AGENT_SHUTDOWN,
            'coordination_request': EventType.COORDINATION_REQUEST,
        }
        
        # Map legacy priorities to enhanced priorities
        self.legacy_priority_mapping = {
            1: EventPriority.LOW,
            2: EventPriority.MEDIUM,
            3: EventPriority.HIGH,
            4: EventPriority.CRITICAL,
        }
        
        # Reverse mappings
        self.enhanced_to_legacy_type = {v: k for k, v in self.legacy_type_mapping.items()}
        self.enhanced_to_legacy_priority = {v: k for k, v in self.legacy_priority_mapping.items()}
    
    def migrate_legacy_to_enhanced(self, legacy_event: Union[LegacyAgentEvent, Dict[str, Any]]) -> AgentEvent:
        """
        Convert a legacy AgentEvent to an enhanced AgentEvent.
        
        Args:
            legacy_event: Legacy event (LegacyAgentEvent instance or dict)
            
        Returns:
            Enhanced AgentEvent
            
        Raises:
            EventMigrationError: If migration fails
        """
        try:\n            # Handle dict input
            if isinstance(legacy_event, dict):
                legacy_data = legacy_event
            else:
                # Convert LegacyAgentEvent to dict
                legacy_data = {
                    'id': legacy_event.id,
                    'type': legacy_event.type,
                    'source_agent': legacy_event.source_agent,
                    'target_agent': legacy_event.target_agent,
                    'data': legacy_event.data,
                    'priority': legacy_event.priority,
                    'timestamp': legacy_event.timestamp,
                    'expires_at': legacy_event.expires_at,
                    'processed_by': list(legacy_event.processed_by) if legacy_event.processed_by else []
                }
            
            # Convert event type
            legacy_type = legacy_data.get('type')
            if isinstance(legacy_type, LegacyEventType):
                event_type = self.legacy_type_mapping.get(legacy_type.value, EventType.CUSTOM)
            elif isinstance(legacy_type, str):
                event_type = self.legacy_type_mapping.get(legacy_type, EventType.CUSTOM)
            else:
                event_type = EventType.CUSTOM
            
            # Convert priority
            legacy_priority = legacy_data.get('priority', Priority.MEDIUM)
            if hasattr(legacy_priority, 'value'):\n                priority = self.legacy_priority_mapping.get(legacy_priority.value, EventPriority.MEDIUM)
            elif isinstance(legacy_priority, int):
                priority = self.legacy_priority_mapping.get(legacy_priority, EventPriority.MEDIUM)
            else:
                priority = EventPriority.MEDIUM
            
            # Convert data to payload
            legacy_data_dict = legacy_data.get('data', {})
            payload = self._convert_data_to_payload(event_type, legacy_data_dict)
            
            # Infer category from event type
            category = self._infer_category_from_type(event_type)
            
            # Create enhanced event
            enhanced_event = AgentEvent(
                id=legacy_data.get('id'),
                type=event_type,
                schema_version=EventSchemaVersion.CURRENT,
                source_agent=legacy_data.get('source_agent'),
                target_agent=legacy_data.get('target_agent'),
                category=category,
                priority=priority,
                severity=EventSeverity.INFO,  # Default severity
                payload=payload,
                timestamp=legacy_data.get('timestamp', datetime.utcnow()),
                expires_at=legacy_data.get('expires_at'),
                processed_by=set(legacy_data.get('processed_by', [])),
                metadata={'migrated_from': 'legacy', 'migration_timestamp': datetime.utcnow().isoformat()}
            )
            
            return enhanced_event
            
        except Exception as e:
            logger.error(f"Failed to migrate legacy event to enhanced: {e}")
            raise EventMigrationError(f"Legacy to enhanced migration failed: {str(e)}") from e
    
    def migrate_enhanced_to_legacy(self, enhanced_event: AgentEvent) -> Dict[str, Any]:
        """
        Convert an enhanced AgentEvent to legacy format.
        
        Args:
            enhanced_event: Enhanced AgentEvent
            
        Returns:
            Legacy event data as dict
            
        Raises:
            EventMigrationError: If migration fails
        """
        try:
            # Convert event type back to legacy format
            legacy_type = self.enhanced_to_legacy_type.get(enhanced_event.type)
            if legacy_type is None:
                logger.warning(f"No legacy mapping for event type {enhanced_event.type}, using string value")
                legacy_type = enhanced_event.type.value
            
            # Convert priority back to legacy format
            legacy_priority = self.enhanced_to_legacy_priority.get(enhanced_event.priority, 2)
            
            # Convert payload to data dict
            if hasattr(enhanced_event.payload, 'to_dict'):
                data = enhanced_event.payload.to_dict()
            else:
                data = {}
            
            # Add metadata to data
            if enhanced_event.metadata:
                data.update(enhanced_event.metadata)
            
            legacy_data = {
                'id': enhanced_event.id,
                'type': legacy_type,
                'source_agent': enhanced_event.source_agent,
                'target_agent': enhanced_event.target_agent,
                'data': data,
                'priority': legacy_priority,
                'timestamp': enhanced_event.timestamp,
                'expires_at': enhanced_event.expires_at,
                'processed_by': list(enhanced_event.processed_by)
            }
            
            return legacy_data
            
        except Exception as e:
            logger.error(f"Failed to migrate enhanced event to legacy: {e}")
            raise EventMigrationError(f"Enhanced to legacy migration failed: {str(e)}") from e
    
    def _convert_data_to_payload(self, event_type: EventType, data: Dict[str, Any]) -> BaseEventPayload:
        """Convert legacy data dict to appropriate payload type."""
        try:
            # Determine payload type based on event type
            if event_type in [EventType.TASK_START, EventType.TASK_COMPLETE, EventType.TASK_FAILED]:
                # Try to create TaskEventPayload
                payload_data = {}
                if 'task_id' in data:
                    payload_data['task_id'] = data['task_id']
                elif 'task' in data:
                    payload_data['task_id'] = data['task']
                else:
                    payload_data['task_id'] = 'unknown'
                
                if 'task_name' in data:
                    payload_data['task_name'] = data['task_name']
                if 'progress' in data:
                    payload_data['progress'] = data['progress']
                if 'result' in data:
                    payload_data['result'] = data['result']
                if 'error' in data:
                    payload_data['error'] = data['error']
                
                return TaskEventPayload(**payload_data)
            
            elif event_type in [EventType.AGENT_START, EventType.AGENT_READY, EventType.AGENT_SHUTDOWN]:
                # Try to create AgentEventPayload
                payload_data = {}
                if 'agent_id' in data:
                    payload_data['agent_id'] = data['agent_id']
                elif 'agent' in data:
                    payload_data['agent_id'] = data['agent']
                else:
                    payload_data['agent_id'] = 'unknown'
                
                if 'status' in data:
                    payload_data['status'] = data['status']
                else:
                    payload_data['status'] = 'unknown'
                
                if 'role' in data:
                    payload_data['agent_role'] = data['role']
                if 'capabilities' in data:
                    payload_data['capabilities'] = data['capabilities']
                
                return AgentEventPayload(**payload_data)
            
            elif event_type == EventType.DECISION_MADE:
                # Try to create DecisionEventPayload
                payload_data = {}
                if 'decision_id' in data:
                    payload_data['decision_id'] = data['decision_id']
                else:
                    payload_data['decision_id'] = 'unknown'
                
                if 'decision_type' in data:
                    payload_data['decision_type'] = data['decision_type']
                else:
                    payload_data['decision_type'] = 'unknown'
                
                if 'summary' in data:
                    payload_data['decision_summary'] = data['summary']
                elif 'content' in data:
                    payload_data['decision_summary'] = data['content']
                else:
                    payload_data['decision_summary'] = 'Decision made'
                
                if 'reasoning' in data:
                    payload_data['reasoning'] = data['reasoning']
                if 'confidence' in data:
                    payload_data['confidence'] = data['confidence']
                
                return DecisionEventPayload(**payload_data)
            
            else:
                # Fall back to BaseEventPayload with data as metadata
                return BaseEventPayload(**data)
                
        except Exception as e:
            logger.warning(f"Failed to create specific payload type, falling back to base: {e}")
            # Fall back to BaseEventPayload
            try:
                return BaseEventPayload(**data)
            except Exception:
                return BaseEventPayload()
    
    def _infer_category_from_type(self, event_type: EventType) -> EventCategory:
        """Infer event category from event type."""
        type_category_map = {
            EventType.TASK_START: EventCategory.LIFECYCLE,
            EventType.TASK_COMPLETE: EventCategory.LIFECYCLE,
            EventType.TASK_FAILED: EventCategory.LIFECYCLE,
            EventType.AGENT_START: EventCategory.LIFECYCLE,
            EventType.AGENT_READY: EventCategory.LIFECYCLE,
            EventType.AGENT_SHUTDOWN: EventCategory.LIFECYCLE,
            EventType.DECISION_MADE: EventCategory.COORDINATION,
            EventType.COORDINATION_REQUEST: EventCategory.COORDINATION,
            EventType.ERROR_OCCURRED: EventCategory.ERROR,
            EventType.CONTEXT_UPDATED: EventCategory.SYSTEM,
        }
        return type_category_map.get(event_type, EventCategory.CUSTOM)
    
    def migrate_schema_version(self, event_data: Dict[str, Any], target_version: EventSchemaVersion) -> Dict[str, Any]:
        """
        Migrate event data between schema versions.
        
        Args:
            event_data: Event data dict
            target_version: Target schema version
            
        Returns:
            Migrated event data
        """
        current_version = event_data.get('schema_version', EventSchemaVersion.V1_0.value)
        
        if current_version == target_version.value:
            return event_data  # No migration needed
        
        try:
            # Apply version-specific migrations
            if current_version == EventSchemaVersion.V1_0.value and target_version == EventSchemaVersion.CURRENT:
                return self._migrate_v1_0_to_current(event_data)
            elif current_version == EventSchemaVersion.V1_1.value and target_version == EventSchemaVersion.CURRENT:
                return self._migrate_v1_1_to_current(event_data)
            else:
                logger.warning(f"No migration path from {current_version} to {target_version.value}")
                return event_data
                
        except Exception as e:
            logger.error(f"Schema migration failed: {e}")
            raise EventMigrationError(f"Schema migration failed: {str(e)}") from e
    
    def _migrate_v1_0_to_current(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from schema version 1.0 to current."""
        migrated = event_data.copy()
        
        # Convert 'data' field to 'payload' if exists
        if 'data' in migrated and 'payload' not in migrated:
            data = migrated.pop('data')
            migrated['payload'] = BaseEventPayload(**data).model_dump()
        
        # Add missing required fields with defaults
        if 'category' not in migrated:
            migrated['category'] = EventCategory.CUSTOM.value
        
        if 'severity' not in migrated:
            migrated['severity'] = EventSeverity.INFO.value
        
        if 'schema_version' not in migrated:
            migrated['schema_version'] = EventSchemaVersion.CURRENT.value
        
        # Convert sets to lists for JSON compatibility
        if 'processed_by' in migrated and isinstance(migrated['processed_by'], list):
            migrated['processed_by'] = migrated['processed_by']
        
        if 'tags' in migrated and isinstance(migrated['tags'], list):
            migrated['tags'] = migrated['tags']
        
        return migrated
    
    def _migrate_v1_1_to_current(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from schema version 1.1 to current."""
        migrated = event_data.copy()
        
        # Add any new fields introduced in current version
        if 'severity' not in migrated:
            migrated['severity'] = EventSeverity.INFO.value
        
        migrated['schema_version'] = EventSchemaVersion.CURRENT.value
        
        return migrated
    
    def batch_migrate_legacy_events(self, legacy_events: List[Union[LegacyAgentEvent, Dict[str, Any]]]) -> List[AgentEvent]:
        """
        Migrate a batch of legacy events to enhanced events.
        
        Args:
            legacy_events: List of legacy events
            
        Returns:
            List of enhanced events
        """
        enhanced_events = []
        migration_errors = []
        
        for i, legacy_event in enumerate(legacy_events):
            try:
                enhanced_event = self.migrate_legacy_to_enhanced(legacy_event)
                enhanced_events.append(enhanced_event)
            except EventMigrationError as e:
                migration_errors.append((i, str(e)))
                logger.error(f"Failed to migrate event {i}: {e}")
        
        if migration_errors:
            logger.warning(f"Migration completed with {len(migration_errors)} errors out of {len(legacy_events)} events")
        
        return enhanced_events
    
    def validate_migration(self, original: Union[LegacyAgentEvent, Dict[str, Any]], migrated: AgentEvent) -> bool:
        """
        Validate that migration preserved essential event information.
        
        Args:
            original: Original legacy event
            migrated: Migrated enhanced event
            
        Returns:
            True if migration is valid, False otherwise
        """
        try:
            if isinstance(original, dict):
                orig_data = original
            else:
                orig_data = {
                    'id': original.id,
                    'type': original.type,
                    'source_agent': original.source_agent,
                    'target_agent': original.target_agent,
                }
            
            # Check essential fields
            if migrated.id != orig_data.get('id'):
                return False
            
            if migrated.source_agent != orig_data.get('source_agent'):
                return False
            
            if migrated.target_agent != orig_data.get('target_agent'):
                return False
            
            # Check type mapping
            orig_type = orig_data.get('type')
            if isinstance(orig_type, str) and orig_type in self.legacy_type_mapping:
                expected_type = self.legacy_type_mapping[orig_type]
                if migrated.type != expected_type:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Migration validation failed: {e}")
            return False


class BackwardCompatibilityLayer:
    """
    Provides backward compatibility for code using legacy event interfaces.
    
    Acts as an adapter layer that allows existing code to work with
    enhanced events while gradually migrating to the new interfaces.
    """
    
    def __init__(self):
        self.migrator = EventMigrator()
    
    def create_legacy_compatible_event(
        self,
        event_type: str,
        source_agent: str,
        data: Dict[str, Any],
        target_agent: Optional[str] = None,
        priority: int = 2
    ) -> AgentEvent:
        """
        Create an enhanced event using legacy parameters.
        
        Args:
            event_type: Legacy event type string
            source_agent: Source agent ID
            data: Legacy event data
            target_agent: Target agent ID
            priority: Legacy priority (integer)
            
        Returns:
            Enhanced AgentEvent
        """
        # Create a legacy-style event dict
        legacy_data = {
            'type': event_type,
            'source_agent': source_agent,
            'target_agent': target_agent,
            'data': data,
            'priority': priority,
            'timestamp': datetime.utcnow(),
        }
        
        # Migrate to enhanced event
        return self.migrator.migrate_legacy_to_enhanced(legacy_data)
    
    def extract_legacy_data(self, enhanced_event: AgentEvent) -> Dict[str, Any]:
        """
        Extract legacy-compatible data from an enhanced event.
        
        Args:
            enhanced_event: Enhanced AgentEvent
            
        Returns:
            Legacy-compatible event data
        """
        return self.migrator.migrate_enhanced_to_legacy(enhanced_event)
    
    def wrap_enhanced_event_as_legacy(self, enhanced_event: AgentEvent) -> 'LegacyEventWrapper':
        """
        Wrap an enhanced event to provide legacy interface.
        
        Args:
            enhanced_event: Enhanced AgentEvent
            
        Returns:
            Legacy-compatible event wrapper
        """
        return LegacyEventWrapper(enhanced_event, self.migrator)


class LegacyEventWrapper:
    """
    Wrapper that provides legacy AgentEvent interface for enhanced events.
    
    Allows existing code to work with enhanced events without modification
    by providing the same attribute access patterns as legacy events.
    """
    
    def __init__(self, enhanced_event: AgentEvent, migrator: EventMigrator):
        self._enhanced_event = enhanced_event
        self._migrator = migrator
        self._legacy_data = None
    
    @property
    def _legacy(self) -> Dict[str, Any]:
        """Get legacy data representation (cached)."""
        if self._legacy_data is None:
            self._legacy_data = self._migrator.migrate_enhanced_to_legacy(self._enhanced_event)
        return self._legacy_data
    
    # Legacy interface properties
    @property
    def id(self) -> str:
        return self._enhanced_event.id
    
    @property
    def type(self) -> str:
        return self._legacy.get('type', self._enhanced_event.type.value)
    
    @property
    def source_agent(self) -> str:
        return self._enhanced_event.source_agent
    
    @property
    def target_agent(self) -> Optional[str]:
        return self._enhanced_event.target_agent
    
    @property
    def data(self) -> Dict[str, Any]:
        return self._legacy.get('data', {})
    
    @property
    def priority(self) -> int:
        return self._legacy.get('priority', 2)
    
    @property
    def timestamp(self) -> datetime:
        return self._enhanced_event.timestamp
    
    @property
    def expires_at(self) -> Optional[datetime]:
        return self._enhanced_event.expires_at
    
    @property
    def processed_by(self) -> List[str]:
        return list(self._enhanced_event.processed_by)
    
    # Legacy interface methods
    @property
    def is_expired(self) -> bool:
        return self._enhanced_event.is_expired
    
    @property
    def is_broadcast(self) -> bool:
        return self._enhanced_event.is_broadcast
    
    def mark_processed(self, agent_id: str) -> None:
        self._enhanced_event.mark_processing_completed(agent_id)
    
    # Access to enhanced event
    def get_enhanced_event(self) -> AgentEvent:
        """Get the underlying enhanced event."""
        return self._enhanced_event


# Global instances
_migrator: Optional[EventMigrator] = None
_compatibility_layer: Optional[BackwardCompatibilityLayer] = None


def get_event_migrator() -> EventMigrator:
    """Get the global event migrator instance."""
    global _migrator
    if _migrator is None:
        _migrator = EventMigrator()
    return _migrator


def get_compatibility_layer() -> BackwardCompatibilityLayer:
    """Get the global backward compatibility layer."""
    global _compatibility_layer
    if _compatibility_layer is None:
        _compatibility_layer = BackwardCompatibilityLayer()
    return _compatibility_layer


# Convenience functions
def migrate_legacy_event(legacy_event: Union[LegacyAgentEvent, Dict[str, Any]]) -> AgentEvent:
    """Migrate a single legacy event to enhanced format."""
    migrator = get_event_migrator()
    return migrator.migrate_legacy_to_enhanced(legacy_event)


def create_legacy_compatible_event(
    event_type: str,
    source_agent: str,
    data: Dict[str, Any],
    target_agent: Optional[str] = None,
    priority: int = 2
) -> AgentEvent:
    """Create an enhanced event using legacy parameters."""
    layer = get_compatibility_layer()
    return layer.create_legacy_compatible_event(event_type, source_agent, data, target_agent, priority)


def wrap_as_legacy(enhanced_event: AgentEvent) -> LegacyEventWrapper:
    """Wrap an enhanced event to provide legacy interface."""
    layer = get_compatibility_layer()
    return layer.wrap_enhanced_event_as_legacy(enhanced_event)