"""
Interactive Context Editor - Advanced context editing with undo/redo functionality.

This module extends the Context Inspector with comprehensive interactive editing
capabilities, supporting the PRD's core principle of "Context Transparency First".

Features:
- Interactive context editing with real-time preview
- Undo/redo stack with configurable depth
- Context validation during editing
- Integration with existing context transparency system
- CLI interface for interactive editing sessions
- Support for both text-based and structured context editing
- Automatic backup and recovery of context states
"""

import asyncio
import json
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from copy import deepcopy

from pydantic import BaseModel, Field

from autom8.models.context import (
    ContextPreview,
    ContextSource,
    ContextSourceType,
    ContextWarning,
    ContextWarningType,
    ContextOptimization,
)
from autom8.core.context.inspector import ContextInspector
from autom8.utils.tokens import get_token_counter
from autom8.utils.logging import get_logger
from autom8.config.settings import get_settings

logger = get_logger(__name__)


class EditAction(str, Enum):
    """Types of edit actions available"""
    ADD_SOURCE = "add_source"
    REMOVE_SOURCE = "remove_source"
    MODIFY_SOURCE = "modify_source"
    REORDER_SOURCES = "reorder_sources"
    UPDATE_PRIORITY = "update_priority"
    APPLY_OPTIMIZATION = "apply_optimization"
    BATCH_EDIT = "batch_edit"


class ValidationLevel(str, Enum):
    """Validation strictness levels"""
    STRICT = "strict"      # Full validation, block invalid edits
    PERMISSIVE = "permissive"  # Warn but allow most edits
    DISABLED = "disabled"  # No validation


class EditMode(str, Enum):
    """Context editing modes"""
    TEXT = "text"          # Text-based editing
    STRUCTURED = "structured"  # Structured object editing
    VISUAL = "visual"      # Visual editing interface


@dataclass
class EditState:
    """Represents the state of context at a point in time"""
    preview: ContextPreview
    timestamp: datetime
    edit_count: int
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate state hash for change detection"""
        content_hash = hashlib.sha256(
            json.dumps(self.preview.dict(), sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        self.metadata["content_hash"] = content_hash
    
    @property
    def is_valid(self) -> bool:
        """Check if the current state is valid"""
        return len(self.validation_errors) == 0
    
    @property
    def content_hash(self) -> str:
        """Get content hash for this state"""
        return self.metadata.get("content_hash", "")


class ContextEditCommand(ABC):
    """Abstract base class for context edit commands"""
    
    def __init__(self, action: EditAction, description: str, timestamp: Optional[datetime] = None):
        self.action = action
        self.description = description
        self.timestamp = timestamp or datetime.utcnow()
        self.command_id = f"{action.value}_{self.timestamp.timestamp()}"
    
    @abstractmethod
    async def execute(self, preview: ContextPreview) -> ContextPreview:
        """Execute the command and return modified preview"""
        pass
    
    @abstractmethod
    async def undo(self, preview: ContextPreview) -> ContextPreview:
        """Undo the command and return previous state"""
        pass
    
    @abstractmethod
    def can_merge_with(self, other: 'ContextEditCommand') -> bool:
        """Check if this command can be merged with another for efficiency"""
        pass
    
    def get_summary(self) -> str:
        """Get human-readable summary of this command"""
        return f"{self.action.value}: {self.description}"


class AddSourceCommand(ContextEditCommand):
    """Command to add a new context source"""
    
    def __init__(self, source: ContextSource, position: Optional[int] = None):
        super().__init__(EditAction.ADD_SOURCE, f"Add source: {source.display_name}")
        self.source = source
        self.position = position
        self.added_index = None  # Track where it was actually added
    
    async def execute(self, preview: ContextPreview) -> ContextPreview:
        """Add the source to the preview"""
        new_preview = deepcopy(preview)
        
        if self.position is not None and 0 <= self.position <= len(new_preview.sources):
            new_preview.sources.insert(self.position, self.source)
            self.added_index = self.position
        else:
            new_preview.sources.append(self.source)
            self.added_index = len(new_preview.sources) - 1
        
        # Update total tokens
        new_preview.total_tokens = sum(s.tokens for s in new_preview.sources)
        
        return new_preview
    
    async def undo(self, preview: ContextPreview) -> ContextPreview:
        """Remove the added source"""
        new_preview = deepcopy(preview)
        
        if self.added_index is not None and 0 <= self.added_index < len(new_preview.sources):
            new_preview.sources.pop(self.added_index)
            new_preview.total_tokens = sum(s.tokens for s in new_preview.sources)
        
        return new_preview
    
    def can_merge_with(self, other: 'ContextEditCommand') -> bool:
        """Add commands can be merged if they're adding multiple sources in sequence"""
        return (isinstance(other, AddSourceCommand) and 
                abs((self.timestamp - other.timestamp).total_seconds()) < 2.0)


class RemoveSourceCommand(ContextEditCommand):
    """Command to remove a context source"""
    
    def __init__(self, source_index: int):
        super().__init__(EditAction.REMOVE_SOURCE, f"Remove source at index {source_index}")
        self.source_index = source_index
        self.removed_source = None  # Store for undo
    
    async def execute(self, preview: ContextPreview) -> ContextPreview:
        """Remove the source from the preview"""
        new_preview = deepcopy(preview)
        
        if 0 <= self.source_index < len(new_preview.sources):
            self.removed_source = new_preview.sources.pop(self.source_index)
            new_preview.total_tokens = sum(s.tokens for s in new_preview.sources)
        
        return new_preview
    
    async def undo(self, preview: ContextPreview) -> ContextPreview:
        """Restore the removed source"""
        new_preview = deepcopy(preview)
        
        if self.removed_source is not None:
            if 0 <= self.source_index <= len(new_preview.sources):
                new_preview.sources.insert(self.source_index, self.removed_source)
            else:
                new_preview.sources.append(self.removed_source)
            
            new_preview.total_tokens = sum(s.tokens for s in new_preview.sources)
        
        return new_preview
    
    def can_merge_with(self, other: 'ContextEditCommand') -> bool:
        """Remove commands generally don't merge"""
        return False


class ModifySourceCommand(ContextEditCommand):
    """Command to modify an existing context source"""
    
    def __init__(self, source_index: int, modifications: Dict[str, Any]):
        content_mod = modifications.get('content', '')
        content_preview = content_mod[:50] + "..." if len(content_mod) > 50 else content_mod
        super().__init__(EditAction.MODIFY_SOURCE, f"Modify source {source_index}: {content_preview}")
        
        self.source_index = source_index
        self.modifications = modifications
        self.original_values = {}  # Store for undo
    
    async def execute(self, preview: ContextPreview) -> ContextPreview:
        """Apply modifications to the source"""
        new_preview = deepcopy(preview)
        
        if 0 <= self.source_index < len(new_preview.sources):
            source = new_preview.sources[self.source_index]
            
            # Store original values for undo
            for key in self.modifications:
                if hasattr(source, key):
                    self.original_values[key] = getattr(source, key)
            
            # Apply modifications
            for key, value in self.modifications.items():
                if hasattr(source, key):
                    setattr(source, key, value)
            
            # Recalculate tokens if content was modified
            if 'content' in self.modifications:
                token_counter = get_token_counter()
                source.tokens = token_counter.count_tokens(source.content, preview.model_target)
            
            # Update total tokens
            new_preview.total_tokens = sum(s.tokens for s in new_preview.sources)
        
        return new_preview
    
    async def undo(self, preview: ContextPreview) -> ContextPreview:
        """Restore original values"""
        new_preview = deepcopy(preview)
        
        if (0 <= self.source_index < len(new_preview.sources) and 
            self.original_values):
            
            source = new_preview.sources[self.source_index]
            
            # Restore original values
            for key, value in self.original_values.items():
                setattr(source, key, value)
            
            # Recalculate tokens if content was restored
            if 'content' in self.original_values:
                token_counter = get_token_counter()
                source.tokens = token_counter.count_tokens(source.content, preview.model_target)
            
            # Update total tokens
            new_preview.total_tokens = sum(s.tokens for s in new_preview.sources)
        
        return new_preview
    
    def can_merge_with(self, other: 'ContextEditCommand') -> bool:
        """Modify commands can merge if they're modifying the same source within a short time"""
        return (isinstance(other, ModifySourceCommand) and 
                self.source_index == other.source_index and
                abs((self.timestamp - other.timestamp).total_seconds()) < 5.0)


class BatchEditCommand(ContextEditCommand):
    """Command that groups multiple edit commands together"""
    
    def __init__(self, commands: List[ContextEditCommand], description: str = "Batch edit"):
        super().__init__(EditAction.BATCH_EDIT, description)
        self.commands = commands
    
    async def execute(self, preview: ContextPreview) -> ContextPreview:
        """Execute all commands in sequence"""
        current_preview = preview
        
        for command in self.commands:
            current_preview = await command.execute(current_preview)
        
        return current_preview
    
    async def undo(self, preview: ContextPreview) -> ContextPreview:
        """Undo all commands in reverse order"""
        current_preview = preview
        
        for command in reversed(self.commands):
            current_preview = await command.undo(current_preview)
        
        return current_preview
    
    def can_merge_with(self, other: 'ContextEditCommand') -> bool:
        """Batch commands generally don't merge"""
        return False


class ContextValidator:
    """Validates context edits and provides warnings/errors"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        self.validation_level = validation_level
        self.token_counter = get_token_counter()
    
    async def validate_edit(
        self, 
        original: ContextPreview, 
        modified: ContextPreview,
        command: ContextEditCommand
    ) -> Tuple[List[str], List[str]]:
        """
        Validate an edit operation.
        
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        if self.validation_level == ValidationLevel.DISABLED:
            return errors, warnings
        
        # Basic validation
        basic_errors, basic_warnings = self._validate_basic_constraints(modified)
        errors.extend(basic_errors)
        warnings.extend(basic_warnings)
        
        # Token budget validation
        token_errors, token_warnings = self._validate_token_budget(modified)
        errors.extend(token_errors)
        warnings.extend(token_warnings)
        
        # Source integrity validation
        integrity_errors, integrity_warnings = self._validate_source_integrity(modified)
        errors.extend(integrity_errors)
        warnings.extend(integrity_warnings)
        
        # Context coherence validation
        coherence_warnings = self._validate_context_coherence(original, modified, command)
        warnings.extend(coherence_warnings)
        
        return errors, warnings
    
    def _validate_basic_constraints(self, preview: ContextPreview) -> Tuple[List[str], List[str]]:
        """Validate basic context constraints"""
        errors = []
        warnings = []
        
        # Must have at least one source
        if not preview.sources:
            errors.append("Context must contain at least one source")
        
        # Must have a query source
        query_sources = [s for s in preview.sources if s.type == ContextSourceType.QUERY]
        if not query_sources:
            if self.validation_level == ValidationLevel.STRICT:
                errors.append("Context must contain a query source")
            else:
                warnings.append("No query source found - this may cause issues")
        
        # Check for empty sources
        empty_sources = [i for i, s in enumerate(preview.sources) if not s.content.strip()]
        if empty_sources:
            warnings.append(f"Empty sources found at indices: {empty_sources}")
        
        return errors, warnings
    
    def _validate_token_budget(self, preview: ContextPreview) -> Tuple[List[str], List[str]]:
        """Validate token usage and budget constraints"""
        errors = []
        warnings = []
        
        # Check token count consistency
        calculated_tokens = sum(s.tokens for s in preview.sources)
        if abs(calculated_tokens - preview.total_tokens) > 10:  # Allow small discrepancy
            errors.append(f"Token count mismatch: calculated {calculated_tokens}, stored {preview.total_tokens}")
        
        # Warn about large contexts
        if preview.total_tokens > 4000:
            warnings.append(f"Large context ({preview.total_tokens} tokens) may be expensive")
        elif preview.total_tokens > 8000:
            if self.validation_level == ValidationLevel.STRICT:
                errors.append(f"Context too large ({preview.total_tokens} tokens) - exceeds maximum")
            else:
                warnings.append(f"Very large context ({preview.total_tokens} tokens) - consider optimization")
        
        return errors, warnings
    
    def _validate_source_integrity(self, preview: ContextPreview) -> Tuple[List[str], List[str]]:
        """Validate individual source integrity"""
        errors = []
        warnings = []
        
        for i, source in enumerate(preview.sources):
            # Validate token count
            expected_tokens = self.token_counter.count_tokens(source.content, preview.model_target)
            if abs(expected_tokens - source.tokens) > 5:  # Allow small discrepancy
                warnings.append(f"Source {i}: token count may be incorrect ({source.tokens} vs {expected_tokens})")
            
            # Validate source type consistency
            if source.type == ContextSourceType.QUERY and i != 0:
                warnings.append(f"Query source found at position {i} - typically should be first")
            
            # Check priority bounds
            if source.priority < 0 or source.priority > 100:
                warnings.append(f"Source {i}: priority {source.priority} outside recommended range (0-100)")
        
        return errors, warnings
    
    def _validate_context_coherence(
        self, 
        original: ContextPreview, 
        modified: ContextPreview, 
        command: ContextEditCommand
    ) -> List[str]:
        """Validate that the edit maintains context coherence"""
        warnings = []
        
        # Check for dramatic changes that might affect quality
        original_tokens = original.total_tokens
        modified_tokens = modified.total_tokens
        
        if original_tokens > 0:
            change_ratio = abs(modified_tokens - original_tokens) / original_tokens
            
            if change_ratio > 0.5:  # 50% change
                warnings.append(f"Large context change ({change_ratio:.1%}) may significantly impact quality")
        
        # Check for loss of important source types
        original_types = set(s.type for s in original.sources)
        modified_types = set(s.type for s in modified.sources)
        lost_types = original_types - modified_types
        
        if lost_types:
            warnings.append(f"Lost source types: {[t.value for t in lost_types]}")
        
        return warnings


class ContextEditSession:
    """
    Manages an interactive context editing session with undo/redo functionality.
    
    This is the main class for interactive context editing, providing:
    - Undo/redo stack management
    - Real-time validation
    - State persistence and recovery
    - Edit command execution and tracking
    """
    
    def __init__(
        self,
        initial_preview: ContextPreview,
        max_history: int = 50,
        validation_level: ValidationLevel = ValidationLevel.STRICT,
        auto_backup: bool = True
    ):
        self.session_id = f"edit_session_{datetime.utcnow().timestamp()}"
        self.initial_preview = deepcopy(initial_preview)
        self.max_history = max_history
        self.validation_level = validation_level
        self.auto_backup = auto_backup
        
        # Edit state management
        self.history: List[EditState] = []
        self.current_index = -1
        self.unsaved_changes = False
        
        # Initialize with the initial state
        initial_state = EditState(
            preview=deepcopy(initial_preview),
            timestamp=datetime.utcnow(),
            edit_count=0
        )
        self.history.append(initial_state)
        self.current_index = 0
        
        # Command tracking
        self.command_history: List[ContextEditCommand] = []
        self.pending_commands: List[ContextEditCommand] = []
        
        # Validation and monitoring
        self.validator = ContextValidator(validation_level)
        self.edit_listeners: List[Callable[[EditState], None]] = []
        
        # Auto-save configuration
        self.auto_save_interval = 30  # seconds
        self.last_auto_save = datetime.utcnow()
        
        logger.info(f"Created context edit session {self.session_id}")
    
    @property
    def current_state(self) -> EditState:
        """Get the current edit state"""
        return self.history[self.current_index]
    
    @property
    def current_preview(self) -> ContextPreview:
        """Get the current context preview"""
        return self.current_state.preview
    
    @property
    def can_undo(self) -> bool:
        """Check if undo is available"""
        return self.current_index > 0
    
    @property
    def can_redo(self) -> bool:
        """Check if redo is available"""
        return self.current_index < len(self.history) - 1
    
    @property
    def edit_count(self) -> int:
        """Get total number of edits performed"""
        return self.current_state.edit_count
    
    async def execute_command(self, command: ContextEditCommand) -> bool:
        """
        Execute an edit command with validation and state management.
        
        Args:
            command: The command to execute
            
        Returns:
            True if command was executed successfully, False otherwise
        """
        try:
            logger.debug(f"Executing command: {command.get_summary()}")
            
            # Execute the command
            new_preview = await command.execute(self.current_preview)
            
            # Validate the result
            errors, warnings = await self.validator.validate_edit(
                self.current_preview, new_preview, command
            )
            
            # Handle validation errors
            if errors and self.validation_level == ValidationLevel.STRICT:
                logger.warning(f"Command blocked by validation errors: {errors}")
                return False
            
            # Create new state
            new_state = EditState(
                preview=new_preview,
                timestamp=datetime.utcnow(),
                edit_count=self.edit_count + 1,
                validation_errors=errors,
                warnings=warnings
            )
            
            # Add to history (truncate future if we're not at the end)
            if self.current_index < len(self.history) - 1:
                self.history = self.history[:self.current_index + 1]
            
            self.history.append(new_state)
            self.current_index += 1
            
            # Limit history size
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]
                self.current_index = len(self.history) - 1
            
            # Track command
            self.command_history.append(command)
            self.unsaved_changes = True
            
            # Notify listeners
            await self._notify_edit_listeners(new_state)
            
            # Auto-backup if enabled
            if self.auto_backup:
                await self._check_auto_backup()
            
            logger.info(f"Command executed successfully: {command.get_summary()}")
            if warnings:
                logger.warning(f"Command warnings: {warnings}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute command {command.get_summary()}: {e}")
            return False
    
    async def undo(self) -> bool:
        """Undo the last edit operation"""
        if not self.can_undo:
            return False
        
        try:
            self.current_index -= 1
            await self._notify_edit_listeners(self.current_state)
            self.unsaved_changes = True
            
            logger.info(f"Undo successful, now at edit {self.edit_count}")
            return True
            
        except Exception as e:
            logger.error(f"Undo failed: {e}")
            self.current_index += 1  # Restore position
            return False
    
    async def redo(self) -> bool:
        """Redo the next edit operation"""
        if not self.can_redo:
            return False
        
        try:
            self.current_index += 1
            await self._notify_edit_listeners(self.current_state)
            self.unsaved_changes = True
            
            logger.info(f"Redo successful, now at edit {self.edit_count}")
            return True
            
        except Exception as e:
            logger.error(f"Redo failed: {e}")
            self.current_index -= 1  # Restore position
            return False
    
    async def reset_to_initial(self) -> bool:
        """Reset the context to its initial state"""
        try:
            self.current_index = 0
            await self._notify_edit_listeners(self.current_state)
            self.unsaved_changes = True
            
            logger.info("Reset to initial state successful")
            return True
            
        except Exception as e:
            logger.error(f"Reset failed: {e}")
            return False
    
    async def add_source(
        self, 
        source: ContextSource, 
        position: Optional[int] = None
    ) -> bool:
        """Add a new context source"""
        command = AddSourceCommand(source, position)
        return await self.execute_command(command)
    
    async def remove_source(self, source_index: int) -> bool:
        """Remove a context source by index"""
        if 0 <= source_index < len(self.current_preview.sources):
            command = RemoveSourceCommand(source_index)
            return await self.execute_command(command)
        return False
    
    async def modify_source(
        self, 
        source_index: int, 
        modifications: Dict[str, Any]
    ) -> bool:
        """Modify an existing context source"""
        if 0 <= source_index < len(self.current_preview.sources):
            command = ModifySourceCommand(source_index, modifications)
            return await self.execute_command(command)
        return False
    
    async def batch_edit(
        self, 
        commands: List[ContextEditCommand], 
        description: str = "Batch edit"
    ) -> bool:
        """Execute multiple commands as a single undoable operation"""
        batch_command = BatchEditCommand(commands, description)
        return await self.execute_command(batch_command)
    
    def get_edit_summary(self) -> Dict[str, Any]:
        """Get a summary of the current editing session"""
        return {
            "session_id": self.session_id,
            "edit_count": self.edit_count,
            "total_tokens": self.current_preview.total_tokens,
            "source_count": len(self.current_preview.sources),
            "validation_errors": len(self.current_state.validation_errors),
            "warnings": len(self.current_state.warnings),
            "can_undo": self.can_undo,
            "can_redo": self.can_redo,
            "unsaved_changes": self.unsaved_changes,
            "content_hash": self.current_state.content_hash
        }
    
    def get_history_summary(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get a summary of recent edit history"""
        start_index = max(0, len(self.history) - limit)
        history_items = []
        
        for i, state in enumerate(self.history[start_index:], start_index):
            is_current = (i == self.current_index)
            history_items.append({
                "index": i,
                "edit_count": state.edit_count,
                "timestamp": state.timestamp.isoformat(),
                "token_count": state.preview.total_tokens,
                "source_count": len(state.preview.sources),
                "errors": len(state.validation_errors),
                "warnings": len(state.warnings),
                "is_current": is_current,
                "content_hash": state.content_hash
            })
        
        return history_items
    
    async def save_session(self, filepath: Optional[str] = None) -> str:
        """Save the current session to a file"""
        if filepath is None:
            filepath = f"/tmp/autom8_edit_session_{self.session_id}.json"
        
        try:
            session_data = {
                "session_id": self.session_id,
                "created_at": self.history[0].timestamp.isoformat(),
                "saved_at": datetime.utcnow().isoformat(),
                "initial_preview": self.initial_preview.dict(),
                "current_preview": self.current_preview.dict(),
                "current_index": self.current_index,
                "edit_count": self.edit_count,
                "validation_level": self.validation_level.value,
                "summary": self.get_edit_summary(),
                "history_summary": self.get_history_summary()
            }
            
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            self.unsaved_changes = False
            logger.info(f"Session saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            raise
    
    async def _notify_edit_listeners(self, state: EditState):
        """Notify all registered edit listeners of state changes"""
        for listener in self.edit_listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(state)
                else:
                    listener(state)
            except Exception as e:
                logger.warning(f"Edit listener failed: {e}")
    
    async def _check_auto_backup(self):
        """Check if auto-backup is needed and perform it"""
        now = datetime.utcnow()
        if (now - self.last_auto_save).total_seconds() >= self.auto_save_interval:
            try:
                backup_path = f"/tmp/autom8_auto_backup_{self.session_id}_{now.timestamp()}.json"
                await self.save_session(backup_path)
                self.last_auto_save = now
                logger.debug(f"Auto-backup saved to {backup_path}")
            except Exception as e:
                logger.warning(f"Auto-backup failed: {e}")
    
    def add_edit_listener(self, listener: Callable[[EditState], None]):
        """Add a listener for edit state changes"""
        self.edit_listeners.append(listener)
    
    def remove_edit_listener(self, listener: Callable[[EditState], None]):
        """Remove an edit state listener"""
        if listener in self.edit_listeners:
            self.edit_listeners.remove(listener)