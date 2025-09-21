"""
Structured Context Editor - High-level interface for programmatic context editing.

Provides a simplified, structured interface for editing context that can be used
programmatically or through API endpoints. This complements the full interactive
editor with a more focused set of operations.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from autom8.models.context import (
    ContextPreview,
    ContextSource,
    ContextSourceType,
    ContextWarning,
    ContextOptimization,
)
from autom8.core.context.inspector import ContextInspector
from autom8.core.context.editor import ContextEditSession, ValidationLevel
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class EditOperation(str, Enum):
    """High-level edit operations"""
    REPLACE_CONTENT = "replace_content"
    UPDATE_PRIORITY = "update_priority"
    ADD_SOURCE = "add_source"
    REMOVE_SOURCE = "remove_source"
    REORDER_SOURCES = "reorder_sources"
    APPLY_OPTIMIZATION = "apply_optimization"
    MERGE_SIMILAR = "merge_similar"
    SPLIT_SOURCE = "split_source"


@dataclass
class EditRequest:
    """Represents a structured edit request"""
    operation: EditOperation
    target_index: Optional[int] = None
    source_type: Optional[ContextSourceType] = None
    content: Optional[str] = None
    priority: Optional[int] = None
    position: Optional[int] = None
    optimization_profile: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EditResponse(BaseModel):
    """Response from a structured edit operation"""
    success: bool
    operation: str
    message: str
    tokens_changed: int = 0
    sources_changed: int = 0
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    new_total_tokens: int = 0
    edit_summary: Optional[Dict[str, Any]] = None


class StructuredContextEditor:
    """
    High-level interface for structured context editing operations.
    
    This class provides a simplified API for common context editing tasks,
    making it easy to integrate context editing into applications and scripts.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        self.validation_level = validation_level
        self.inspector = ContextInspector()
        self.current_session: Optional[ContextEditSession] = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the structured editor"""
        try:
            success = await self.inspector.initialize()
            if success:
                self._initialized = True
                logger.info("Structured context editor initialized")
            return success
        except Exception as e:
            logger.error(f"Failed to initialize structured editor: {e}")
            return False
    
    async def start_editing(
        self, 
        preview: ContextPreview,
        auto_backup: bool = True,
        max_history: int = 30
    ) -> str:
        """
        Start a new editing session.
        
        Args:
            preview: Context preview to edit
            auto_backup: Enable automatic backup
            max_history: Maximum undo history
            
        Returns:
            Session ID
        """
        if not self._initialized:
            raise RuntimeError("Editor not initialized")
        
        self.current_session = await self.inspector.create_edit_session(
            preview=preview,
            max_history=max_history,
            validation_level=self.validation_level.value,
            auto_backup=auto_backup
        )
        
        logger.info(f"Started structured editing session {self.current_session.session_id}")
        return self.current_session.session_id
    
    async def apply_edit(self, edit_request: EditRequest) -> EditResponse:
        """
        Apply a structured edit request.
        
        Args:
            edit_request: The edit operation to perform
            
        Returns:
            EditResponse with results
        """
        if not self.current_session:
            raise RuntimeError("No active editing session")
        
        original_tokens = self.current_session.current_preview.total_tokens
        original_sources = len(self.current_session.current_preview.sources)
        
        try:
            success = False
            message = ""
            warnings = []
            errors = []
            
            if edit_request.operation == EditOperation.REPLACE_CONTENT:
                success = await self._handle_replace_content(edit_request)
                message = f"Replaced content for source {edit_request.target_index}"
            
            elif edit_request.operation == EditOperation.UPDATE_PRIORITY:
                success = await self._handle_update_priority(edit_request)
                message = f"Updated priority for source {edit_request.target_index}"
            
            elif edit_request.operation == EditOperation.ADD_SOURCE:
                success = await self._handle_add_source(edit_request)
                message = f"Added new {edit_request.source_type.value} source"
            
            elif edit_request.operation == EditOperation.REMOVE_SOURCE:
                success = await self._handle_remove_source(edit_request)
                message = f"Removed source {edit_request.target_index}"
            
            elif edit_request.operation == EditOperation.REORDER_SOURCES:
                success = await self._handle_reorder_sources(edit_request)
                message = "Reordered context sources"
            
            elif edit_request.operation == EditOperation.APPLY_OPTIMIZATION:
                success, opt_warnings = await self._handle_apply_optimization(edit_request)
                warnings.extend(opt_warnings)
                message = f"Applied {edit_request.optimization_profile} optimization"
            
            elif edit_request.operation == EditOperation.MERGE_SIMILAR:
                success = await self._handle_merge_similar(edit_request)
                message = "Merged similar sources"
            
            elif edit_request.operation == EditOperation.SPLIT_SOURCE:
                success = await self._handle_split_source(edit_request)
                message = f"Split source {edit_request.target_index}"
            
            else:
                raise ValueError(f"Unknown operation: {edit_request.operation}")
            
            # Calculate changes
            new_tokens = self.current_session.current_preview.total_tokens
            new_sources = len(self.current_session.current_preview.sources)
            
            # Get validation state
            current_state = self.current_session.current_state
            if current_state.validation_errors:
                errors.extend(current_state.validation_errors)
            if current_state.warnings:
                warnings.extend(current_state.warnings)
            
            return EditResponse(
                success=success,
                operation=edit_request.operation.value,
                message=message,
                tokens_changed=new_tokens - original_tokens,
                sources_changed=new_sources - original_sources,
                warnings=warnings,
                errors=errors,
                new_total_tokens=new_tokens,
                edit_summary=self.current_session.get_edit_summary()
            )
            
        except Exception as e:
            logger.error(f"Edit operation failed: {e}")
            return EditResponse(
                success=False,
                operation=edit_request.operation.value,
                message=f"Operation failed: {str(e)}",
                errors=[str(e)],
                new_total_tokens=original_tokens
            )
    
    async def batch_edit(self, edit_requests: List[EditRequest]) -> List[EditResponse]:
        """
        Apply multiple edit requests as a batch.
        
        Args:
            edit_requests: List of edit operations to perform
            
        Returns:
            List of EditResponse objects
        """
        if not self.current_session:
            raise RuntimeError("No active editing session")
        
        responses = []
        
        # Apply each edit sequentially
        for edit_request in edit_requests:
            response = await self.apply_edit(edit_request)
            responses.append(response)
            
            # Stop on critical errors
            if not response.success and response.errors:
                logger.warning(f"Stopping batch edit due to error: {response.errors}")
                break
        
        return responses
    
    async def undo_last_edit(self) -> EditResponse:
        """Undo the last edit operation"""
        if not self.current_session:
            raise RuntimeError("No active editing session")
        
        if not self.current_session.can_undo:
            return EditResponse(
                success=False,
                operation="undo",
                message="Nothing to undo"
            )
        
        success = await self.current_session.undo()
        
        return EditResponse(
            success=success,
            operation="undo",
            message="Undo successful" if success else "Undo failed",
            new_total_tokens=self.current_session.current_preview.total_tokens,
            edit_summary=self.current_session.get_edit_summary()
        )
    
    async def redo_last_edit(self) -> EditResponse:
        """Redo the last undone edit operation"""
        if not self.current_session:
            raise RuntimeError("No active editing session")
        
        if not self.current_session.can_redo:
            return EditResponse(
                success=False,
                operation="redo",
                message="Nothing to redo"
            )
        
        success = await self.current_session.redo()
        
        return EditResponse(
            success=success,
            operation="redo",
            message="Redo successful" if success else "Redo failed",
            new_total_tokens=self.current_session.current_preview.total_tokens,
            edit_summary=self.current_session.get_edit_summary()
        )
    
    async def get_current_preview(self) -> ContextPreview:
        """Get the current context preview"""
        if not self.current_session:
            raise RuntimeError("No active editing session")
        
        return self.current_session.current_preview
    
    async def finalize_editing(self) -> ContextPreview:
        """
        Finalize the editing session and return the final context preview.
        
        Returns:
            Final context preview with all edits applied
        """
        if not self.current_session:
            raise RuntimeError("No active editing session")
        
        final_preview = await self.inspector.apply_interactive_edits(
            original_preview=self.current_session.initial_preview,
            edit_session=self.current_session
        )
        
        # Save session for potential recovery
        await self.current_session.save_session()
        
        session_id = self.current_session.session_id
        self.current_session = None
        
        logger.info(f"Finalized editing session {session_id}")
        return final_preview
    
    def get_session_summary(self) -> Optional[Dict[str, Any]]:
        """Get a summary of the current editing session"""
        if not self.current_session:
            return None
        
        return self.current_session.get_edit_summary()
    
    # Private helper methods for handling specific operations
    
    async def _handle_replace_content(self, edit_request: EditRequest) -> bool:
        """Handle content replacement operation"""
        if edit_request.target_index is None or edit_request.content is None:
            raise ValueError("target_index and content required for replace_content")
        
        return await self.current_session.modify_source(
            edit_request.target_index,
            {"content": edit_request.content}
        )
    
    async def _handle_update_priority(self, edit_request: EditRequest) -> bool:
        """Handle priority update operation"""
        if edit_request.target_index is None or edit_request.priority is None:
            raise ValueError("target_index and priority required for update_priority")
        
        return await self.current_session.modify_source(
            edit_request.target_index,
            {"priority": edit_request.priority}
        )
    
    async def _handle_add_source(self, edit_request: EditRequest) -> bool:
        """Handle add source operation"""
        if not edit_request.source_type or not edit_request.content:
            raise ValueError("source_type and content required for add_source")
        
        from autom8.utils.tokens import get_token_counter
        
        token_counter = get_token_counter()
        new_source = ContextSource(
            type=edit_request.source_type,
            content=edit_request.content,
            tokens=token_counter.count_tokens(
                edit_request.content, 
                self.current_session.current_preview.model_target
            ),
            source=edit_request.metadata.get("source_id", "structured_edit"),
            priority=edit_request.priority or 50,
            timestamp=datetime.utcnow()
        )
        
        return await self.current_session.add_source(new_source, edit_request.position)
    
    async def _handle_remove_source(self, edit_request: EditRequest) -> bool:
        """Handle remove source operation"""
        if edit_request.target_index is None:
            raise ValueError("target_index required for remove_source")
        
        return await self.current_session.remove_source(edit_request.target_index)
    
    async def _handle_reorder_sources(self, edit_request: EditRequest) -> bool:
        """Handle reorder sources operation"""
        new_order = edit_request.metadata.get("new_order")
        if not new_order:
            raise ValueError("new_order required in metadata for reorder_sources")
        
        # This would require implementing the reorder functionality in the session
        # For now, return a simple implementation
        logger.warning("Reorder sources not fully implemented in structured editor")
        return False
    
    async def _handle_apply_optimization(self, edit_request: EditRequest) -> Tuple[bool, List[str]]:
        """Handle apply optimization operation"""
        profile = edit_request.optimization_profile or "balanced"
        warnings = []
        
        try:
            optimized_preview, report = await self.inspector.optimize_with_edits(
                self.current_session.current_preview,
                optimization_profile=profile
            )
            
            # Create new session with optimized preview
            new_session = await self.inspector.create_edit_session(
                optimized_preview,
                max_history=self.current_session.max_history,
                validation_level=self.current_session.validation_level.value,
                auto_backup=self.current_session.auto_backup
            )
            
            # Replace current session
            old_session_id = self.current_session.session_id
            self.current_session = new_session
            
            # Add optimization warnings
            if report.get("quality_retention", 1.0) < 0.8:
                warnings.append(f"Quality retention: {report['quality_retention']:.1%}")
            
            logger.info(f"Applied optimization, replaced session {old_session_id} with {new_session.session_id}")
            return True, warnings
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return False, [str(e)]
    
    async def _handle_merge_similar(self, edit_request: EditRequest) -> bool:
        """Handle merge similar sources operation"""
        # This would require implementing similarity detection and merging
        # For now, return a placeholder
        logger.warning("Merge similar sources not implemented in structured editor")
        return False
    
    async def _handle_split_source(self, edit_request: EditRequest) -> bool:
        """Handle split source operation"""
        if edit_request.target_index is None:
            raise ValueError("target_index required for split_source")
        
        # This would require implementing source splitting logic
        # For now, return a placeholder
        logger.warning("Split source not implemented in structured editor")
        return False


# Convenience functions for common operations

async def quick_edit_context(
    preview: ContextPreview,
    edits: List[Dict[str, Any]],
    validation_level: str = "strict"
) -> Tuple[ContextPreview, List[EditResponse]]:
    """
    Perform quick edits on a context preview.
    
    Args:
        preview: Context preview to edit
        edits: List of edit operation dictionaries
        validation_level: Validation strictness
        
    Returns:
        Tuple of (final preview, list of responses)
    """
    editor = StructuredContextEditor(ValidationLevel(validation_level))
    await editor.initialize()
    
    await editor.start_editing(preview)
    
    # Convert dictionaries to EditRequest objects
    edit_requests = []
    for edit_dict in edits:
        edit_request = EditRequest(
            operation=EditOperation(edit_dict["operation"]),
            target_index=edit_dict.get("target_index"),
            source_type=ContextSourceType(edit_dict["source_type"]) if edit_dict.get("source_type") else None,
            content=edit_dict.get("content"),
            priority=edit_dict.get("priority"),
            position=edit_dict.get("position"),
            optimization_profile=edit_dict.get("optimization_profile"),
            metadata=edit_dict.get("metadata", {})
        )
        edit_requests.append(edit_request)
    
    # Apply edits
    responses = await editor.batch_edit(edit_requests)
    
    # Finalize and return
    final_preview = await editor.finalize_editing()
    
    return final_preview, responses


async def add_context_sources(
    preview: ContextPreview,
    sources: List[Dict[str, Any]]
) -> ContextPreview:
    """
    Add multiple context sources to a preview.
    
    Args:
        preview: Context preview to extend
        sources: List of source dictionaries with keys: type, content, priority, source_id
        
    Returns:
        Updated context preview
    """
    edits = []
    for source in sources:
        edits.append({
            "operation": "add_source",
            "source_type": source["type"],
            "content": source["content"],
            "priority": source.get("priority", 50),
            "metadata": {"source_id": source.get("source_id", "added_source")}
        })
    
    final_preview, responses = await quick_edit_context(preview, edits)
    return final_preview


async def optimize_context_structured(
    preview: ContextPreview,
    profile: str = "balanced"
) -> Tuple[ContextPreview, Dict[str, Any]]:
    """
    Apply optimization to a context preview using structured editing.
    
    Args:
        preview: Context preview to optimize
        profile: Optimization profile
        
    Returns:
        Tuple of (optimized preview, optimization report)
    """
    editor = StructuredContextEditor()
    await editor.initialize()
    
    await editor.start_editing(preview)
    
    edit_request = EditRequest(
        operation=EditOperation.APPLY_OPTIMIZATION,
        optimization_profile=profile
    )
    
    response = await editor.apply_edit(edit_request)
    final_preview = await editor.finalize_editing()
    
    report = {
        "success": response.success,
        "tokens_changed": response.tokens_changed,
        "sources_changed": response.sources_changed,
        "warnings": response.warnings,
        "final_tokens": response.new_total_tokens
    }
    
    return final_preview, report