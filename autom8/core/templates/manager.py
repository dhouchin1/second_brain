"""
TemplateManager - Core template management functionality.

This is the main entry point for all template operations, orchestrating
template storage, rendering, composition, validation, and analytics.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

from autom8.models.template import (
    ContextTemplate,
    TemplateType,
    TemplateStatus,
    TemplateExecutionContext,
    TemplateExecutionResult,
    TemplateLibrary,
    TemplateImportExport,
    TemplateVariable,
    TemplateSource,
    TemplateMetadata,
)
from autom8.models.context import (
    ContextPreview,
    ContextSource,
    ContextSourceType,
)
from autom8.core.context.inspector import ContextInspector
from autom8.utils.tokens import get_token_counter
from autom8.utils.logging import get_logger
from autom8.config.settings import get_settings

logger = get_logger(__name__)


class TemplateManager:
    """
    Core template management system.
    
    Provides comprehensive template lifecycle management including creation,
    storage, execution, composition, validation, and analytics.
    """
    
    def __init__(self):
        self.storage = None
        self.renderer = None
        self.composer = None
        self.validator = None
        self.analytics = None
        self.context_inspector = None
        self.token_counter = get_token_counter()
        self._initialized = False
        
        # Cache for frequently used templates
        self._template_cache = {}
        self._cache_ttl = timedelta(minutes=30)
        self._last_cache_cleanup = datetime.utcnow()
    
    async def initialize(self) -> bool:
        """Initialize the template manager and all dependencies."""
        if self._initialized:
            return True
        
        try:
            logger.info("Initializing TemplateManager...")
            
            # Initialize dependencies
            from .storage import get_template_storage
            from .renderer import TemplateRenderer
            from .composer import TemplateComposer
            from .validator import TemplateValidator
            from .analytics import get_template_analytics
            
            self.storage = await get_template_storage()
            self.renderer = TemplateRenderer()
            self.composer = TemplateComposer()
            self.validator = TemplateValidator()
            self.analytics = await get_template_analytics()
            
            # Initialize context inspector for integration
            self.context_inspector = ContextInspector()
            await self.context_inspector.initialize()
            
            # Initialize sub-components
            await self.renderer.initialize()
            await self.composer.initialize(self)
            await self.validator.initialize()
            
            self._initialized = True
            logger.info("TemplateManager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TemplateManager: {e}")
            return False
    
    # Template Lifecycle Management
    
    async def create_template(
        self,
        template_id: str,
        template_type: TemplateType,
        metadata: TemplateMetadata,
        variables: Optional[List[TemplateVariable]] = None,
        sources: Optional[List[TemplateSource]] = None,
        created_by: Optional[str] = None
    ) -> ContextTemplate:
        """Create a new context template."""
        logger.debug(f"Creating template: {template_id}")
        
        # Validate template_id uniqueness
        existing = await self.storage.get_template(template_id)
        if existing:
            raise ValueError(f"Template with ID '{template_id}' already exists")
        
        # Create template
        template = ContextTemplate(
            template_id=template_id,
            type=template_type,
            status=TemplateStatus.DRAFT,
            metadata=metadata,
            variables=variables or [],
            sources=sources or [],
            created_by=created_by
        )
        
        # Validate template structure
        validation_errors = await self.validator.validate_template(template)
        if validation_errors:
            raise ValueError(f"Template validation failed: {validation_errors}")
        
        # Store template
        success = await self.storage.store_template(template)
        if not success:
            raise RuntimeError("Failed to store template")
        
        logger.info(f"Created template: {template_id}")
        return template
    
    async def get_template(self, template_id: str) -> Optional[ContextTemplate]:
        """Get template by ID with caching."""
        # Check cache first
        cached = self._get_from_cache(template_id)
        if cached:
            return cached
        
        # Load from storage
        template = await self.storage.get_template(template_id)
        if template:
            self._add_to_cache(template_id, template)
        
        return template
    
    async def update_template(
        self,
        template_id: str,
        updates: Dict[str, Any],
        updated_by: Optional[str] = None
    ) -> Optional[ContextTemplate]:
        """Update an existing template."""
        logger.debug(f"Updating template: {template_id}")
        
        # Get existing template
        template = await self.get_template(template_id)
        if not template:
            raise ValueError(f"Template '{template_id}' not found")
        
        # Apply updates
        for field, value in updates.items():
            if hasattr(template, field):
                setattr(template, field, value)
        
        # Update timestamps
        template.updated_at = datetime.utcnow()
        
        # Validate updated template
        validation_errors = await self.validator.validate_template(template)
        if validation_errors:
            raise ValueError(f"Template validation failed: {validation_errors}")
        
        # Store updated template
        success = await self.storage.store_template(template)
        if not success:
            raise RuntimeError("Failed to update template")
        
        # Update cache
        self._add_to_cache(template_id, template)
        
        logger.info(f"Updated template: {template_id}")
        return template
    
    async def delete_template(self, template_id: str) -> bool:
        """Delete a template."""
        logger.debug(f"Deleting template: {template_id}")
        
        # Check if template exists
        template = await self.get_template(template_id)
        if not template:
            return False
        
        # Delete from storage
        success = await self.storage.delete_template(template_id)
        if success:
            # Remove from cache
            self._remove_from_cache(template_id)
            logger.info(f"Deleted template: {template_id}")
        
        return success
    
    async def list_templates(
        self,
        template_type: Optional[TemplateType] = None,
        status: Optional[TemplateStatus] = None,
        created_by: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[ContextTemplate]:
        """List templates with filtering."""
        return await self.storage.list_templates(
            template_type=template_type,
            status=status,
            created_by=created_by,
            tags=tags,
            limit=limit,
            offset=offset
        )
    
    async def search_templates(
        self,
        query: str,
        template_type: Optional[TemplateType] = None,
        limit: int = 20
    ) -> List[ContextTemplate]:
        """Search templates by content."""
        return await self.storage.search_templates(
            query=query,
            template_type=template_type,
            limit=limit
        )
    
    # Template Execution
    
    async def execute_template(
        self,
        template_id: str,
        variables: Dict[str, Any],
        executor: Optional[str] = None,
        dry_run: bool = False,
        agent_id: str = "template-executor"
    ) -> TemplateExecutionResult:
        """
        Execute a template with provided variables.
        
        This is the main template execution method that orchestrates
        rendering, composition, validation, and context creation.
        """
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        start_time = datetime.utcnow()
        
        logger.debug(f"Executing template {template_id} (execution: {execution_id})")
        
        # Create execution context
        context = TemplateExecutionContext(
            template_id=template_id,
            variables=variables,
            execution_id=execution_id,
            executor=executor,
            dry_run=dry_run
        )
        
        try:
            # Get template
            template = await self.get_template(template_id)
            if not template:
                raise ValueError(f"Template '{template_id}' not found")
            
            # Validate template is active
            if template.status not in [TemplateStatus.ACTIVE, TemplateStatus.DRAFT]:
                raise ValueError(f"Template '{template_id}' is not active (status: {template.status})")
            
            # Validate variables
            validation_errors = template.validate_variable_values(variables)
            if validation_errors:
                return TemplateExecutionResult(
                    execution_context=context,
                    success=False,
                    validation_errors=validation_errors,
                    error_message="Variable validation failed"
                )
            
            # Handle composition if needed
            if template.has_composition:
                logger.debug(f"Template {template_id} uses composition")
                template = await self.composer.compose_template(
                    template, variables, context
                )
            
            # Render template sources
            render_start = datetime.utcnow()
            rendered_sources = await self.renderer.render_template(
                template, variables, context
            )
            render_time = (datetime.utcnow() - render_start).total_seconds() * 1000
            
            # Count tokens
            token_start = datetime.utcnow()
            total_tokens = sum(
                self.token_counter.count_tokens(source.get('content', ''), 'gpt-3.5-turbo')
                for source in rendered_sources
            )
            token_time = (datetime.utcnow() - token_start).total_seconds() * 1000
            
            # Apply template validation rules
            warnings = []
            if template.validation:
                validation_warnings = await self.validator.validate_execution_result(
                    template, rendered_sources, total_tokens
                )
                warnings.extend(validation_warnings)
            
            # Create execution result
            context.completed_at = datetime.utcnow()
            result = TemplateExecutionResult(
                execution_context=context,
                success=True,
                rendered_sources=rendered_sources,
                total_tokens=total_tokens,
                warnings=warnings,
                render_time_ms=int(render_time),
                token_count_time_ms=int(token_time)
            )
            
            # Record analytics (async, don't block)
            if not dry_run:
                asyncio.create_task(self._record_execution_analytics(template, result))
            
            logger.info(
                f"Template execution completed: {template_id} -> "
                f"{len(rendered_sources)} sources, {total_tokens} tokens"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Template execution failed: {e}")
            
            context.completed_at = datetime.utcnow()
            return TemplateExecutionResult(
                execution_context=context,
                success=False,
                error_message=str(e),
                error_details={"exception_type": type(e).__name__}
            )
    
    async def execute_template_to_context_preview(
        self,
        template_id: str,
        variables: Dict[str, Any],
        agent_id: str = "template-executor",
        model_target: Optional[str] = None,
        complexity_score: Optional[float] = None
    ) -> Tuple[bool, Union[ContextPreview, str]]:
        """
        Execute template and convert result to ContextPreview.
        
        This method bridges templates with the existing context system,
        allowing templates to be used seamlessly in the preview/approval workflow.
        """
        logger.debug(f"Executing template {template_id} to context preview")
        
        try:
            # Execute template
            result = await self.execute_template(
                template_id=template_id,
                variables=variables,
                agent_id=agent_id
            )
            
            if not result.success:
                error_msg = result.error_message or "Template execution failed"
                if result.validation_errors:
                    error_msg += f": {', '.join(result.validation_errors)}"
                return False, error_msg
            
            # Convert rendered sources to ContextSource objects
            context_sources = []
            for source_data in result.rendered_sources:
                context_source = ContextSource(
                    type=ContextSourceType(source_data.get('type', 'reference')),
                    content=source_data['content'],
                    tokens=self.token_counter.count_tokens(
                        source_data['content'], model_target or 'gpt-3.5-turbo'
                    ),
                    source=source_data.get('source', f"template:{template_id}"),
                    location=source_data.get('location'),
                    expandable=source_data.get('expandable', False),
                    priority=source_data.get('priority', 50),
                    summary=source_data.get('summary'),
                    timestamp=datetime.utcnow()
                )
                context_sources.append(context_source)
            
            # Generate a query from template if not provided in sources
            query_sources = [s for s in context_sources if s.type == ContextSourceType.QUERY]
            if not query_sources:
                # Create a default query describing the template
                template = await self.get_template(template_id)
                query_content = f"Execute template: {template.metadata.title}"
                if template.metadata.description:
                    query_content += f" - {template.metadata.description}"
                
                query_source = ContextSource(
                    type=ContextSourceType.QUERY,
                    content=query_content,
                    tokens=self.token_counter.count_tokens(query_content, model_target),
                    source="template_query",
                    priority=100
                )
                context_sources.insert(0, query_source)
                query_content = query_source.content
            else:
                query_content = query_sources[0].content
            
            # Create context preview using inspector
            preview = await self.context_inspector.preview(
                query=query_content,
                agent_id=agent_id,
                context_sources=context_sources[1:],  # Skip query source
                model_target=model_target,
                complexity_score=complexity_score
            )
            
            # Add template execution metadata to preview
            preview.metadata = getattr(preview, 'metadata', {})
            preview.metadata.update({
                'template_id': template_id,
                'template_execution_id': result.execution_context.execution_id,
                'template_variables': variables,
                'template_render_time_ms': result.render_time_ms
            })
            
            return True, preview
            
        except Exception as e:
            logger.error(f"Failed to execute template to context preview: {e}")
            return False, str(e)
    
    # Template Composition
    
    async def compose_templates(
        self,
        base_template_ids: List[str],
        variables: Dict[str, Any],
        merge_strategy: str = "append",
        conflict_resolution: str = "latest"
    ) -> ContextTemplate:
        """Compose multiple templates into a single template."""
        return await self.composer.compose_templates(
            base_template_ids, variables, merge_strategy, conflict_resolution
        )
    
    # Template Validation and Testing
    
    async def validate_template(self, template: ContextTemplate) -> List[str]:
        """Validate template structure and content."""
        return await self.validator.validate_template(template)
    
    async def test_template(
        self,
        template_id: str,
        test_variables: Dict[str, Any],
        expected_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Test template execution with provided variables."""
        logger.debug(f"Testing template: {template_id}")
        
        # Execute template
        result = await self.execute_template(
            template_id=template_id,
            variables=test_variables,
            dry_run=True
        )
        
        # Analyze results
        test_result = {
            "template_id": template_id,
            "success": result.success,
            "execution_time_ms": result.render_time_ms + result.token_count_time_ms,
            "total_tokens": result.total_tokens,
            "source_count": len(result.rendered_sources),
            "validation_errors": result.validation_errors,
            "warnings": result.warnings
        }
        
        # Compare with expected results if provided
        if expected_results:
            test_result["expectations"] = self._compare_with_expectations(
                result, expected_results
            )
        
        return test_result
    
    def _compare_with_expectations(
        self, 
        result: TemplateExecutionResult,
        expected: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare execution result with expected values."""
        comparisons = {}
        
        if "min_sources" in expected:
            comparisons["min_sources"] = {
                "expected": expected["min_sources"],
                "actual": len(result.rendered_sources),
                "passed": len(result.rendered_sources) >= expected["min_sources"]
            }
        
        if "max_tokens" in expected:
            comparisons["max_tokens"] = {
                "expected": expected["max_tokens"],
                "actual": result.total_tokens,
                "passed": result.total_tokens <= expected["max_tokens"]
            }
        
        if "should_succeed" in expected:
            comparisons["should_succeed"] = {
                "expected": expected["should_succeed"],
                "actual": result.success,
                "passed": result.success == expected["should_succeed"]
            }
        
        return comparisons
    
    # Template Libraries and Sharing
    
    async def create_library(
        self,
        library_id: str,
        name: str,
        description: str,
        author: Optional[str] = None
    ) -> TemplateLibrary:
        """Create a new template library."""
        library = TemplateLibrary(
            library_id=library_id,
            name=name,
            description=description,
            author=author
        )
        
        success = await self.storage.store_library(library)
        if not success:
            raise RuntimeError("Failed to create library")
        
        logger.info(f"Created template library: {library_id}")
        return library
    
    async def add_template_to_library(
        self,
        library_id: str,
        template_id: str
    ) -> bool:
        """Add template to library."""
        return await self.storage.add_template_to_library(library_id, template_id)
    
    async def export_template(
        self,
        template_id: str,
        format: str = "json",
        include_metadata: bool = True
    ) -> Union[str, Dict[str, Any]]:
        """Export template to specified format."""
        template = await self.get_template(template_id)
        if not template:
            raise ValueError(f"Template '{template_id}' not found")
        
        export_config = TemplateImportExport(
            format=format,
            include_metadata=include_metadata
        )
        
        if format.lower() == "json":
            template_dict = template.dict()
            if not include_metadata:
                template_dict.pop('metadata', None)
            return template_dict
        
        elif format.lower() == "yaml":
            import yaml
            template_dict = template.dict()
            if not include_metadata:
                template_dict.pop('metadata', None)
            return yaml.dump(template_dict, default_flow_style=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def import_template(
        self,
        template_data: Union[str, Dict[str, Any]],
        format: str = "json",
        merge_strategy: str = "replace"
    ) -> ContextTemplate:
        """Import template from data."""
        if format.lower() == "json":
            if isinstance(template_data, str):
                template_dict = json.loads(template_data)
            else:
                template_dict = template_data
        
        elif format.lower() == "yaml":
            import yaml
            if isinstance(template_data, dict):
                template_dict = template_data
            else:
                template_dict = yaml.safe_load(template_data)
        
        else:
            raise ValueError(f"Unsupported import format: {format}")
        
        # Create template from dict
        template = ContextTemplate(**template_dict)
        
        # Handle existing template
        existing = await self.get_template(template.template_id)
        if existing:
            if merge_strategy == "replace":
                pass  # Will overwrite
            elif merge_strategy == "skip":
                return existing
            elif merge_strategy == "error":
                raise ValueError(f"Template '{template.template_id}' already exists")
        
        # Store template
        success = await self.storage.store_template(template)
        if not success:
            raise RuntimeError("Failed to import template")
        
        logger.info(f"Imported template: {template.template_id}")
        return template
    
    # Analytics and Insights
    
    async def get_template_analytics(self, template_id: str) -> Dict[str, Any]:
        """Get analytics for a specific template."""
        return await self.analytics.get_template_analytics(template_id)
    
    async def get_system_analytics(self) -> Dict[str, Any]:
        """Get system-wide template analytics."""
        return await self.analytics.get_system_analytics()
    
    async def get_popular_templates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular templates."""
        return await self.analytics.get_popular_templates(limit)
    
    # Cache Management
    
    def _get_from_cache(self, template_id: str) -> Optional[ContextTemplate]:
        """Get template from cache."""
        self._cleanup_cache()
        
        cache_entry = self._template_cache.get(template_id)
        if cache_entry:
            if datetime.utcnow() - cache_entry['timestamp'] < self._cache_ttl:
                return cache_entry['template']
            else:
                del self._template_cache[template_id]
        
        return None
    
    def _add_to_cache(self, template_id: str, template: ContextTemplate):
        """Add template to cache."""
        self._template_cache[template_id] = {
            'template': template,
            'timestamp': datetime.utcnow()
        }
    
    def _remove_from_cache(self, template_id: str):
        """Remove template from cache."""
        self._template_cache.pop(template_id, None)
    
    def _cleanup_cache(self):
        """Remove expired cache entries."""
        now = datetime.utcnow()
        
        # Only cleanup every 5 minutes
        if now - self._last_cache_cleanup < timedelta(minutes=5):
            return
        
        expired_keys = [
            key for key, entry in self._template_cache.items()
            if now - entry['timestamp'] > self._cache_ttl
        ]
        
        for key in expired_keys:
            del self._template_cache[key]
        
        self._last_cache_cleanup = now
    
    # Analytics Recording
    
    async def _record_execution_analytics(
        self,
        template: ContextTemplate,
        result: TemplateExecutionResult
    ):
        """Record template execution analytics (async)."""
        try:
            await self.analytics.record_execution(template, result)
        except Exception as e:
            logger.warning(f"Failed to record template analytics: {e}")
    
    # Integration Methods
    
    async def create_template_from_context(
        self,
        context_preview: ContextPreview,
        template_id: str,
        metadata: TemplateMetadata,
        extract_variables: bool = True
    ) -> ContextTemplate:
        """
        Create a template from an existing context preview.
        
        This allows users to "templateize" contexts they've built through
        the normal context editing workflow.
        """
        logger.debug(f"Creating template from context: {template_id}")
        
        # Convert context sources to template sources
        template_sources = []
        extracted_variables = []
        
        for source in context_preview.sources:
            if source.type == ContextSourceType.QUERY:
                continue  # Handle query separately
            
            # Create template source
            template_source = TemplateSource(
                type=source.type.value,
                content_template=source.content,
                priority=source.priority,
                expandable=source.expandable,
                source_id_template=source.source
            )
            
            # Extract variables if requested
            if extract_variables:
                source_variables = self._extract_variables_from_content(source.content)
                extracted_variables.extend(source_variables)
            
            template_sources.append(template_source)
        
        # Create template
        template = ContextTemplate(
            template_id=template_id,
            type=TemplateType.CONTEXT,
            status=TemplateStatus.DRAFT,
            metadata=metadata,
            variables=extracted_variables,
            sources=template_sources
        )
        
        # Store template
        success = await self.storage.store_template(template)
        if not success:
            raise RuntimeError("Failed to create template from context")
        
        logger.info(f"Created template from context: {template_id}")
        return template
    
    def _extract_variables_from_content(self, content: str) -> List[TemplateVariable]:
        """Extract potential variables from content."""
        import re
        
        # Simple variable extraction - look for {{variable}} patterns
        variable_pattern = r'\{\{(\w+)\}\}'
        matches = re.findall(variable_pattern, content)
        
        variables = []
        for match in set(matches):  # Remove duplicates
            variables.append(TemplateVariable(
                name=match,
                type="string",
                description=f"Extracted variable: {match}",
                required=True
            ))
        
        return variables


# Global instance
_template_manager = None


async def get_template_manager() -> TemplateManager:
    """Get global template manager instance."""
    global _template_manager
    
    if _template_manager is None:
        _template_manager = TemplateManager()
        await _template_manager.initialize()
    
    return _template_manager