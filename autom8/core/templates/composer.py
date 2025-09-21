"""
Template Composition Engine

This module provides sophisticated template composition capabilities, allowing
templates to be combined, merged, and extended to create complex reusable patterns.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Set, Tuple
from copy import deepcopy

from autom8.models.template import (
    ContextTemplate,
    TemplateType,
    TemplateStatus,
    TemplateVariable,
    TemplateSource,
    TemplateComposition,
    TemplateInheritance,
    TemplateExecutionContext,
    TemplateMetadata,
    VariableType,
)
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class CompositionEngine:
    """
    Core template composition logic for merging and combining templates.
    """
    
    def merge_sources(
        self,
        sources_lists: List[List[TemplateSource]],
        strategy: str = "append",
        conflict_resolution: str = "latest"
    ) -> List[TemplateSource]:
        """
        Merge multiple lists of template sources.
        
        Args:
            sources_lists: Lists of sources to merge
            strategy: How to merge ("append", "replace", "merge")
            conflict_resolution: How to resolve conflicts ("latest", "priority", "error")
        
        Returns:
            Merged list of template sources
        """
        if not sources_lists:
            return []
        
        if strategy == "append":
            # Simple concatenation
            merged = []
            for sources in sources_lists:
                merged.extend(deepcopy(sources))
            return merged
        
        elif strategy == "replace":
            # Later templates replace earlier ones entirely
            return deepcopy(sources_lists[-1]) if sources_lists else []
        
        elif strategy == "merge":
            # Intelligent merging with conflict resolution
            return self._merge_sources_intelligent(sources_lists, conflict_resolution)
        
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")
    
    def merge_variables(
        self,
        variables_lists: List[List[TemplateVariable]],
        strategy: str = "merge",
        conflict_resolution: str = "latest"
    ) -> List[TemplateVariable]:
        """
        Merge multiple lists of template variables.
        
        Args:
            variables_lists: Lists of variables to merge
            strategy: How to merge variables
            conflict_resolution: How to resolve conflicts
        
        Returns:
            Merged list of template variables
        """
        if not variables_lists:
            return []
        
        # Create a map of variables by name for conflict resolution
        variable_map = {}
        
        for variables in variables_lists:
            for var in variables:
                if var.name in variable_map:
                    # Handle conflict
                    existing_var = variable_map[var.name]
                    resolved_var = self._resolve_variable_conflict(
                        existing_var, var, conflict_resolution
                    )
                    variable_map[var.name] = resolved_var
                else:
                    variable_map[var.name] = deepcopy(var)
        
        return list(variable_map.values())
    
    def _merge_sources_intelligent(
        self,
        sources_lists: List[List[TemplateSource]],
        conflict_resolution: str
    ) -> List[TemplateSource]:
        """Intelligent source merging with conflict resolution."""
        merged_sources = []
        
        # Create a comprehensive list of all sources
        all_sources = []
        for i, sources in enumerate(sources_lists):
            for source in sources:
                source_copy = deepcopy(source)
                source_copy._template_index = i  # Track which template this came from
                all_sources.append(source_copy)
        
        # Group sources by type for intelligent merging
        sources_by_type = {}
        for source in all_sources:
            if source.type not in sources_by_type:
                sources_by_type[source.type] = []
            sources_by_type[source.type].append(source)
        
        # Merge each type group
        for source_type, type_sources in sources_by_type.items():
            merged_type_sources = self._merge_sources_by_type(
                type_sources, source_type, conflict_resolution
            )
            merged_sources.extend(merged_type_sources)
        
        # Sort by priority (descending)
        merged_sources.sort(key=lambda s: s.priority, reverse=True)
        
        return merged_sources
    
    def _merge_sources_by_type(
        self,
        sources: List[TemplateSource],
        source_type: str,
        conflict_resolution: str
    ) -> List[TemplateSource]:
        """Merge sources of the same type."""
        if len(sources) <= 1:
            return sources
        
        if source_type == "query":
            # Query sources: typically merge content
            return [self._merge_query_sources(sources, conflict_resolution)]
        
        elif source_type in ["reference", "context", "memory"]:
            # These can often be combined
            return self._merge_reference_sources(sources, conflict_resolution)
        
        else:
            # Default: apply conflict resolution
            if conflict_resolution == "latest":
                return [sources[-1]]  # Take the last one
            elif conflict_resolution == "priority":
                return [max(sources, key=lambda s: s.priority)]
            elif conflict_resolution == "error":
                raise ValueError(f"Conflicting sources of type {source_type}")
            else:
                return sources  # Keep all
    
    def _merge_query_sources(
        self,
        sources: List[TemplateSource],
        conflict_resolution: str
    ) -> TemplateSource:
        """Merge multiple query sources into one."""
        if conflict_resolution == "latest":
            return sources[-1]
        
        # Merge content
        merged_content = []
        highest_priority = 0
        
        for source in sources:
            merged_content.append(source.content_template)
            highest_priority = max(highest_priority, source.priority)
        
        # Create merged source
        merged_source = deepcopy(sources[0])
        merged_source.content_template = " ".join(merged_content)
        merged_source.priority = highest_priority
        
        return merged_source
    
    def _merge_reference_sources(
        self,
        sources: List[TemplateSource],
        conflict_resolution: str
    ) -> List[TemplateSource]:
        """Merge reference-type sources."""
        if conflict_resolution == "latest":
            return [sources[-1]]
        elif conflict_resolution == "priority":
            return [max(sources, key=lambda s: s.priority)]
        else:
            # Keep all reference sources
            return sources
    
    def _resolve_variable_conflict(
        self,
        existing: TemplateVariable,
        new: TemplateVariable,
        resolution: str
    ) -> TemplateVariable:
        """Resolve conflict between two variables with the same name."""
        if resolution == "latest":
            return new
        elif resolution == "priority":
            # Prefer required variables, then more specific types
            if new.required and not existing.required:
                return new
            elif existing.required and not new.required:
                return existing
            else:
                return new  # Default to latest
        elif resolution == "merge":
            # Merge variable properties
            merged = deepcopy(existing)
            
            # Take the more restrictive type if different
            if new.type != existing.type:
                if new.type == VariableType.ENUM:
                    merged.type = new.type
                    merged.allowed_values = new.allowed_values
            
            # Merge constraints (take the more restrictive)
            if new.min_length is not None:
                if merged.min_length is None or new.min_length > merged.min_length:
                    merged.min_length = new.min_length
            
            if new.max_length is not None:
                if merged.max_length is None or new.max_length < merged.max_length:
                    merged.max_length = new.max_length
            
            # Update description if new one is more detailed
            if len(new.description) > len(merged.description):
                merged.description = new.description
            
            # Merge allowed values
            if new.allowed_values and merged.allowed_values:
                merged.allowed_values = list(set(merged.allowed_values + new.allowed_values))
            elif new.allowed_values:
                merged.allowed_values = new.allowed_values
            
            return merged
        elif resolution == "error":
            raise ValueError(f"Variable conflict: {existing.name}")
        
        return new
    
    def merge_inherited_properties(self, parent: ContextTemplate, child: ContextTemplate) -> ContextTemplate:
        """
        Merge inherited properties from parent template into child template.
        
        Args:
            parent: Parent template to inherit from
            child: Child template that inherits
        
        Returns:
            New template with merged properties
        """
        if not child.inheritance:
            return child
        
        inheritance = child.inheritance
        
        # Start with child template as base
        merged_dict = child.dict()
        
        # Merge variables if enabled
        if inheritance.inherit_variables:
            merged_variables = []
            child_var_names = {v.name for v in child.variables}
            
            # Add inherited variables that don't conflict
            for parent_var in parent.variables:
                if parent_var.name not in child_var_names:
                    # Apply variable overrides if defined
                    if parent_var.name in inheritance.variable_overrides:
                        override_data = inheritance.variable_overrides[parent_var.name]
                        var_dict = parent_var.dict()
                        if isinstance(override_data, dict):
                            var_dict.update(override_data)
                        else:
                            # Handle simple value overrides
                            var_dict["default_value"] = override_data
                        merged_variables.append(TemplateVariable(**var_dict))
                    else:
                        merged_variables.append(parent_var)
            
            # Add child variables (which override parent)
            merged_variables.extend(child.variables)
            merged_dict["variables"] = merged_variables
        
        # Merge sources if enabled
        if inheritance.inherit_sources:
            merged_sources = []
            
            # Add inherited sources with adjusted priority
            for parent_source in parent.sources:
                source_dict = parent_source.dict()
                # Apply source overrides if defined
                if parent_source.type in inheritance.source_overrides:
                    source_dict.update(inheritance.source_overrides[parent_source.type])
                else:
                    # Default: reduce priority of inherited sources
                    source_dict["priority"] = max(0, source_dict.get("priority", 50) - 10)
                
                merged_sources.append(TemplateSource(**source_dict))
            
            # Add child sources
            merged_sources.extend(child.sources)
            
            # Sort by priority
            merged_sources.sort(key=lambda s: s.priority, reverse=True)
            merged_dict["sources"] = merged_sources
        
        # Merge validation if enabled  
        if inheritance.inherit_validation and parent.validation:
            if not child.validation:
                merged_dict["validation"] = parent.validation.dict()
            else:
                # Merge validation rules
                merged_validation = child.validation.dict()
                
                # Merge constraints
                parent_constraints = parent.validation.constraints or []
                child_constraints = child.validation.constraints or []
                
                # Apply validation overrides
                if inheritance.validation_overrides:
                    merged_validation.update(inheritance.validation_overrides)
                
                # Combine constraints, child overrides parent
                constraint_names = {c.name for c in child_constraints}
                merged_constraints = child_constraints[:]
                
                for parent_constraint in parent_constraints:
                    if parent_constraint.name not in constraint_names:
                        merged_constraints.append(parent_constraint)
                
                merged_validation["constraints"] = merged_constraints
                merged_dict["validation"] = merged_validation
        
        # Update inheritance chain
        if not merged_dict.get("inheritance"):
            merged_dict["inheritance"] = {}
        
        merged_dict["inheritance"]["inheritance_chain"] = [parent.template_id] + inheritance.inheritance_chain
        
        return ContextTemplate(**merged_dict)


class InheritanceEngine:
    """
    Core template inheritance logic for resolving inherited properties.
    """
    
    def __init__(self, template_manager=None):
        self.template_manager = template_manager
        self.composition_engine = CompositionEngine()
    
    async def resolve_template_inheritance(self, template: ContextTemplate) -> ContextTemplate:
        """
        Resolve a template with all inherited properties from its inheritance chain.
        
        Args:
            template: Template to resolve inheritance for
        
        Returns:
            Template with all inherited properties resolved
        """
        if not template.has_inheritance:
            return template
        
        # Get inheritance chain
        inheritance_chain = await self._get_inheritance_chain(template)
        
        # Resolve inheritance from root to current
        resolved_template = template
        
        for i in range(len(inheritance_chain) - 1):
            parent_id = inheritance_chain[i]
            child_id = inheritance_chain[i + 1]
            
            parent_template = await self._get_template(parent_id)
            child_template = await self._get_template(child_id) if i < len(inheritance_chain) - 2 else resolved_template
            
            if parent_template and child_template:
                resolved_template = self.composition_engine.merge_inherited_properties(parent_template, child_template)
        
        return resolved_template
    
    async def _get_inheritance_chain(self, template: ContextTemplate) -> List[str]:
        """Get complete inheritance chain from root to current template."""
        if not template.inheritance or not template.inheritance.parent_template_id:
            return [template.template_id]
        
        chain = []
        current_template = template
        visited = set()
        
        while current_template and current_template.template_id not in visited:
            visited.add(current_template.template_id)
            chain.append(current_template.template_id)
            
            if current_template.inheritance and current_template.inheritance.parent_template_id:
                parent_id = current_template.inheritance.parent_template_id
                current_template = await self._get_template(parent_id)
            else:
                break
        
        return list(reversed(chain))  # Root to current order
    
    async def _get_template(self, template_id: str) -> Optional[ContextTemplate]:
        """Get template by ID."""
        if self.template_manager:
            return await self.template_manager.get_template(template_id)
        return None
    
    async def create_child_template(self, parent_template_id: str, child_template_id: str,
                                  child_metadata: TemplateMetadata,
                                  inheritance_config: Optional[Dict[str, Any]] = None) -> ContextTemplate:
        """
        Create a new child template that inherits from a parent template.
        
        Args:
            parent_template_id: ID of parent template
            child_template_id: ID for new child template
            child_metadata: Metadata for child template
            inheritance_config: Inheritance configuration options
        
        Returns:
            New child template
        """
        parent_template = await self._get_template(parent_template_id)
        if not parent_template:
            raise ValueError(f"Parent template '{parent_template_id}' not found")
        
        # Check if parent is sealed
        if parent_template.is_sealed:
            raise ValueError(f"Cannot inherit from sealed template '{parent_template_id}'")
        
        # Default inheritance configuration
        default_inheritance = {
            "parent_template_id": parent_template_id,
            "inherit_variables": True,
            "inherit_sources": True,
            "inherit_validation": True,
            "inherit_metadata": False,
            "allow_override": True,
            "sealed": False,
            "abstract": False
        }
        
        if inheritance_config:
            default_inheritance.update(inheritance_config)
        
        # Build inheritance chain
        parent_chain = parent_template.inheritance.inheritance_chain if parent_template.inheritance else []
        inheritance_chain = parent_chain + [parent_template_id]
        default_inheritance["inheritance_chain"] = inheritance_chain
        
        # Create child template
        child_template = ContextTemplate(
            template_id=child_template_id,
            type=parent_template.type,
            status=TemplateStatus.DRAFT,
            metadata=child_metadata,
            variables=[],
            sources=[],
            inheritance=TemplateInheritance(**default_inheritance)
        )
        
        return child_template


class TemplateComposer:
    """
    Main template composition orchestrator that handles template inheritance,
    composition, and advanced template building patterns.
    """
    
    def __init__(self):
        self.composition_engine = CompositionEngine()
        self.inheritance_engine = InheritanceEngine()
        self._template_manager = None
        self._initialized = False
    
    async def initialize(self, template_manager) -> bool:
        """Initialize the template composer."""
        if self._initialized:
            return True
        
        try:
            logger.info("Initializing TemplateComposer...")
            self._template_manager = template_manager
            self.inheritance_engine.template_manager = template_manager
            self._initialized = True
            logger.info("TemplateComposer initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize TemplateComposer: {e}")
            return False
    
    async def compose_template(
        self,
        template: ContextTemplate,
        variables: Dict[str, Any],
        context: TemplateExecutionContext
    ) -> ContextTemplate:
        """
        Compose a template with its base templates.
        
        This is called during template execution when a template uses composition.
        
        Args:
            template: Template with composition configuration
            variables: Variable values for composition
            context: Execution context
        
        Returns:
            Fully composed template ready for rendering
        """
        if not template.composition or not template.composition.base_templates:
            return template
        
        logger.debug(f"Composing template {template.template_id} with bases: {template.composition.base_templates}")
        
        # Load base templates
        base_templates = []
        for base_id in template.composition.base_templates:
            base_template = await self._template_manager.get_template(base_id)
            if not base_template:
                logger.warning(f"Base template not found: {base_id}")
                continue
            
            # Recursively compose if base template also has composition
            if base_template.has_composition:
                base_template = await self.compose_template(base_template, variables, context)
            
            base_templates.append(base_template)
        
        if not base_templates:
            logger.warning("No valid base templates found for composition")
            return template
        
        # Compose templates
        composed = await self._compose_templates_internal(
            base_templates + [template],
            variables,
            template.composition.merge_strategy,
            template.composition.conflict_resolution
        )
        
        # Update composition tracking
        context.execution_path.extend(template.composition.base_templates)
        
        logger.debug(f"Composed template {template.template_id} with {len(base_templates)} base templates")
        return composed
    
    async def compose_templates(
        self,
        base_template_ids: List[str],
        variables: Dict[str, Any],
        merge_strategy: str = "append",
        conflict_resolution: str = "latest"
    ) -> ContextTemplate:
        """
        Compose multiple templates into a new template.
        
        This is for creating new composed templates programmatically.
        
        Args:
            base_template_ids: Template IDs to compose
            variables: Variable values for composition
            merge_strategy: How to merge sources
            conflict_resolution: How to resolve conflicts
        
        Returns:
            New composed template
        """
        logger.debug(f"Composing templates: {base_template_ids}")
        
        # Load templates
        templates = []
        for template_id in base_template_ids:
            template = await self._template_manager.get_template(template_id)
            if template:
                templates.append(template)
            else:
                logger.warning(f"Template not found: {template_id}")
        
        if not templates:
            raise ValueError("No valid templates found for composition")
        
        return await self._compose_templates_internal(
            templates, variables, merge_strategy, conflict_resolution
        )
    
    async def _compose_templates_internal(
        self,
        templates: List[ContextTemplate],
        variables: Dict[str, Any],
        merge_strategy: str,
        conflict_resolution: str
    ) -> ContextTemplate:
        """Internal template composition logic."""
        
        # Collect all sources and variables
        all_sources = [t.sources for t in templates]
        all_variables = [t.variables for t in templates]
        
        # Merge sources
        merged_sources = self.composition_engine.merge_sources(
            all_sources, merge_strategy, conflict_resolution
        )
        
        # Merge variables
        merged_variables = self.composition_engine.merge_variables(
            all_variables, "merge", conflict_resolution
        )
        
        # Apply variable overrides from composition configuration
        final_template = templates[-1]  # Use the last template as the base
        if final_template.composition and final_template.composition.override_variables:
            merged_variables = self._apply_variable_overrides(
                merged_variables, final_template.composition.override_variables
            )
        
        # Create composed template
        composed_metadata = self._merge_metadata([t.metadata for t in templates])
        
        composed_template = ContextTemplate(
            template_id=f"composed_{final_template.template_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            type=final_template.type,
            status=TemplateStatus.ACTIVE,
            metadata=composed_metadata,
            variables=merged_variables,
            sources=merged_sources,
            composition=None,  # Remove composition from final template
            validation=final_template.validation,
            created_by="template_composer"
        )
        
        return composed_template
    
    async def resolve_template_with_inheritance(self, template: ContextTemplate) -> ContextTemplate:
        """Resolve template with full inheritance chain."""
        return await self.inheritance_engine.resolve_template_inheritance(template)
    
    async def create_inherited_template(self, parent_template_id: str, child_template_id: str,
                                      child_metadata: TemplateMetadata,
                                      inheritance_options: Optional[Dict[str, Any]] = None) -> ContextTemplate:
        """Create a new template that inherits from a parent."""
        return await self.inheritance_engine.create_child_template(
            parent_template_id, child_template_id, child_metadata, inheritance_options
        )
    
    async def create_template_from_pattern(self, pattern_name: str, template_id: str,
                                         customizations: Optional[Dict[str, Any]] = None) -> ContextTemplate:
        """Create template from predefined pattern."""
        patterns = await self._get_predefined_patterns()
        
        if pattern_name not in patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        
        pattern = patterns[pattern_name]
        
        # Apply customizations
        if customizations:
            pattern = self._apply_pattern_customizations(pattern, customizations)
        
        # Create template from pattern
        template_dict = pattern.copy()
        template_dict["template_id"] = template_id
        template_dict["created_at"] = datetime.utcnow()
        template_dict["updated_at"] = datetime.utcnow()
        
        return ContextTemplate(**template_dict)
    
    async def _get_predefined_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Get predefined template patterns."""
        return {
            "code_review": {
                "type": "context",
                "status": "active",
                "metadata": {
                    "title": "Code Review Template",
                    "description": "Comprehensive code review with security and performance analysis",
                    "tags": ["code", "review", "analysis"],
                    "category": "development"
                },
                "variables": [
                    {
                        "name": "code",
                        "type": "string",
                        "description": "Code to review",
                        "required": True
                    },
                    {
                        "name": "language",
                        "type": "enum",
                        "description": "Programming language",
                        "allowed_values": ["python", "javascript", "java", "cpp", "go"],
                        "default_value": "python"
                    },
                    {
                        "name": "review_type",
                        "type": "enum",
                        "description": "Type of review",
                        "allowed_values": ["basic", "security", "performance", "comprehensive"],
                        "default_value": "comprehensive"
                    }
                ],
                "sources": [
                    {
                        "type": "query",
                        "content_template": "Please review this {{language}} code for {{review_type}} issues:\\n\\n```{{language}}\\n{{code}}\\n```",
                        "priority": 100
                    }
                ],
                "validation": {
                    "max_tokens": 8000,
                    "min_sources": 1,
                    "constraints": [
                        {
                            "name": "code",
                            "type": "variable_pattern",
                            "value": ".+",
                            "operator": "matches",
                            "message": "Code cannot be empty"
                        }
                    ]
                }
            },
            "documentation": {
                "type": "context",
                "status": "active", 
                "metadata": {
                    "title": "Documentation Writer Template",
                    "description": "Generate comprehensive documentation",
                    "tags": ["documentation", "writing"],
                    "category": "documentation"
                },
                "variables": [
                    {
                        "name": "subject",
                        "type": "string",
                        "description": "Subject to document",
                        "required": True
                    },
                    {
                        "name": "doc_type",
                        "type": "enum", 
                        "description": "Type of documentation",
                        "allowed_values": ["api", "user_guide", "technical", "tutorial"],
                        "default_value": "technical"
                    },
                    {
                        "name": "audience",
                        "type": "enum",
                        "description": "Target audience",
                        "allowed_values": ["developers", "end_users", "administrators"],
                        "default_value": "developers"
                    }
                ],
                "sources": [
                    {
                        "type": "query",
                        "content_template": "Create {{doc_type}} documentation for {{subject}} targeting {{audience}}. Include examples and best practices.",
                        "priority": 100
                    }
                ]
            },
            "bug_analysis": {
                "type": "context",
                "status": "active",
                "metadata": {
                    "title": "Bug Analysis Template",
                    "description": "Systematic bug analysis and troubleshooting",
                    "tags": ["bug", "debugging", "analysis"],
                    "category": "troubleshooting"
                },
                "variables": [
                    {
                        "name": "error_description",
                        "type": "string",
                        "description": "Description of the error or bug",
                        "required": True
                    },
                    {
                        "name": "code_snippet",
                        "type": "string",
                        "description": "Relevant code snippet",
                        "required": False
                    },
                    {
                        "name": "environment",
                        "type": "string",
                        "description": "Environment details",
                        "required": False
                    },
                    {
                        "name": "steps_to_reproduce",
                        "type": "string",
                        "description": "Steps to reproduce the issue", 
                        "required": False
                    }
                ],
                "sources": [
                    {
                        "type": "query",
                        "content_template": "Analyze this bug: {{error_description}}\\n\\nCode: {{code_snippet}}\\nEnvironment: {{environment}}\\nSteps: {{steps_to_reproduce}}\\n\\nProvide root cause analysis and solutions.",
                        "priority": 100
                    }
                ]
            }
        }
    
    def _apply_pattern_customizations(self, pattern: Dict[str, Any], 
                                    customizations: Dict[str, Any]) -> Dict[str, Any]:
        """Apply customizations to pattern."""
        customized_pattern = pattern.copy()
        
        # Apply deep merge of customizations
        for key, value in customizations.items():
            if key in customized_pattern and isinstance(customized_pattern[key], dict) and isinstance(value, dict):
                customized_pattern[key].update(value)
            else:
                customized_pattern[key] = value
        
        return customized_pattern
    
    def _apply_variable_overrides(
        self,
        variables: List[TemplateVariable],
        overrides: Dict[str, Any]
    ) -> List[TemplateVariable]:
        """Apply variable overrides from composition configuration."""
        
        variable_map = {var.name: var for var in variables}
        
        for name, override_value in overrides.items():
            if name in variable_map:
                var = variable_map[name]
                if isinstance(override_value, dict):
                    # Override specific properties
                    for prop, value in override_value.items():
                        if hasattr(var, prop):
                            setattr(var, prop, value)
                else:
                    # Override default value
                    var.default_value = override_value
        
        return list(variable_map.values())
    
    def _merge_metadata(self, metadata_list: List[TemplateMetadata]) -> TemplateMetadata:
        """Merge metadata from multiple templates."""
        if not metadata_list:
            return TemplateMetadata(title="Composed Template", description="")
        
        base_metadata = metadata_list[-1]  # Start with the last template's metadata
        
        # Merge tags
        all_tags = set()
        for metadata in metadata_list:
            all_tags.update(metadata.tags)
        
        # Create merged metadata
        merged = TemplateMetadata(
            title=f"Composed: {base_metadata.title}",
            description=base_metadata.description + " (Composed template)",
            author=base_metadata.author,
            tags=list(all_tags),
            category=base_metadata.category,
            version="1.0.0",  # Reset version for composed template
            is_public=False,  # Composed templates are private by default
            license=base_metadata.license
        )
        
        return merged
    
    async def create_template_from_pattern(
        self,
        pattern_name: str,
        template_id: str,
        customizations: Dict[str, Any] = None
    ) -> ContextTemplate:
        """
        Create a template from a predefined pattern.
        
        Patterns are pre-built template compositions for common use cases.
        
        Args:
            pattern_name: Name of the pattern to use
            template_id: ID for the new template
            customizations: Custom values to apply
        
        Returns:
            New template based on the pattern
        """
        patterns = {
            "code_review": {
                "base_templates": ["query.code_review", "context.git_diff", "reference.best_practices"],
                "merge_strategy": "merge",
                "variables": {
                    "review_type": {"type": "enum", "allowed_values": ["full", "quick", "security"]},
                    "language": {"type": "string", "required": True},
                    "focus_areas": {"type": "list", "default_value": ["bugs", "performance", "style"]}
                }
            },
            "documentation": {
                "base_templates": ["query.documentation", "context.codebase", "reference.doc_standards"],
                "merge_strategy": "append",
                "variables": {
                    "doc_type": {"type": "enum", "allowed_values": ["api", "user", "technical"]},
                    "format": {"type": "enum", "allowed_values": ["markdown", "rst", "html"]},
                    "include_examples": {"type": "boolean", "default_value": True}
                }
            },
            "bug_analysis": {
                "base_templates": ["query.bug_analysis", "context.error_logs", "context.code_context"],
                "merge_strategy": "merge",
                "variables": {
                    "severity": {"type": "enum", "allowed_values": ["low", "medium", "high", "critical"]},
                    "component": {"type": "string", "required": True},
                    "reproduction_steps": {"type": "string"}
                }
            }
        }
        
        if pattern_name not in patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}. Available: {list(patterns.keys())}")
        
        pattern = patterns[pattern_name]
        
        # Create metadata
        metadata = TemplateMetadata(
            title=f"{pattern_name.replace('_', ' ').title()} Template",
            description=f"Template created from {pattern_name} pattern",
            category=pattern_name,
            tags=[pattern_name, "pattern", "generated"]
        )
        
        # Create variables
        variables = []
        for var_name, var_config in pattern["variables"].items():
            var = TemplateVariable(
                name=var_name,
                type=VariableType(var_config["type"]),
                description=f"{var_name} for {pattern_name}",
                **{k: v for k, v in var_config.items() if k != "type"}
            )
            variables.append(var)
        
        # Apply customizations
        if customizations:
            for var in variables:
                if var.name in customizations:
                    var.default_value = customizations[var.name]
        
        # Create composition
        composition = TemplateComposition(
            base_templates=pattern["base_templates"],
            merge_strategy=pattern["merge_strategy"],
            override_variables=customizations or {}
        )
        
        # Create template
        template = ContextTemplate(
            template_id=template_id,
            type=TemplateType.CONTEXT,
            status=TemplateStatus.DRAFT,
            metadata=metadata,
            variables=variables,
            sources=[],  # Sources will come from composition
            composition=composition
        )
        
        logger.info(f"Created template from pattern {pattern_name}: {template_id}")
        return template
    
    def analyze_composition_complexity(
        self,
        template: ContextTemplate
    ) -> Dict[str, Any]:
        """
        Analyze the complexity of template composition.
        
        Returns:
            Dictionary with composition analysis
        """
        if not template.composition:
            return {"has_composition": False}
        
        analysis = {
            "has_composition": True,
            "base_template_count": len(template.composition.base_templates),
            "merge_strategy": template.composition.merge_strategy,
            "conflict_resolution": template.composition.conflict_resolution,
            "has_overrides": bool(template.composition.override_variables),
            "override_count": len(template.composition.override_variables),
            "complexity_score": 0
        }
        
        # Calculate complexity score
        score = 0
        score += analysis["base_template_count"] * 10  # Each base template adds complexity
        score += len(template.composition.override_variables) * 5  # Overrides add complexity
        
        if template.composition.merge_strategy == "merge":
            score += 15  # Merge strategy is more complex
        
        analysis["complexity_score"] = min(score, 100)
        
        return analysis
    
    async def validate_composition(
        self,
        template: ContextTemplate
    ) -> List[str]:
        """
        Validate template composition configuration.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        if not template.composition:
            return errors
        
        # Check base templates exist
        for base_id in template.composition.base_templates:
            base_template = await self._template_manager.get_template(base_id)
            if not base_template:
                errors.append(f"Base template not found: {base_id}")
            elif base_template.status == TemplateStatus.ARCHIVED:
                errors.append(f"Base template is archived: {base_id}")
        
        # Check for circular dependencies
        circular_deps = await self._check_circular_dependencies(
            template.template_id, template.composition.base_templates
        )
        if circular_deps:
            errors.append(f"Circular dependency detected: {' -> '.join(circular_deps)}")
        
        # Validate override variables
        for var_name, override in template.composition.override_variables.items():
            if not any(var.name == var_name for var in template.variables):
                errors.append(f"Override variable not defined: {var_name}")
        
        return errors
    
    async def _check_circular_dependencies(
        self,
        template_id: str,
        base_template_ids: List[str],
        visited: Set[str] = None
    ) -> Optional[List[str]]:
        """Check for circular dependencies in template composition."""
        if visited is None:
            visited = set()
        
        if template_id in visited:
            return [template_id]  # Circular dependency found
        
        visited.add(template_id)
        
        for base_id in base_template_ids:
            base_template = await self._template_manager.get_template(base_id)
            if base_template and base_template.composition:
                circular = await self._check_circular_dependencies(
                    base_id, base_template.composition.base_templates, visited.copy()
                )
                if circular:
                    return [template_id] + circular
        
        return None