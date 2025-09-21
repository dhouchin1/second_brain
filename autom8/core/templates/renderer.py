"""
Template Rendering Engine

This module provides powerful template rendering capabilities with variable substitution,
conditional logic, and template context management.
"""

import re
import json
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Set, Tuple
from jinja2 import Environment, DictLoader, StrictUndefined, TemplateSyntaxError
from jinja2.exceptions import UndefinedError

from autom8.models.template import (
    ContextTemplate,
    TemplateVariable,
    TemplateSource,
    TemplateExecutionContext,
    VariableType,
)
from autom8.models.context import ContextSourceType
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class VariableSubstitutionEngine:
    """
    Core variable substitution engine supporting multiple template formats.
    """
    
    def __init__(self):
        self._jinja_env = Environment(
            loader=DictLoader({}),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        self._jinja_env.filters.update({
            'capitalize_first': self._capitalize_first,
            'truncate_tokens': self._truncate_tokens,
            'format_list': self._format_list,
            'json_format': self._json_format,
            'default_if_empty': self._default_if_empty
        })
    
    def substitute_simple(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Simple variable substitution using {{variable}} syntax.
        Fast for simple templates without complex logic.
        """
        result = template
        
        # Simple regex substitution for basic templates
        for name, value in variables.items():
            pattern = r'\{\{\s*' + re.escape(name) + r'\s*\}\}'
            str_value = self._convert_to_string(value)
            result = re.sub(pattern, str_value, result)
        
        return result
    
    def substitute_advanced(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Advanced variable substitution using Jinja2 for complex templates.
        Supports conditionals, loops, filters, etc.
        """
        try:
            jinja_template = self._jinja_env.from_string(template)
            return jinja_template.render(**variables)
        except (TemplateSyntaxError, UndefinedError) as e:
            logger.error(f"Template rendering error: {e}")
            raise ValueError(f"Template rendering failed: {e}")
    
    def detect_template_variables(self, template: str) -> Set[str]:
        """
        Detect all variable references in a template.
        Supports both simple {{var}} and Jinja2 syntax.
        """
        variables = set()
        
        # Simple variable pattern
        simple_pattern = r'\{\{\s*(\w+)\s*\}\}'
        variables.update(re.findall(simple_pattern, template))
        
        # Jinja2 variable pattern (more complex)
        jinja_pattern = r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*(?:\|[^}]*)?\}\}'
        jinja_matches = re.findall(jinja_pattern, template)
        for match in jinja_matches:
            # Extract base variable name (before dots)
            base_var = match.split('.')[0]
            variables.add(base_var)
        
        return variables
    
    def is_advanced_template(self, template: str) -> bool:
        """
        Determine if template requires advanced Jinja2 rendering.
        """
        # Check for Jinja2-specific syntax
        jinja_patterns = [
            r'\{%.*?%\}',      # Control structures
            r'\{\{.*?\|.*?\}\}', # Filters
            r'\{#.*?#\}',      # Comments
            r'\{\{.*?\..*?\}\}', # Object access
            r'\bif\b.*?\bendif\b',  # If statements
            r'\bfor\b.*?\bendfor\b'  # For loops
        ]
        
        for pattern in jinja_patterns:
            if re.search(pattern, template, re.DOTALL | re.IGNORECASE):
                return True
        
        return False
    
    def _convert_to_string(self, value: Any) -> str:
        """Convert variable value to string for substitution."""
        if value is None:
            return ""
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (list, dict)):
            return json.dumps(value)
        else:
            return str(value)
    
    # Custom Jinja2 filters
    
    def _capitalize_first(self, text: str) -> str:
        """Capitalize first letter of text."""
        return text[0].upper() + text[1:] if text else ""
    
    def _truncate_tokens(self, text: str, max_tokens: int = 100) -> str:
        """Truncate text to approximate token limit."""
        # Rough approximation: 4 characters per token
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            return text[:max_chars] + "..."
        return text
    
    def _format_list(self, items: List[Any], separator: str = ", ") -> str:
        """Format list as string with separator."""
        return separator.join(str(item) for item in items)
    
    def _json_format(self, obj: Any, indent: int = 2) -> str:
        """Format object as JSON."""
        return json.dumps(obj, indent=indent)
    
    def _default_if_empty(self, value: Any, default: Any = "") -> Any:
        """Return default if value is empty."""
        if not value:
            return default
        return value


class TemplateRenderer:
    """
    Main template rendering engine that orchestrates variable substitution,
    conditional source inclusion, and content generation.
    """
    
    def __init__(self):
        self.substitution_engine = VariableSubstitutionEngine()
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the template renderer."""
        if self._initialized:
            return True
        
        try:
            logger.info("Initializing TemplateRenderer...")
            self._initialized = True
            logger.info("TemplateRenderer initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize TemplateRenderer: {e}")
            return False
    
    async def render_template(
        self,
        template: ContextTemplate,
        variables: Dict[str, Any],
        context: TemplateExecutionContext
    ) -> List[Dict[str, Any]]:
        """
        Render a complete template with all sources.
        
        Args:
            template: Template to render
            variables: Variable values for substitution
            context: Execution context
        
        Returns:
            List of rendered context sources
        """
        logger.debug(f"Rendering template: {template.template_id}")
        
        # Apply default values for missing variables
        merged_variables = self._apply_default_values(template, variables)
        
        rendered_sources = []
        
        for source in template.sources:
            # Check conditional inclusion
            if not self._evaluate_source_condition(source, merged_variables):
                logger.debug(f"Skipping source due to condition: {source.condition}")
                continue
            
            try:
                rendered_source = await self._render_source(
                    source, merged_variables, context
                )
                rendered_sources.append(rendered_source)
                
            except Exception as e:
                logger.error(f"Failed to render source: {e}")
                # Add error as a source for debugging
                error_source = {
                    'type': 'error',
                    'content': f"Template rendering error: {e}",
                    'source': f"template:{template.template_id}:error",
                    'priority': 0,
                    'expandable': False
                }
                rendered_sources.append(error_source)
        
        logger.debug(f"Rendered {len(rendered_sources)} sources for template {template.template_id}")
        return rendered_sources
    
    async def render_source(
        self,
        source: TemplateSource,
        variables: Dict[str, Any],
        template_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Render a single template source.
        
        Args:
            source: Source to render
            variables: Variable values
            template_id: Template ID for context
        
        Returns:
            Rendered source dictionary
        """
        context = TemplateExecutionContext(
            template_id=template_id,
            variables=variables,
            execution_id=f"single_{datetime.utcnow().isoformat()}"
        )
        
        return await self._render_source(source, variables, context)
    
    async def _render_source(
        self,
        source: TemplateSource,
        variables: Dict[str, Any],
        context: TemplateExecutionContext
    ) -> Dict[str, Any]:
        """Render a single template source."""
        
        # Render content
        content = await self._render_content(source.content_template, variables)
        
        # Render metadata templates
        source_id = source.source_id_template
        if source_id:
            source_id = await self._render_content(source_id, variables)
        else:
            source_id = f"template:{context.template_id}"
        
        location = source.location_template
        if location:
            location = await self._render_content(location, variables)
        
        summary = source.summary_template
        if summary:
            summary = await self._render_content(summary, variables)
        
        # Create rendered source
        rendered = {
            'type': source.type,
            'content': content,
            'source': source_id,
            'priority': source.priority,
            'expandable': source.expandable
        }
        
        if location:
            rendered['location'] = location
        
        if summary:
            rendered['summary'] = summary
        
        return rendered
    
    async def _render_content(self, template: str, variables: Dict[str, Any]) -> str:
        """Render content template with variable substitution."""
        if not template:
            return ""
        
        # Detect if advanced rendering is needed
        if self.substitution_engine.is_advanced_template(template):
            return self.substitution_engine.substitute_advanced(template, variables)
        else:
            return self.substitution_engine.substitute_simple(template, variables)
    
    def _apply_default_values(
        self,
        template: ContextTemplate,
        variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply default values for missing variables."""
        merged = variables.copy()
        
        for var in template.variables:
            if var.name not in merged and var.default_value is not None:
                merged[var.name] = var.default_value
        
        return merged
    
    def _evaluate_source_condition(
        self,
        source: TemplateSource,
        variables: Dict[str, Any]
    ) -> bool:
        """
        Evaluate source inclusion condition.
        
        Supports simple variable checks:
        - variable_name: Include if variable exists and is truthy
        - !variable_name: Include if variable is falsy or doesn't exist
        - variable_name == "value": Include if variable equals value
        - variable_name != "value": Include if variable doesn't equal value
        """
        if not source.condition:
            return True
        
        condition = source.condition.strip()
        
        try:
            # Simple existence check
            if condition.startswith('!'):
                var_name = condition[1:].strip()
                return not variables.get(var_name)
            
            # Equality check
            if '==' in condition:
                var_name, expected = condition.split('==', 1)
                var_name = var_name.strip()
                expected = expected.strip().strip('"\'')
                return str(variables.get(var_name, '')) == expected
            
            # Inequality check
            if '!=' in condition:
                var_name, expected = condition.split('!=', 1)
                var_name = var_name.strip()
                expected = expected.strip().strip('"\'')
                return str(variables.get(var_name, '')) != expected
            
            # Simple truthiness check
            return bool(variables.get(condition))
            
        except Exception as e:
            logger.warning(f"Failed to evaluate condition '{condition}': {e}")
            return True  # Include by default if condition evaluation fails
    
    async def validate_template_syntax(self, template: ContextTemplate) -> List[str]:
        """
        Validate template syntax without rendering.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate each source template
        for i, source in enumerate(template.sources):
            source_errors = await self._validate_source_syntax(source, i)
            errors.extend(source_errors)
        
        return errors
    
    async def _validate_source_syntax(
        self,
        source: TemplateSource,
        source_index: int
    ) -> List[str]:
        """Validate syntax of a single source template."""
        errors = []
        prefix = f"Source {source_index}"
        
        # Validate content template
        try:
            if self.substitution_engine.is_advanced_template(source.content_template):
                self.substitution_engine._jinja_env.from_string(source.content_template)
        except TemplateSyntaxError as e:
            errors.append(f"{prefix} content template syntax error: {e}")
        
        # Validate metadata templates
        for field_name, template in [
            ("source_id", source.source_id_template),
            ("location", source.location_template),
            ("summary", source.summary_template)
        ]:
            if not template:
                continue
            
            try:
                if self.substitution_engine.is_advanced_template(template):
                    self.substitution_engine._jinja_env.from_string(template)
            except TemplateSyntaxError as e:
                errors.append(f"{prefix} {field_name} template syntax error: {e}")
        
        return errors
    
    def extract_template_variables(self, template: ContextTemplate) -> Set[str]:
        """
        Extract all variable references from template sources.
        
        Returns:
            Set of variable names referenced in the template
        """
        variables = set()
        
        for source in template.sources:
            # Extract from content template
            content_vars = self.substitution_engine.detect_template_variables(
                source.content_template
            )
            variables.update(content_vars)
            
            # Extract from metadata templates
            for template_field in [
                source.source_id_template,
                source.location_template,
                source.summary_template
            ]:
                if template_field:
                    field_vars = self.substitution_engine.detect_template_variables(template_field)
                    variables.update(field_vars)
        
        return variables
    
    def analyze_template_complexity(self, template: ContextTemplate) -> Dict[str, Any]:
        """
        Analyze template complexity and provide insights.
        
        Returns:
            Dictionary with complexity analysis
        """
        analysis = {
            'total_sources': len(template.sources),
            'conditional_sources': 0,
            'advanced_templates': 0,
            'simple_templates': 0,
            'total_variables': len(self.extract_template_variables(template)),
            'defined_variables': len(template.variables),
            'complexity_score': 0.0
        }
        
        for source in template.sources:
            if source.condition:
                analysis['conditional_sources'] += 1
            
            if self.substitution_engine.is_advanced_template(source.content_template):
                analysis['advanced_templates'] += 1
            else:
                analysis['simple_templates'] += 1
        
        # Calculate complexity score (0-100)
        score = 0
        score += analysis['total_sources'] * 5  # Base complexity
        score += analysis['conditional_sources'] * 10  # Conditionals add complexity
        score += analysis['advanced_templates'] * 15  # Advanced templates are complex
        score += analysis['total_variables'] * 3  # More variables = more complex
        
        analysis['complexity_score'] = min(score, 100)
        
        return analysis