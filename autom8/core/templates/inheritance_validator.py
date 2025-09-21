"""
Template Inheritance Validation Engine

This module provides comprehensive template inheritance validation capabilities,
including inheritance chain validation, constraint resolution, and advanced
template composition validation.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Callable

from autom8.models.template import (
    ContextTemplate,
    TemplateVariable,
    TemplateSource,
    TemplateValidation,
    TemplateConstraint,
    TemplateInheritance,
    VariableType,
)
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class TemplateInheritanceRule:
    """Validate template inheritance structure and constraints."""
    
    def __init__(self):
        self.name = "template_inheritance"
        self.description = "Validate template inheritance structure and rules"
        self.severity = "error"
    
    async def validate(self, template: ContextTemplate) -> List[Dict[str, Any]]:
        issues = []
        
        if not template.inheritance:
            return issues
        
        inheritance = template.inheritance
        
        # Check for circular inheritance
        if template.template_id in inheritance.inheritance_chain:
            issues.append({
                "rule": self.name,
                "severity": "error",
                "message": f"Circular inheritance detected in chain: {' -> '.join(inheritance.inheritance_chain)}",
                "field": "inheritance.inheritance_chain"
            })
        
        # Check inheritance depth limits
        if inheritance.inheritance_depth > 5:
            issues.append({
                "rule": self.name,
                "severity": "warning",
                "message": f"Deep inheritance chain ({inheritance.inheritance_depth} levels) may impact performance",
                "field": "inheritance.inheritance_chain"
            })
        
        # Abstract templates must have inheritance configuration
        if inheritance.abstract and not inheritance.allow_override:
            issues.append({
                "rule": self.name,
                "severity": "error",
                "message": "Abstract templates must allow override by child templates",
                "field": "inheritance.allow_override"
            })
        
        # Sealed templates cannot be abstract
        if inheritance.sealed and inheritance.abstract:
            issues.append({
                "rule": self.name,
                "severity": "error",
                "message": "Templates cannot be both sealed and abstract",
                "field": "inheritance"
            })
        
        return issues


class CustomConstraintRule:
    """Validate custom template constraints."""
    
    def __init__(self):
        self.name = "custom_constraints"
        self.description = "Validate custom template constraints and rules"
        self.severity = "error"
    
    async def validate(self, template: ContextTemplate) -> List[Dict[str, Any]]:
        issues = []
        
        if not template.validation or not template.validation.constraints:
            return issues
        
        for constraint in template.validation.constraints:
            # Validate constraint definition
            constraint_issues = await self._validate_constraint(constraint, template)
            issues.extend(constraint_issues)
        
        return issues
    
    async def _validate_constraint(self, constraint: TemplateConstraint, template: ContextTemplate) -> List[Dict[str, Any]]:
        """Validate individual constraint definition"""
        issues = []
        
        # Check constraint references valid variables
        if constraint.type in ["variable_required", "variable_pattern", "value_range"]:
            if constraint.name not in [v.name for v in template.variables]:
                issues.append({
                    "rule": self.name,
                    "severity": "error",
                    "message": f"Constraint references unknown variable '{constraint.name}'",
                    "field": f"validation.constraints.{constraint.name}"
                })
        
        # Validate constraint operators
        valid_operators = ["==", "!=", "<", "<=", ">", ">=", "in", "not_in", "matches", "not_matches"]
        if constraint.operator not in valid_operators:
            issues.append({
                "rule": self.name,
                "severity": "error",
                "message": f"Invalid constraint operator '{constraint.operator}'. Must be one of: {valid_operators}",
                "field": f"validation.constraints.{constraint.name}.operator"
            })
        
        # Validate constraint dependencies
        if constraint.depends_on:
            variable_names = [v.name for v in template.variables]
            for dep in constraint.depends_on:
                if dep not in variable_names:
                    issues.append({
                        "rule": self.name,
                        "severity": "error",
                        "message": f"Constraint dependency '{dep}' references unknown variable",
                        "field": f"validation.constraints.{constraint.name}.depends_on"
                    })
        
        return issues


class InheritanceAwareValidator:
    """Validator that understands template inheritance."""
    
    def __init__(self):
        self.template_manager = None  # Will be set by TemplateValidator
    
    async def validate_with_inheritance(self, template: ContextTemplate) -> Dict[str, Any]:
        """Validate template including inherited properties."""
        issues = []
        
        # Collect inheritance chain
        inheritance_chain = await self._get_inheritance_chain(template)
        
        # Validate each template in chain
        for template_id in inheritance_chain:
            chain_template = await self._get_template(template_id)
            if chain_template:
                # Validate individual template
                chain_issues = await self._validate_single_template(chain_template)
                issues.extend(chain_issues)
        
        # Validate inheritance consistency
        consistency_issues = await self._validate_inheritance_consistency(template, inheritance_chain)
        issues.extend(consistency_issues)
        
        return {
            "inheritance_chain": inheritance_chain,
            "issues": issues,
            "is_valid": not any(issue["severity"] == "error" for issue in issues)
        }
    
    async def _get_inheritance_chain(self, template: ContextTemplate) -> List[str]:
        """Get full inheritance chain for template."""
        if not template.inheritance or not template.inheritance.parent_template_id:
            return [template.template_id]
        
        chain = [template.template_id]
        current_id = template.inheritance.parent_template_id
        visited = set()
        
        while current_id and current_id not in visited:
            visited.add(current_id)
            chain.append(current_id)
            
            parent = await self._get_template(current_id)
            if parent and parent.inheritance:
                current_id = parent.inheritance.parent_template_id
            else:
                break
        
        return list(reversed(chain))  # Root to leaf order
    
    async def _get_template(self, template_id: str) -> Optional[ContextTemplate]:
        """Get template by ID through manager."""
        if self.template_manager:
            return await self.template_manager.get_template(template_id)
        return None
    
    async def _validate_single_template(self, template: ContextTemplate) -> List[Dict[str, Any]]:
        """Validate individual template without inheritance context."""
        # This would use the standard validation rules
        return []
    
    async def _validate_inheritance_consistency(self, template: ContextTemplate, chain: List[str]) -> List[Dict[str, Any]]:
        """Validate that inheritance chain is consistent."""
        issues = []
        
        # Check for sealed templates in chain
        for i, template_id in enumerate(chain[:-1]):  # Exclude leaf
            chain_template = await self._get_template(template_id)
            if chain_template and chain_template.is_sealed:
                issues.append({
                    "rule": "inheritance_consistency",
                    "severity": "error",
                    "message": f"Cannot inherit from sealed template '{template_id}'",
                    "field": "inheritance.parent_template_id"
                })
        
        return issues


class InheritanceResolver:
    """Resolves inherited properties from template inheritance chain."""
    
    def __init__(self, template_manager):
        self.template_manager = template_manager
    
    async def resolve_inherited_template(self, template: ContextTemplate) -> ContextTemplate:
        """Resolve a template with all inherited properties."""
        
        if not template.has_inheritance:
            return template
        
        # Get inheritance chain
        chain = await self._get_inheritance_chain(template)
        
        # Resolve from root to leaf
        resolved_template = template
        
        for template_id in chain[:-1]:  # Exclude the current template
            parent_template = await self.template_manager.get_template(template_id)
            if parent_template:
                resolved_template = await self._merge_inherited_properties(
                    parent_template, resolved_template
                )
        
        return resolved_template
    
    async def _get_inheritance_chain(self, template: ContextTemplate) -> List[str]:
        """Get full inheritance chain."""
        if not template.inheritance or not template.inheritance.parent_template_id:
            return [template.template_id]
        
        chain = [template.template_id]
        current_id = template.inheritance.parent_template_id
        visited = set()
        
        while current_id and current_id not in visited:
            visited.add(current_id)
            chain.append(current_id)
            
            parent = await self.template_manager.get_template(current_id)
            if parent and parent.inheritance:
                current_id = parent.inheritance.parent_template_id
            else:
                break
        
        return list(reversed(chain))
    
    async def _merge_inherited_properties(self, parent: ContextTemplate, child: ContextTemplate) -> ContextTemplate:
        """Merge inherited properties from parent into child."""
        
        # Create a copy of the child template
        merged = child.dict()
        
        # Inherit variables if enabled
        if child.inheritance and child.inheritance.inherit_variables:
            inherited_variables = []
            existing_var_names = {v.name for v in child.variables}
            
            # Add parent variables that don't exist in child
            for parent_var in parent.variables:
                if parent_var.name not in existing_var_names:
                    inherited_variables.append(parent_var)
            
            merged["variables"] = inherited_variables + child.variables
        
        # Inherit sources if enabled
        if child.inheritance and child.inheritance.inherit_sources:
            inherited_sources = []
            
            # Add parent sources with lower priority
            for parent_source in parent.sources:
                inherited_source = parent_source.dict()
                inherited_source["priority"] = inherited_source.get("priority", 50) - 10  # Lower priority
                inherited_sources.append(TemplateSource(**inherited_source))
            
            merged["sources"] = inherited_sources + child.sources
        
        # Inherit validation rules if enabled
        if child.inheritance and child.inheritance.inherit_validation and parent.validation:
            if not child.validation:
                merged["validation"] = parent.validation.dict()
            else:
                # Merge validation rules
                merged_validation = child.validation.dict()
                
                # Inherit constraints
                parent_constraints = parent.validation.constraints if parent.validation.constraints else []
                child_constraints = child.validation.constraints if child.validation.constraints else []
                
                merged_validation["constraints"] = parent_constraints + child_constraints
                
                merged["validation"] = merged_validation
        
        return ContextTemplate(**merged)


class ValidationEngine:
    """Enhanced template validation engine with inheritance support."""
    
    def __init__(self):
        self.inheritance_validator = InheritanceAwareValidator()
        self.custom_validators = {}
        self.validation_cache = {}
        self.inheritance_resolver = None
    
    def set_template_manager(self, manager):
        """Set template manager reference."""
        self.inheritance_validator.template_manager = manager
        self.inheritance_resolver = InheritanceResolver(manager)
    
    def register_custom_validator(self, name: str, validator_func: Callable):
        """Register a custom validation function."""
        self.custom_validators[name] = validator_func
    
    async def validate_template_comprehensive(self, template: ContextTemplate, 
                                           include_inheritance: bool = True) -> Dict[str, Any]:
        """Comprehensive template validation including inheritance."""
        
        cache_key = f"{template.template_id}_{hash(str(template.dict()))}"
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        result = {
            "template_id": template.template_id,
            "validation_timestamp": datetime.utcnow().isoformat(),
            "issues": [],
            "summary": {},
            "inheritance_info": None
        }
        
        # Standard validation
        issues = []
        
        # Run standard validation rules
        rules = [
            TemplateInheritanceRule(),
            CustomConstraintRule()
        ]
        
        for rule in rules:
            try:
                rule_issues = await rule.validate(template)
                issues.extend(rule_issues)
            except Exception as e:
                issues.append({
                    "rule": rule.name,
                    "severity": "error",
                    "message": f"Validation rule failed: {e}",
                    "field": "validation_system"
                })
        
        # Inheritance validation
        if include_inheritance and template.has_inheritance:
            try:
                inheritance_result = await self.inheritance_validator.validate_with_inheritance(template)
                result["inheritance_info"] = inheritance_result
                issues.extend(inheritance_result["issues"])
            except Exception as e:
                issues.append({
                    "rule": "inheritance_validation",
                    "severity": "error",
                    "message": f"Inheritance validation failed: {e}",
                    "field": "inheritance"
                })
        
        # Custom constraint validation
        if template.validation and template.validation.custom_rules:
            for rule_name in template.validation.custom_rules:
                if rule_name in self.custom_validators:
                    try:
                        custom_issues = await self.custom_validators[rule_name](template)
                        if custom_issues:
                            issues.extend(custom_issues)
                    except Exception as e:
                        issues.append({
                            "rule": rule_name,
                            "severity": "error",
                            "message": f"Custom validation rule '{rule_name}' failed: {e}",
                            "field": "validation.custom_rules"
                        })
        
        # Generate summary
        result["issues"] = issues
        result["summary"] = self._generate_validation_summary(issues)
        
        # Cache result
        self.validation_cache[cache_key] = result
        return result
    
    def _generate_validation_summary(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate validation summary from issues."""
        errors = [i for i in issues if i.get("severity") == "error"]
        warnings = [i for i in issues if i.get("severity") == "warning"]
        info = [i for i in issues if i.get("severity") == "info"]
        
        return {
            "is_valid": len(errors) == 0,
            "total_issues": len(issues),
            "errors": len(errors),
            "warnings": len(warnings),
            "info": len(info),
            "validation_status": "valid" if len(errors) == 0 else "invalid",
            "quality_score": max(0, 100 - (len(errors) * 25) - (len(warnings) * 5) - (len(info) * 1))
        }