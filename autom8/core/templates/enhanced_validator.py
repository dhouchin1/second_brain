"""
Enhanced Template Validation Engine

This module provides enhanced template validation with inheritance support,
advanced constraint validation, and comprehensive testing capabilities.
"""

import re
import json
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Set, Tuple, Callable
from pathlib import Path

from autom8.models.template import (
    ContextTemplate,
    TemplateVariable,
    TemplateSource,
    TemplateValidation,
    TemplateConstraint,
    TemplateInheritance,
    TemplateExecutionResult,
    TemplateType,
    TemplateStatus,
    VariableType,
)
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class EnhancedTemplateValidator:
    """Enhanced template validation coordinator with inheritance support."""
    
    def __init__(self):
        from .inheritance_validator import ValidationEngine
        self.validation_engine = ValidationEngine()
        self.initialized = False
    
    async def initialize(self):
        """Initialize the validator."""
        if self.initialized:
            return
        
        # Register built-in custom validators
        await self._register_builtin_validators()
        
        self.initialized = True
    
    def set_template_manager(self, manager):
        """Set template manager reference for inheritance validation."""
        self.validation_engine.set_template_manager(manager)
    
    async def _register_builtin_validators(self):
        """Register built-in custom validation functions."""
        
        async def validate_token_efficiency(template: ContextTemplate) -> List[Dict[str, Any]]:
            """Check if template is token-efficient."""
            issues = []
            
            # Check for redundant sources
            source_types = [s.type for s in template.sources]
            if len(source_types) != len(set(source_types)):
                issues.append({
                    "rule": "token_efficiency",
                    "severity": "warning",
                    "message": "Template has duplicate source types which may be inefficient",
                    "field": "sources"
                })
            
            return issues
        
        async def validate_security_patterns(template: ContextTemplate) -> List[Dict[str, Any]]:
            """Check for potential security issues in templates."""
            issues = []
            
            # Check for potentially dangerous patterns
            dangerous_patterns = ["eval", "exec", "__import__", "file://", "../"]
            
            for source in template.sources:
                content = source.content_template.lower()
                for pattern in dangerous_patterns:
                    if pattern in content:
                        issues.append({
                            "rule": "security_patterns",
                            "severity": "warning",
                            "message": f"Template contains potentially dangerous pattern: {pattern}",
                            "field": "sources.content_template"
                        })
            
            return issues
        
        self.validation_engine.register_custom_validator("token_efficiency", validate_token_efficiency)
        self.validation_engine.register_custom_validator("security_patterns", validate_security_patterns)
    
    async def validate_template_detailed(self, template: ContextTemplate, 
                                       include_warnings: bool = False,
                                       include_inheritance: bool = True) -> Dict[str, Any]:
        """Perform detailed template validation."""
        
        if not self.initialized:
            await self.initialize()
        
        result = await self.validation_engine.validate_template_comprehensive(
            template, 
            include_inheritance=include_inheritance
        )
        
        # Filter out warnings if not requested
        if not include_warnings:
            result["issues"] = [issue for issue in result["issues"] if issue.get("severity") != "warning"]
            result["summary"] = self.validation_engine._generate_validation_summary(result["issues"])
        
        return result
    
    async def validate_inheritance_chain(self, template: ContextTemplate) -> Dict[str, Any]:
        """Validate complete inheritance chain for a template."""
        
        if not template.has_inheritance:
            return {
                "has_inheritance": False,
                "chain_length": 0,
                "issues": [],
                "is_valid": True
            }
        
        return await self.validation_engine.inheritance_validator.validate_with_inheritance(template)
    
    async def create_test_suite(self, template: ContextTemplate, 
                              include_edge_cases: bool = False) -> List[Dict[str, Any]]:
        """Create automated test suite for template."""
        test_cases = []
        
        # Basic success case with all required variables
        required_vars = {var.name: self._get_test_value(var) for var in template.variables if var.required}
        test_cases.append({
            "test_id": "basic_success",
            "description": "Basic template execution with required variables",
            "variables": required_vars,
            "expected": {
                "assertions": {
                    "should_succeed": True,
                    "source_count": {"operator": ">=", "value": 1}
                }
            }
        })
        
        # Test with all variables (including optional)
        all_vars = {var.name: self._get_test_value(var) for var in template.variables}
        test_cases.append({
            "test_id": "complete_variables",
            "description": "Template execution with all variables provided",
            "variables": all_vars,
            "expected": {
                "assertions": {
                    "should_succeed": True,
                    "source_count": {"operator": ">=", "value": len(template.sources) if template.sources else 1}
                }
            }
        })
        
        if include_edge_cases:
            # Missing required variables
            test_cases.append({
                "test_id": "missing_required",
                "description": "Template execution with missing required variables",
                "variables": {},
                "expected": {
                    "assertions": {
                        "should_succeed": False,
                        "error_type": "validation_error"
                    }
                }
            })
            
            # Invalid variable types
            if template.variables:
                invalid_vars = required_vars.copy()
                first_var = template.variables[0]
                if first_var.type == VariableType.STRING:
                    invalid_vars[first_var.name] = 12345  # Number instead of string
                elif first_var.type == VariableType.NUMBER:
                    invalid_vars[first_var.name] = "not_a_number"
                
                test_cases.append({
                    "test_id": "invalid_types",
                    "description": "Template execution with invalid variable types",
                    "variables": invalid_vars,
                    "expected": {
                        "assertions": {
                            "should_succeed": False,
                            "error_type": "validation_error"
                        }
                    }
                })
        
        return test_cases
    
    def _get_test_value(self, var: TemplateVariable) -> Any:
        """Generate appropriate test value for variable."""
        if var.default_value is not None:
            return var.default_value
        
        if var.type == VariableType.STRING:
            return "test_string"
        elif var.type == VariableType.NUMBER:
            return 42
        elif var.type == VariableType.BOOLEAN:
            return True
        elif var.type == VariableType.LIST:
            return ["item1", "item2"]
        elif var.type == VariableType.OBJECT:
            return {"key": "value"}
        elif var.type == VariableType.ENUM and var.allowed_values:
            return var.allowed_values[0]
        else:
            return "test_value"
    
    async def run_template_tests(self, template: ContextTemplate, 
                               test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run test cases against template."""
        
        results = {
            "template_id": template.template_id,
            "test_run_timestamp": datetime.utcnow().isoformat(),
            "total_tests": len(test_cases),
            "passed_tests": 0,
            "failed_tests": 0,
            "success_rate": 0.0,
            "test_cases": []
        }
        
        for test_case in test_cases:
            test_result = await self._run_single_test(template, test_case)
            results["test_cases"].append(test_result)
            
            if test_result["passed"]:
                results["passed_tests"] += 1
            else:
                results["failed_tests"] += 1
        
        results["success_rate"] = results["passed_tests"] / results["total_tests"] if results["total_tests"] > 0 else 0.0
        
        return results
    
    async def _run_single_test(self, template: ContextTemplate, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case."""
        
        test_result = {
            "test_id": test_case["test_id"],
            "description": test_case["description"],
            "passed": False,
            "errors": [],
            "execution_time_ms": 0,
            "details": {}
        }
        
        start_time = datetime.utcnow()
        
        try:
            # Validate variables
            validation_errors = template.validate_variable_values(test_case["variables"])
            
            expected = test_case.get("expected", {})
            assertions = expected.get("assertions", {})
            
            # Check if test should succeed
            should_succeed = assertions.get("should_succeed", True)
            
            if should_succeed and validation_errors:
                test_result["errors"].append(f"Expected success but got validation errors: {validation_errors}")
            elif not should_succeed and not validation_errors:
                test_result["errors"].append("Expected validation errors but template passed validation")
            else:
                # Test passed basic validation expectations
                test_result["passed"] = True
            
            test_result["details"]["validation_errors"] = validation_errors
            
        except Exception as e:
            test_result["errors"].append(f"Test execution failed: {e}")
        
        end_time = datetime.utcnow()
        test_result["execution_time_ms"] = int((end_time - start_time).total_seconds() * 1000)
        
        return test_result