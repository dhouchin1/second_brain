"""
Template Validation Engine

This module provides comprehensive template validation capabilities including
syntax validation, constraint checking, quality assessment, and testing frameworks.
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


class ValidationRule:
    """Base class for template validation rules."""
    
    def __init__(self, name: str, description: str, severity: str = "error"):
        self.name = name
        self.description = description
        self.severity = severity  # "error", "warning", "info"
    
    async def validate(self, template: ContextTemplate) -> List[Dict[str, Any]]:
        """
        Validate template against this rule.
        
        Returns:
            List of validation issues
        """
        raise NotImplementedError


class TemplateStructureRule(ValidationRule):
    """Validate basic template structure."""
    
    def __init__(self):
        super().__init__(
            "template_structure",
            "Validate basic template structure and required fields"
        )
    
    async def validate(self, template: ContextTemplate) -> List[Dict[str, Any]]:
        issues = []
        
        # Check required fields
        if not template.template_id:
            issues.append({
                "rule": self.name,
                "severity": "error",
                "message": "Template ID is required",
                "field": "template_id"
            })
        
        if not template.metadata.title:
            issues.append({
                "rule": self.name,
                "severity": "error",
                "message": "Template title is required",
                "field": "metadata.title"
            })
        
        if not template.metadata.description:
            issues.append({
                "rule": self.name,
                "severity": "warning",
                "message": "Template description is recommended",
                "field": "metadata.description"
            })
        
        # Check sources exist
        if not template.sources:
            issues.append({
                "rule": self.name,
                "severity": "warning",
                "message": "Template has no sources defined",
                "field": "sources"
            })
        
        return issues


class VariableDefinitionRule(ValidationRule):
    """Validate variable definitions."""
    
    def __init__(self):
        super().__init__(
            "variable_definitions",
            "Validate template variable definitions and constraints"
        )
    
    async def validate(self, template: ContextTemplate) -> List[Dict[str, Any]]:
        issues = []
        
        variable_names = set()
        
        for i, var in enumerate(template.variables):
            # Check for duplicate names
            if var.name in variable_names:
                issues.append({
                    "rule": self.name,
                    "severity": "error",
                    "message": f"Duplicate variable name: {var.name}",
                    "field": f"variables[{i}].name"
                })
            variable_names.add(var.name)
            
            # Validate variable constraints
            var_issues = await self._validate_variable(var, i)
            issues.extend(var_issues)
        
        return issues
    
    async def _validate_variable(self, var: TemplateVariable, index: int) -> List[Dict[str, Any]]:
        """Validate a single variable definition."""
        issues = []
        
        # Type-specific validation
        if var.type == VariableType.ENUM:
            if not var.allowed_values:
                issues.append({
                    "rule": self.name,
                    "severity": "error",
                    "message": f"Enum variable '{var.name}' must have allowed_values",
                    "field": f"variables[{index}].allowed_values"
                })
        
        if var.type == VariableType.STRING:
            if var.min_length is not None and var.max_length is not None:
                if var.min_length > var.max_length:
                    issues.append({
                        "rule": self.name,
                        "severity": "error",
                        "message": f"Variable '{var.name}' min_length > max_length",
                        "field": f"variables[{index}].min_length"
                    })
        
        if var.type == VariableType.NUMBER:
            if var.min_value is not None and var.max_value is not None:
                if var.min_value > var.max_value:
                    issues.append({
                        "rule": self.name,
                        "severity": "error",
                        "message": f"Variable '{var.name}' min_value > max_value",
                        "field": f"variables[{index}].min_value"
                    })
        
        # Pattern validation
        if var.pattern:
            try:
                re.compile(var.pattern)
            except re.error as e:
                issues.append({
                    "rule": self.name,
                    "severity": "error",
                    "message": f"Invalid regex pattern for variable '{var.name}': {e}",
                    "field": f"variables[{index}].pattern"
                })
        
        return issues


class SourceContentRule(ValidationRule):
    """Validate source content and templates."""
    
    def __init__(self):
        super().__init__(
            "source_content",
            "Validate source content templates and structure"
        )
    
    async def validate(self, template: ContextTemplate) -> List[Dict[str, Any]]:
        issues = []
        
        for i, source in enumerate(template.sources):
            source_issues = await self._validate_source(source, i, template)
            issues.extend(source_issues)
        
        return issues
    
    async def _validate_source(
        self,
        source: TemplateSource,
        index: int,
        template: ContextTemplate
    ) -> List[Dict[str, Any]]:
        """Validate a single source."""
        issues = []
        
        # Check content exists
        if not source.content_template:
            issues.append({
                "rule": self.name,
                "severity": "error",
                "message": f"Source {index} has no content template",
                "field": f"sources[{index}].content_template"
            })
            return issues
        
        # Validate variable references
        defined_vars = {var.name for var in template.variables}
        referenced_vars = self._extract_variables(source.content_template)
        
        for var_name in referenced_vars:
            if var_name not in defined_vars:
                issues.append({
                    "rule": self.name,
                    "severity": "error",
                    "message": f"Source {index} references undefined variable: {var_name}",
                    "field": f"sources[{index}].content_template",
                    "variable": var_name
                })
        
        # Validate priority range
        if source.priority < 0 or source.priority > 100:
            issues.append({
                "rule": self.name,
                "severity": "warning",
                "message": f"Source {index} priority should be 0-100 (got {source.priority})",
                "field": f"sources[{index}].priority"
            })
        
        return issues
    
    def _extract_variables(self, template: str) -> Set[str]:
        """Extract variable names from template content."""
        # Simple variable pattern {{variable}}
        pattern = r'\{\{\s*(\w+)(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?\s*(?:\|[^}]*)?\}\}'
        matches = re.findall(pattern, template)
        return set(matches)


class ValidationConstraintsRule(ValidationRule):
    """Validate template validation constraints."""
    
    def __init__(self):
        super().__init__(
            "validation_constraints",
            "Validate template validation rules and constraints"
        )
    
    async def validate(self, template: ContextTemplate) -> List[Dict[str, Any]]:
        issues = []
        
        if not template.validation:
            return issues
        
        validation = template.validation
        
        # Check constraint values
        if validation.max_tokens is not None and validation.max_tokens <= 0:
            issues.append({
                "rule": self.name,
                "severity": "error",
                "message": "max_tokens must be positive",
                "field": "validation.max_tokens"
            })
        
        if validation.min_sources is not None and validation.min_sources < 0:
            issues.append({
                "rule": self.name,
                "severity": "error",
                "message": "min_sources cannot be negative",
                "field": "validation.min_sources"
            })
        
        if validation.max_sources is not None and validation.max_sources <= 0:
            issues.append({
                "rule": self.name,
                "severity": "error",
                "message": "max_sources must be positive",
                "field": "validation.max_sources"
            })
        
        # Check min/max consistency
        if (validation.min_sources is not None and 
            validation.max_sources is not None and
            validation.min_sources > validation.max_sources):
            issues.append({
                "rule": self.name,
                "severity": "error",
                "message": "min_sources cannot be greater than max_sources",
                "field": "validation"
            })
        
        # Validate forbidden patterns
        if validation.forbidden_patterns:
            for i, pattern in enumerate(validation.forbidden_patterns):
                try:
                    re.compile(pattern)
                except re.error as e:
                    issues.append({
                        "rule": self.name,
                        "severity": "error",
                        "message": f"Invalid forbidden pattern regex: {e}",
                        "field": f"validation.forbidden_patterns[{i}]"
                    })
        
        return issues


class QualityRule(ValidationRule):
    """Validate template quality metrics."""
    
    def __init__(self):
        super().__init__(
            "quality_metrics",
            "Validate template quality and best practices",
            severity="warning"
        )
    
    async def validate(self, template: ContextTemplate) -> List[Dict[str, Any]]:
        issues = []
        
        # Check for good metadata
        if len(template.metadata.description) < 10:
            issues.append({
                "rule": self.name,
                "severity": "warning",
                "message": "Template description should be more descriptive",
                "field": "metadata.description"
            })
        
        if not template.metadata.tags:
            issues.append({
                "rule": self.name,
                "severity": "info",
                "message": "Adding tags helps with template discovery",
                "field": "metadata.tags"
            })
        
        # Check for reasonable variable count
        if len(template.variables) > 20:
            issues.append({
                "rule": self.name,
                "severity": "warning",
                "message": "Template has many variables, consider composition",
                "field": "variables"
            })
        
        # Check for reasonable source count
        if len(template.sources) > 15:
            issues.append({
                "rule": self.name,
                "severity": "warning",
                "message": "Template has many sources, consider splitting",
                "field": "sources"
            })
        
        return issues


class ValidationEngine:
    """Core validation engine that coordinates all validation rules."""
    
    def __init__(self):
        self.rules = [
            TemplateStructureRule(),
            VariableDefinitionRule(),
            SourceContentRule(),
            ValidationConstraintsRule(),
            QualityRule(),
        ]
        self.custom_rules = []
    
    def add_rule(self, rule: ValidationRule):
        """Add a custom validation rule."""
        self.custom_rules.append(rule)
    
    async def validate_template(self, template: ContextTemplate) -> List[Dict[str, Any]]:
        """
        Run all validation rules against a template.
        
        Returns:
            List of validation issues
        """
        all_issues = []
        
        # Run built-in rules
        for rule in self.rules:
            try:
                issues = await rule.validate(template)
                all_issues.extend(issues)
            except Exception as e:
                logger.error(f"Validation rule {rule.name} failed: {e}")
                all_issues.append({
                    "rule": rule.name,
                    "severity": "error",
                    "message": f"Validation rule failed: {e}",
                    "field": "template"
                })
        
        # Run custom rules
        for rule in self.custom_rules:
            try:
                issues = await rule.validate(template)
                all_issues.extend(issues)
            except Exception as e:
                logger.error(f"Custom validation rule {rule.name} failed: {e}")
        
        return all_issues
    
    def get_validation_summary(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get a summary of validation results."""
        summary = {
            "total_issues": len(issues),
            "errors": 0,
            "warnings": 0,
            "info": 0,
            "rules_triggered": set(),
            "fields_with_issues": set(),
            "is_valid": True
        }
        
        for issue in issues:
            severity = issue.get("severity", "error")
            summary[f"{severity}s"] = summary.get(f"{severity}s", 0) + 1
            summary["rules_triggered"].add(issue.get("rule", "unknown"))
            if "field" in issue:
                summary["fields_with_issues"].add(issue["field"])
            
            if severity == "error":
                summary["is_valid"] = False
        
        summary["rules_triggered"] = list(summary["rules_triggered"])
        summary["fields_with_issues"] = list(summary["fields_with_issues"])
        
        return summary


class TemplateValidator:
    """
    Main template validation orchestrator providing comprehensive
    template validation, testing, and quality assessment capabilities.
    """
    
    def __init__(self):
        self.validation_engine = ValidationEngine()
        self._initialized = False
        
        # Test execution tracking
        self._test_results_cache = {}
        self._cache_ttl = timedelta(minutes=30)
    
    async def initialize(self) -> bool:
        """Initialize the template validator."""
        if self._initialized:
            return True
        
        try:
            logger.info("Initializing TemplateValidator...")
            self._initialized = True
            logger.info("TemplateValidator initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize TemplateValidator: {e}")
            return False
    
    async def validate_template(self, template: ContextTemplate) -> List[str]:
        """
        Validate template and return error messages.
        
        This is the main validation method called by TemplateManager.
        
        Args:
            template: Template to validate
        
        Returns:
            List of validation error messages (empty if valid)
        """
        issues = await self.validation_engine.validate_template(template)
        
        # Return only error-level issues as strings
        errors = []
        for issue in issues:
            if issue.get("severity") == "error":
                message = issue.get("message", "Unknown error")
                field = issue.get("field", "")
                if field:
                    errors.append(f"{field}: {message}")
                else:
                    errors.append(message)
        
        return errors
    
    async def validate_template_detailed(
        self,
        template: ContextTemplate,
        include_warnings: bool = True
    ) -> Dict[str, Any]:
        """
        Perform detailed validation with full issue details.
        
        Args:
            template: Template to validate
            include_warnings: Include warnings and info in results
        
        Returns:
            Detailed validation results
        """
        logger.debug(f"Performing detailed validation for template: {template.template_id}")
        
        issues = await self.validation_engine.validate_template(template)
        
        # Filter issues based on settings
        if not include_warnings:
            issues = [issue for issue in issues if issue.get("severity") == "error"]
        
        # Get validation summary
        summary = self.validation_engine.get_validation_summary(issues)
        
        return {
            "template_id": template.template_id,
            "summary": summary,
            "issues": issues,
            "validation_timestamp": datetime.utcnow(),
            "is_valid": summary["is_valid"]
        }
    
    async def validate_execution_result(
        self,
        template: ContextTemplate,
        rendered_sources: List[Dict[str, Any]],
        total_tokens: int
    ) -> List[str]:
        """
        Validate template execution result against validation rules.
        
        Returns:
            List of validation warnings
        """
        warnings = []
        
        if not template.validation:
            return warnings
        
        validation = template.validation
        
        # Check token limit
        if validation.max_tokens and total_tokens > validation.max_tokens:
            warnings.append(
                f"Token count ({total_tokens}) exceeds maximum ({validation.max_tokens})"
            )
        
        # Check source count limits
        source_count = len(rendered_sources)
        if validation.min_sources and source_count < validation.min_sources:
            warnings.append(
                f"Source count ({source_count}) below minimum ({validation.min_sources})"
            )
        
        if validation.max_sources and source_count > validation.max_sources:
            warnings.append(
                f"Source count ({source_count}) exceeds maximum ({validation.max_sources})"
            )
        
        # Check required source types
        if validation.required_source_types:
            present_types = {source.get("type") for source in rendered_sources}
            missing_types = set(validation.required_source_types) - present_types
            if missing_types:
                warnings.append(f"Missing required source types: {list(missing_types)}")
        
        # Check forbidden patterns
        if validation.forbidden_patterns:
            for pattern in validation.forbidden_patterns:
                try:
                    regex = re.compile(pattern)
                    for i, source in enumerate(rendered_sources):
                        content = source.get("content", "")
                        if regex.search(content):
                            warnings.append(
                                f"Source {i} contains forbidden pattern: {pattern}"
                            )
                except re.error:
                    warnings.append(f"Invalid forbidden pattern regex: {pattern}")
        
        return warnings
    
    async def run_template_tests(
        self,
        template: ContextTemplate,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run comprehensive tests on a template.
        
        Args:
            template: Template to test
            test_cases: List of test cases with variables and expected results
        
        Returns:
            Test results summary
        """
        logger.debug(f"Running template tests: {template.template_id}")
        
        test_results = {
            "template_id": template.template_id,
            "test_timestamp": datetime.utcnow(),
            "total_tests": len(test_cases),
            "passed_tests": 0,
            "failed_tests": 0,
            "test_cases": []
        }
        
        for i, test_case in enumerate(test_cases):
            test_id = f"test_{i+1}"
            logger.debug(f"Running test case {test_id}")
            
            try:
                case_result = await self._run_single_test(
                    template, test_case, test_id
                )
                test_results["test_cases"].append(case_result)
                
                if case_result["passed"]:
                    test_results["passed_tests"] += 1
                else:
                    test_results["failed_tests"] += 1
                    
            except Exception as e:
                logger.error(f"Test case {test_id} failed with exception: {e}")
                test_results["test_cases"].append({
                    "test_id": test_id,
                    "passed": False,
                    "error": str(e),
                    "exception_type": type(e).__name__
                })
                test_results["failed_tests"] += 1
        
        # Calculate success rate
        test_results["success_rate"] = (
            test_results["passed_tests"] / test_results["total_tests"]
            if test_results["total_tests"] > 0 else 0.0
        )
        
        # Cache results
        self._test_results_cache[template.template_id] = {
            "results": test_results,
            "timestamp": datetime.utcnow()
        }
        
        logger.info(
            f"Template tests completed: {template.template_id} - "
            f"{test_results['passed_tests']}/{test_results['total_tests']} passed"
        )
        
        return test_results
    
    async def _run_single_test(
        self,
        template: ContextTemplate,
        test_case: Dict[str, Any],
        test_id: str
    ) -> Dict[str, Any]:
        """Run a single test case."""
        
        variables = test_case.get("variables", {})
        expected = test_case.get("expected", {})
        description = test_case.get("description", f"Test case {test_id}")
        
        result = {
            "test_id": test_id,
            "description": description,
            "variables": variables,
            "passed": True,
            "errors": [],
            "warnings": [],
            "assertions": {}
        }
        
        # Validate variables first
        validation_errors = template.validate_variable_values(variables)
        if validation_errors:
            result["passed"] = False
            result["errors"].extend(validation_errors)
            return result
        
        # Test expectations
        assertions = expected.get("assertions", {})
        
        for assertion_name, expected_value in assertions.items():
            try:
                actual_value = await self._evaluate_assertion(
                    template, variables, assertion_name
                )
                
                assertion_passed = self._compare_assertion_values(
                    expected_value, actual_value
                )
                
                result["assertions"][assertion_name] = {
                    "expected": expected_value,
                    "actual": actual_value,
                    "passed": assertion_passed
                }
                
                if not assertion_passed:
                    result["passed"] = False
                    result["errors"].append(
                        f"Assertion '{assertion_name}' failed: "
                        f"expected {expected_value}, got {actual_value}"
                    )
                    
            except Exception as e:
                result["passed"] = False
                result["errors"].append(f"Assertion '{assertion_name}' error: {e}")
                result["assertions"][assertion_name] = {
                    "expected": expected_value,
                    "actual": None,
                    "passed": False,
                    "error": str(e)
                }
        
        return result
    
    async def _evaluate_assertion(
        self,
        template: ContextTemplate,
        variables: Dict[str, Any],
        assertion_name: str
    ) -> Any:
        """Evaluate an assertion against the template."""
        
        # Supported assertions
        if assertion_name == "variable_count":
            return len(template.variables)
        elif assertion_name == "source_count":
            return len(template.sources)
        elif assertion_name == "has_composition":
            return template.has_composition
        elif assertion_name == "template_type":
            return template.type.value
        elif assertion_name == "status":
            return template.status.value
        elif assertion_name.startswith("variable_"):
            var_name = assertion_name[9:]  # Remove "variable_" prefix
            var = template.get_variable(var_name)
            return var is not None
        elif assertion_name == "complexity_score":
            # This would need access to the renderer for full analysis
            # For now, return a simple estimate
            return len(template.sources) * 10 + len(template.variables) * 5
        else:
            raise ValueError(f"Unknown assertion: {assertion_name}")
    
    def _compare_assertion_values(self, expected: Any, actual: Any) -> bool:
        """Compare expected and actual assertion values."""
        if isinstance(expected, dict) and "operator" in expected:
            # Handle complex assertions like {"operator": ">=", "value": 5}
            operator = expected["operator"]
            expected_value = expected["value"]
            
            if operator == "==":
                return actual == expected_value
            elif operator == "!=":
                return actual != expected_value
            elif operator == ">":
                return actual > expected_value
            elif operator == ">=":
                return actual >= expected_value
            elif operator == "<":
                return actual < expected_value
            elif operator == "<=":
                return actual <= expected_value
            elif operator == "in":
                return actual in expected_value
            elif operator == "not_in":
                return actual not in expected_value
            else:
                raise ValueError(f"Unknown operator: {operator}")
        else:
            # Simple equality check
            return actual == expected
    
    def get_cached_test_results(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get cached test results if available and not expired."""
        if template_id not in self._test_results_cache:
            return None
        
        cache_entry = self._test_results_cache[template_id]
        if datetime.utcnow() - cache_entry["timestamp"] > self._cache_ttl:
            del self._test_results_cache[template_id]
            return None
        
        return cache_entry["results"]
    
    def clear_test_cache(self, template_id: Optional[str] = None):
        """Clear test results cache."""
        if template_id:
            self._test_results_cache.pop(template_id, None)
        else:
            self._test_results_cache.clear()
    
    def add_custom_validation_rule(self, rule: ValidationRule):
        """Add a custom validation rule."""
        self.validation_engine.add_rule(rule)
        logger.info(f"Added custom validation rule: {rule.name}")
    
    async def create_test_suite(
        self,
        template: ContextTemplate,
        include_edge_cases: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Automatically generate a test suite for a template.
        
        Args:
            template: Template to create tests for
            include_edge_cases: Include edge case tests
        
        Returns:
            Generated test cases
        """
        test_cases = []
        
        # Generate basic test with default values
        default_variables = {}
        for var in template.variables:
            if var.default_value is not None:
                default_variables[var.name] = var.default_value
            elif var.type == VariableType.STRING:
                default_variables[var.name] = "test_value"
            elif var.type == VariableType.NUMBER:
                default_variables[var.name] = 42
            elif var.type == VariableType.BOOLEAN:
                default_variables[var.name] = True
            elif var.type == VariableType.LIST:
                default_variables[var.name] = ["item1", "item2"]
            elif var.type == VariableType.OBJECT:
                default_variables[var.name] = {"key": "value"}
            elif var.type == VariableType.ENUM and var.allowed_values:
                default_variables[var.name] = var.allowed_values[0]
        
        test_cases.append({
            "description": "Basic test with default values",
            "variables": default_variables,
            "expected": {
                "assertions": {
                    "source_count": {"operator": ">=", "value": 1},
                    "template_type": template.type.value
                }
            }
        })
        
        # Generate edge case tests
        if include_edge_cases:
            edge_cases = await self._generate_edge_case_tests(template)
            test_cases.extend(edge_cases)
        
        return test_cases
    
    async def _generate_edge_case_tests(
        self,
        template: ContextTemplate
    ) -> List[Dict[str, Any]]:
        """Generate edge case tests for a template."""
        edge_cases = []
        
        # Test with minimal required variables only
        minimal_variables = {}
        for var in template.variables:
            if var.required and var.default_value is None:
                if var.type == VariableType.STRING:
                    minimal_variables[var.name] = "x"
                elif var.type == VariableType.NUMBER:
                    minimal_variables[var.name] = 1
                elif var.type == VariableType.BOOLEAN:
                    minimal_variables[var.name] = False
                elif var.type == VariableType.ENUM and var.allowed_values:
                    minimal_variables[var.name] = var.allowed_values[0]
        
        if minimal_variables:
            edge_cases.append({
                "description": "Test with minimal required variables",
                "variables": minimal_variables,
                "expected": {
                    "assertions": {
                        "source_count": {"operator": ">=", "value": 1}
                    }
                }
            })
        
        # Test string length constraints
        for var in template.variables:
            if var.type == VariableType.STRING:
                base_vars = minimal_variables.copy()
                
                if var.min_length is not None:
                    # Test minimum length
                    base_vars[var.name] = "x" * var.min_length
                    edge_cases.append({
                        "description": f"Test {var.name} minimum length",
                        "variables": base_vars,
                        "expected": {"assertions": {"source_count": {"operator": ">=", "value": 1}}}
                    })
                
                if var.max_length is not None:
                    # Test maximum length
                    base_vars[var.name] = "x" * var.max_length
                    edge_cases.append({
                        "description": f"Test {var.name} maximum length",
                        "variables": base_vars,
                        "expected": {"assertions": {"source_count": {"operator": ">=", "value": 1}}}
                    })
        
        return edge_cases