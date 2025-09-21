"""
Complexity Analyzer - Intelligent query complexity assessment for optimal model routing.

Analyzes queries across multiple dimensions to determine the most appropriate
model tier for handling the request.
"""

import re
import time
from typing import Dict, List, Optional, Set, Tuple

from autom8.models.complexity import (
    ComplexityScore,
    ComplexityDimensions,
    ComplexityTier,
    TaskPattern,
    ComplexityHistory,
)
from autom8.utils.tokens import get_token_counter
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class ComplexityAnalyzer:
    """
    Accurately assesses query complexity to enable proper model routing.
    
    Uses multi-dimensional analysis to match queries with appropriate model tiers,
    ensuring optimal resource utilization across the full model spectrum.
    """
    
    def __init__(self):
        self.token_counter = get_token_counter()
        self.analysis_history: List[ComplexityHistory] = []
        self._initialized = False
        
        # Pre-compiled patterns for efficiency
        self._compile_patterns()
        
        # Domain complexity profiles
        self.domain_profiles = self._load_domain_profiles()
    
    async def initialize(self) -> bool:
        """Initialize the complexity analyzer."""
        try:
            self._initialized = True
            logger.info("Complexity analyzer initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize complexity analyzer: {e}")
            return False
        
    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance."""
        self.patterns = {
            'code_blocks': re.compile(r'```[\s\S]*?```'),
            'inline_code': re.compile(r'`[^`]+`'),
            'function_def': re.compile(r'def\s+\w+\('),
            'class_def': re.compile(r'class\s+\w+[\(:]'),
            'imports': re.compile(r'(?:from\s+\w+\s+)?import\s+[\w.,\s]+'),
            'loops': re.compile(r'\b(?:for|while)\s+'),
            'conditionals': re.compile(r'\b(?:if|elif|else|switch|case)\b'),
            'async_await': re.compile(r'\b(?:async|await)\b'),
            'decorators': re.compile(r'@\w+'),
            'sql_queries': re.compile(r'\b(?:SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b', re.IGNORECASE),
            'regex_patterns': re.compile(r'r["\'].*?["\']'),
            'error_handling': re.compile(r'\b(?:try|except|catch|throw|raise|finally)\b'),
            'architecture_terms': re.compile(r'\b(?:microservice|distributed|scalable|architecture|design pattern|SOLID|MVC|REST|GraphQL)\b', re.IGNORECASE),
            'debugging_terms': re.compile(r'\b(?:debug|trace|breakpoint|stack trace|error|exception|bug|fix)\b', re.IGNORECASE),
            'refactor_terms': re.compile(r'\b(?:refactor|optimize|restructure|clean up|improve|simplify)\b', re.IGNORECASE),
            'test_terms': re.compile(r'\b(?:test|unittest|pytest|mock|assert|coverage)\b', re.IGNORECASE),
        }
    
    def _load_domain_profiles(self) -> Dict[str, Dict[str, float]]:
        """Load domain-specific complexity profiles."""
        return {
            'cryptography': {
                'base_complexity': 0.8,
                'keywords': ['encrypt', 'decrypt', 'hash', 'cipher', 'key', 'certificate', 'ssl', 'tls', 'crypto'],
                'multiplier': 1.5
            },
            'distributed_systems': {
                'base_complexity': 0.7,
                'keywords': ['distributed', 'microservice', 'cluster', 'load balancer', 'consensus', 'raft', 'eventual consistency'],
                'multiplier': 1.4
            },
            'machine_learning': {
                'base_complexity': 0.6,
                'keywords': ['neural network', 'deep learning', 'tensorflow', 'pytorch', 'model', 'training', 'inference'],
                'multiplier': 1.3
            },
            'database_optimization': {
                'base_complexity': 0.5,
                'keywords': ['query optimization', 'index', 'performance', 'explain plan', 'database tuning'],
                'multiplier': 1.2
            },
            'web_frontend': {
                'base_complexity': 0.3,
                'keywords': ['react', 'vue', 'angular', 'html', 'css', 'javascript', 'dom', 'responsive'],
                'multiplier': 1.0
            },
            'scripting': {
                'base_complexity': 0.2,
                'keywords': ['script', 'automation', 'bash', 'shell', 'cron', 'simple task'],
                'multiplier': 0.8
            }
        }
    
    async def analyze(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> ComplexityScore:
        """
        Perform comprehensive complexity analysis of a query.
        
        Args:
            query: The query/task to analyze
            context: Additional context information
            
        Returns:
            Complete complexity assessment
        """
        start_time = time.perf_counter()
        logger.debug(f"Starting complexity analysis for query: {query[:100]}...")
        
        context = context or {}
        
        # Multi-dimensional analysis
        dimensions = ComplexityDimensions(
            syntactic=self._analyze_syntax(query),
            semantic=self._analyze_semantics(query),
            contextual=self._analyze_context_needs(query, context),
            interdependency=self._analyze_dependencies(query, context),
            domain=self._analyze_domain_complexity(query)
        )
        
        # Task pattern recognition
        task_pattern = self._identify_task_pattern(query)
        
        # Calculate weighted complexity score
        raw_score = self._calculate_weighted_score(dimensions, task_pattern)
        
        # Determine confidence based on pattern recognition
        confidence = self._calculate_confidence(dimensions, task_pattern, query)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(dimensions, task_pattern, raw_score)
        key_factors = self._identify_key_factors(dimensions, task_pattern)
        
        # Domain classification
        domain_classification = self._classify_domain(query)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        score = ComplexityScore(
            raw_score=raw_score,
            confidence=confidence,
            dimensions=dimensions,
            recommended_tier=self._map_to_tier(raw_score),
            task_pattern=task_pattern,
            domain_classification=domain_classification,
            reasoning=reasoning,
            key_factors=key_factors,
            processing_time_ms=processing_time
        )
        
        logger.info(
            f"Complexity analysis complete: score={raw_score:.3f}, "
            f"tier={score.recommended_tier}, confidence={confidence:.3f}, "
            f"time={processing_time:.1f}ms"
        )
        
        return score
    
    def _analyze_syntax(self, query: str) -> float:
        """Analyze syntactic complexity of the query."""
        complexity = 0.0
        
        # Code complexity indicators
        code_blocks = len(self.patterns['code_blocks'].findall(query))
        inline_code = len(self.patterns['inline_code'].findall(query))
        
        if code_blocks > 0 or inline_code > 0:
            # Extract and analyze code
            code_content = ""
            for match in self.patterns['code_blocks'].finditer(query):
                code_content += match.group()
            for match in self.patterns['inline_code'].finditer(query):
                code_content += match.group()
            
            if code_content:
                complexity += self._analyze_code_complexity(code_content)
        
        # Natural language complexity
        sentences = len([s for s in query.split('.') if s.strip()])
        complexity += min(0.3, sentences * 0.02)
        
        # Question complexity (multiple questions = higher complexity)
        questions = query.count('?')
        complexity += min(0.2, questions * 0.05)
        
        # Length factor (very long queries are more complex)
        length_factor = min(0.2, len(query) / 5000)
        complexity += length_factor
        
        return min(1.0, complexity)
    
    def _analyze_code_complexity(self, code: str) -> float:
        """Analyze complexity of code snippets."""
        complexity = 0.0
        
        # Cyclomatic complexity indicators
        conditionals = len(self.patterns['conditionals'].findall(code))
        loops = len(self.patterns['loops'].findall(code))
        error_handling = len(self.patterns['error_handling'].findall(code))
        
        complexity += conditionals * 0.05
        complexity += loops * 0.06
        complexity += error_handling * 0.04
        
        # Advanced patterns
        async_patterns = len(self.patterns['async_await'].findall(code))
        decorators = len(self.patterns['decorators'].findall(code))
        regex_patterns = len(self.patterns['regex_patterns'].findall(code))
        
        complexity += async_patterns * 0.08
        complexity += decorators * 0.03
        complexity += regex_patterns * 0.07
        
        # Function and class definitions
        functions = len(self.patterns['function_def'].findall(code))
        classes = len(self.patterns['class_def'].findall(code))
        
        complexity += functions * 0.04
        complexity += classes * 0.08
        
        # SQL complexity
        sql_queries = len(self.patterns['sql_queries'].findall(code))
        complexity += sql_queries * 0.06
        
        # Nesting depth approximation
        nesting_chars = code.count('{') + code.count('(') + code.count('[')
        nesting_complexity = min(0.3, nesting_chars * 0.01)
        complexity += nesting_complexity
        
        return min(1.0, complexity)
    
    def _analyze_semantics(self, query: str) -> float:
        """Analyze semantic complexity and meaning depth."""
        complexity = 0.0
        query_lower = query.lower()
        
        # Base programming complexity - common development tasks
        programming_terms = [
            'function', 'class', 'method', 'variable', 'array', 'list', 'dictionary',
            'loop', 'condition', 'if', 'else', 'for', 'while', 'try', 'catch',
            'import', 'export', 'module', 'package', 'library', 'framework'
        ]
        
        programming_count = sum(1 for term in programming_terms if term in query_lower)
        complexity += min(0.3, programming_count * 0.05)  # Base programming complexity
        
        # Abstract concept indicators
        abstract_terms = [
            'architecture', 'design', 'pattern', 'paradigm', 'methodology',
            'algorithm', 'optimization', 'scalability', 'performance',
            'security', 'reliability', 'maintainability', 'extensibility'
        ]
        
        abstract_count = sum(1 for term in abstract_terms if term in query_lower)
        complexity += min(0.4, abstract_count * 0.1)  # Increased weight for abstract concepts
        
        # Technical depth indicators
        technical_terms = [
            'implementation', 'integration', 'configuration', 'deployment',
            'monitoring', 'debugging', 'profiling', 'benchmarking',
            'synchronization', 'concurrency', 'parallelism', 'distributed',
            'sort', 'search', 'parse', 'validate', 'compile', 'build'
        ]
        
        technical_count = sum(1 for term in technical_terms if term in query_lower)
        complexity += min(0.4, technical_count * 0.08)  # Increased weight
        
        # Problem-solving indicators
        problem_terms = [
            'solve', 'fix', 'troubleshoot', 'diagnose', 'analyze', 'investigate',
            'optimize', 'improve', 'enhance', 'refactor', 'redesign'
        ]
        
        problem_count = sum(1 for term in problem_terms if term in query_lower)
        complexity += min(0.3, problem_count * 0.07)
        
        # Multi-step process indicators
        step_indicators = ['first', 'then', 'next', 'finally', 'step', 'phase']
        step_count = sum(1 for indicator in step_indicators if indicator in query_lower)
        complexity += min(0.2, step_count * 0.04)
        
        return min(1.0, complexity)
    
    def _analyze_context_needs(self, query: str, context: Dict) -> float:
        """Analyze how much context understanding is required."""
        complexity = 0.0
        query_lower = query.lower()
        
        # Basic task complexity - even simple tasks need some context
        if len(query.strip()) > 0:
            complexity += 0.1  # Base complexity for any non-empty query
        
        # Reference to external systems/files
        reference_indicators = [
            'this file', 'current project', 'existing code', 'previous implementation',
            'our system', 'the database', 'configuration', 'environment', 'write',
            'create', 'build', 'make', 'implement', 'add', 'develop'
        ]
        
        reference_count = sum(1 for ref in reference_indicators if ref in query_lower)
        complexity += min(0.4, reference_count * 0.08)  # Slightly reduced but more comprehensive
        
        # Pronouns indicating context dependency
        pronouns = ['this', 'that', 'these', 'those', 'it', 'they']
        pronoun_count = sum(query_lower.count(pronoun) for pronoun in pronouns)
        complexity += min(0.2, pronoun_count * 0.03)  # Increased weight
        
        # Context size factor
        context_size = len(str(context)) if context else 0
        context_factor = min(0.3, context_size / 10000)
        complexity += context_factor
        
        # Integration requirements
        integration_terms = ['integrate', 'connect', 'interface', 'api', 'service']
        integration_count = sum(1 for term in integration_terms if term in query_lower)
        complexity += min(0.3, integration_count * 0.08)
        
        return min(1.0, complexity)
    
    def _analyze_dependencies(self, query: str, context: Dict) -> float:
        """Analyze system interaction and dependency complexity."""
        complexity = 0.0
        query_lower = query.lower()
        
        # Explicit dependencies
        explicit_deps = [
            'import', 'include', 'require', 'using', 'from', 'library', 'module',
            'package', 'dependency', 'framework', 'tool', 'utility'
        ]
        
        explicit_count = sum(1 for term in explicit_deps if term in query_lower)
        complexity += min(0.3, explicit_count * 0.06)  # Explicit dependencies
        
        # Implicit programming dependencies (any programming task has some dependencies)
        programming_indicators = [
            'function', 'class', 'method', 'algorithm', 'code', 'program',
            'script', 'application', 'system', 'software'
        ]
        
        programming_count = sum(1 for term in programming_indicators if term in query_lower)
        if programming_count > 0:
            complexity += 0.1  # Base interdependency for any programming task
        
        # System interaction indicators
        system_terms = [
            'database', 'api', 'service', 'microservice', 'server', 'client',
            'network', 'protocol', 'interface', 'endpoint', 'queue', 'cache',
            'file', 'data', 'input', 'output', 'stream', 'connection'
        ]
        
        system_count = sum(1 for term in system_terms if term in query_lower)
        complexity += min(0.4, system_count * 0.08)  # Increased weight
        
        # Cross-cutting concerns
        concern_terms = [
            'security', 'authentication', 'authorization', 'logging', 'monitoring',
            'error handling', 'transaction', 'consistency', 'backup', 'recovery'
        ]
        
        concern_count = sum(1 for term in concern_terms if term in query_lower)
        complexity += min(0.3, concern_count * 0.06)
        
        # Multi-component interactions
        if any(word in query_lower for word in ['between', 'among', 'across', 'throughout']):
            complexity += 0.15
        
        # Dependency chain indicators
        chain_terms = ['depends on', 'requires', 'needs', 'relies on', 'uses']
        chain_count = sum(1 for term in chain_terms if term in query_lower)
        complexity += min(0.2, chain_count * 0.08)
        
        return min(1.0, complexity)
    
    def _analyze_domain_complexity(self, query: str) -> float:
        """Analyze domain-specific complexity."""
        query_lower = query.lower()
        max_complexity = 0.0
        
        for domain, profile in self.domain_profiles.items():
            # Check for domain keywords
            keyword_matches = sum(1 for keyword in profile['keywords'] 
                                if keyword in query_lower)
            
            if keyword_matches > 0:
                # Calculate domain complexity
                domain_complexity = profile['base_complexity']
                keyword_factor = min(0.3, keyword_matches * 0.1)
                total_complexity = (domain_complexity + keyword_factor) * profile['multiplier']
                
                max_complexity = max(max_complexity, min(1.0, total_complexity))
        
        # Default complexity if no domain detected
        return max_complexity if max_complexity > 0 else 0.3
    
    def _identify_task_pattern(self, query: str) -> TaskPattern:
        """Identify the primary task pattern."""
        query_lower = query.lower()
        
        # Pattern scoring
        pattern_scores = {}
        
        # Boilerplate generation
        if any(term in query_lower for term in ['generate', 'create template', 'boilerplate', 'scaffold']):
            pattern_scores[TaskPattern.BOILERPLATE] = 0.8
        
        # Refactoring
        refactor_matches = len(self.patterns['refactor_terms'].findall(query))
        if refactor_matches > 0:
            pattern_scores[TaskPattern.REFACTORING] = min(0.9, 0.3 + refactor_matches * 0.2)
        
        # Debugging
        debug_matches = len(self.patterns['debugging_terms'].findall(query))
        if debug_matches > 0:
            pattern_scores[TaskPattern.DEBUGGING] = min(0.9, 0.4 + debug_matches * 0.15)
        
        # Architecture
        arch_matches = len(self.patterns['architecture_terms'].findall(query))
        if arch_matches > 0:
            pattern_scores[TaskPattern.ARCHITECTURE] = min(0.9, 0.5 + arch_matches * 0.1)
        
        # Testing
        test_matches = len(self.patterns['test_terms'].findall(query))
        if test_matches > 0:
            pattern_scores[TaskPattern.TESTING] = min(0.8, 0.3 + test_matches * 0.2)
        
        # Documentation
        if any(term in query_lower for term in ['document', 'explain', 'describe', 'comment']):
            pattern_scores[TaskPattern.DOCUMENTATION] = 0.6
        
        # Analysis
        if any(term in query_lower for term in ['analyze', 'review', 'examine', 'assess']):
            pattern_scores[TaskPattern.ANALYSIS] = 0.7
        
        # Creative tasks
        if any(term in query_lower for term in ['design', 'creative', 'brainstorm', 'ideate']):
            pattern_scores[TaskPattern.CREATIVE] = 0.5
        
        # Research
        if any(term in query_lower for term in ['research', 'investigate', 'study', 'explore']):
            pattern_scores[TaskPattern.RESEARCH] = 0.8
        
        # Return highest scoring pattern
        if pattern_scores:
            return max(pattern_scores.items(), key=lambda x: x[1])[0]
        
        return TaskPattern.UNKNOWN
    
    def _calculate_weighted_score(self, dimensions: ComplexityDimensions, pattern: TaskPattern) -> float:
        """Calculate weighted complexity score."""
        # Base weights
        weights = {
            'syntactic': 0.2,
            'semantic': 0.25,
            'contextual': 0.2,
            'interdependency': 0.2,
            'domain': 0.15
        }
        
        # Adjust weights based on task pattern
        if pattern == TaskPattern.BOILERPLATE:
            weights['syntactic'] = 0.4
            weights['semantic'] = 0.1
        elif pattern == TaskPattern.ARCHITECTURE:
            weights['semantic'] = 0.3
            weights['interdependency'] = 0.3
        elif pattern == TaskPattern.DEBUGGING:
            weights['contextual'] = 0.3
            weights['interdependency'] = 0.25
        elif pattern == TaskPattern.RESEARCH:
            weights['semantic'] = 0.35
            weights['domain'] = 0.25
        
        # Calculate weighted score
        score = (
            dimensions.syntactic * weights['syntactic'] +
            dimensions.semantic * weights['semantic'] +
            dimensions.contextual * weights['contextual'] +
            dimensions.interdependency * weights['interdependency'] +
            dimensions.domain * weights['domain']
        )
        
        # Apply scaling factor to align with expected ranges
        # Current scores are too conservative, scale up by ~2.5x
        scaled_score = score * 2.5
        
        return min(1.0, scaled_score)
    
    def _calculate_confidence(self, dimensions: ComplexityDimensions, pattern: TaskPattern, query: str) -> float:
        """Calculate confidence in the complexity assessment."""
        confidence = 0.7  # Base confidence
        
        # Higher confidence for recognized patterns
        if pattern != TaskPattern.UNKNOWN:
            confidence += 0.2
        
        # Higher confidence for clear domain classification
        if dimensions.domain > 0.5:
            confidence += 0.1
        
        # Lower confidence for very short queries
        if len(query) < 50:
            confidence -= 0.2
        
        # Lower confidence for very long queries without structure
        if len(query) > 2000 and dimensions.syntactic < 0.3:
            confidence -= 0.15
        
        # Higher confidence for balanced dimensions
        dimension_variance = self._calculate_variance([
            dimensions.syntactic, dimensions.semantic, dimensions.contextual,
            dimensions.interdependency, dimensions.domain
        ])
        
        if dimension_variance < 0.1:  # Low variance = more consistent assessment
            confidence += 0.1
        
        return max(0.1, min(1.0, confidence))
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _generate_reasoning(self, dimensions: ComplexityDimensions, pattern: TaskPattern, score: float) -> str:
        """Generate human-readable reasoning for the complexity assessment."""
        reasons = []
        
        # Pattern-based reasoning
        if pattern == TaskPattern.BOILERPLATE:
            reasons.append("Standard template generation task")
        elif pattern == TaskPattern.REFACTORING:
            reasons.append("Code refactoring with moderate complexity")
        elif pattern == TaskPattern.DEBUGGING:
            reasons.append("Debugging task requiring analysis and problem-solving")
        elif pattern == TaskPattern.ARCHITECTURE:
            reasons.append("System architecture design requiring holistic understanding")
        elif pattern == TaskPattern.RESEARCH:
            reasons.append("Research task requiring deep domain knowledge")
        
        # Dimension-based reasoning
        if dimensions.syntactic > 0.6:
            reasons.append("High syntactic complexity with complex code structures")
        elif dimensions.syntactic > 0.3:
            reasons.append("Moderate syntactic complexity")
        
        if dimensions.semantic > 0.6:
            reasons.append("Abstract concepts requiring deep understanding")
        
        if dimensions.contextual > 0.6:
            reasons.append("Significant context dependency")
        
        if dimensions.interdependency > 0.6:
            reasons.append("Complex system interactions and dependencies")
        
        if dimensions.domain > 0.7:
            reasons.append("Specialized domain expertise required")
        
        # Overall assessment
        if score > 0.8:
            reasons.append("Frontier-level complexity requiring maximum model capability")
        elif score > 0.6:
            reasons.append("Complex task requiring capable model")
        elif score > 0.4:
            reasons.append("Moderate complexity suitable for mid-tier models")
        elif score > 0.2:
            reasons.append("Simple task suitable for efficient models")
        else:
            reasons.append("Trivial task suitable for lightweight models")
        
        return "; ".join(reasons) if reasons else "Standard complexity assessment"
    
    def _identify_key_factors(self, dimensions: ComplexityDimensions, pattern: TaskPattern) -> List[str]:
        """Identify the key factors driving complexity."""
        factors = []
        
        # Get sorted dimensions
        dim_values = [
            ('syntactic', dimensions.syntactic),
            ('semantic', dimensions.semantic),
            ('contextual', dimensions.contextual),
            ('interdependency', dimensions.interdependency),
            ('domain', dimensions.domain)
        ]
        dim_values.sort(key=lambda x: x[1], reverse=True)
        
        # Add top contributing dimensions
        for dim_name, value in dim_values[:3]:
            if value > 0.4:
                factors.append(f"{dim_name} complexity ({value:.2f})")
        
        # Add pattern factor
        if pattern != TaskPattern.UNKNOWN:
            factors.append(f"task pattern: {pattern.value}")
        
        return factors
    
    def _classify_domain(self, query: str) -> Optional[str]:
        """Classify the domain of the query."""
        query_lower = query.lower()
        
        for domain, profile in self.domain_profiles.items():
            keyword_matches = sum(1 for keyword in profile['keywords'] 
                                if keyword in query_lower)
            if keyword_matches >= 2:  # Require at least 2 keyword matches
                return domain
        
        return None
    
    def _map_to_tier(self, score: float) -> ComplexityTier:
        """Map complexity score to model tier."""
        if score < 0.2:
            return ComplexityTier.TRIVIAL
        elif score < 0.4:
            return ComplexityTier.SIMPLE
        elif score < 0.6:
            return ComplexityTier.MODERATE
        elif score < 0.8:
            return ComplexityTier.COMPLEX
        else:
            return ComplexityTier.FRONTIER
    
    def add_feedback(
        self,
        query: str,
        assessed_complexity: ComplexityScore,
        actual_model: str,
        performance_metrics: Dict[str, float],
        user_feedback: Optional[str] = None
    ) -> None:
        """Add feedback for continuous learning."""
        import hashlib
        
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        history_entry = ComplexityHistory(
            query_hash=query_hash,
            assessed_complexity=assessed_complexity,
            actual_model_used=actual_model,
            actual_performance=performance_metrics,
            user_feedback=user_feedback,
            routing_correct=performance_metrics.get('success', 0.0) > 0.8
        )
        
        self.analysis_history.append(history_entry)
        
        # Keep only recent history (last 1000 entries)
        if len(self.analysis_history) > 1000:
            self.analysis_history = self.analysis_history[-1000:]
        
        logger.debug(f"Added complexity analysis feedback for query hash {query_hash[:8]}")
    
    def get_accuracy_metrics(self) -> Dict[str, float]:
        """Get accuracy metrics for the complexity analyzer."""
        if not self.analysis_history:
            return {}
        
        total_assessments = len(self.analysis_history)
        correct_routings = sum(1 for h in self.analysis_history if h.routing_correct)
        
        accuracy = correct_routings / total_assessments if total_assessments > 0 else 0.0
        
        # Calculate average confidence for correct vs incorrect assessments
        correct_entries = [h for h in self.analysis_history if h.routing_correct]
        incorrect_entries = [h for h in self.analysis_history if not h.routing_correct]
        
        avg_confidence_correct = (
            sum(h.assessed_complexity.confidence for h in correct_entries) / len(correct_entries)
            if correct_entries else 0.0
        )
        
        avg_confidence_incorrect = (
            sum(h.assessed_complexity.confidence for h in incorrect_entries) / len(incorrect_entries)
            if incorrect_entries else 0.0
        )
        
        return {
            'overall_accuracy': accuracy,
            'total_assessments': total_assessments,
            'correct_routings': correct_routings,
            'avg_confidence_correct': avg_confidence_correct,
            'avg_confidence_incorrect': avg_confidence_incorrect,
            'confidence_correlation': avg_confidence_correct - avg_confidence_incorrect
        }
