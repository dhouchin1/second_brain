"""
Automated Model Performance Benchmark Suite.

Comprehensive benchmarking system for evaluating model performance
across different scenarios, complexity levels, and task types.
"""

import asyncio
import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union

from autom8.core.benchmarks.scenarios import BenchmarkScenario, ScenarioGenerator, ScenarioType, DifficultyLevel
from autom8.core.benchmarks.results import (
    BenchmarkResult, BenchmarkReport, ExecutionStatus, QualityRating, PerformanceMetrics
)
from autom8.core.routing.router import ModelRouter
from autom8.core.complexity.analyzer import ComplexityAnalyzer
from autom8.models.routing import Model, RoutingPreferences
from autom8.models.complexity import ComplexityScore
from autom8.utils.tokens import estimate_tokens
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class EvaluationEngine:
    """
    Automated evaluation system for benchmark responses.
    """
    
    def __init__(self):
        self.evaluation_functions = {
            'exact_match': self._exact_match_evaluation,
            'contains_answer': self._contains_answer_evaluation,
            'code_execution_test': self._code_execution_evaluation,
            'code_structure_test': self._code_structure_evaluation,
            'algorithm_performance_test': self._algorithm_performance_evaluation,
            'logic_evaluation': self._logic_evaluation,
            'summary_quality_check': self._summary_quality_evaluation,
            'schema_validation': self._schema_validation_evaluation,
            'bug_fix_validation': self._bug_fix_evaluation
        }
    
    async def evaluate_response(
        self,
        scenario: BenchmarkScenario,
        response: str,
        expected_output: Optional[str] = None
    ) -> tuple[QualityRating, Dict[str, float], List[str]]:
        """
        Evaluate a model response against a benchmark scenario.
        
        Returns:
            Tuple of (quality_rating, evaluation_scores, evaluation_notes)
        """
        
        if not scenario.auto_evaluable or not scenario.evaluation_function:
            return await self._manual_evaluation_placeholder(scenario, response, expected_output)
        
        evaluation_func = self.evaluation_functions.get(scenario.evaluation_function)
        if not evaluation_func:
            logger.warning(f"Unknown evaluation function: {scenario.evaluation_function}")
            return QualityRating.ACCEPTABLE, {}, ["Automatic evaluation not available"]
        
        try:
            return await evaluation_func(scenario, response, expected_output)
        except Exception as e:
            logger.error(f"Evaluation failed for scenario {scenario.id}: {e}")
            return QualityRating.FAILED, {}, [f"Evaluation error: {str(e)}"]
    
    async def _exact_match_evaluation(
        self, scenario: BenchmarkScenario, response: str, expected: Optional[str]
    ) -> tuple[QualityRating, Dict[str, float], List[str]]:
        """Exact match evaluation for precise answers."""
        
        if not expected:
            return QualityRating.ACCEPTABLE, {}, ["No expected output for comparison"]
        
        # Clean and normalize responses
        response_clean = response.strip().lower()
        expected_clean = expected.strip().lower()
        
        if response_clean == expected_clean:
            return QualityRating.EXCELLENT, {'accuracy': 1.0}, ["Exact match achieved"]
        elif expected_clean in response_clean:
            return QualityRating.GOOD, {'accuracy': 0.8}, ["Expected answer found in response"]
        else:
            # Check for partial matches
            words_expected = set(expected_clean.split())
            words_response = set(response_clean.split())
            overlap = len(words_expected.intersection(words_response))
            
            if overlap > 0:
                accuracy = overlap / len(words_expected)
                if accuracy > 0.5:
                    return QualityRating.ACCEPTABLE, {'accuracy': accuracy}, ["Partial match found"]
            
            return QualityRating.POOR, {'accuracy': 0.0}, ["No match found"]
    
    async def _contains_answer_evaluation(
        self, scenario: BenchmarkScenario, response: str, expected: Optional[str]
    ) -> tuple[QualityRating, Dict[str, float], List[str]]:
        """Check if response contains the expected answer."""
        
        if not expected:
            return QualityRating.ACCEPTABLE, {}, ["No expected output for comparison"]
        
        response_lower = response.lower()
        expected_lower = expected.lower()
        
        if expected_lower in response_lower:
            # Check response quality beyond just containing the answer
            response_length = len(response.split())
            if response_length < 5:
                return QualityRating.GOOD, {'accuracy': 0.9, 'completeness': 0.5}, ["Answer found but response is brief"]
            elif response_length > 100:
                return QualityRating.GOOD, {'accuracy': 0.9, 'completeness': 0.7}, ["Answer found but response is verbose"]
            else:
                return QualityRating.EXCELLENT, {'accuracy': 1.0, 'completeness': 1.0}, ["Answer found with good detail"]
        else:
            return QualityRating.POOR, {'accuracy': 0.0}, ["Expected answer not found"]
    
    async def _code_execution_evaluation(
        self, scenario: BenchmarkScenario, response: str, expected: Optional[str]
    ) -> tuple[QualityRating, Dict[str, float], List[str]]:
        """Evaluate code by attempting to execute it."""
        
        notes = []
        scores = {}
        
        # Extract code blocks from response
        code_blocks = self._extract_code_blocks(response)
        if not code_blocks:
            return QualityRating.POOR, {'syntax': 0.0}, ["No code blocks found in response"]
        
        # Test syntax validity
        syntax_valid = False
        for code in code_blocks:
            try:
                compile(code, '<string>', 'exec')
                syntax_valid = True
                notes.append("Syntax is valid")
                scores['syntax'] = 1.0
                break
            except SyntaxError:
                continue
        
        if not syntax_valid:
            scores['syntax'] = 0.0
            notes.append("Syntax errors found")
            return QualityRating.POOR, scores, notes
        
        # For simple functions, try basic functionality tests
        if 'prime' in scenario.prompt.lower():
            functionality_score = self._test_prime_function(code_blocks[0])
            scores['functionality'] = functionality_score
            notes.append(f"Functionality test score: {functionality_score:.2f}")
        
        # Overall assessment
        overall_score = sum(scores.values()) / len(scores)
        if overall_score >= 0.9:
            return QualityRating.EXCELLENT, scores, notes
        elif overall_score >= 0.7:
            return QualityRating.GOOD, scores, notes
        elif overall_score >= 0.5:
            return QualityRating.ACCEPTABLE, scores, notes
        else:
            return QualityRating.POOR, scores, notes
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks from markdown-formatted text."""
        import re
        
        # Find code blocks with triple backticks
        pattern = r'```(?:python)?\n?(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            return [match.strip() for match in matches]
        
        # Try to find code without markdown formatting
        lines = text.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.strip().startswith('def ') or line.strip().startswith('class '):
                in_code = True
            
            if in_code:
                code_lines.append(line)
                
                # Stop at empty line or non-indented line (simple heuristic)
                if not line.strip() and code_lines:
                    break
        
        if code_lines:
            return ['\n'.join(code_lines)]
        
        return []
    
    def _test_prime_function(self, code: str) -> float:
        """Test a prime number function implementation."""
        try:
            # Execute the code
            namespace = {}
            exec(code, namespace)
            
            # Find the function (assuming it's the main function)
            func = None
            for name, obj in namespace.items():
                if callable(obj) and name != '__builtins__':
                    func = obj
                    break
            
            if not func:
                return 0.0
            
            # Test with known values
            test_cases = [
                (2, True), (3, True), (4, False), (5, True),
                (8, False), (11, True), (15, False), (17, True)
            ]
            
            correct = 0
            for n, expected in test_cases:
                try:
                    result = func(n)
                    if bool(result) == expected:
                        correct += 1
                except:
                    pass
            
            return correct / len(test_cases)
            
        except Exception:
            return 0.0
    
    async def _code_structure_evaluation(
        self, scenario: BenchmarkScenario, response: str, expected: Optional[str]
    ) -> tuple[QualityRating, Dict[str, float], List[str]]:
        """Evaluate code structure and organization."""
        
        notes = []
        scores = {}
        
        code_blocks = self._extract_code_blocks(response)
        if not code_blocks:
            return QualityRating.POOR, {'structure': 0.0}, ["No code found"]
        
        code = code_blocks[0]
        
        # Check for class definition
        if 'class ' in code:
            scores['has_class'] = 1.0
            notes.append("Class definition found")
        else:
            scores['has_class'] = 0.0
            notes.append("No class definition found")
        
        # Check for required methods
        required_methods = ['insert', 'search', 'delete']
        methods_found = 0
        for method in required_methods:
            if f'def {method}' in code:
                methods_found += 1
        
        scores['methods'] = methods_found / len(required_methods)
        notes.append(f"Found {methods_found}/{len(required_methods)} required methods")
        
        # Check for proper structure
        if '__init__' in code:
            scores['constructor'] = 1.0
            notes.append("Constructor found")
        else:
            scores['constructor'] = 0.0
        
        overall_score = sum(scores.values()) / len(scores)
        
        if overall_score >= 0.8:
            return QualityRating.EXCELLENT, scores, notes
        elif overall_score >= 0.6:
            return QualityRating.GOOD, scores, notes
        elif overall_score >= 0.4:
            return QualityRating.ACCEPTABLE, scores, notes
        else:
            return QualityRating.POOR, scores, notes
    
    async def _algorithm_performance_evaluation(
        self, scenario: BenchmarkScenario, response: str, expected: Optional[str]
    ) -> tuple[QualityRating, Dict[str, float], List[str]]:
        """Evaluate algorithm implementation and performance."""
        
        notes = []
        scores = {}
        
        # Check for algorithmic concepts
        algorithmic_terms = ['dijkstra', 'shortest', 'path', 'graph', 'distance', 'priority', 'queue']
        terms_found = sum(1 for term in algorithmic_terms if term.lower() in response.lower())
        scores['algorithmic_concepts'] = min(1.0, terms_found / 5)
        notes.append(f"Found {terms_found} relevant algorithmic terms")
        
        # Check for implementation completeness
        code_blocks = self._extract_code_blocks(response)
        if code_blocks:
            code = code_blocks[0]
            if len(code.split('\n')) > 10:
                scores['implementation_length'] = 1.0
                notes.append("Substantial implementation provided")
            else:
                scores['implementation_length'] = 0.5
                notes.append("Brief implementation")
        else:
            scores['implementation_length'] = 0.0
            notes.append("No implementation found")
        
        overall_score = sum(scores.values()) / len(scores)
        
        if overall_score >= 0.8:
            return QualityRating.EXCELLENT, scores, notes
        elif overall_score >= 0.6:
            return QualityRating.GOOD, scores, notes
        else:
            return QualityRating.ACCEPTABLE, scores, notes
    
    async def _logic_evaluation(
        self, scenario: BenchmarkScenario, response: str, expected: Optional[str]
    ) -> tuple[QualityRating, Dict[str, float], List[str]]:
        """Evaluate logical reasoning and deduction."""
        
        notes = []
        scores = {}
        
        # Check if response contains logical reasoning
        reasoning_indicators = ['because', 'therefore', 'since', 'given that', 'if', 'then']
        reasoning_count = sum(1 for indicator in reasoning_indicators if indicator in response.lower())
        scores['reasoning_structure'] = min(1.0, reasoning_count / 3)
        
        # Check for the expected answer pattern
        if expected and expected.lower() in response.lower():
            scores['correct_answer'] = 1.0
            notes.append("Correct answer found")
        else:
            scores['correct_answer'] = 0.0
            notes.append("Expected answer not found")
        
        # Check for step-by-step reasoning
        if any(phrase in response.lower() for phrase in ['step 1', 'first', 'next', 'finally']):
            scores['structured_approach'] = 1.0
            notes.append("Structured reasoning approach")
        else:
            scores['structured_approach'] = 0.5
        
        overall_score = sum(scores.values()) / len(scores)
        
        if overall_score >= 0.8:
            return QualityRating.EXCELLENT, scores, notes
        elif overall_score >= 0.6:
            return QualityRating.GOOD, scores, notes
        else:
            return QualityRating.ACCEPTABLE, scores, notes
    
    async def _summary_quality_evaluation(
        self, scenario: BenchmarkScenario, response: str, expected: Optional[str]
    ) -> tuple[QualityRating, Dict[str, float], List[str]]:
        """Evaluate summary quality and completeness."""
        
        notes = []
        scores = {}
        
        # Check length appropriateness
        response_sentences = response.count('.') + response.count('!') + response.count('?')
        if 1 <= response_sentences <= 3:
            scores['length'] = 1.0
            notes.append("Appropriate summary length")
        else:
            scores['length'] = 0.5
            notes.append(f"Summary has {response_sentences} sentences")
        
        # Check for key terms (simplified)
        key_terms = ['machine learning', 'ai', 'artificial intelligence', 'algorithm']
        terms_found = sum(1 for term in key_terms if term.lower() in response.lower())
        scores['key_concepts'] = min(1.0, terms_found / 2)
        
        overall_score = sum(scores.values()) / len(scores)
        
        if overall_score >= 0.8:
            return QualityRating.EXCELLENT, scores, notes
        else:
            return QualityRating.GOOD, scores, notes
    
    async def _schema_validation_evaluation(
        self, scenario: BenchmarkScenario, response: str, expected: Optional[str]
    ) -> tuple[QualityRating, Dict[str, float], List[str]]:
        """Evaluate database schema design."""
        
        notes = []
        scores = {}
        
        # Check for key database concepts
        db_concepts = ['table', 'primary key', 'foreign key', 'index', 'relationship']
        concepts_found = sum(1 for concept in db_concepts if concept.lower() in response.lower())
        scores['db_concepts'] = min(1.0, concepts_found / 3)
        
        # Check for expected entities
        entities = ['user', 'product', 'order', 'inventory']
        entities_found = sum(1 for entity in entities if entity.lower() in response.lower())
        scores['entities'] = min(1.0, entities_found / 3)
        
        overall_score = sum(scores.values()) / len(scores)
        
        if overall_score >= 0.7:
            return QualityRating.GOOD, scores, notes
        else:
            return QualityRating.ACCEPTABLE, scores, notes
    
    async def _bug_fix_evaluation(
        self, scenario: BenchmarkScenario, response: str, expected: Optional[str]
    ) -> tuple[QualityRating, Dict[str, float], List[str]]:
        """Evaluate bug identification and fixing."""
        
        notes = []
        scores = {}
        
        # Check if base case is mentioned
        base_case_terms = ['base case', 'stopping condition', 'n == 0', 'n <= 1']
        if any(term in response.lower() for term in base_case_terms):
            scores['identifies_issue'] = 1.0
            notes.append("Correctly identifies missing base case")
        else:
            scores['identifies_issue'] = 0.0
            notes.append("Does not identify the core issue")
        
        # Check for correct fix
        if 'return 1' in response or 'n <= 1' in response:
            scores['provides_fix'] = 1.0
            notes.append("Provides correct fix")
        else:
            scores['provides_fix'] = 0.0
        
        overall_score = sum(scores.values()) / len(scores)
        
        if overall_score >= 0.8:
            return QualityRating.EXCELLENT, scores, notes
        else:
            return QualityRating.GOOD, scores, notes
    
    async def _manual_evaluation_placeholder(
        self, scenario: BenchmarkScenario, response: str, expected: Optional[str]
    ) -> tuple[QualityRating, Dict[str, float], List[str]]:
        """Placeholder for scenarios requiring manual evaluation."""
        
        # Simple heuristic evaluation based on response characteristics
        response_length = len(response.split())
        
        if response_length < 10:
            return QualityRating.POOR, {'completeness': 0.3}, ["Response too brief for manual evaluation"]
        elif response_length > 500:
            return QualityRating.ACCEPTABLE, {'completeness': 0.7}, ["Detailed response requiring manual review"]
        else:
            return QualityRating.ACCEPTABLE, {'completeness': 0.6}, ["Response requires manual evaluation"]


class BenchmarkSuite:
    """
    Comprehensive automated benchmarking system for model performance evaluation.
    """
    
    def __init__(self, 
                 model_router: Optional[ModelRouter] = None,
                 complexity_analyzer: Optional[ComplexityAnalyzer] = None):
        self.model_router = model_router
        self.complexity_analyzer = complexity_analyzer
        self.scenario_generator = ScenarioGenerator()
        self.evaluation_engine = EvaluationEngine()
        
        # Execution state
        self.running_benchmarks: Dict[str, asyncio.Task] = {}
        self.completed_reports: Dict[str, BenchmarkReport] = {}
        
        # Configuration
        self.default_timeout = 30  # seconds
        self.max_concurrent_benchmarks = 5
    
    async def run_benchmark(
        self,
        models: List[Model],
        scenarios: Optional[List[BenchmarkScenario]] = None,
        scenario_filters: Optional[Dict[str, Any]] = None,
        report_name: str = "Model Performance Benchmark"
    ) -> BenchmarkReport:
        """
        Run comprehensive benchmark across models and scenarios.
        
        Args:
            models: List of models to benchmark
            scenarios: Specific scenarios to run (None for default set)
            scenario_filters: Filters for scenario selection
            report_name: Name for the benchmark report
            
        Returns:
            Complete benchmark report with results and analysis
        """
        
        start_time = time.perf_counter()
        report_id = str(uuid.uuid4())
        
        # Initialize report
        report = BenchmarkReport(
            id=report_id,
            name=report_name,
            description=f"Benchmarking {len(models)} models across multiple scenarios"
        )
        
        logger.info(f"Starting benchmark {report_id} with {len(models)} models")
        
        try:
            # Select scenarios
            if scenarios is None:
                scenarios = self._select_scenarios(scenario_filters or {})
            
            logger.info(f"Running {len(scenarios)} scenarios per model")
            
            # Create all benchmark tasks
            all_tasks = []
            for model in models:
                for scenario in scenarios:
                    task_id = f"{model.name}_{scenario.id}"
                    task = asyncio.create_task(
                        self._run_single_benchmark(model, scenario, task_id)
                    )
                    all_tasks.append(task)
            
            # Execute benchmarks with concurrency control
            semaphore = asyncio.Semaphore(self.max_concurrent_benchmarks)
            
            async def run_with_semaphore(task):
                async with semaphore:
                    return await task
            
            # Wait for all benchmarks to complete
            results = await asyncio.gather(
                *[run_with_semaphore(task) for task in all_tasks],
                return_exceptions=True
            )
            
            # Process results
            for result in results:
                if isinstance(result, BenchmarkResult):
                    report.add_result(result)
                elif isinstance(result, Exception):
                    logger.error(f"Benchmark task failed: {result}")
            
            # Generate analysis
            report.execution_time_total_ms = (time.perf_counter() - start_time) * 1000
            report.generate_model_rankings()
            report.generate_insights()
            
            # Store report
            self.completed_reports[report_id] = report
            
            logger.info(f"Benchmark complete: {report.success_rate:.1%} success rate, "
                       f"{report.average_score:.2f} avg score")
            
            return report
            
        except Exception as e:
            logger.error(f"Benchmark execution failed: {e}")
            raise
    
    async def _run_single_benchmark(
        self,
        model: Model,
        scenario: BenchmarkScenario,
        task_id: str
    ) -> BenchmarkResult:
        """Execute a single benchmark scenario with a specific model."""
        
        result = BenchmarkResult(
            id=str(uuid.uuid4()),
            scenario_id=scenario.id,
            model_name=model.name,
            input_prompt=scenario.prompt,
            status=ExecutionStatus.RUNNING
        )
        
        logger.debug(f"Running benchmark {task_id}")
        
        try:
            # Prepare context and complexity analysis
            if self.complexity_analyzer:
                complexity = await self.complexity_analyzer.analyze(scenario.prompt)
            else:
                # Simple fallback complexity estimation
                complexity = self._estimate_complexity_simple(scenario)
            
            # Execute model inference
            start_inference = time.perf_counter()
            
            # For now, simulate model execution (in real implementation, call actual models)
            response = await self._simulate_model_execution(model, scenario, complexity)
            
            inference_time = (time.perf_counter() - start_inference) * 1000
            
            # Calculate performance metrics
            input_tokens = estimate_tokens(scenario.prompt)
            output_tokens = estimate_tokens(response)
            estimated_cost = model.estimate_cost(input_tokens, output_tokens) if hasattr(model, 'estimate_cost') else 0.0
            
            result.metrics = PerformanceMetrics(
                total_time_ms=inference_time,
                inference_time_ms=inference_time,
                tokens_per_second=output_tokens / (inference_time / 1000) if inference_time > 0 else 0,
                estimated_cost=estimated_cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            
            # Evaluate response quality
            quality_rating, eval_scores, eval_notes = await self.evaluation_engine.evaluate_response(
                scenario, response, scenario.expected_output
            )
            
            # Update result
            result.mark_completed(response, quality_rating)
            result.evaluation_scores = eval_scores
            result.evaluation_notes = eval_notes
            
            # Update performance metrics with evaluation scores
            if eval_scores:
                result.metrics.accuracy_score = eval_scores.get('accuracy', 0.0)
                result.metrics.quality_score = sum(eval_scores.values()) / len(eval_scores)
                result.metrics.completeness_score = eval_scores.get('completeness', 
                                                                   result.metrics.quality_score)
            
            logger.debug(f"Benchmark {task_id} completed: {quality_rating.value}")
            
        except asyncio.TimeoutError:
            result.status = ExecutionStatus.TIMEOUT
            result.error_message = f"Benchmark timed out after {self.default_timeout} seconds"
            logger.warning(f"Benchmark {task_id} timed out")
            
        except Exception as e:
            result.mark_failed(str(e), type(e).__name__)
            logger.error(f"Benchmark {task_id} failed: {e}")
        
        return result
    
    async def _simulate_model_execution(
        self,
        model: Model,
        scenario: BenchmarkScenario,
        complexity: ComplexityScore
    ) -> str:
        """
        Simulate model execution for testing purposes.
        In a real implementation, this would call the actual model APIs.
        """
        
        # Simulate processing time based on model characteristics
        base_latency = getattr(model, 'avg_latency_ms', 1000) / 1000
        await asyncio.sleep(base_latency + (complexity.raw_score * 0.5))
        
        # Generate simulated responses based on scenario type
        if scenario.scenario_type == ScenarioType.SIMPLE_QA:
            if "15 + 27" in scenario.prompt:
                return "42"
            elif "capital of France" in scenario.prompt:
                return "The capital of France is Paris."
            elif "mitosis and meiosis" in scenario.prompt:
                return "Mitosis is cell division that produces two identical diploid cells, while meiosis produces four genetically different haploid gametes for reproduction."
            else:
                return "This is a simulated response to a simple Q&A question."
        
        elif scenario.scenario_type == ScenarioType.CODE_GENERATION:
            if "prime" in scenario.prompt:
                return """```python
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```"""
            elif "binary search tree" in scenario.prompt:
                return """```python
class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        # Implementation here
        pass
    
    def search(self, value):
        # Implementation here
        pass
    
    def delete(self, value):
        # Implementation here
        pass
```"""
            else:
                return "```python\n# Simulated code generation response\nprint('Hello, World!')\n```"
        
        elif scenario.scenario_type == ScenarioType.REASONING:
            if "Alice, Bob, and Carol" in scenario.prompt:
                return "Let me work through this step by step:\n1. Carol likes green (given)\n2. Alice doesn't like red (given)\n3. Bob's favorite is not blue (given)\n\nSince Carol likes green, Alice and Bob must like red and blue. Since Alice doesn't like red, Alice likes blue. Therefore Bob likes red.\n\nAnswer: Alice likes blue, Bob likes red, Carol likes green."
            else:
                return "This requires logical reasoning to solve step by step."
        
        elif scenario.scenario_type == ScenarioType.CREATIVE_WRITING:
            return "Once upon a time, there was a robot who discovered the beauty of human emotions through its interactions with the world around it."
        
        elif scenario.scenario_type == ScenarioType.DEBUGGING:
            if "factorial" in scenario.prompt:
                return "The bug is a missing base case. The function will recurse infinitely. Fix: Add 'if n <= 1: return 1' at the beginning."
        
        else:
            return f"This is a simulated response for {scenario.scenario_type.value} scenario."
    
    def _estimate_complexity_simple(self, scenario: BenchmarkScenario) -> ComplexityScore:
        """Simple complexity estimation fallback."""
        from autom8.models.complexity import ComplexityScore, ComplexityDimensions, ComplexityTier, TaskPattern
        
        # Map difficulty to complexity score
        difficulty_scores = {
            DifficultyLevel.TRIVIAL: 0.1,
            DifficultyLevel.SIMPLE: 0.3,
            DifficultyLevel.MODERATE: 0.5,
            DifficultyLevel.COMPLEX: 0.7,
            DifficultyLevel.FRONTIER: 0.9
        }
        
        raw_score = difficulty_scores.get(scenario.difficulty_level, 0.5)
        
        # Map scenario type to task pattern
        type_patterns = {
            ScenarioType.CODE_GENERATION: TaskPattern.BOILERPLATE,
            ScenarioType.DEBUGGING: TaskPattern.DEBUGGING,
            ScenarioType.REASONING: TaskPattern.ANALYSIS,
            ScenarioType.CREATIVE_WRITING: TaskPattern.CREATIVE,
        }
        
        return ComplexityScore(
            raw_score=raw_score,
            confidence=0.8,
            dimensions=ComplexityDimensions(
                syntactic=raw_score * 0.8,
                semantic=raw_score,
                contextual=raw_score * 0.6,
                interdependency=raw_score * 0.4,
                domain=raw_score * 0.7
            ),
            recommended_tier=self._score_to_tier(raw_score),
            task_pattern=type_patterns.get(scenario.scenario_type, TaskPattern.UNKNOWN),
            reasoning=f"Estimated from scenario difficulty: {scenario.difficulty_level.value}",
            processing_time_ms=1.0
        )
    
    def _score_to_tier(self, score: float) -> 'ComplexityTier':
        """Convert score to complexity tier."""
        from autom8.models.complexity import ComplexityTier
        
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
    
    def _select_scenarios(self, filters: Dict[str, Any]) -> List[BenchmarkScenario]:
        """Select scenarios based on filters."""
        
        all_scenarios = self.scenario_generator.get_all_scenarios()
        
        # Apply filters
        filtered_scenarios = all_scenarios
        
        if 'scenario_types' in filters:
            types = filters['scenario_types']
            if not isinstance(types, list):
                types = [types]
            filtered_scenarios = [s for s in filtered_scenarios if s.scenario_type in types]
        
        if 'difficulty_levels' in filters:
            levels = filters['difficulty_levels']
            if not isinstance(levels, list):
                levels = [levels]
            filtered_scenarios = [s for s in filtered_scenarios if s.difficulty_level in levels]
        
        if 'tags' in filters:
            tags = filters['tags']
            if not isinstance(tags, list):
                tags = [tags]
            filtered_scenarios = [s for s in filtered_scenarios 
                                if any(tag in s.tags for tag in tags)]
        
        if 'max_scenarios' in filters:
            filtered_scenarios = filtered_scenarios[:filters['max_scenarios']]
        
        # If no scenarios match, return a basic set
        if not filtered_scenarios:
            filtered_scenarios = all_scenarios[:5]
        
        return filtered_scenarios
    
    async def run_quick_benchmark(
        self,
        models: List[Model],
        scenario_types: Optional[List[ScenarioType]] = None
    ) -> BenchmarkReport:
        """Run a quick benchmark with a subset of scenarios."""
        
        filters = {
            'max_scenarios': 10,
            'difficulty_levels': [DifficultyLevel.SIMPLE, DifficultyLevel.MODERATE]
        }
        
        if scenario_types:
            filters['scenario_types'] = scenario_types
        
        return await self.run_benchmark(
            models=models,
            scenario_filters=filters,
            report_name="Quick Performance Benchmark"
        )
    
    async def run_capability_assessment(
        self,
        model: Model,
        target_capability: float = 0.5
    ) -> BenchmarkReport:
        """Assess a model's capability level across different complexity tiers."""
        
        # Select scenarios around the target capability level
        scenarios = self.scenario_generator.get_scenarios_for_model(target_capability)
        
        return await self.run_benchmark(
            models=[model],
            scenarios=scenarios,
            report_name=f"Capability Assessment - {model.name}"
        )
    
    def get_report(self, report_id: str) -> Optional[BenchmarkReport]:
        """Get a completed benchmark report by ID."""
        return self.completed_reports.get(report_id)
    
    def list_reports(self) -> List[Dict[str, Any]]:
        """List all completed benchmark reports."""
        return [
            {
                'id': report.id,
                'name': report.name,
                'created_at': report.created_at,
                'total_scenarios': report.total_scenarios,
                'success_rate': report.success_rate,
                'average_score': report.average_score
            }
            for report in self.completed_reports.values()
        ]


# Global benchmark suite instance
_benchmark_suite: Optional[BenchmarkSuite] = None


async def get_benchmark_suite(
    model_router: Optional[ModelRouter] = None,
    complexity_analyzer: Optional[ComplexityAnalyzer] = None
) -> BenchmarkSuite:
    """Get global benchmark suite instance."""
    global _benchmark_suite
    
    if _benchmark_suite is None:
        _benchmark_suite = BenchmarkSuite(model_router, complexity_analyzer)
    
    return _benchmark_suite