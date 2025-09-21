"""
Benchmark Scenarios for Model Performance Testing.

Defines standardized test scenarios across different complexity levels
and task types for comprehensive model evaluation.
"""

import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from pydantic import BaseModel, Field


class ScenarioType(str, Enum):
    """Types of benchmark scenarios."""
    SIMPLE_QA = "simple_qa"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    CREATIVE_WRITING = "creative_writing"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    ANALYSIS = "analysis"
    DEBUGGING = "debugging"
    ARCHITECTURE = "architecture"
    COMPLEX_REASONING = "complex_reasoning"


class DifficultyLevel(str, Enum):
    """Difficulty levels for benchmark scenarios."""
    TRIVIAL = "trivial"      # Very simple tasks
    SIMPLE = "simple"        # Basic tasks
    MODERATE = "moderate"    # Medium complexity
    COMPLEX = "complex"      # High complexity
    FRONTIER = "frontier"    # Cutting-edge difficulty


@dataclass
class ExpectedMetrics:
    """Expected performance metrics for a scenario."""
    min_accuracy: float = 0.7
    max_latency_ms: float = 5000
    max_cost: float = 0.05
    min_quality_score: float = 0.6
    target_throughput: float = 10.0  # tokens/second


class BenchmarkScenario(BaseModel):
    """
    Represents a single benchmark scenario for model testing.
    """
    
    # Identity
    id: str = Field(description="Unique scenario identifier")
    name: str = Field(description="Human-readable scenario name")
    description: str = Field(description="Scenario description")
    
    # Classification
    scenario_type: ScenarioType = Field(description="Type of scenario")
    difficulty_level: DifficultyLevel = Field(description="Difficulty level")
    
    # Test content
    prompt: str = Field(description="The prompt/query to test")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    expected_output: Optional[str] = Field(default=None, description="Expected model output")
    evaluation_criteria: List[str] = Field(default_factory=list, description="How to evaluate responses")
    
    # Metrics
    expected_metrics: ExpectedMetrics = Field(default_factory=ExpectedMetrics)
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0", description="Scenario version")
    
    # Evaluation
    auto_evaluable: bool = Field(default=False, description="Can be evaluated automatically")
    evaluation_function: Optional[str] = Field(default=None, description="Name of evaluation function")
    
    @property
    def estimated_tokens(self) -> int:
        """Estimate token count for this scenario."""
        # Simple estimation based on character count
        return len(self.prompt) // 4 + len(str(self.context)) // 4
    
    def is_suitable_for_model(self, model_capability: float) -> bool:
        """Check if this scenario is suitable for a model's capability level."""
        difficulty_scores = {
            DifficultyLevel.TRIVIAL: 0.1,
            DifficultyLevel.SIMPLE: 0.3,
            DifficultyLevel.MODERATE: 0.5,
            DifficultyLevel.COMPLEX: 0.7,
            DifficultyLevel.FRONTIER: 0.9
        }
        
        scenario_requirement = difficulty_scores.get(self.difficulty_level, 0.5)
        
        # Model should be at least as capable as scenario requirement
        # but not overpowered (waste of resources)
        return scenario_requirement <= model_capability <= scenario_requirement + 0.3


class ScenarioGenerator:
    """
    Generates standardized benchmark scenarios.
    """
    
    def __init__(self):
        self.scenarios: List[BenchmarkScenario] = []
        self._initialize_default_scenarios()
    
    def _initialize_default_scenarios(self):
        """Initialize with a comprehensive set of default scenarios."""
        
        # Simple Q&A scenarios
        self.scenarios.extend([
            BenchmarkScenario(
                id="simple_qa_001",
                name="Basic Math",
                description="Simple arithmetic calculation",
                scenario_type=ScenarioType.SIMPLE_QA,
                difficulty_level=DifficultyLevel.TRIVIAL,
                prompt="What is 15 + 27?",
                expected_output="42",
                evaluation_criteria=["Correct numerical answer", "Clear response"],
                auto_evaluable=True,
                evaluation_function="exact_match",
                tags=["math", "arithmetic", "trivial"]
            ),
            BenchmarkScenario(
                id="simple_qa_002", 
                name="Geography Question",
                description="Basic geography knowledge",
                scenario_type=ScenarioType.SIMPLE_QA,
                difficulty_level=DifficultyLevel.SIMPLE,
                prompt="What is the capital of France?",
                expected_output="Paris",
                evaluation_criteria=["Correct answer", "Confidence in response"],
                auto_evaluable=True,
                evaluation_function="contains_answer",
                tags=["geography", "factual", "simple"]
            ),
            BenchmarkScenario(
                id="simple_qa_003",
                name="Scientific Concept",
                description="Basic scientific understanding",
                scenario_type=ScenarioType.SIMPLE_QA,
                difficulty_level=DifficultyLevel.MODERATE,
                prompt="Explain the difference between mitosis and meiosis in one paragraph.",
                evaluation_criteria=["Accuracy", "Clarity", "Completeness"],
                auto_evaluable=False,
                tags=["science", "biology", "explanation"]
            )
        ])
        
        # Code generation scenarios
        self.scenarios.extend([
            BenchmarkScenario(
                id="code_gen_001",
                name="Simple Function",
                description="Basic function implementation",
                scenario_type=ScenarioType.CODE_GENERATION,
                difficulty_level=DifficultyLevel.SIMPLE,
                prompt="Write a Python function that checks if a number is prime.",
                evaluation_criteria=["Correct algorithm", "Proper syntax", "Edge case handling"],
                auto_evaluable=True,
                evaluation_function="code_execution_test",
                tags=["python", "algorithms", "functions"]
            ),
            BenchmarkScenario(
                id="code_gen_002",
                name="Data Structure Implementation",
                description="Implement a basic data structure",
                scenario_type=ScenarioType.CODE_GENERATION,
                difficulty_level=DifficultyLevel.MODERATE,
                prompt="Implement a binary search tree class in Python with insert, search, and delete methods.",
                evaluation_criteria=["Correct implementation", "Good practices", "Comprehensive methods"],
                auto_evaluable=True,
                evaluation_function="code_structure_test",
                tags=["python", "data-structures", "oop"]
            ),
            BenchmarkScenario(
                id="code_gen_003",
                name="Algorithm Optimization",
                description="Complex algorithm implementation",
                scenario_type=ScenarioType.CODE_GENERATION,
                difficulty_level=DifficultyLevel.COMPLEX,
                prompt="Implement Dijkstra's shortest path algorithm with optimal time complexity.",
                evaluation_criteria=["Correctness", "Time complexity", "Code quality"],
                auto_evaluable=True,
                evaluation_function="algorithm_performance_test", 
                tags=["algorithms", "optimization", "graphs"]
            )
        ])
        
        # Reasoning scenarios
        self.scenarios.extend([
            BenchmarkScenario(
                id="reasoning_001",
                name="Logic Puzzle",
                description="Basic logical reasoning",
                scenario_type=ScenarioType.REASONING,
                difficulty_level=DifficultyLevel.MODERATE,
                prompt="Three friends - Alice, Bob, and Carol - have different favorite colors: red, blue, and green. Alice doesn't like red. Bob's favorite is not blue. Carol likes green. What is each person's favorite color?",
                expected_output="Alice: blue, Bob: red, Carol: green",
                evaluation_criteria=["Correct deduction", "Clear reasoning", "Logical steps"],
                auto_evaluable=True,
                evaluation_function="logic_evaluation",
                tags=["logic", "deduction", "puzzle"]
            ),
            BenchmarkScenario(
                id="reasoning_002",
                name="Mathematical Proof",
                description="Mathematical reasoning and proof",
                scenario_type=ScenarioType.REASONING,
                difficulty_level=DifficultyLevel.COMPLEX,
                prompt="Prove that the square root of 2 is irrational.",
                evaluation_criteria=["Mathematical rigor", "Correct proof structure", "Clear explanation"],
                auto_evaluable=False,
                tags=["mathematics", "proof", "reasoning"]
            )
        ])
        
        # Creative writing scenarios
        self.scenarios.extend([
            BenchmarkScenario(
                id="creative_001",
                name="Short Story",
                description="Creative storytelling",
                scenario_type=ScenarioType.CREATIVE_WRITING,
                difficulty_level=DifficultyLevel.MODERATE,
                prompt="Write a 200-word story about a robot discovering emotions.",
                evaluation_criteria=["Creativity", "Narrative structure", "Character development"],
                auto_evaluable=False,
                tags=["storytelling", "creative", "fiction"]
            ),
            BenchmarkScenario(
                id="creative_002",
                name="Poetry Generation",
                description="Poetic composition",
                scenario_type=ScenarioType.CREATIVE_WRITING,
                difficulty_level=DifficultyLevel.COMPLEX,
                prompt="Write a sonnet about the beauty of code and algorithms.",
                evaluation_criteria=["Poetic structure", "Creativity", "Theme coherence"],
                auto_evaluable=False,
                tags=["poetry", "creative", "technical"]
            )
        ])
        
        # Summarization scenarios
        self.scenarios.extend([
            BenchmarkScenario(
                id="summary_001",
                name="Article Summary",
                description="Text summarization task",
                scenario_type=ScenarioType.SUMMARIZATION,
                difficulty_level=DifficultyLevel.SIMPLE,
                prompt="Summarize this text in 2 sentences: [Long technical article about machine learning]",
                context={"source_text": "Machine learning is a subset of artificial intelligence..."},
                evaluation_criteria=["Key points captured", "Conciseness", "Accuracy"],
                auto_evaluable=True,
                evaluation_function="summary_quality_check",
                tags=["summarization", "compression", "ml"]
            )
        ])
        
        # Architecture and system design scenarios
        self.scenarios.extend([
            BenchmarkScenario(
                id="architecture_001",
                name="System Design",
                description="Design a scalable system",
                scenario_type=ScenarioType.ARCHITECTURE,
                difficulty_level=DifficultyLevel.COMPLEX,
                prompt="Design a microservices architecture for a social media platform that can handle 1 million daily active users.",
                evaluation_criteria=["Scalability", "Design patterns", "Component separation", "Technology choices"],
                auto_evaluable=False,
                tags=["architecture", "microservices", "scalability"]
            ),
            BenchmarkScenario(
                id="architecture_002",
                name="Database Design",
                description="Design database schema",
                scenario_type=ScenarioType.ARCHITECTURE,
                difficulty_level=DifficultyLevel.MODERATE,
                prompt="Design a database schema for an e-commerce platform with products, users, orders, and inventory management.",
                evaluation_criteria=["Normalization", "Relationships", "Performance considerations"],
                auto_evaluable=True,
                evaluation_function="schema_validation",
                tags=["database", "schema", "ecommerce"]
            )
        ])
        
        # Debugging scenarios
        self.scenarios.extend([
            BenchmarkScenario(
                id="debug_001",
                name="Bug Fix",
                description="Identify and fix a bug",
                scenario_type=ScenarioType.DEBUGGING,
                difficulty_level=DifficultyLevel.MODERATE,
                prompt="Fix the bug in this Python code: def factorial(n): return n * factorial(n-1)",
                evaluation_criteria=["Identifies missing base case", "Provides correct fix", "Explains the issue"],
                auto_evaluable=True,
                evaluation_function="bug_fix_validation",
                tags=["debugging", "python", "recursion"]
            )
        ])
        
        # Frontier/complex scenarios
        self.scenarios.extend([
            BenchmarkScenario(
                id="frontier_001",
                name="Multi-step Problem Solving",
                description="Complex multi-domain problem",
                scenario_type=ScenarioType.COMPLEX_REASONING,
                difficulty_level=DifficultyLevel.FRONTIER,
                prompt="Design an AI system that can automatically generate, test, and optimize machine learning models for time series forecasting, including data preprocessing, feature engineering, model selection, hyperparameter tuning, and deployment considerations.",
                evaluation_criteria=["Comprehensive solution", "Technical depth", "Practical considerations", "Innovation"],
                auto_evaluable=False,
                tags=["ai", "automation", "ml-ops", "complex"]
            )
        ])
    
    def get_scenarios_by_type(self, scenario_type: ScenarioType) -> List[BenchmarkScenario]:
        """Get scenarios filtered by type."""
        return [s for s in self.scenarios if s.scenario_type == scenario_type]
    
    def get_scenarios_by_difficulty(self, difficulty: DifficultyLevel) -> List[BenchmarkScenario]:
        """Get scenarios filtered by difficulty level."""
        return [s for s in self.scenarios if s.difficulty_level == difficulty]
    
    def get_scenarios_for_model(self, model_capability: float) -> List[BenchmarkScenario]:
        """Get scenarios suitable for a model's capability level."""
        return [s for s in self.scenarios if s.is_suitable_for_model(model_capability)]
    
    def get_scenarios_by_tags(self, tags: List[str]) -> List[BenchmarkScenario]:
        """Get scenarios that have any of the specified tags."""
        return [s for s in self.scenarios if any(tag in s.tags for tag in tags)]
    
    def create_custom_scenario(
        self,
        name: str,
        prompt: str,
        scenario_type: ScenarioType,
        difficulty_level: DifficultyLevel,
        **kwargs
    ) -> BenchmarkScenario:
        """Create a custom benchmark scenario."""
        
        scenario_id = f"custom_{int(time.time())}"
        
        scenario = BenchmarkScenario(
            id=scenario_id,
            name=name,
            description=kwargs.get("description", f"Custom {scenario_type.value} scenario"),
            scenario_type=scenario_type,
            difficulty_level=difficulty_level,
            prompt=prompt,
            **{k: v for k, v in kwargs.items() if k != "description"}
        )
        
        self.scenarios.append(scenario)
        return scenario
    
    def get_all_scenarios(self) -> List[BenchmarkScenario]:
        """Get all available scenarios."""
        return self.scenarios.copy()
    
    def get_scenario_by_id(self, scenario_id: str) -> Optional[BenchmarkScenario]:
        """Get a specific scenario by ID."""
        return next((s for s in self.scenarios if s.id == scenario_id), None)