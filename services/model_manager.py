from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ModelTask(Enum):
    """Different tasks requiring different models"""
    CHAT = "chat"
    MEMORY_EXTRACTION = "memory_extraction"
    SUMMARIZATION = "summarization"
    TITLE_GENERATION = "title_generation"

@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    context_length: int
    supports_json: bool
    speed: str  # 'fast', 'medium', 'slow'
    quality: str  # 'low', 'medium', 'high'
    description: str = ""

class ModelManager:
    """
    Manages different Ollama models for different tasks
    Allows runtime model switching
    """

    # Recommended models for different tasks
    RECOMMENDED_MODELS = {
        ModelTask.CHAT: [
            ModelConfig("llama3.2", 128000, False, "fast", "medium",
                       "Fast general-purpose chat model"),
            ModelConfig("llama3.1:8b", 128000, True, "medium", "high",
                       "High-quality chat with JSON support"),
            ModelConfig("qwen2.5:7b", 32768, True, "medium", "high",
                       "Excellent reasoning and coding"),
        ],
        ModelTask.MEMORY_EXTRACTION: [
            ModelConfig("llama3.1:8b", 128000, True, "medium", "high",
                       "Best for structured extraction"),
            ModelConfig("mistral:7b", 32768, True, "medium", "high",
                       "Good balance of speed and quality"),
            ModelConfig("qwen2.5:7b", 32768, True, "medium", "high",
                       "Strong analytical capabilities"),
        ],
        ModelTask.SUMMARIZATION: [
            ModelConfig("llama3.2", 128000, False, "fast", "medium",
                       "Fast summarization"),
            ModelConfig("llama3.1:8b", 128000, False, "medium", "high",
                       "High-quality summaries"),
        ],
        ModelTask.TITLE_GENERATION: [
            ModelConfig("llama3.2", 128000, False, "fast", "medium",
                       "Quick title generation"),
        ]
    }

    def __init__(self):
        self.model_overrides: Dict[ModelTask, str] = {}
        logger.info("ModelManager initialized")

    def get_model_for_task(self, task: ModelTask) -> str:
        """Get the best model for a specific task"""

        # Check for override
        if task in self.model_overrides:
            model = self.model_overrides[task]
            logger.debug(f"Using override model '{model}' for task {task.value}")
            return model

        # Return first recommended model
        recommended = self.RECOMMENDED_MODELS.get(task, [])
        if recommended:
            model = recommended[0].name
            logger.debug(f"Using recommended model '{model}' for task {task.value}")
            return model

        # Fallback to default
        logger.warning(f"No recommended model for task {task.value}, using default")
        return "llama3.2"

    def set_model_for_task(self, task: ModelTask, model_name: str):
        """Override model for a specific task"""
        self.model_overrides[task] = model_name
        logger.info(f"Model override set: {task.value} -> {model_name}")

    def clear_override(self, task: ModelTask):
        """Remove override for a task"""
        if task in self.model_overrides:
            del self.model_overrides[task]
            logger.info(f"Cleared override for task {task.value}")

    def get_recommended_models(self, task: ModelTask) -> List[ModelConfig]:
        """Get list of recommended models for a task"""
        return self.RECOMMENDED_MODELS.get(task, [])

    def is_model_suitable(self, model_name: str, task: ModelTask) -> bool:
        """Check if a model is suitable for a task"""
        recommended = self.RECOMMENDED_MODELS.get(task, [])
        return any(m.name == model_name for m in recommended)

    def get_all_overrides(self) -> Dict[str, str]:
        """Get all current overrides"""
        return {task.value: model for task, model in self.model_overrides.items()}

# Global instance
_model_manager = None

def get_model_manager() -> ModelManager:
    """Get global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
