import json
import requests
from typing import List, Dict, Optional
import logging
from pydantic import ValidationError
from services.memory_service import MemoryService
from services.model_manager import ModelManager, ModelTask
from services.security_utils import ExtractionResult, sanitize_for_log

logger = logging.getLogger(__name__)

class MemoryExtractionService:
    """Extract memories from conversations using Ollama"""

    def __init__(
        self,
        memory_service: MemoryService,
        ollama_url: str,
        model_manager: ModelManager
    ):
        self.memory = memory_service
        self.ollama_url = ollama_url
        self.model_manager = model_manager
        logger.info("MemoryExtractionService initialized")

    def extract_from_conversation(
        self,
        user_id: int,
        conversation: List[Dict[str, str]],
        model_override: Optional[str] = None
    ) -> Dict:
        """
        Extract episodic and semantic memories from conversation

        Args:
            user_id: User ID
            conversation: List of {'role': 'user/assistant', 'content': '...'}
            model_override: Optional model to use instead of default

        Returns:
            {'episodic': [...], 'semantic': [...], 'extraction_metadata': {...}}
        """

        if len(conversation) < 2:
            logger.debug("Conversation too short for extraction")
            return {'episodic': [], 'semantic': []}

        # Format conversation for prompt
        conv_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in conversation[-10:]  # Last 10 messages max
        ])

        prompt = self._build_extraction_prompt(conv_text)

        # Get model for extraction
        model = model_override or self.model_manager.get_model_for_task(ModelTask.MEMORY_EXTRACTION)
        logger.debug(f"Using model '{model}' for memory extraction")

        # Call Ollama
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.3,  # Lower temperature for consistent extraction
                        "num_predict": 1000
                    }
                },
                timeout=30
            )

            if response.status_code != 200:
                logger.error(f"Ollama request failed: {response.status_code}")
                return {'episodic': [], 'semantic': []}

            # Parse and validate LLM response
            try:
                raw_result = json.loads(response.json()['response'])
                # Validate using Pydantic model
                validated_result = ExtractionResult(**raw_result)
                result = validated_result.dict()
                logger.info(
                    f"Extracted {len(result.get('episodic', []))} episodic, "
                    f"{len(result.get('semantic', []))} semantic memories"
                )
            except ValidationError as e:
                logger.error(f"LLM extraction validation failed: {e}")
                logger.debug(f"Raw LLM response: {sanitize_for_log(str(raw_result), 200)}")
                return {'episodic': [], 'semantic': []}

        except requests.Timeout:
            logger.error("Memory extraction timed out")
            return {'episodic': [], 'semantic': []}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse extraction JSON: {e}")
            return {'episodic': [], 'semantic': []}
        except Exception as e:
            logger.error(f"Memory extraction failed: {e}")
            return {'episodic': [], 'semantic': []}

        # Store extracted memories
        self._store_memories(user_id, result, conv_text)

        return result

    def _build_extraction_prompt(self, conversation_text: str) -> str:
        """Build prompt for memory extraction"""
        return f"""Analyze this conversation and extract important information.

Conversation:
{conversation_text}

Extract:
1. EPISODIC memories - specific things that happened, actions taken, decisions made, outcomes
2. SEMANTIC facts - stable facts about the user (preferences, knowledge, skills, context)

Rules:
- Only extract meaningful, specific information
- Episodic: Focus on what happened, not general statements
- Semantic: Focus on enduring facts about the user
- Rate importance/confidence honestly
- Skip trivial or unclear information

Respond with ONLY valid JSON:
{{
  "episodic": [
    {{
      "summary": "User completed project X with result Y",
      "importance": 0.7,
      "context": "Work project completion"
    }}
  ],
  "semantic": [
    {{
      "fact": "User prefers Python for data analysis",
      "confidence": 0.9,
      "category": "preference"
    }},
    {{
      "fact": "User works as a data scientist",
      "confidence": 1.0,
      "category": "context"
    }}
  ]
}}

Categories: preference, knowledge, context, skill"""

    def _store_memories(self, user_id: int, extraction_result: Dict, conversation_text: str):
        """Store extracted memories in database"""

        # Store episodic memories
        for episode in extraction_result.get('episodic', []):
            try:
                self.memory.add_episodic_memory(
                    user_id=user_id,
                    content=conversation_text[:1000],  # Truncate long conversations
                    summary=episode.get('summary', ''),
                    importance=float(episode.get('importance', 0.5)),
                    context=episode.get('context', '')
                )
            except Exception as e:
                logger.error(f"Failed to store episodic memory: {e}")

        # Store semantic memories
        for fact_data in extraction_result.get('semantic', []):
            try:
                fact = fact_data.get('fact', '')
                if not fact:
                    continue

                # Check for similar existing facts to avoid duplicates
                similar = self.memory.search_semantic(
                    user_id=user_id,
                    query=fact,
                    limit=1
                )

                # If very similar fact exists (high FTS rank), skip or update
                if similar and similar[0].get('rank', 0) < -10:  # Strong match
                    logger.debug(f"Skipping duplicate semantic fact: {fact[:50]}")
                    continue

                self.memory.add_semantic_memory(
                    user_id=user_id,
                    fact=fact,
                    category=fact_data.get('category', 'general'),
                    confidence=float(fact_data.get('confidence', 1.0)),
                    source='conversation_extraction'
                )
            except Exception as e:
                logger.error(f"Failed to store semantic memory: {e}")
