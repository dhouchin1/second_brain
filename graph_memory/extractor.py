"""Simple fact extraction helpers for graph memory ingestion."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from config import settings
from llm_utils import ollama_generate

logger = logging.getLogger(__name__)


@dataclass
class FactCandidate:
    """Represents a semantic triple extracted from text."""

    subject: str
    predicate: str
    object: str
    confidence: float = 0.6
    object_type: str = "string"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "object_type": self.object_type,
        }


class GraphFactExtractor:
    """Lightweight extractor that prefers LLM output with heuristic fallback."""

    def __init__(self, *, use_llm: Optional[bool] = None, max_facts: Optional[int] = None) -> None:
        self.use_llm = settings.graph_memory_use_llm if use_llm is None else use_llm
        self.max_facts = max_facts or settings.graph_memory_max_facts

    def extract(self, text: str) -> List[Dict[str, Any]]:
        if not text or not text.strip():
            return []

        facts: List[FactCandidate] = []
        if self.use_llm and settings.ai_processing_enabled:
            llm_facts = self._extract_via_llm(text)
            if llm_facts:
                facts.extend(llm_facts)

        if not facts:
            facts.extend(self._fallback_extract(text))

        # Trim and serialise
        serialised = [fact.to_dict() for fact in facts[: self.max_facts]]
        logger.debug("GraphFactExtractor produced %s facts", len(serialised))
        return serialised

    # ------------------------------------------------------------------
    # LLM extraction
    # ------------------------------------------------------------------
    def _extract_via_llm(self, text: str) -> List[FactCandidate]:
        """Extract facts using local Ollama models only.

        If Ollama fails, we notify the user and give them options to select
        from multiple available models or fall back to heuristic extraction.
        """
        limit = self.max_facts
        prompt = (
            "Extract up to {limit} factual triples from the text below.\\n"
            "Return ONLY a JSON array of objects with keys subject, predicate, object,"
            " object_type (string|number|entity), and confidence (0-1).\\n"
            "Keep predicates short in snake_case. Only include verifiable, atomic facts."
        ).format(limit=limit)
        request = f"{prompt}\n\nTEXT:\n{text.strip()}\n"

        # Try available Ollama models
        models_to_try = self._get_available_ollama_models()

        for model in models_to_try:
            try:
                response = self._try_ollama_model(model, request)
                if response:
                    payload = self._parse_json_array(response)
                    facts = self._process_llm_response(payload, limit)
                    if facts:
                        logger.debug(f"LLM extraction succeeded with {model} model: {len(facts)} facts")
                        return facts
            except Exception as exc:
                logger.debug(f"Ollama extraction failed with model {model}: {exc}")
                continue

        # If all Ollama models failed, notify user and try fallback
        logger.warning("All Ollama models failed for fact extraction. Using heuristic fallback.")
        self._notify_llm_failure(text)
        return []

    def _get_available_ollama_models(self) -> List[str]:
        """Get list of available Ollama models to try."""
        # Primary models to try in order
        default_models = [
            "llama3.1",  # Default model
            "llama3.2",  # Alternative version
            "llama3",    # Base llama3
            "mistral",   # Alternative model
            "codellama", # Code-focused model
        ]

        # Add any custom models from environment
        custom_models = os.getenv("OLLAMA_MODELS", "")
        if custom_models:
            custom_list = [m.strip() for m in custom_models.split(",") if m.strip()]
            default_models = custom_list + default_models

        return default_models

    def _try_ollama_model(self, model: str, request: str) -> Optional[str]:
        """Try to generate response with a specific Ollama model."""
        try:
            # Temporarily patch the model in settings
            from config import settings
            original_model = getattr(settings, 'ollama_model', 'llama3.2')
            settings.ollama_model = model

            response = ollama_generate(request)

            # Restore original model
            settings.ollama_model = original_model

            return response
        except Exception:
            # Restore original model even on failure
            from config import settings
            original_model = getattr(settings, 'ollama_model', 'llama3.2')
            settings.ollama_model = original_model
            raise

    def _notify_llm_failure(self, text: str) -> None:
        """Notify user when LLM extraction fails and give options."""
        # Create a notification for the user
        try:
            from services.notification_service import get_notification_service
            notification_service = get_notification_service()

            message = (
                "Graph memory fact extraction failed with all available Ollama models. "
                "The system has fallen back to heuristic extraction, which may be less accurate. "
                "Consider:\n"
                "1. Starting Ollama service if not running\n"
                "2. Pulling additional models: `ollama pull llama3.1`\n"
                "3. Setting OLLAMA_MODELS environment variable with custom model list\n"
                "4. Checking system resources (CPU/memory)"
            )

            # Try to send notification (will be silent if notification service unavailable)
            try:
                notification_service.send_notification(
                    title="LLM Fact Extraction Failed",
                    message=message,
                    notification_type="warning",
                    metadata={
                        "text_length": len(text),
                        "available_models": self._get_available_ollama_models(),
                        "fallback_used": "heuristic"
                    }
                )
            except Exception:
                # If notification fails, just log the warning
                logger.warning(message)

        except ImportError:
            # If notification service not available, just log
            logger.warning("LLM fact extraction failed - all Ollama models unavailable")
        except Exception as e:
            logger.error(f"Failed to send LLM failure notification: {e}")

    def _process_llm_response(self, payload: List[Dict[str, Any]], limit: int) -> List[FactCandidate]:
        """Process LLM response payload into FactCandidate objects."""
        facts: List[FactCandidate] = []
        for item in payload[:limit]:
            subject = str(item.get("subject", "")).strip()
            predicate = str(item.get("predicate", "")).strip()
            obj = str(item.get("object", "")).strip()
            if not subject or not predicate or not obj:
                continue
            confidence = float(item.get("confidence", 0.7))
            obj_type = str(item.get("object_type", "string"))
            facts.append(FactCandidate(subject, predicate, obj, confidence, obj_type))
        return facts

    def _parse_json_array(self, payload: str) -> List[Dict[str, Any]]:
        try:
            data = json.loads(payload)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            # Attempt to salvage JSON substring
            start = payload.find("[")
            end = payload.rfind("]")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(payload[start : end + 1])
                except json.JSONDecodeError:
                    logger.debug("Failed to parse JSON substring from LLM output")
        return []

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------
    def _fallback_extract(self, text: str) -> List[FactCandidate]:
        sentences = re.split(r"(?<=[.!?])\\s+", text)
        facts: List[FactCandidate] = []
        for sentence in sentences:
            snippet = sentence.strip()
            if not snippet:
                continue
            # Simple pattern: X is Y, X are Y
            match = re.match(r"([^,]+?)\\s+(is|are|was|were)\\s+(.*)", snippet, re.IGNORECASE)
            if match:
                subject, verb, obj = match.groups()
                subject = subject.strip().rstrip(" :")
                obj = obj.strip().rstrip(" .")
                if subject and obj:
                    facts.append(
                        FactCandidate(
                            subject=subject,
                            predicate=verb.lower(),
                            object=obj,
                            confidence=0.35,
                            object_type="string",
                        )
                    )
            if len(facts) >= self.max_facts:
                break
        return facts
