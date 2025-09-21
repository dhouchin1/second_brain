"""
Enhanced LLM Service with Autom8 Integration

Provides intelligent AI model routing with fallback to existing Ollama functionality.
Maintains backward compatibility while adding cost optimization and better model selection.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from config import settings
from services.autom8_client import get_autom8_service, Autom8Request

logger = logging.getLogger(__name__)

class EnhancedLLMService:
    """Enhanced LLM service with Autom8 routing and Ollama fallback."""

    def __init__(self):
        self._autom8_service = None
        self._initialized = False

    async def _ensure_initialized(self):
        """Ensure the service is initialized."""
        if not self._initialized:
            self._autom8_service = await get_autom8_service()
            self._initialized = True

    async def summarize(self, text: str, prompt: Optional[str] = None,
                      context_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Summarize text and extract tags and actions using intelligent routing.

        Args:
            text: Text to summarize
            prompt: Optional custom prompt
            context_metadata: Optional metadata for better context optimization

        Returns:
            Dict with keys: summary, tags, actions, cost_info
        """
        logger.info(f"[enhanced_summarize] Called with text: {repr(text[:200])}")

        if not text or not text.strip():
            return {"summary": "", "tags": [], "actions": [], "cost_info": {}}

        await self._ensure_initialized()

        system_prompt = (
            prompt or
            "Summarize and extract tags and action items from this transcript of conversation snippet or note."
        )

        full_prompt = (
            f"{system_prompt}\n\n{text}\n\n"
            "Respond in JSON with keys 'summary', 'tags', and 'actions'."
        )

        try:
            # Optimize context based on metadata
            optimized_context = self._optimize_context(text, context_metadata, task_type="summarization")

            # Use Autom8 with intelligent routing
            response = await self._autom8_service.generate_with_fallback(
                prompt=full_prompt,
                context=optimized_context,
                task_type="summarization"
            )

            # Parse the JSON response
            content = response["content"]
            parsed_result = self._parse_json_response(content)

            # Add cost information
            parsed_result["cost_info"] = {
                "source": response["source"],
                "model_used": response["model_used"],
                "cost": response["cost"],
                "provider": response["provider"],
                "response_time": response["response_time"]
            }

            logger.info(f"Summarization completed via {response['source']} "
                       f"using {response['model_used']} (${response['cost']:.4f})")

            return parsed_result

        except Exception as e:
            logger.error(f"Enhanced summarization failed: {e}")
            # Fall back to legacy method
            return await self._legacy_summarize(text, prompt)

    async def generate_title(self, text: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a title for the given text using intelligent routing.

        Args:
            text: Text to generate title for
            prompt: Optional custom prompt

        Returns:
            Dict with keys: title, cost_info
        """
        logger.info(f"[enhanced_generate_title] Called with text: {repr(text[:200])}")

        if not text or not text.strip():
            return {"title": "", "cost_info": {}}

        await self._ensure_initialized()

        system_prompt = (
            prompt or
            "Generate a concise, descriptive title for this content. "
            "Focus on the main topic or key insight. Keep it under 10 words."
        )

        full_prompt = f"{system_prompt}\n\nContent:\n{text}\n\nTitle:"

        try:
            # Use Autom8 with intelligent routing
            response = await self._autom8_service.generate_with_fallback(
                prompt=full_prompt,
                context=text[:500],  # Shorter context for title generation
                task_type="title_generation"
            )

            # Clean up the title
            title = response["content"].strip()
            title = title.replace('"', '').replace("Title:", "").strip()

            result = {
                "title": title,
                "cost_info": {
                    "source": response["source"],
                    "model_used": response["model_used"],
                    "cost": response["cost"],
                    "provider": response["provider"],
                    "response_time": response["response_time"]
                }
            }

            logger.info(f"Title generation completed via {response['source']} "
                       f"using {response['model_used']} (${response['cost']:.4f})")

            return result

        except Exception as e:
            logger.error(f"Enhanced title generation failed: {e}")
            # Fall back to legacy method
            return await self._legacy_generate_title(text, prompt)

    async def generate_tags(self, text: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate relevant tags for the given text.

        Args:
            text: Text to generate tags for
            prompt: Optional custom prompt

        Returns:
            Dict with keys: tags, cost_info
        """
        logger.info(f"[enhanced_generate_tags] Called with text: {repr(text[:200])}")

        if not text or not text.strip():
            return {"tags": [], "cost_info": {}}

        await self._ensure_initialized()

        system_prompt = (
            prompt or
            "Generate 3-7 relevant tags for this content. "
            "Focus on topics, themes, and key concepts. "
            "Return as a JSON array of strings."
        )

        full_prompt = f"{system_prompt}\n\nContent:\n{text}\n\nTags (JSON array):"

        try:
            response = await self._autom8_service.generate_with_fallback(
                prompt=full_prompt,
                context=text[:500],
                task_type="tagging"
            )

            # Parse tags
            content = response["content"].strip()
            tags = self._parse_tags_response(content)

            result = {
                "tags": tags,
                "cost_info": {
                    "source": response["source"],
                    "model_used": response["model_used"],
                    "cost": response["cost"],
                    "provider": response["provider"],
                    "response_time": response["response_time"]
                }
            }

            logger.info(f"Tag generation completed via {response['source']} "
                       f"using {response['model_used']} (${response['cost']:.4f})")

            return result

        except Exception as e:
            logger.error(f"Enhanced tag generation failed: {e}")
            return {"tags": [], "cost_info": {"error": str(e)}}

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON response from LLM, handling various formats."""
        try:
            parsed = json.loads(content)
            return {
                "summary": parsed.get("summary", ""),
                "tags": parsed.get("tags", []),
                "actions": parsed.get("actions", [])
            }
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end >= start:
                try:
                    candidate = content[start:end + 1]
                    parsed = json.loads(candidate)
                    return {
                        "summary": parsed.get("summary", ""),
                        "tags": parsed.get("tags", []),
                        "actions": parsed.get("actions", [])
                    }
                except json.JSONDecodeError:
                    pass

            # Fallback: return content as summary
            logger.warning("Could not parse JSON response, using as summary")
            return {
                "summary": content.strip(),
                "tags": [],
                "actions": []
            }

    def _parse_tags_response(self, content: str) -> List[str]:
        """Parse tags from LLM response."""
        try:
            # Try to parse as JSON array
            tags = json.loads(content)
            if isinstance(tags, list):
                return [str(tag).strip() for tag in tags if tag]
        except json.JSONDecodeError:
            pass

        # Try to extract array from response
        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1:
            try:
                tags_str = content[start:end + 1]
                tags = json.loads(tags_str)
                if isinstance(tags, list):
                    return [str(tag).strip() for tag in tags if tag]
            except json.JSONDecodeError:
                pass

        # Fallback: split by common delimiters
        content = content.replace("[", "").replace("]", "").replace('"', "")
        tags = [tag.strip() for tag in content.split(",") if tag.strip()]
        return tags[:7]  # Limit to 7 tags

    def _optimize_context(self, text: str, metadata: Optional[Dict[str, Any]] = None,
                         task_type: str = "general") -> str:
        """
        Optimize context for better AI model selection and performance.

        Args:
            text: Original text content
            metadata: Optional metadata about the content
            task_type: Type of task being performed

        Returns:
            Optimized context string
        """
        if not metadata:
            # Default context optimization based on content
            if task_type == "summarization":
                return text[:1500]  # Longer context for summarization
            elif task_type == "title_generation":
                return text[:500]   # Shorter context for titles
            elif task_type == "tagging":
                return text[:800]   # Medium context for tagging
            else:
                return text[:1000]  # Default context

        # Enhanced context optimization using metadata
        context_parts = []

        # Add content type information
        if metadata.get("content_type"):
            context_parts.append(f"Content Type: {metadata['content_type']}")

        # Add source information
        if metadata.get("source"):
            context_parts.append(f"Source: {metadata['source']}")

        # Add timestamp if available
        if metadata.get("timestamp"):
            context_parts.append(f"Date: {metadata['timestamp']}")

        # Add user context if available
        if metadata.get("user_context"):
            context_parts.append(f"Context: {metadata['user_context']}")

        # Add domain/topic hints
        if metadata.get("domain"):
            context_parts.append(f"Domain: {metadata['domain']}")

        # Calculate dynamic context length based on complexity
        complexity = self._estimate_complexity(text, metadata)
        if complexity == "high":
            max_content_length = 2000
        elif complexity == "medium":
            max_content_length = 1200
        else:
            max_content_length = 800

        # Combine metadata and content
        metadata_str = " | ".join(context_parts)
        content_preview = text[:max_content_length]

        if metadata_str:
            return f"{metadata_str}\n\nContent Preview:\n{content_preview}"
        else:
            return content_preview

    def _estimate_complexity(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Estimate content complexity for better model selection.

        Returns:
            Complexity level: "low", "medium", or "high"
        """
        # Length-based complexity
        if len(text) > 5000:
            base_complexity = "high"
        elif len(text) > 2000:
            base_complexity = "medium"
        else:
            base_complexity = "low"

        # Content-based complexity indicators
        complexity_indicators = [
            "analysis", "research", "technical", "academic", "scientific",
            "algorithm", "methodology", "framework", "architecture",
            "implementation", "optimization", "evaluation"
        ]

        complex_words = sum(1 for word in complexity_indicators if word in text.lower())

        # Adjust based on metadata
        if metadata:
            # Academic or technical content is typically more complex
            if metadata.get("domain") in ["academic", "technical", "scientific", "research"]:
                if base_complexity == "low":
                    base_complexity = "medium"
                elif base_complexity == "medium":
                    base_complexity = "high"

            # Code or documentation content
            if metadata.get("content_type") in ["code", "documentation", "api"]:
                if base_complexity == "low":
                    base_complexity = "medium"

        # Final adjustment based on complexity indicators
        if complex_words >= 3 and base_complexity == "low":
            return "medium"
        elif complex_words >= 6 and base_complexity == "medium":
            return "high"

        return base_complexity

    async def _legacy_summarize(self, text: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Fallback to legacy Ollama summarization."""
        try:
            from llm_utils import ollama_summarize
            result = ollama_summarize(text, prompt)
            result["cost_info"] = {
                "source": "ollama_legacy",
                "model_used": settings.ollama_model,
                "cost": 0.0,
                "provider": "ollama",
                "response_time": 0.0
            }
            return result
        except Exception as e:
            logger.error(f"Legacy summarization failed: {e}")
            return {
                "summary": "",
                "tags": [],
                "actions": [],
                "cost_info": {"error": str(e)}
            }

    async def _legacy_generate_title(self, text: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Fallback to legacy Ollama title generation."""
        try:
            from llm_utils import ollama_generate_title
            title = ollama_generate_title(text, prompt)
            return {
                "title": title,
                "cost_info": {
                    "source": "ollama_legacy",
                    "model_used": settings.ollama_model,
                    "cost": 0.0,
                    "provider": "ollama",
                    "response_time": 0.0
                }
            }
        except Exception as e:
            logger.error(f"Legacy title generation failed: {e}")
            return {
                "title": "",
                "cost_info": {"error": str(e)}
            }

    async def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        if not self._initialized:
            return {"status": "not_initialized"}

        try:
            autom8_stats = await self._autom8_service.get_stats()
            return {
                "status": "initialized",
                "autom8_stats": autom8_stats
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Global service instance
_enhanced_llm_service = None

async def get_enhanced_llm_service() -> EnhancedLLMService:
    """Get or create the global enhanced LLM service instance."""
    global _enhanced_llm_service
    if _enhanced_llm_service is None:
        _enhanced_llm_service = EnhancedLLMService()
    return _enhanced_llm_service

# Async wrapper functions for backward compatibility
async def enhanced_ollama_summarize(text: str, prompt: Optional[str] = None) -> Dict[str, Any]:
    """Async version of ollama_summarize with Autom8 integration."""
    service = await get_enhanced_llm_service()
    return await service.summarize(text, prompt)

async def enhanced_ollama_generate_title(text: str, prompt: Optional[str] = None) -> str:
    """Async version of ollama_generate_title with Autom8 integration."""
    service = await get_enhanced_llm_service()
    result = await service.generate_title(text, prompt)
    return result.get("title", "")

async def enhanced_ollama_generate_tags(text: str, prompt: Optional[str] = None) -> List[str]:
    """Generate tags using Autom8 integration."""
    service = await get_enhanced_llm_service()
    result = await service.generate_tags(text, prompt)
    return result.get("tags", [])