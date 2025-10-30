# ──────────────────────────────────────────────────────────────────────────────
# File: api/routes_chat.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Memory-augmented chat API endpoints
Provides conversational interface with episodic and semantic memory support
"""
from __future__ import annotations
import os
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import requests

from services.memory_service import MemoryService
from services.memory_consolidation_service import get_consolidation_queue
from services.search_adapter import SearchService
from services.model_manager import get_model_manager, ModelTask
from services.security_utils import (
    sanitize_prompt_input,
    sanitize_for_log,
    ExtractionResult,
    verify_user_access
)
from database import get_db_connection
from services.embeddings import get_embeddings_service
from config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])

# ─── Request/Response Models ─────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_id: int
    message: str
    session_id: Optional[str] = None
    conversation_history: List[ChatMessage] = []
    use_memory: bool = True
    model_override: Optional[str] = None

class ModelOverrideRequest(BaseModel):
    task: str  # 'chat', 'memory_extraction', 'summarization'
    model_name: str

# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/query")
async def chat_with_memory(request: ChatRequest):
    """
    Chat endpoint with memory augmentation
    """
    settings = get_settings()

    try:
        db = get_db_connection()
        embeddings = get_embeddings_service()
        memory = MemoryService(db, embeddings)
        search = SearchService(db_path=settings.db_path, vec_ext_path=os.getenv('SQLITE_VEC_PATH'))
        model_manager = get_model_manager()

        # Start or continue conversation session
        if not request.session_id:
            session_id = memory.start_conversation(request.user_id)
            logger.info(f"Started new session {session_id} for user {request.user_id}")
        else:
            session_id = request.session_id

        # Sanitize user input to prevent prompt injection
        try:
            sanitized_message = sanitize_prompt_input(request.message)
        except ValueError as e:
            raise HTTPException(400, f"Invalid input: {str(e)}")

        # Log sanitized version only
        logger.info(f"Chat query from user {request.user_id}: {sanitize_for_log(sanitized_message)}")

        # Add user message to session
        memory.add_message(session_id, "user", sanitized_message)

        # Search with memory augmentation
        search_results = search.search_with_memory(
            user_id=request.user_id,
            query=sanitized_message,
            mode="hybrid",
            limit=10,
            include_memory=request.use_memory
        )

        # Build enhanced prompt for Ollama
        system_prompt = search_results['context_summary']

        if not system_prompt:
            system_prompt = "You are a helpful AI assistant."

        # Get appropriate model
        model = request.model_override or model_manager.get_model_for_task(ModelTask.CHAT)

        # Build full prompt
        full_prompt = f"""{system_prompt}

Current query: {request.message}

Respond naturally and helpfully. If using information from the knowledge base, cite sources using [1], [2] format. If relevant past interactions or user preferences inform your response, acknowledge them naturally."""

        # Call Ollama for response
        logger.debug(f"Calling Ollama with model '{model}'")

        try:
            ollama_response = requests.post(
                settings.ollama_api_url,
                json={
                    "model": model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 2000
                    }
                },
                timeout=60
            )

            if ollama_response.status_code != 200:
                raise HTTPException(500, f"LLM generation failed: {ollama_response.status_code}")

            assistant_message = ollama_response.json()['response']

        except requests.Timeout:
            raise HTTPException(504, "LLM request timed out")
        except requests.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            raise HTTPException(500, f"LLM generation failed: {str(e)}")

        # Add assistant response to session
        memory.add_message(session_id, "assistant", assistant_message)

        # Get conversation for extraction
        full_conversation = memory.get_conversation(session_id)

        # Background: extract and store memories if conversation is long enough
        if len(full_conversation) >= settings.memory_extraction_threshold:
            try:
                consolidation_queue = get_consolidation_queue()
                consolidation_queue.enqueue(
                    user_id=request.user_id,
                    conversation=full_conversation
                )
                logger.debug(f"Enqueued conversation for memory extraction")
            except Exception as e:
                logger.warning(f"Failed to enqueue memory extraction: {e}")

        return {
            "response": assistant_message,
            "session_id": session_id,
            "sources": search_results['documents'][:5],
            "memory_used": {
                "episodic_count": len(search_results['episodic']),
                "semantic_count": len(search_results['semantic']),
                "documents_count": len(search_results['documents'])
            },
            "model_used": model
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(500, f"Internal server error: {str(e)}")

@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get conversation session details"""
    try:
        db = get_db_connection()
        embeddings = get_embeddings_service()
        memory = MemoryService(db, embeddings)

        messages = memory.get_conversation(session_id)

        return {
            "session_id": session_id,
            "message_count": len(messages),
            "messages": messages
        }
    except Exception as e:
        logger.error(f"Get session error: {e}")
        raise HTTPException(500, str(e))

@router.get("/memory/profile/{user_id}")
async def get_user_memory_profile(user_id: int):
    """Get user's memory profile"""
    try:
        db = get_db_connection()
        embeddings = get_embeddings_service()
        memory = MemoryService(db, embeddings)

        semantic_facts = memory.get_all_user_facts(user_id)
        recent_episodes = memory.get_recent_episodes(user_id, limit=10)

        # Group semantic facts by category
        facts_by_category = {}
        for fact in semantic_facts:
            category = fact['category']
            if category not in facts_by_category:
                facts_by_category[category] = []
            facts_by_category[category].append(fact)

        return {
            "user_id": user_id,
            "semantic_facts_count": len(semantic_facts),
            "semantic_facts": semantic_facts[:20],  # Limit for API response
            "facts_by_category": facts_by_category,
            "recent_episodes_count": len(recent_episodes),
            "recent_episodes": recent_episodes
        }
    except Exception as e:
        logger.error(f"Get memory profile error: {e}")
        raise HTTPException(500, str(e))

@router.post("/memory/semantic/add")
async def add_semantic_fact(
    user_id: int,
    fact: str,
    category: str = "general",
    confidence: float = 1.0
):
    """Manually add a semantic fact"""
    try:
        db = get_db_connection()
        embeddings = get_embeddings_service()
        memory = MemoryService(db, embeddings)

        fact_id = memory.add_semantic_memory(
            user_id=user_id,
            fact=fact,
            category=category,
            confidence=confidence,
            source='manual'
        )

        return {
            "status": "success",
            "fact_id": fact_id
        }
    except Exception as e:
        logger.error(f"Add semantic fact error: {e}")
        raise HTTPException(500, str(e))

@router.delete("/memory/semantic/{fact_id}")
async def delete_semantic_fact(fact_id: str):
    """Delete a semantic fact"""
    try:
        db = get_db_connection()
        embeddings = get_embeddings_service()
        memory = MemoryService(db, embeddings)

        memory.delete_semantic_memory(fact_id)

        return {"status": "success"}
    except Exception as e:
        logger.error(f"Delete semantic fact error: {e}")
        raise HTTPException(500, str(e))

@router.get("/models")
async def get_available_models():
    """Get information about available models"""
    model_manager = get_model_manager()

    return {
        "current_assignments": model_manager.get_all_overrides(),
        "recommended_models": {
            "chat": [m.__dict__ for m in model_manager.get_recommended_models(ModelTask.CHAT)],
            "memory_extraction": [m.__dict__ for m in model_manager.get_recommended_models(ModelTask.MEMORY_EXTRACTION)],
            "summarization": [m.__dict__ for m in model_manager.get_recommended_models(ModelTask.SUMMARIZATION)]
        }
    }

@router.post("/models/override")
async def set_model_override(request: ModelOverrideRequest):
    """Override model for a specific task"""
    try:
        model_manager = get_model_manager()

        # Map string to enum
        task_map = {
            'chat': ModelTask.CHAT,
            'memory_extraction': ModelTask.MEMORY_EXTRACTION,
            'summarization': ModelTask.SUMMARIZATION,
            'title_generation': ModelTask.TITLE_GENERATION
        }

        if request.task not in task_map:
            raise HTTPException(400, f"Invalid task: {request.task}")

        task = task_map[request.task]
        model_manager.set_model_for_task(task, request.model_name)

        logger.info(f"Model override set: {request.task} -> {request.model_name}")

        return {
            "status": "success",
            "task": request.task,
            "model": request.model_name
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Set model override error: {e}")
        raise HTTPException(500, str(e))

@router.get("/queue/status")
async def get_queue_status():
    """Get memory consolidation queue status"""
    try:
        queue = get_consolidation_queue()
        return {
            "queue_size": queue.get_queue_size(),
            "status": "running" if queue.running else "stopped"
        }
    except RuntimeError:
        return {
            "queue_size": 0,
            "status": "not_initialized"
        }
