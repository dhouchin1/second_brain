# ──────────────────────────────────────────────────────────────────────────────
# File: services/search_router.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Consolidated search router that replaces all redundant search endpoints from app.py.
Combines functionality from:
- /api/search/enhanced
- /api/search  
- /api/search/hybrid
- /api/search/suggestions
- /api/search/enhance
"""
from __future__ import annotations
import os
import json
import time
import re
from typing import Optional, Dict, List, Any
from functools import lru_cache

from fastapi import APIRouter, Depends, Query, Body
from pydantic import BaseModel

from services.search_adapter import SearchService
# get_conn will be passed from app.py context
from services.auth_service import User
from config import settings

# Global variables to hold functions from app.py context
get_conn = None
get_current_user = None

class SearchRequest(BaseModel):
    query: str
    filters: Optional[dict] = {}
    limit: int = 20


class SearchSuggestion(BaseModel):
    text: str
    type: str
    icon: str
    source: Optional[str] = None
    score: Optional[str] = None
    count: Optional[int] = None


class EnhancementResponse(BaseModel):
    enhanced_query: str
    alternative_queries: List[str] = []
    suggested_tags: List[str] = []
    intent: str = "search"
    query_type: str = "hybrid"
    spelling_corrections: List[str] = []
    expansion_terms: List[str] = []


# Initialize router
router = APIRouter(prefix="/api/search", tags=["search"])

def init_search_router(get_conn_func, get_current_user_func):
    """Initialize search router with functions from app.py"""
    global get_conn, get_current_user
    get_conn = get_conn_func
    get_current_user = get_current_user_func


# ─── Utility Functions ──────────────────────────────────────────────────────

def _get_search_service() -> SearchService:
    """Get SearchService instance"""
    from services.search_adapter import SearchService
    db_path = os.getenv('SQLITE_DB', str(settings.db_path))
    return SearchService(db_path=db_path, vec_ext_path=os.getenv('SQLITE_VEC_PATH'))


def _resolve_search_mode(filters: Optional[dict]) -> str:
    """Resolve search mode from filters"""
    mode = 'hybrid'
    if filters and isinstance(filters, dict):
        t = filters.get('type') or filters.get('mode')
        if t in {'fts', 'keyword'}:
            mode = 'keyword'
        elif t in {'semantic', 'vector'}:
            mode = 'semantic'
        elif t in {'hybrid', 'both'}:
            mode = 'hybrid'
    return mode


def _fuzzy_match(query: str, text: str, threshold: float = 0.6) -> bool:
    """Simple fuzzy matching based on character overlap"""
    if len(query) < 3 or len(text) < 3:
        return False
    
    query_chars = set(query.lower())
    text_chars = set(text.lower())
    overlap = len(query_chars.intersection(text_chars))
    return overlap / len(query_chars) >= threshold


def _extract_phrases(content: str, query: str, max_phrases: int = 2) -> List[str]:
    """Extract meaningful phrases containing the query from content"""
    phrases = []
    sentences = re.split(r'[.!?]+', content)
    
    for sentence in sentences[:10]:  # Check first 10 sentences
        sentence = sentence.strip()
        if query.lower() in sentence.lower() and len(sentence) > 10:
            # Clean up the sentence
            words = sentence.split()
            if len(words) > 15:  # Truncate long sentences
                # Find query position and extract around it
                query_pos = -1
                for i, word in enumerate(words):
                    if query.lower() in word.lower():
                        query_pos = i
                        break
                
                if query_pos != -1:
                    start = max(0, query_pos - 5)
                    end = min(len(words), query_pos + 8)
                    phrase = " ".join(words[start:end])
                else:
                    phrase = " ".join(words[:12])
            else:
                phrase = sentence
            
            if len(phrase) <= 80 and phrase not in phrases:
                phrases.append(phrase)
                
            if len(phrases) >= max_phrases:
                break
    
    return phrases


def _basic_query_enhancement(query: str, tags: set, searches: list) -> Dict[str, Any]:
    """Basic query enhancement without LLM"""
    from difflib import get_close_matches
    
    # Find similar tags
    suggested_tags = []
    if tags:
        for tag in tags:
            if any(word.lower() in tag.lower() for word in query.split()):
                suggested_tags.append(tag)
    
    # Find similar searches  
    similar_searches = get_close_matches(query, searches, n=3, cutoff=0.6)
    
    # Basic intent detection
    intent = "search"
    query_lower = query.lower()
    if any(word in query_lower for word in ["how", "why", "what", "when", "where"]):
        intent = "research"
    elif any(word in query_lower for word in ["remember", "recall", "find"]):
        intent = "recall"
    elif any(word in query_lower for word in ["plan", "todo", "schedule"]):
        intent = "planning"
    
    return {
        "enhanced_query": query,
        "alternative_queries": similar_searches,
        "suggested_tags": suggested_tags[:5],
        "intent": intent,
        "query_type": "hybrid",
        "expansion_terms": query.split()
    }


def _spell_correct_query(query: str, reference_texts: list) -> str:
    """Simple spell correction using reference texts"""
    from difflib import get_close_matches
    
    words = query.split()
    corrected_words = []
    
    # Create a set of reference words
    reference_words = set()
    for text in reference_texts:
        if text:
            reference_words.update(text.lower().split())
    
    for word in words:
        if len(word) > 3:  # Only correct longer words
            matches = get_close_matches(word.lower(), reference_words, n=1, cutoff=0.8)
            if matches:
                corrected_words.append(matches[0])
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
    
    return " ".join(corrected_words)


# ─── Search Endpoints ───────────────────────────────────────────────────────

@router.post("/enhanced")
async def enhanced_search(
    request: SearchRequest,
    current_user: User = Depends(get_current_user)
):
    """Enhanced search delegating to the unified SearchService with optional filters"""
    svc = _get_search_service()
    f = request.filters or {}
    mode = _resolve_search_mode(f)
    rows = svc.search(request.query, mode=mode, k=request.limit or 20)
    notes = [{k: row[k] for k in row.keys()} for row in rows]
    
    # Apply simple filters client-side to match legacy endpoint
    if 'tags' in f and f['tags']:
        tag_q = str(f['tags']).strip()
        notes = [n for n in notes if tag_q in (n.get('tags') or '')]
    if 'type' in f and f['type'] not in {'fts','keyword','semantic','vector','hybrid','both'}:
        notes = [n for n in notes if n.get('type') == f['type']]
    
    return {
        "results": notes,
        "total": len(notes),
        "query": request.query,
        "mode": mode
    }


@router.post("")
@router.post("/")
async def search(
    request: SearchRequest,
    current_user: User = Depends(get_current_user)
):
    """Advanced search with filters and semantic similarity via SearchService"""
    svc = _get_search_service()
    mode = _resolve_search_mode(request.filters)
    rows = svc.search(request.query, mode=mode, k=request.limit or 20)
    # Convert sqlite3.Row to dict
    notes = [{k: row[k] for k in row.keys()} for row in rows]
    return {
        "results": notes,
        "total": len(notes),
        "query": request.query,
        "mode": mode
    }


@router.post("/hybrid")
async def hybrid_search(
    request: dict,
    current_user: User = Depends(get_current_user)
):
    """Hybrid search endpoint for Advanced Search interface"""
    svc = _get_search_service()
    
    # Extract search parameters from request
    query = request.get('query', '')
    search_type = request.get('search_type', 'hybrid')
    fts_weight = request.get('fts_weight', 0.4)
    semantic_weight = request.get('semantic_weight', 0.6)
    min_fts_score = request.get('min_fts_score', 0.1)
    min_semantic_score = request.get('min_semantic_score', 0.1)
    limit = request.get('limit', 20)
    
    # Get filter parameters
    date_start = request.get('date_start')
    date_end = request.get('date_end') 
    tags = request.get('tags', [])
    types = request.get('types', [])
    status = request.get('status', [])
    
    # Determine search mode
    mode = search_type if search_type in ['fts', 'semantic', 'hybrid'] else 'hybrid'
    
    # Perform search
    rows = svc.search(query, mode=mode, k=limit)
    
    # Convert to dict format expected by frontend
    results = []
    for row in rows:
        result = {k: row[k] for k in row.keys()}
        
        # Add score information for frontend display
        if mode == 'hybrid' or mode == 'fts':
            # Simulate FTS score if not present
            result['fts_score'] = result.get('fts_score', 0.5)
        if mode == 'hybrid' or mode == 'semantic':
            # Simulate semantic score if not present  
            result['semantic_score'] = result.get('semantic_score', 0.5)
        if mode == 'hybrid':
            # Calculate combined score
            fts = result.get('fts_score', 0)
            semantic = result.get('semantic_score', 0)
            result['combined_score'] = (fts * fts_weight) + (semantic * semantic_weight)
        
        # Create snippet from content/summary
        content = result.get('content', '') or result.get('summary', '')
        if content and len(content) > 200:
            result['snippet'] = content[:200] + '...'
        else:
            result['snippet'] = content
            
        results.append(result)
    
    # Filter by minimum scores if specified
    if min_fts_score > 0 or min_semantic_score > 0:
        filtered_results = []
        for r in results:
            fts_ok = r.get('fts_score', 0) >= min_fts_score
            sem_ok = r.get('semantic_score', 0) >= min_semantic_score
            if mode == 'fts' and fts_ok:
                filtered_results.append(r)
            elif mode == 'semantic' and sem_ok:
                filtered_results.append(r)
            elif mode == 'hybrid' and (fts_ok or sem_ok):
                filtered_results.append(r)
        results = filtered_results
    
    return {
        "results": results,
        "total": len(results),
        "search_type": mode,
        "query": query
    }


@router.get("/suggestions")
async def search_suggestions(
    q: str = Query(..., description="Search query for suggestions"),
    current_user: User = Depends(get_current_user)
):
    """Get intelligent search suggestions with AI-powered enhancements"""
    if len(q.strip()) < 2:
        return {"suggestions": []}
    
    # Cache key includes user ID and query
    cache_key = f"suggestions_{current_user.id}_{q.lower()}"
    
    # Simple in-memory cache with TTL (5 minutes)
    if not hasattr(search_suggestions, '_cache'):
        search_suggestions._cache = {}
        search_suggestions._cache_times = {}
    
    current_time = time.time()
    if cache_key in search_suggestions._cache:
        cache_time = search_suggestions._cache_times.get(cache_key, 0)
        if current_time - cache_time < 300:  # 5 minutes TTL
            return search_suggestions._cache[cache_key]
    
    conn = get_conn()
    c = conn.cursor()
    
    suggestions = []
    
    # 1. Recent searches by this user
    recent_searches = c.execute(
        """SELECT DISTINCT query FROM search_history 
           WHERE user_id = ? AND query LIKE ? 
           ORDER BY created_at DESC LIMIT 3""",
        (current_user.id, f"%{q}%")
    ).fetchall()
    suggestions.extend([{"text": row["query"], "type": "recent", "icon": "🕐"} for row in recent_searches])
    
    # 2. Smart title suggestions with fuzzy matching
    title_matches = c.execute(
        """SELECT DISTINCT title, created_at FROM notes 
           WHERE user_id = ? AND (title LIKE ? OR title LIKE ?) 
           AND title != '' ORDER BY created_at DESC LIMIT 6""",
        (current_user.id, f"%{q}%", f"%{q.replace(' ', '%')}%")
    ).fetchall()
    suggestions.extend([{"text": row["title"], "type": "title", "icon": "📄"} for row in title_matches])
    
    # 3. Intelligent tag suggestions
    tag_matches = c.execute(
        "SELECT DISTINCT tags FROM notes WHERE user_id = ? AND tags != '' LIMIT 20",
        (current_user.id,)
    ).fetchall()
    
    all_tags = set()
    for row in tag_matches:
        tags = [t.strip() for t in (row["tags"] or "").split(",") if t.strip()]
        all_tags.update(tags)
    
    # Find tags that match query with fuzzy matching
    matching_tags = []
    q_lower = q.lower()
    for tag in all_tags:
        tag_lower = tag.lower()
        if (q_lower in tag_lower or 
            any(q_part in tag_lower for q_part in q_lower.split()) or
            _fuzzy_match(q_lower, tag_lower)):
            matching_tags.append(tag)
    
    suggestions.extend([{"text": f"tag:{tag}", "type": "tag", "icon": "🏷️"} for tag in sorted(matching_tags)[:4]])
    
    # 4. Content-based phrase suggestions
    content_matches = c.execute(
        """SELECT DISTINCT content, title FROM notes 
           WHERE user_id = ? AND content LIKE ? AND content != '' 
           ORDER BY created_at DESC LIMIT 4""",
        (current_user.id, f"%{q}%")
    ).fetchall()
    
    for row in content_matches:
        content = row["content"] or ""
        title = row["title"] or "Untitled"
        # Extract meaningful phrases around the query
        phrases = _extract_phrases(content, q)
        for phrase in phrases[:2]:  # Limit to 2 phrases per note
            suggestions.append({
                "text": phrase, 
                "type": "phrase", 
                "icon": "💭",
                "source": title[:30] + "..." if len(title) > 30 else title
            })
    
    # 5. Semantic suggestions if available
    try:
        svc = _get_search_service()
        if svc._vec_table_exists():
            semantic_results = svc.search(q, mode='semantic', k=3)
            for result in semantic_results:
                if result['title'] and result['title'].lower() != q.lower():
                    suggestions.append({
                        "text": result['title'], 
                        "type": "semantic", 
                        "icon": "🧠",
                        "score": f"{result.get('score', 0):.2f}"
                    })
    except Exception as e:
        print(f"Semantic suggestions error: {e}")
    
    # 6. Popular queries (global suggestions)
    popular_queries = c.execute(
        """SELECT query, COUNT(*) as count FROM search_history 
           WHERE query LIKE ? 
           GROUP BY query ORDER BY count DESC LIMIT 3""",
        (f"%{q}%",)
    ).fetchall()
    suggestions.extend([{"text": row["query"], "type": "popular", "icon": "🔥", "count": row["count"]} for row in popular_queries])
    
    conn.close()
    
    # Remove duplicates while preserving order and type info
    seen = set()
    unique_suggestions = []
    for sugg in suggestions:
        text = sugg["text"].lower()
        if text not in seen and len(unique_suggestions) < 12:
            seen.add(text)
            unique_suggestions.append(sugg)
    
    result = {"suggestions": unique_suggestions}
    
    # Cache the result
    search_suggestions._cache[cache_key] = result
    search_suggestions._cache_times[cache_key] = current_time
    
    return result


@router.post("/enhance")
async def enhance_search_query(
    query: str = Body(..., embed=True),
    current_user: User = Depends(get_current_user)
):
    """AI-powered query enhancement and expansion"""
    if not query or len(query.strip()) < 2:
        return {"enhanced_query": query, "suggestions": [], "intent": "unknown"}
    
    try:
        # Get user's recent notes context for better enhancement
        conn = get_conn()
        c = conn.cursor()
        
        # Get recent note titles and tags for context
        recent_notes = c.execute(
            """SELECT title, tags FROM notes 
               WHERE user_id = ? AND (title != '' OR tags != '') 
               ORDER BY created_at DESC LIMIT 20""",
            (current_user.id,)
        ).fetchall()
        
        # Get user's search history for intent analysis
        search_history = c.execute(
            """SELECT DISTINCT query FROM search_history 
               WHERE user_id = ? ORDER BY created_at DESC LIMIT 10""",
            (current_user.id,)
        ).fetchall()
        
        conn.close()
        
        # Build context for LLM
        context_notes = [row["title"] for row in recent_notes if row["title"]]
        context_tags = set()
        for row in recent_notes:
            if row["tags"]:
                tags = [t.strip() for t in row["tags"].split(",") if t.strip()]
                context_tags.update(tags)
        
        context_searches = [row["query"] for row in search_history]
        
        # Try LLM enhancement
        try:
            from llm_utils import ollama_summarize
            
            # Create enhancement prompt
            enhancement_prompt = f"""
You are a search query enhancement assistant. Given a search query, enhance it to be more effective while preserving the user's intent.

User's Query: "{query}"

Context from user's notes:
- Recent note titles: {context_notes[:10]}
- Available tags: {list(context_tags)[:15]}
- Recent searches: {context_searches[:5]}

Provide enhancements in JSON format:
{{
    "enhanced_query": "improved version of the query",
    "alternative_queries": ["alternative phrasing 1", "alternative phrasing 2"],
    "suggested_tags": ["relevant", "tags"],
    "intent": "search_intent_category",
    "query_type": "keyword|semantic|hybrid",
    "spelling_corrections": ["corrections if needed"],
    "expansion_terms": ["related", "terms", "to", "include"]
}}

Intent categories: research, recall, reference, planning, analysis, creative, troubleshooting, learning
Query types: keyword (exact matches), semantic (meaning-based), hybrid (both)
"""

            # Call LLM for enhancement
            llm_result = ollama_summarize(enhancement_prompt, 
                "Enhance the search query using the provided context. Return only valid JSON.")
            
            if isinstance(llm_result, dict) and "summary" in llm_result:
                # Parse LLM response
                try:
                    enhancement_data = json.loads(llm_result["summary"])
                except (json.JSONDecodeError, TypeError):
                    # Fallback to basic enhancement
                    enhancement_data = _basic_query_enhancement(query, context_tags, context_searches)
            else:
                enhancement_data = _basic_query_enhancement(query, context_tags, context_searches)
        except ImportError:
            # LLM not available, use basic enhancement
            enhancement_data = _basic_query_enhancement(query, context_tags, context_searches)
        
        # Add automatic spelling correction
        corrected_query = _spell_correct_query(query, context_notes + context_searches)
        if corrected_query != query:
            enhancement_data.setdefault("spelling_corrections", []).append(corrected_query)
        
        # Ensure required fields
        enhancement_data.setdefault("enhanced_query", query)
        enhancement_data.setdefault("alternative_queries", [])
        enhancement_data.setdefault("suggested_tags", [])
        enhancement_data.setdefault("intent", "search")
        enhancement_data.setdefault("query_type", "hybrid")
        enhancement_data.setdefault("expansion_terms", [])
        
        return enhancement_data
        
    except Exception as e:
        print(f"Query enhancement error: {e}")
        # Return basic enhancement as fallback
        return {
            "enhanced_query": query,
            "alternative_queries": [query],
            "suggested_tags": [],
            "intent": "search",
            "query_type": "hybrid",
            "expansion_terms": []
        }