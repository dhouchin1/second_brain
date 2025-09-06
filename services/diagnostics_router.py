"""
Diagnostics Router for Second Brain

Provides lightweight diagnostics endpoints for search/index status.
"""
import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pathlib import Path

from config import settings
from services.search_index import SearchIndexer, SearchConfig

get_conn = None
get_current_user = None  # optional dependency for authenticated endpoints

router = APIRouter(prefix="/api/diagnostics", tags=["diagnostics"])


def init_diagnostics_router(get_conn_func, get_current_user_func: Optional[callable] = None):
    global get_conn, get_current_user
    get_conn = get_conn_func
    get_current_user = get_current_user_func


@router.get("/health")
async def health():
    return {
        "service": "diagnostics",
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/search")
async def search_status():
    """Report FTS and vector index presence and basic counts."""
    try:
        conn = get_conn()
        cur = conn.cursor()

        # notes count
        notes_count = cur.execute("SELECT COUNT(*) FROM notes").fetchone()[0]

        # FTS
        fts_exists = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='notes_fts'"
        ).fetchone() is not None
        fts_rows = 0
        if fts_exists:
            try:
                fts_rows = cur.execute("SELECT COUNT(*) FROM notes_fts").fetchone()[0]
            except Exception:
                fts_rows = None

        # sqlite-vec note vectors
        vec_exists = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='note_vecs'"
        ).fetchone() is not None
        vec_rows = 0
        if vec_exists:
            try:
                vec_rows = cur.execute("SELECT COUNT(*) FROM note_vecs").fetchone()[0]
            except Exception:
                vec_rows = None

        # JSON fallback embeddings table
        emb_exists = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'"
        ).fetchone() is not None
        emb_rows = 0
        if emb_exists:
            try:
                emb_rows = cur.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            except Exception:
                emb_rows = None

        # Extension presence hint
        vec_path = os.getenv("SQLITE_VEC_PATH")

        return {
            "notes": {"count": notes_count},
            "fts": {"exists": fts_exists, "rows": fts_rows},
            "vectors": {"exists": vec_exists, "rows": vec_rows},
            "embeddings_fallback": {"exists": emb_exists, "rows": emb_rows},
            "env": {"SQLITE_VEC_PATH": bool(vec_path)},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostics failed: {e}")


@router.post("/reindex")
async def reindex(embeddings: bool = Query(True, description="Rebuild embeddings as well as FTS")):
    """Trigger a best-effort index rebuild (FTS and optionally embeddings)."""
    try:
        cfg = SearchConfig(
            db_path=Path(settings.db_path),
            embed_model=getattr(settings, 'auto_seeding_embed_model', 'all-MiniLM-L6-v2'),
            enable_embeddings=embeddings,
            ollama_url=settings.ollama_api_url.replace('/api/generate', '')
        )
        indexer = SearchIndexer(cfg)
        indexer.ensure_fts()
        vec_available = indexer.ensure_vec()
        result = indexer.rebuild_all(embeddings=embeddings and vec_available)
        result.update({
            "vec_available": vec_available,
            "embeddings_requested": embeddings,
        })
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reindex failed: {e}")
