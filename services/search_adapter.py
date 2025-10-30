# ──────────────────────────────────────────────────────────────────────────────
# File: services/search_adapter.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Search adapter over SQLite FTS5 + (optional) sqlite-vec.
- Loads sqlite-vec extension if SQLITE_VEC_PATH is set.
- Runs migrations from db/migrations/*
- Provides keyword, semantic, and hybrid search.
"""
from __future__ import annotations
import os
import sqlite3
import json
from pathlib import Path
from typing import Optional

from services.embeddings import Embeddings

MIGRATIONS = [
    Path('db/migrations/001_core.sql'),
    Path('db/migrations/002_vec.sql'),
    Path('db/migrations/004_search_features.sql'),
    Path('db/migrations/005_github_integration.sql'),
    Path('db/migrations/006_search_benchmarking.sql'),
]

class SearchService:
    def __init__(self, db_path: str = 'notes.db', vec_ext_path: Optional[str] = None):
        self.db_path = db_path
        self.vec_ext_path = vec_ext_path or os.getenv('SQLITE_VEC_PATH')
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._enable_extensions()
        self._run_migrations()
        self.embedder = Embeddings()

    def _enable_extensions(self):
        self.conn.execute('PRAGMA foreign_keys=ON;')
        try:
            self.conn.enable_load_extension(True)
            path = self.vec_ext_path
            # Treat placeholder paths as unset
            if path and path.startswith('/absolute/path/to/'):
                path = None
            # Try env path first
            load_ok = False
            if path:
                try:
                    self.conn.load_extension(path)
                    load_ok = True
                except Exception as e:
                    print(f"[search] sqlite-vec load failed for {path}: {e}")
            # Fallback: auto-detect from package
            if not load_ok:
                try:
                    import sqlite_vec  # type: ignore
                    auto_path = getattr(sqlite_vec, 'loadable_path', lambda: None)()
                    if auto_path:
                        self.conn.load_extension(auto_path)
                        load_ok = True
                except Exception as e:
                    print(f"[search] sqlite-vec auto-detect load failed: {e}")
        except Exception as e:
            # Extension loading is optional; log and continue
            print(f"[search] sqlite-vec not enabled: {e}")

    def _run_migrations(self):
        cur = self.conn.cursor()
        for path in MIGRATIONS:
            if not path.exists():
                continue
            sql = path.read_text(encoding='utf-8')
            # Strip shell-style comment lines to avoid tokenizer issues
            lines = [ln for ln in sql.splitlines() if not ln.lstrip().startswith('#')]
            sql = "\n".join(lines)
            try:
                cur.executescript(sql)
                self.conn.commit()
            except sqlite3.OperationalError as e:
                print(f"[search] migration {path.name} skipped/error: {e}")
                self.conn.rollback()

    # ─── Indexing ────────────────────────────────────────────────────────────
    def upsert_note(self, note_id: Optional[int], title: str, body: str, tags: str = '') -> int:
        cur = self.conn.cursor()
        if note_id is None:
            cur.execute("INSERT INTO notes(title, body, tags) VALUES (?,?,?)", (title, body, tags))
            note_id = cur.lastrowid
        else:
            cur.execute("UPDATE notes SET title=?, body=?, tags=?, updated_at=datetime('now') WHERE id=?",
                        (title, body, tags, note_id))
        self.conn.commit()
        # FTS5 is updated by triggers. Now (optionally) update vectors.
        self._upsert_vector(note_id, f"{title}\n\n{body}")
        return note_id

    def _vec_table_exists(self) -> bool:
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='note_vecs'")
        return cur.fetchone() is not None

    def _upsert_vector(self, note_id: int, text: str):
        if not self._vec_table_exists():
            return
        vec = self.embedder.embed(text)
        # Try JSON text insert first (supported by sqlite-vec), then fall back to BLOB
        cur = self.conn.cursor()
        try:
            cur.execute("INSERT OR REPLACE INTO note_vecs(note_id, embedding) VALUES (?, ?)", (note_id, json.dumps(vec)))
            self.conn.commit()
            return
        except Exception:
            pass
        try:
            from services.embeddings import Embeddings as _E
            blob = _E.pack_f32(vec)
            cur.execute("INSERT OR REPLACE INTO note_vecs(note_id, embedding) VALUES (?, ?)", (note_id, sqlite3.Binary(blob)))
            self.conn.commit()
        except Exception as e:
            print(f"[search] vector upsert failed (note {note_id}): {e}")
            self.conn.rollback()

    def _sanitize_fts_query(self, q: str) -> str:
        """Sanitize query for FTS5 compatibility"""
        if not q or not q.strip():
            return ""
        
        # Remove leading/trailing whitespace
        q = q.strip()
        
        # Handle special characters that break FTS5
        import re
        
        # Remove or escape problematic characters
        # FTS5 special characters: " * ( ) : < > = ^ - 
        q = re.sub(r'[<>=^]', '', q)  # Remove these entirely
        q = re.sub(r'[@#$%&]', '', q)  # Remove email/social chars
        q = re.sub(r'[():]', ' ', q)  # Replace with space
        q = re.sub(r'[-]', ' ', q)    # Replace dash with space
        
        # Handle quotes - remove unmatched quotes
        quote_count = q.count('"')
        if quote_count % 2 != 0:
            q = q.replace('"', '')
        
        # Clean up multiple spaces
        q = re.sub(r'\s+', ' ', q).strip()
        
        # If query is still empty or too short, return empty
        if not q or len(q) < 2:
            return ""
        
        # For single words, return as-is
        # For multiple words, wrap in quotes for phrase search
        words = q.split()
        if len(words) == 1:
            return words[0]
        else:
            # Use phrase search for multi-word queries
            return f'"{q}"'

    # ─── Search ─────────────────────────────────────────────────────────────
    def search(self, q: str, mode: str = 'hybrid', k: int = 20) -> list[sqlite3.Row]:
        if mode not in {'hybrid','keyword','semantic'}:
            mode = 'hybrid'
        if mode == 'keyword' or not self._vec_table_exists():
            return self._keyword(q, k)
        if mode == 'semantic':
            return self._semantic(q, k)
        return self._hybrid(q, k)

    def _keyword(self, q: str, k: int) -> list[sqlite3.Row]:
        # Sanitize query for FTS5
        sanitized_query = self._sanitize_fts_query(q)
        if not sanitized_query:
            return []
            
        cur = self.conn.cursor()
        try:
            rows = cur.execute(
                """
                SELECT n.*,
                       bm25(notes_fts) AS kw_rank,
                       snippet(notes_fts, 1, '<b>', '</b>', '…', 12) AS snippet
                FROM notes_fts JOIN notes n ON notes_fts.rowid = n.id
                WHERE notes_fts MATCH ?
                ORDER BY kw_rank
                LIMIT ?
                """, (sanitized_query, k)).fetchall()
            return rows
        except Exception as e:
            print(f"[search] FTS query failed for '{sanitized_query}': {e}")
            return []

    def _semantic(self, q: str, k: int) -> list[sqlite3.Row]:
        if not self._vec_table_exists():
            return []
        qvec = self.embedder.embed(q)
        cur = self.conn.cursor()
        rows = cur.execute(
            """
            WITH vs AS (
              SELECT note_id AS id,
                     1.0 - vec_distance_cosine(embedding, ?) AS vs_rank
              FROM note_vecs
              ORDER BY vs_rank DESC
              LIMIT ?
            )
            SELECT n.*, vs.vs_rank AS score FROM vs JOIN notes n ON n.id = vs.id
            ORDER BY score DESC
            """, (json.dumps(qvec), k)).fetchall()
        return rows

    def _hybrid(self, q: str, k: int) -> list[sqlite3.Row]:
        if not self._vec_table_exists():
            return self._keyword(q, k)
        
        # Sanitize query for FTS part
        sanitized_query = self._sanitize_fts_query(q)
        if not sanitized_query:
            # If query can't be sanitized, fall back to semantic search only
            return self._semantic(q, k)
            
        qvec = self.embedder.embed(q)
        cur = self.conn.cursor()
        try:
            rows = cur.execute(
            """
            WITH kw AS (
              SELECT rowid AS id, bm25(notes_fts) AS kw_rank
              FROM notes_fts
              WHERE notes_fts MATCH ?
              ORDER BY kw_rank
              LIMIT 50
            ),
            vs AS (
              SELECT note_id AS id, 1.0 - vec_distance_cosine(embedding, ?) AS vs_rank
              FROM note_vecs
              ORDER BY vs_rank DESC
              LIMIT 50
            ),
            unioned AS (
              SELECT id, (1.0/(1.0+kw_rank)) AS kw_s, 0.0 AS vs_s FROM kw
              UNION ALL
              SELECT id, 0.0, vs_rank FROM vs
            )
            SELECT n.*,
                   COALESCE(SUM(kw_s),0)*0.6 + COALESCE(SUM(vs_s),0)*0.4 AS score
            FROM unioned u JOIN notes n ON n.id = u.id
            GROUP BY n.id
            ORDER BY score DESC
            LIMIT ?
            """, (sanitized_query, json.dumps(qvec), k)).fetchall()
            return rows
        except Exception as e:
            print(f"[search] Hybrid search failed for '{sanitized_query}': {e}")
            # Fallback to semantic search only
            return self._semantic(q, k)

    # ─── Memory-Augmented Search ────────────────────────────────────────────
    def search_with_memory(
        self,
        user_id: int,
        query: str,
        mode: str = "hybrid",
        limit: int = 20,
        include_memory: bool = True
    ) -> dict:
        """
        Enhanced search that includes episodic and semantic memories

        Args:
            user_id: User ID for memory retrieval
            query: Search query
            mode: Search mode ('hybrid', 'keyword', 'semantic')
            limit: Max results
            include_memory: Whether to include memory context

        Returns:
            {
                'documents': [...],  # Document results
                'episodic': [...],   # Relevant past interactions
                'semantic': [...],   # User facts
                'context_summary': '...'  # Assembled context for LLM
            }
        """
        import logging
        logger = logging.getLogger(__name__)

        # Get regular document results (existing search)
        doc_results = self.search(query, mode=mode, k=limit)

        # Convert Row objects to dicts for easier JSON serialization
        documents = []
        for row in doc_results:
            doc = dict(row)
            documents.append(doc)

        if not include_memory:
            return {
                'documents': documents,
                'episodic': [],
                'semantic': [],
                'context_summary': ''
            }

        # Get memory service
        try:
            from services.memory_service import MemoryService
            from services.embeddings import get_embeddings_service
            from database import get_db_connection
            from config import get_settings

            settings = get_settings()
            memory = MemoryService(get_db_connection(), get_embeddings_service())

            # Search episodic memories
            episodic_results = memory.search_episodic(
                user_id=user_id,
                query=query,
                limit=settings.max_episodic_memories,
                min_importance=settings.episodic_importance_threshold
            )

            # Search semantic memories
            semantic_results = memory.search_semantic(
                user_id=user_id,
                query=query,
                limit=settings.max_semantic_memories
            )

            # Build context summary
            context_summary = self._build_context_summary(
                documents=documents[:settings.max_document_results],
                episodic=episodic_results,
                semantic=semantic_results
            )

            logger.debug(f"Memory-augmented search: {len(documents)} docs, {len(episodic_results)} episodes, {len(semantic_results)} facts")

            return {
                'documents': documents,
                'episodic': episodic_results,
                'semantic': semantic_results,
                'context_summary': context_summary
            }

        except Exception as e:
            logger.error(f"Memory search failed, falling back to documents only: {e}")
            return {
                'documents': documents,
                'episodic': [],
                'semantic': [],
                'context_summary': ''
            }

    def _build_context_summary(
        self,
        documents: list,
        episodic: list,
        semantic: list
    ) -> str:
        """Build LLM context from all memory sources"""

        parts = []

        # User profile (semantic)
        if semantic:
            parts.append("USER PROFILE:")
            for fact in semantic:
                parts.append(f"- {fact['fact']}")
            parts.append("")

        # Past interactions (episodic)
        if episodic:
            parts.append("RELEVANT PAST INTERACTIONS:")
            for ep in episodic:
                summary = ep.get('summary', ep.get('content', ''))[:200]
                parts.append(f"- {summary}")
            parts.append("")

        # Knowledge base (documents)
        if documents:
            parts.append("RELEVANT KNOWLEDGE BASE:")
            for i, doc in enumerate(documents, 1):
                title = doc.get('title', 'Untitled')
                # Try to get content from different possible fields
                content = doc.get('body', doc.get('content', doc.get('snippet', '')))
                snippet = content[:200] if content else ''
                parts.append(f"[{i}] {title}: {snippet}...")
            parts.append("")

        return "\n".join(parts)
