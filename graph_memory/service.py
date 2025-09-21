"""High-level service that coordinates graph memory ingestion."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from config import settings
from database import get_db_manager
from graph_memory.storage_adapter import GraphStorageAdapter
from graph_memory.extractor import GraphFactExtractor

logger = logging.getLogger(__name__)


class GraphMemoryService:
    """Persist raw documents and extracted facts into the graph memory tables."""

    def __init__(self) -> None:
        self.db_manager = get_db_manager()
        self.storage = GraphStorageAdapter()
        self.extractor = GraphFactExtractor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ingest_text(
        self,
        *,
        text: str,
        title: Optional[str] = None,
        source_type: str,
        uri: Optional[str] = None,
        checksum: Optional[str] = None,
        path: Optional[str] = None,
        mime: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not settings.graph_memory_enabled:
            return {"enabled": False, "stored": False, "facts": 0}

        metadata = self._normalise_metadata(metadata)

        with self.db_manager.get_db_context() as conn:
            source_id = self._get_or_create_source(
                conn,
                source_type=source_type,
                uri=uri,
                checksum=checksum,
                metadata=metadata,
            )

        storage_info = self.storage.get_storage_identity()
        storage_rowid = self.storage.upsert_text(
            text,
            title=title,
            source_id=source_id,
            path=path,
            mime=mime,
            metadata=metadata,
            user_id=user_id,
        )

        with self.db_manager.get_db_context() as conn:
            self._register_document(
                conn,
                source_id=source_id,
                storage_table=storage_info["table"],
                storage_rowid=storage_rowid,
                checksum=checksum,
                path=path,
                mime=mime,
            )

        facts_inserted = 0
        if settings.graph_memory_extract_on_ingest:
            facts = self.extractor.extract(text)
            if facts:
                with self.db_manager.get_db_context() as conn:
                    facts_inserted = self._store_facts(conn, source_id, facts)

        return {
            "enabled": True,
            "stored": True,
            "facts": facts_inserted,
            "source_id": source_id,
            "storage_rowid": storage_rowid,
            "storage_table": storage_info["table"],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_or_create_source(
        self,
        conn,
        *,
        source_type: str,
        uri: Optional[str],
        checksum: Optional[str],
        metadata: Dict[str, Any],
    ) -> int:
        key = self._make_source_key(source_type, uri, checksum)
        row = conn.execute(
            "SELECT id FROM gm_sources WHERE source_key = ?",
            (key,),
        ).fetchone()
        if row:
            return int(row[0])

        conn.execute(
            """
            INSERT INTO gm_sources (source_key, source_type, uri, checksum, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (key, source_type, uri, checksum, json.dumps(metadata)),
        )
        return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])

    def _register_document(
        self,
        conn,
        *,
        source_id: int,
        storage_table: str,
        storage_rowid: int,
        checksum: Optional[str],
        path: Optional[str],
        mime: Optional[str],
    ) -> None:
        conn.execute(
            """
            INSERT INTO gm_documents (storage_table, storage_rowid, source_id, checksum, path, mime)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(storage_table, storage_rowid) DO UPDATE SET
              source_id = excluded.source_id,
              checksum = COALESCE(excluded.checksum, gm_documents.checksum),
              path = COALESCE(excluded.path, gm_documents.path),
              mime = COALESCE(excluded.mime, gm_documents.mime)
            """,
            (storage_table, storage_rowid, source_id, checksum, path, mime),
        )

    def _store_facts(self, conn, source_id: int, facts: List[Dict[str, Any]]) -> int:
        now = datetime.utcnow().isoformat(timespec="seconds")
        inserted = 0
        for fact in facts:
            conn.execute(
                """
                INSERT INTO gm_facts (subject, predicate, object, object_type, confidence, source_id, valid_at, last_seen_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(subject, predicate, object, valid_at, source_id) DO UPDATE SET
                  confidence = excluded.confidence,
                  last_seen_at = excluded.last_seen_at,
                  invalid_at = NULL
                """,
                (
                    fact.get("subject"),
                    fact.get("predicate"),
                    fact.get("object"),
                    fact.get("object_type", "string"),
                    float(fact.get("confidence", 0.6)),
                    source_id,
                    now,
                    now,
                ),
            )
            inserted += 1
        return inserted

    def _make_source_key(
        self,
        source_type: str,
        uri: Optional[str],
        checksum: Optional[str],
    ) -> str:
        if checksum:
            return f"{source_type}:{checksum}"
        if uri:
            return f"{source_type}:{uri}"
        return f"{source_type}:unknown"

    def _normalise_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if metadata is None:
            metadata = {}
        try:
            return json.loads(json.dumps(metadata, default=str))
        except Exception:
            return {"raw": str(metadata)}


_graph_memory_service: Optional[GraphMemoryService] = None


def get_graph_memory_service() -> GraphMemoryService:
    global _graph_memory_service
    if _graph_memory_service is None:
        _graph_memory_service = GraphMemoryService()
    return _graph_memory_service
