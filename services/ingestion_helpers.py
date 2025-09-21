"""Utilities for routing captured content through the ingestion job pipeline."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from typing import Iterable, Optional, Dict, Any

from config import settings
from services.ingestion_queue import ingestion_queue, IngestionJobType


def compute_text_hash(parts: Iterable[Optional[str]]) -> Optional[str]:
    """Compute a stable SHA-256 hash from textual parts."""
    normalized = " ".join(
        " ".join((part or "").strip().lower().split())
        for part in parts
        if part
    ).strip()
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def enqueue_ingestion_job(
    note_id: int,
    user_id: Optional[int],
    *,
    job_type: IngestionJobType = IngestionJobType.NOTE_ENRICHMENT,
    payload: Optional[Dict[str, Any]] = None,
    content_hash: Optional[str] = None,
    priority: int = 0,
):
    """Create a job in the ingestion queue for downstream processing."""
    return ingestion_queue.enqueue(
        job_type,
        note_id=note_id,
        user_id=user_id,
        payload=payload,
        priority=priority,
        content_hash=content_hash,
    )


def update_note_status(note_id: int, status: str) -> None:
    """Update status/timestamp for a note."""
    conn = sqlite3.connect(str(settings.db_path))
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            "UPDATE notes SET status = ?, timestamp = ? WHERE id = ?",
            (status, now, note_id),
        )
        conn.commit()
    finally:
        conn.close()
