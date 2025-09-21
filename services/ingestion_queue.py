"""Ingestion job queue service.

Provides a resilient job queue for coordinating ingestion/post-processing tasks.
Jobs are persisted in SQLite so they can be resumed after restarts and exposed to
clients for realtime status updates.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from config import settings


class IngestionJobStatus(str, Enum):
    """Valid states for an ingestion job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestionJobType(str, Enum):
    """Supported ingestion job types."""

    NOTE_ENRICHMENT = "note_enrichment"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    WEB_CAPTURE = "web_capture"
    ARCHIVE_WEB_CAPTURE = "archive_web_capture"


@dataclass
class IngestionJob:
    """Representation of a job row."""

    id: int
    job_key: str
    job_type: str
    note_id: Optional[int]
    user_id: Optional[int]
    status: str
    priority: int
    progress: int
    status_detail: Optional[str]
    payload: Optional[Dict[str, Any]]
    content_hash: Optional[str]
    retry_count: int
    last_error: Optional[str]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]

    def to_api(self) -> Dict[str, Any]:
        """Return a safe payload for API responses."""

        return {
            "job_key": self.job_key,
            "job_type": self.job_type,
            "note_id": self.note_id,
            "status": self.status,
            "priority": self.priority,
            "progress": self.progress,
            "status_detail": self.status_detail,
            "retry_count": self.retry_count,
            "last_error": self.last_error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class IngestionQueue:
    """SQLite-backed job queue used for ingestion tasks."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or str(settings.db_path)
        self._lock = threading.Lock()
        self._active_job_id: Optional[int] = None
        self._ensure_table()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self) -> None:
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_key TEXT NOT NULL UNIQUE,
                job_type TEXT NOT NULL,
                note_id INTEGER,
                user_id INTEGER,
                status TEXT NOT NULL DEFAULT 'queued',
                priority INTEGER NOT NULL DEFAULT 0,
                progress INTEGER NOT NULL DEFAULT 0,
                status_detail TEXT,
                payload TEXT,
                content_hash TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                FOREIGN KEY (note_id) REFERENCES notes (id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status ON ingestion_jobs (status, priority DESC, created_at ASC)"
        )
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def enqueue(
        self,
        job_type: IngestionJobType,
        *,
        note_id: Optional[int] = None,
        user_id: Optional[int] = None,
        payload: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        content_hash: Optional[str] = None,
    ) -> IngestionJob:
        """Insert a new job and return the row."""

        job_key = uuid.uuid4().hex
        conn = self._connect()
        now = datetime.utcnow().isoformat()
        payload_json = json.dumps(payload) if payload else None
        cur = conn.execute(
            """
            INSERT INTO ingestion_jobs (
                job_key, job_type, note_id, user_id, status, priority,
                progress, status_detail, payload, content_hash, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 0, NULL, ?, ?, ?)
            """,
            (
                job_key,
                job_type.value,
                note_id,
                user_id,
                IngestionJobStatus.QUEUED.value,
                priority,
                payload_json,
                content_hash,
                now,
            ),
        )
        job_id = cur.lastrowid
        conn.commit()
        conn.close()
        return self.get_job(job_id)

    def get_job(self, job_id: int) -> Optional[IngestionJob]:
        conn = self._connect()
        row = conn.execute("SELECT * FROM ingestion_jobs WHERE id = ?", (job_id,)).fetchone()
        conn.close()
        if not row:
            return None
        return self._row_to_job(row)

    def get_job_by_key(self, job_key: str) -> Optional[IngestionJob]:
        conn = self._connect()
        row = conn.execute("SELECT * FROM ingestion_jobs WHERE job_key = ?", (job_key,)).fetchone()
        conn.close()
        if not row:
            return None
        return self._row_to_job(row)

    def get_processing_job_for_note(self, note_id: int) -> Optional[IngestionJob]:
        conn = self._connect()
        row = conn.execute(
            """
            SELECT * FROM ingestion_jobs
            WHERE note_id = ? AND status = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (note_id, IngestionJobStatus.PROCESSING.value),
        ).fetchone()
        conn.close()
        if not row:
            return None
        return self._row_to_job(row)

    def get_latest_job_for_note(self, note_id: int) -> Optional[IngestionJob]:
        conn = self._connect()
        row = conn.execute(
            """
            SELECT * FROM ingestion_jobs
            WHERE note_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (note_id,),
        ).fetchone()
        conn.close()
        if not row:
            return None
        return self._row_to_job(row)

    def get_next_job(self) -> Optional[IngestionJob]:
        """Return the next queued job and transition it to processing."""

        with self._lock:
            if self._active_job_id is not None:
                return None

            conn = self._connect()
            row = conn.execute(
                """
                SELECT * FROM ingestion_jobs
                WHERE status = ?
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
                """,
                (IngestionJobStatus.QUEUED.value,),
            ).fetchone()
            if not row:
                conn.close()
                return None

            job_id = row["id"]
            now = datetime.utcnow().isoformat()
            conn.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?, started_at = ?, progress = 5, status_detail = 'Queued for processing'
                WHERE id = ?
                """,
                (IngestionJobStatus.PROCESSING.value, now, job_id),
            )
            conn.commit()
            conn.close()
            self._active_job_id = job_id
            return self._row_to_job(row)

    def mark_progress(self, job_id: int, progress: int, detail: Optional[str] = None) -> None:
        progress = max(0, min(progress, 100))
        conn = self._connect()
        conn.execute(
            "UPDATE ingestion_jobs SET progress = ?, status_detail = ? WHERE id = ?",
            (progress, detail, job_id),
        )
        conn.commit()
        conn.close()

    def mark_started(self, job_id: int, detail: Optional[str] = None) -> None:
        """Transition a queued job to processing if not already started."""

        conn = self._connect()
        now = datetime.utcnow().isoformat()
        conn.execute(
            """
            UPDATE ingestion_jobs
            SET status = ?, started_at = COALESCE(started_at, ?), progress = CASE WHEN progress < 5 THEN 5 ELSE progress END,
                status_detail = COALESCE(?, status_detail)
            WHERE id = ?
            """,
            (IngestionJobStatus.PROCESSING.value, now, detail, job_id),
        )
        conn.commit()
        conn.close()

    def mark_complete(self, job_id: int, *, success: bool, error: Optional[str] = None) -> None:
        status = IngestionJobStatus.COMPLETED.value if success else IngestionJobStatus.FAILED.value
        now = datetime.utcnow().isoformat()
        conn = self._connect()
        conn.execute(
            """
            UPDATE ingestion_jobs
            SET status = ?, completed_at = ?, progress = ?, status_detail = ?, last_error = ?
            WHERE id = ?
            """,
            (
                status,
                now,
                100 if success else 0,
                "Completed" if success else "Failed",
                error,
                job_id,
            ),
        )
        conn.commit()
        conn.close()
        with self._lock:
            if self._active_job_id == job_id:
                self._active_job_id = None

    def release_active_job(self) -> None:
        """Reset the active-job lock after fatal errors."""

        with self._lock:
            self._active_job_id = None

    def _row_to_job(self, row: sqlite3.Row) -> IngestionJob:
        payload = json.loads(row["payload"]) if row["payload"] else None
        return IngestionJob(
            id=row["id"],
            job_key=row["job_key"],
            job_type=row["job_type"],
            note_id=row["note_id"],
            user_id=row["user_id"],
            status=row["status"],
            priority=row["priority"],
            progress=row["progress"],
            status_detail=row["status_detail"],
            payload=payload,
            content_hash=row["content_hash"],
            retry_count=row["retry_count"],
            last_error=row["last_error"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )


ingestion_queue = IngestionQueue()
"""Global queue instance used by the application."""


# ArchiveBox-specific payload schemas
@dataclass
class ArchiveWebCapturePayload:
    """Payload for ARCHIVE_WEB_CAPTURE jobs."""

    url: str
    extract_types: Optional[List[str]] = None
    timeout: Optional[int] = None
    only_new: bool = True
    depth: int = 0
    overwrite: bool = False
    storage_strategy: Optional[str] = None  # symlink, copy, or None for default
    priority_archival: bool = False  # High-priority archival
    integration_metadata: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[int] = None
    note_id: Optional[int] = None
    priority: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for job payload."""
        result = {
            "url": self.url,
            "extract_types": self.extract_types,
            "timeout": self.timeout,
            "only_new": self.only_new,
            "depth": self.depth,
            "overwrite": self.overwrite,
            "storage_strategy": self.storage_strategy,
            "priority_archival": self.priority_archival,
        }
        if self.integration_metadata:
            result["integration_metadata"] = self.integration_metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArchiveWebCapturePayload":
        """Create from dictionary."""
        return cls(
            url=data["url"],
            extract_types=data.get("extract_types"),
            timeout=data.get("timeout"),
            only_new=data.get("only_new", True),
            depth=data.get("depth", 0),
            overwrite=data.get("overwrite", False),
            storage_strategy=data.get("storage_strategy"),
            priority_archival=data.get("priority_archival", False),
            integration_metadata=data.get("integration_metadata") or {},
            priority=data.get("priority"),
            note_id=data.get("note_id"),
            user_id=data.get("user_id"),
        )


def create_archive_job(
    url: str,
    user_id: Optional[int] = None,
    note_id: Optional[int] = None,
    extract_types: Optional[List[str]] = None,
    priority: int = 0,
    **kwargs
) -> IngestionJob:
    """Helper function to create an ArchiveBox job."""
    payload = ArchiveWebCapturePayload(
        url=url,
        extract_types=extract_types,
        **kwargs
    )

    return ingestion_queue.enqueue(
        job_type=IngestionJobType.ARCHIVE_WEB_CAPTURE,
        note_id=note_id,
        user_id=user_id,
        payload=payload.to_dict(),
        priority=priority,
        content_hash=None  # Could use URL hash for deduplication
    )
