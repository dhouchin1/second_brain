"""Storage adapter for graph memory ingestion.

The adapter prefers the existing ``notes`` table + ``notes_fts`` index when the
project schema is available. If those tables are missing it transparently falls
back to creating a lightweight ``docs`` store with its own FTS triggers. Column
names and table selection can be overridden via environment variables with the
``GM_*`` prefix so advanced deployments stay configurable.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

from database import get_db_manager

_DEFAULT_TEXT_COLUMNS = ("content", "body", "text")
_DEFAULT_TITLE_COLUMNS = ("title",)
_DEFAULT_PATH_COLUMNS = ("source_url", "path")
_DEFAULT_MIME_COLUMNS = ("mime", "file_mime_type")


class GraphStorageAdapter:
    """Route raw text ingestion into the primary notes store or a docs fallback."""

    def __init__(self) -> None:
        self.db_manager = get_db_manager()
        # Env overrides
        self._text_table_override = os.getenv("GM_TEXT_TABLE")
        self._text_column_override = os.getenv("GM_TEXT_COLUMN")
        self._title_column_override = os.getenv("GM_TITLE_COLUMN")
        self._path_column_override = os.getenv("GM_PATH_COLUMN")
        self._mime_column_override = os.getenv("GM_MIME_COLUMN")
        self._rowid_column_override = os.getenv("GM_ROWID_COLUMN")
        self._fts_table_override = os.getenv("GM_FTS_TABLE")
        self._fts_column_override = os.getenv("GM_FTS_TEXT_COLUMN")

        # Determined lazily
        self._detected = False
        self._text_table: str = "docs"
        self._rowid_column: str = "id"
        self._text_column: str = "text"
        self._title_column: Optional[str] = None
        self._path_column: Optional[str] = None
        self._mime_column: Optional[str] = None
        self._body_column: Optional[str] = None
        self._fts_table: str = "docs_fts"

    @contextmanager
    def _connection(self):
        with self.db_manager.get_db_context() as conn:
            yield conn

    # ------------------------------------------------------------------
    # Detection / setup helpers
    # ------------------------------------------------------------------
    def _detect(self, conn: sqlite3.Connection) -> None:
        if self._detected:
            return

        table_names = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }

        # Determine backing table
        if self._text_table_override:
            target_table = self._text_table_override
        elif "notes" in table_names:
            target_table = "notes"
        else:
            target_table = "docs"

        if target_table == "docs" and target_table not in table_names:
            self._ensure_docs_tables(conn)
            table_names.add("docs")

        self._text_table = target_table
        self._rowid_column = self._rowid_column_override or self._default_rowid(conn, target_table)
        self._text_column, self._body_column = self._resolve_text_columns(conn, target_table)
        self._title_column = self._resolve_optional_column(
            conn,
            target_table,
            override=self._title_column_override,
            candidates=_DEFAULT_TITLE_COLUMNS,
        )
        self._path_column = self._resolve_optional_column(
            conn,
            target_table,
            override=self._path_column_override,
            candidates=_DEFAULT_PATH_COLUMNS,
        )
        self._mime_column = self._resolve_optional_column(
            conn,
            target_table,
            override=self._mime_column_override,
            candidates=_DEFAULT_MIME_COLUMNS,
        )
        self._fts_table = self._fts_table_override or (
            "notes_fts" if target_table == "notes" else "docs_fts"
        )

        self._detected = True

    def _default_rowid(self, conn: sqlite3.Connection, table: str) -> str:
        pragma = conn.execute(f"PRAGMA table_info({table})").fetchall()
        for column in pragma:
            if column[5]:  # pk flag
                return column[1]
        return "rowid"

    def _resolve_text_columns(
        self, conn: sqlite3.Connection, table: str
    ) -> Tuple[str, Optional[str]]:
        override = self._text_column_override
        if override:
            return override, self._resolve_optional_column(conn, table, candidates=("body",))

        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        text_col = None
        for candidate in _DEFAULT_TEXT_COLUMNS:
            if candidate in columns:
                text_col = candidate
                break
        if not text_col:
            raise RuntimeError(f"No suitable text column found on {table}")

        body_col = "body" if "body" in columns else None
        return text_col, body_col

    def _resolve_optional_column(
        self,
        conn: sqlite3.Connection,
        table: str,
        override: Optional[str] = None,
        candidates: Tuple[str, ...] = (),
    ) -> Optional[str]:
        if override:
            return override
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for candidate in candidates:
            if candidate in existing:
                return candidate
        return None

    def _ensure_docs_tables(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS docs (
              id INTEGER PRIMARY KEY,
              title TEXT,
              text TEXT NOT NULL,
              path TEXT,
              mime TEXT,
              source_id INTEGER,
              created_at TEXT NOT NULL DEFAULT (datetime('now')),
              updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
              text,
              content='docs',
              content_rowid='id'
            );

            CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON docs BEGIN
              INSERT INTO docs_fts(rowid, text) VALUES (new.id, new.text);
            END;

            CREATE TRIGGER IF NOT EXISTS docs_au AFTER UPDATE ON docs BEGIN
              INSERT INTO docs_fts(docs_fts, rowid, text) VALUES ('delete', old.id, old.text);
              INSERT INTO docs_fts(rowid, text) VALUES (new.id, new.text);
              UPDATE docs SET updated_at = datetime('now') WHERE id = new.id;
            END;

            CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON docs BEGIN
              INSERT INTO docs_fts(docs_fts, rowid, text) VALUES ('delete', old.id, old.text);
            END;
            """
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def upsert_text(
        self,
        text: str,
        *,
        title: Optional[str] = None,
        source_id: Optional[int] = None,
        path: Optional[str] = None,
        mime: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
    ) -> int:
        """Insert text into the configured storage.

        Returns the row identifier of the inserted record in the target table.
        """
        if not text:
            raise ValueError("text is required for storage")

        with self._connection() as conn:
            self._detect(conn)
            now = datetime.utcnow().isoformat(timespec="seconds")
            columns = []
            values: list[Any] = []

            # Title/body preparation
            resolved_title = title or self._derive_title(text)
            resolved_body = text

            if self._title_column:
                columns.append(self._title_column)
                values.append(resolved_title)
            if self._body_column and self._body_column != self._text_column:
                columns.append(self._body_column)
                values.append(resolved_body)

            columns.append(self._text_column)
            values.append(resolved_body)

            if self._path_column and path is not None:
                columns.append(self._path_column)
                values.append(path)

            if self._mime_column and mime is not None:
                columns.append(self._mime_column)
                values.append(mime)

            if source_id is not None:
                # Persist to docs table or embed into metadata/notes when column exists
                if self._has_column(conn, self._text_table, "source_id"):
                    columns.append("source_id")
                    values.append(source_id)
                elif metadata is None:
                    metadata = {"source_id": source_id}
                else:
                    metadata.setdefault("source_id", source_id)

            if metadata and self._has_column(conn, self._text_table, "metadata"):
                columns.append("metadata")
                import json

                values.append(json.dumps(metadata))

            if user_id is not None and self._has_column(conn, self._text_table, "user_id"):
                columns.append("user_id")
                values.append(user_id)

            if self._text_table == "notes":
                # Maintain timestamps when columns exist
                if self._has_column(conn, "notes", "timestamp") and "timestamp" not in columns:
                    columns.append("timestamp")
                    values.append(now)
                if self._has_column(conn, "notes", "status") and "status" not in columns:
                    columns.append("status")
                    values.append("completed")
            else:
                if self._has_column(conn, self._text_table, "updated_at") and "updated_at" not in columns:
                    columns.append("updated_at")
                    values.append(now)

            placeholders = ",".join("?" for _ in columns)
            column_list = ",".join(columns)
            sql = f"INSERT INTO {self._text_table} ({column_list}) VALUES ({placeholders})"
            cursor = conn.execute(sql, values)
            return int(cursor.lastrowid)

    def get_storage_identity(self) -> Dict[str, str]:
        """Return a summary of the detected storage backend."""
        with self._connection() as conn:
            self._detect(conn)
            return {
                "table": self._text_table,
                "rowid_column": self._rowid_column,
                "text_column": self._text_column,
                "body_column": self._body_column or "",
                "title_column": self._title_column or "",
                "path_column": self._path_column or "",
                "mime_column": self._mime_column or "",
                "fts_table": self._fts_table,
            }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _has_column(self, conn: sqlite3.Connection, table: str, column: str) -> bool:
        return any(row[1] == column for row in conn.execute(f"PRAGMA table_info({table})"))

    def _derive_title(self, text: str, limit: int = 120) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:limit]
        return text[:limit] if text else "Untitled"
