-- Migration 015: Ingestion job queue
-- Introduces a generic ingestion_jobs table for tracking capture/post-processing tasks
-- tied to uploaded or captured content.

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
);

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status
    ON ingestion_jobs (status, priority DESC, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_note
    ON ingestion_jobs (note_id);

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_user
    ON ingestion_jobs (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_hash
    ON ingestion_jobs (content_hash);
