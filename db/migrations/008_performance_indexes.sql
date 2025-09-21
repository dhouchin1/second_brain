-- Performance optimization indexes for Second Brain database
-- Migration: 008_performance_indexes.sql

-- Index for search by user and status (used in status queries)
CREATE INDEX IF NOT EXISTS idx_notes_user_status
ON notes(user_id, status);

-- Index for user and tags (used in tag-based searches)
CREATE INDEX IF NOT EXISTS idx_notes_user_tags
ON notes(user_id, tags);

-- Index for the FTS5 virtual table queries with user filtering
CREATE INDEX IF NOT EXISTS idx_notes_user_content
ON notes(user_id, content);

-- Composite index for dashboard queries with status and timestamp
CREATE INDEX IF NOT EXISTS idx_notes_user_status_timestamp
ON notes(user_id, status, timestamp DESC, created_at DESC);

-- Index for type-based filtering in API endpoints
CREATE INDEX IF NOT EXISTS idx_notes_user_type
ON notes(user_id, type);

-- Index for user and source filtering (file types, etc)
CREATE INDEX IF NOT EXISTS idx_notes_user_source
ON notes(user_id, file_type);

-- Index for audio file queries
CREATE INDEX IF NOT EXISTS idx_notes_user_audio
ON notes(user_id, audio_filename);

-- Index for file-based queries
CREATE INDEX IF NOT EXISTS idx_notes_user_file
ON notes(user_id, file_filename);

-- Index for external URL queries
CREATE INDEX IF NOT EXISTS idx_notes_user_external
ON notes(user_id, external_url);

-- Analyze the notes table to update SQLite's query planner statistics
ANALYZE notes;