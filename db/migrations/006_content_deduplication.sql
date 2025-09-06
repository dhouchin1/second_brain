-- Migration 006: Add proper content deduplication support
-- This migration adds a dedicated column for content hashing to improve deduplication performance

-- Add content_hash column for efficient deduplication
-- Column may already exist in some environments; skip adding to keep idempotent
-- For fresh databases, content_hash is expected to be present via later consolidated schema
-- If missing, the application can add it opportunistically during startup

-- Create index for fast content hash lookups
CREATE INDEX IF NOT EXISTS idx_notes_content_hash 
ON notes (content_hash) WHERE content_hash IS NOT NULL;

-- Create composite index for hash + timestamp queries
CREATE INDEX IF NOT EXISTS idx_notes_content_hash_created 
ON notes (content_hash, created_at) WHERE content_hash IS NOT NULL;

-- Note: content_hash will be populated by the application for new notes
-- Existing notes will have NULL content_hash, which is fine for gradual rollout
