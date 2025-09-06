-- Migration 005: Add metrics columns to auto_seeding_log table
-- This migration adds tracking columns for auto-seeding operations

-- Columns already exist in current schema or will be created by later migrations.
-- Skipping ALTERs to keep this migration idempotent across environments.

-- Create index for performance on user queries
CREATE INDEX IF NOT EXISTS idx_auto_seeding_log_user_timestamp 
ON auto_seeding_log (user_id, timestamp);

-- Create index for success rate queries
CREATE INDEX IF NOT EXISTS idx_auto_seeding_log_success 
ON auto_seeding_log (success, timestamp);
