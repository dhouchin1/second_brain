# Backup the current notes.db before creating brain.db
# Uses SQLite VACUUM INTO to create a compact backup copy.

PRAGMA wal_checkpoint(FULL);
VACUUM;  -- Clean up the main database first

-- Create a deterministic pre-brain backup file in repo root
VACUUM INTO 'notes.backup.pre_brain.db';

