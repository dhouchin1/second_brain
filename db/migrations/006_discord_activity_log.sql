-- Discord Activity Logging
-- Track Discord bot interactions for monitoring and analytics

CREATE TABLE IF NOT EXISTS discord_activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    discord_user_id TEXT NOT NULL,
    discord_username TEXT,
    command TEXT,
    action TEXT NOT NULL,
    description TEXT,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_discord_activity_created_at 
ON discord_activity_log(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_discord_activity_user 
ON discord_activity_log(discord_user_id);

CREATE INDEX IF NOT EXISTS idx_discord_activity_command 
ON discord_activity_log(command);