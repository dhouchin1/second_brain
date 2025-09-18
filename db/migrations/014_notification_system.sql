-- Notification System Database Schema
-- Migration: 014_notification_system.sql
-- Description: Create tables for real-time notification system

-- Create notifications table
CREATE TABLE IF NOT EXISTS notifications (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    type TEXT NOT NULL,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    priority TEXT NOT NULL DEFAULT 'normal',
    data TEXT, -- JSON data for notification context
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    read_at TIMESTAMP,
    expires_at TIMESTAMP,
    persistent BOOLEAN DEFAULT 1,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

-- Create notification_preferences table
CREATE TABLE IF NOT EXISTS notification_preferences (
    user_id INTEGER PRIMARY KEY,
    enabled_types TEXT NOT NULL DEFAULT '[]', -- JSON array of enabled notification types
    min_priority TEXT NOT NULL DEFAULT 'normal', -- minimum priority to show
    sound_enabled BOOLEAN DEFAULT 1,
    desktop_enabled BOOLEAN DEFAULT 1,
    email_enabled BOOLEAN DEFAULT 0,
    auto_dismiss_seconds INTEGER, -- auto-dismiss timeout in seconds
    max_notifications INTEGER DEFAULT 50, -- maximum notifications to keep in history
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

-- Create indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_notifications_user_created 
ON notifications (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_notifications_user_unread
ON notifications (user_id, read_at) WHERE read_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_notifications_type 
ON notifications (type);

CREATE INDEX IF NOT EXISTS idx_notifications_priority
ON notifications (priority);

CREATE INDEX IF NOT EXISTS idx_notifications_expires
ON notifications (expires_at) WHERE expires_at IS NOT NULL;

-- Create trigger to automatically clean up old notifications
CREATE TRIGGER IF NOT EXISTS cleanup_old_notifications
AFTER INSERT ON notifications
BEGIN
    -- Clean up expired notifications
    DELETE FROM notifications 
    WHERE expires_at IS NOT NULL 
    AND expires_at < datetime('now');
    
    -- Clean up excess notifications per user (keep only latest max_notifications)
    DELETE FROM notifications 
    WHERE id IN (
        SELECT n.id 
        FROM notifications n
        LEFT JOIN notification_preferences np ON n.user_id = np.user_id
        WHERE n.user_id = NEW.user_id
        AND n.persistent = 1
        ORDER BY n.created_at DESC 
        LIMIT -1 OFFSET COALESCE(np.max_notifications, 50)
    );
END;

-- Insert default notification preferences for existing users
INSERT OR IGNORE INTO notification_preferences (user_id, enabled_types)
SELECT id, '["processing_started","processing_completed","processing_failed","note_created","note_updated","note_deleted","search_indexed","audio_transcribed","file_uploaded","system_status","sync_completed","sync_failed"]'
FROM users;