-- Migration: Add monitoring and observability system
-- Created: 2025-09-08
-- Description: Adds comprehensive monitoring, metrics, and alerting tables

-- Monitoring metrics table for time-series data
CREATE TABLE IF NOT EXISTS monitoring_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    labels TEXT, -- JSON string for metric labels/tags
    
    INDEX idx_monitoring_metrics_name_timestamp (metric_name, timestamp),
    INDEX idx_monitoring_metrics_timestamp (timestamp)
);

-- Monitoring alerts table for alert history and status
CREATE TABLE IF NOT EXISTS monitoring_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_name TEXT NOT NULL,
    alert_level TEXT NOT NULL CHECK (alert_level IN ('info', 'warning', 'critical')),
    message TEXT NOT NULL,
    triggered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    resolved_at DATETIME NULL,
    metadata TEXT, -- JSON string for additional alert data
    
    INDEX idx_monitoring_alerts_name (alert_name),
    INDEX idx_monitoring_alerts_level (alert_level),
    INDEX idx_monitoring_alerts_triggered (triggered_at),
    INDEX idx_monitoring_alerts_active (resolved_at) -- NULL values for active alerts
);

-- System health checks history
CREATE TABLE IF NOT EXISTS monitoring_health_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    service_name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('healthy', 'degraded', 'unhealthy')),
    response_time_ms REAL NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    error_message TEXT NULL,
    details TEXT, -- JSON string for additional details
    
    INDEX idx_monitoring_health_service_timestamp (service_name, timestamp),
    INDEX idx_monitoring_health_status (status),
    INDEX idx_monitoring_health_timestamp (timestamp)
);

-- Performance metrics aggregation table
CREATE TABLE IF NOT EXISTS monitoring_performance_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    period_start DATETIME NOT NULL,
    period_end DATETIME NOT NULL,
    endpoint TEXT NOT NULL,
    method TEXT NOT NULL,
    request_count INTEGER NOT NULL DEFAULT 0,
    total_response_time REAL NOT NULL DEFAULT 0.0,
    error_count INTEGER NOT NULL DEFAULT 0,
    avg_response_time_ms REAL GENERATED ALWAYS AS (
        CASE 
            WHEN request_count > 0 THEN (total_response_time / request_count) * 1000
            ELSE 0
        END
    ) STORED,
    error_rate REAL GENERATED ALWAYS AS (
        CASE 
            WHEN request_count > 0 THEN (error_count * 100.0 / request_count)
            ELSE 0
        END
    ) STORED,
    
    UNIQUE(period_start, period_end, endpoint, method),
    INDEX idx_monitoring_perf_period (period_start, period_end),
    INDEX idx_monitoring_perf_endpoint (endpoint),
    INDEX idx_monitoring_perf_method (method)
);

-- Security events tracking
CREATE TABLE IF NOT EXISTS monitoring_security_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    source_ip TEXT,
    user_agent TEXT,
    endpoint TEXT,
    message TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT, -- JSON string for additional event data
    
    INDEX idx_monitoring_security_type (event_type),
    INDEX idx_monitoring_security_severity (severity),
    INDEX idx_monitoring_security_timestamp (timestamp),
    INDEX idx_monitoring_security_ip (source_ip)
);

-- System resource monitoring
CREATE TABLE IF NOT EXISTS monitoring_system_resources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    cpu_percent REAL NOT NULL,
    memory_percent REAL NOT NULL,
    memory_used_mb REAL NOT NULL,
    memory_available_mb REAL NOT NULL,
    disk_usage_percent REAL NOT NULL,
    disk_used_gb REAL NOT NULL,
    disk_free_gb REAL NOT NULL,
    active_connections INTEGER NOT NULL DEFAULT 0,
    load_average TEXT, -- JSON array of load averages [1m, 5m, 15m]
    
    INDEX idx_monitoring_system_timestamp (timestamp)
);

-- Daily monitoring summary for reporting
CREATE TABLE IF NOT EXISTS monitoring_daily_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL UNIQUE,
    total_requests INTEGER NOT NULL DEFAULT 0,
    total_errors INTEGER NOT NULL DEFAULT 0,
    avg_response_time_ms REAL NOT NULL DEFAULT 0,
    unique_users INTEGER NOT NULL DEFAULT 0,
    notes_created INTEGER NOT NULL DEFAULT 0,
    searches_performed INTEGER NOT NULL DEFAULT 0,
    failed_authentications INTEGER NOT NULL DEFAULT 0,
    alerts_triggered INTEGER NOT NULL DEFAULT 0,
    uptime_percent REAL NOT NULL DEFAULT 100.0,
    
    INDEX idx_monitoring_daily_date (date)
);

-- Application-specific monitoring
CREATE TABLE IF NOT EXISTS monitoring_application_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_name TEXT NOT NULL,
    event_category TEXT NOT NULL, -- e.g., 'note_processing', 'search', 'authentication'
    status TEXT NOT NULL CHECK (status IN ('started', 'completed', 'failed')),
    duration_ms REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_id INTEGER,
    metadata TEXT, -- JSON string for event-specific data
    
    FOREIGN KEY (user_id) REFERENCES users(id),
    INDEX idx_monitoring_app_name_timestamp (event_name, timestamp),
    INDEX idx_monitoring_app_category (event_category),
    INDEX idx_monitoring_app_status (status),
    INDEX idx_monitoring_app_user (user_id)
);

-- Create views for common monitoring queries
CREATE VIEW IF NOT EXISTS monitoring_active_alerts AS
SELECT 
    alert_name,
    alert_level,
    message,
    triggered_at,
    julianday('now') - julianday(triggered_at) as hours_active,
    metadata
FROM monitoring_alerts 
WHERE resolved_at IS NULL
ORDER BY triggered_at DESC;

CREATE VIEW IF NOT EXISTS monitoring_recent_health AS
SELECT 
    service_name,
    status,
    response_time_ms,
    timestamp,
    error_message
FROM monitoring_health_checks 
WHERE timestamp > datetime('now', '-1 hour')
ORDER BY service_name, timestamp DESC;

CREATE VIEW IF NOT EXISTS monitoring_performance_hourly AS
SELECT 
    strftime('%Y-%m-%d %H:00:00', period_start) as hour,
    endpoint,
    method,
    SUM(request_count) as total_requests,
    SUM(error_count) as total_errors,
    AVG(avg_response_time_ms) as avg_response_time_ms,
    (SUM(error_count) * 100.0 / SUM(request_count)) as error_rate
FROM monitoring_performance_summary
WHERE period_start > datetime('now', '-24 hours')
GROUP BY strftime('%Y-%m-%d %H:00:00', period_start), endpoint, method
ORDER BY hour DESC, total_requests DESC;

-- Add some indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_notes_created_at ON notes(created_at);
CREATE INDEX IF NOT EXISTS idx_users_last_login ON users(last_login_at);

-- Update schema version
PRAGMA user_version = 13;