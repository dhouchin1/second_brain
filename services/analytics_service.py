"""
Analytics Service

Handles analytics and reporting functionality.
Extracted from app.py to provide clean separation of analytics concerns.
"""

import sqlite3
from typing import Dict, List
from services.auth_service import User


class AnalyticsService:
    """Service for handling analytics operations."""
    
    def __init__(self, get_conn_func):
        """Initialize analytics service with database connection function."""
        self.get_conn = get_conn_func
    
    def get_user_analytics(self, current_user: User) -> Dict:
        """Get user analytics and insights."""
        conn = self.get_conn()
        conn.row_factory = sqlite3.Row  # Enable dictionary-like access
        c = conn.cursor()
        
        # Basic stats
        total_notes = c.execute(
            "SELECT COUNT(*) as count FROM notes WHERE user_id = ?",
            (current_user.id,)
        ).fetchone()["count"]
        
        # This week
        this_week = c.execute(
            "SELECT COUNT(*) as count FROM notes WHERE user_id = ? AND date(COALESCE(timestamp, created_at)) >= date('now', '-7 days')",
            (current_user.id,)
        ).fetchone()["count"]
        
        # Last week for comparison
        last_week = c.execute(
            "SELECT COUNT(*) as count FROM notes WHERE user_id = ? AND date(COALESCE(timestamp, created_at)) BETWEEN date('now', '-14 days') AND date('now', '-7 days')",
            (current_user.id,)
        ).fetchone()["count"]
        
        # This month
        this_month = c.execute(
            "SELECT COUNT(*) as count FROM notes WHERE user_id = ? AND date(COALESCE(timestamp, created_at)) >= date('now', 'start of month')",
            (current_user.id,)
        ).fetchone()["count"]
        
        # Daily activity for the last 30 days
        daily_activity = c.execute("""
            SELECT date(COALESCE(timestamp, created_at)) as date, COUNT(*) as count 
            FROM notes 
            WHERE user_id = ? AND date(COALESCE(timestamp, created_at)) >= date('now', '-30 days')
            GROUP BY date(COALESCE(timestamp, created_at))
            ORDER BY date(COALESCE(timestamp, created_at))
        """, (current_user.id,)).fetchall()
        
        # Weekly activity for the last 12 weeks
        weekly_activity = c.execute("""
            SELECT strftime('%Y-W%W', COALESCE(timestamp, created_at)) as week, COUNT(*) as count
            FROM notes 
            WHERE user_id = ? AND date(COALESCE(timestamp, created_at)) >= date('now', '-84 days')
            GROUP BY strftime('%Y-W%W', COALESCE(timestamp, created_at))
            ORDER BY week
        """, (current_user.id,)).fetchall()
        
        # Hourly patterns (what time of day are most active)
        hourly_patterns = c.execute("""
            SELECT strftime('%H', COALESCE(timestamp, created_at)) as hour, COUNT(*) as count
            FROM notes
            WHERE user_id = ? AND strftime('%H', COALESCE(timestamp, created_at)) IS NOT NULL
            GROUP BY strftime('%H', COALESCE(timestamp, created_at))
            ORDER BY hour
        """, (current_user.id,)).fetchall()
        
        # By type
        by_type = c.execute(
            "SELECT type, COUNT(*) as count FROM notes WHERE user_id = ? GROUP BY type",
            (current_user.id,)
        ).fetchall()
        
        # Popular tags
        tag_counts = {}
        tag_rows = c.execute(
            "SELECT tags FROM notes WHERE user_id = ? AND tags IS NOT NULL",
            (current_user.id,)
        ).fetchall()
        
        for row in tag_rows:
            tags = row["tags"].split(",")
            for tag in tags:
                tag = tag.strip()
                if tag:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        popular_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Average notes per day
        first_note = c.execute(
            "SELECT MIN(date(COALESCE(timestamp, created_at))) as first_date FROM notes WHERE user_id = ?",
            (current_user.id,)
        ).fetchone()["first_date"]
        
        days_active = 1  # Default to prevent division by zero
        if first_note:
            days_result = c.execute(
                "SELECT julianday('now') - julianday(?) as days",
                (first_note,)
            ).fetchone()
            days_active = max(1, int(days_result["days"]))
        
        avg_notes_per_day = round(total_notes / days_active, 1) if total_notes > 0 else 0
        
        # Growth rate (this week vs last week)
        growth_rate = 0
        if last_week > 0:
            growth_rate = round(((this_week - last_week) / last_week) * 100, 1)
        elif this_week > 0:
            growth_rate = 100  # If no notes last week but some this week, that's 100% growth
        
        conn.close()
        
        return {
            "total_notes": total_notes,
            "this_week": this_week,
            "last_week": last_week,
            "this_month": this_month,
            "growth_rate": growth_rate,
            "avg_notes_per_day": avg_notes_per_day,
            "days_active": days_active,
            "by_type": [{"type": row["type"], "count": row["count"]} for row in by_type],
            "popular_tags": [{"name": tag, "count": count} for tag, count in popular_tags],
            "daily_activity": [{"date": row["date"], "count": row["count"]} for row in daily_activity],
            "weekly_activity": [{"week": row["week"], "count": row["count"]} for row in weekly_activity],
            "hourly_patterns": [{"hour": int(row["hour"]), "count": row["count"]} for row in hourly_patterns]
        }
