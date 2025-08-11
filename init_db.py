#!/usr/bin/env python3
"""Initialize the database with tables and sample data"""

from database import init_db, get_conn
from auth import get_password_hash

def create_demo_user():
    """Create a demo user"""
    conn = get_conn()
    c = conn.cursor()
    
    # Check if demo user exists
    existing = c.execute("SELECT id FROM users WHERE username = ?", ("demo",)).fetchone()
    if not existing:
        hashed_password = get_password_hash("demo")
        c.execute(
            "INSERT INTO users (username, hashed_password) VALUES (?, ?)",
            ("demo", hashed_password)
        )
        conn.commit()
        print("✅ Demo user created (username: demo, password: demo)")
    else:
        print("ℹ️ Demo user already exists")
    
    conn.close()

if __name__ == "__main__":
    print("🔧 Initializing Second Brain database...")
    init_db()
    create_demo_user()
    print("✅ Database initialization complete!")
