#!/usr/bin/env python3
"""
Fix corrupted user password hashes in the database.

This script identifies users with invalid password hashes and either:
1. Deletes them (if they're corrupted)
2. Resets their password to a known value

Usage:
    python fix_user_passwords.py
"""

import sqlite3
from passlib.context import CryptContext
from pathlib import Path

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_db_path():
    """Get database path from config"""
    db_path = Path(__file__).parent / "notes.db"
    return str(db_path)

def check_hash_validity(hashed_password):
    """Check if a password hash is valid bcrypt format"""
    if not hashed_password:
        return False
    # Valid bcrypt hashes start with $2a$, $2b$, or $2y$
    return hashed_password.startswith(('$2a$', '$2b$', '$2y$'))

def fix_corrupted_users():
    """Fix users with corrupted password hashes"""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    print("🔍 Checking for corrupted user password hashes...\n")

    # Get all users
    users = c.execute("SELECT id, username, hashed_password FROM users").fetchall()

    corrupted_users = []
    valid_users = []

    for user_id, username, hashed_password in users:
        if check_hash_validity(hashed_password):
            valid_users.append(username)
            print(f"✅ {username}: Valid bcrypt hash")
        else:
            corrupted_users.append((user_id, username, hashed_password))
            print(f"❌ {username}: Invalid hash (starts with: {hashed_password[:20]}...)")

    if not corrupted_users:
        print("\n🎉 All user password hashes are valid!")
        conn.close()
        return

    print(f"\n⚠️  Found {len(corrupted_users)} corrupted user(s)")
    print("\nOptions:")
    print("1. Delete corrupted users")
    print("2. Reset corrupted users' passwords to 'password123'")
    print("3. Cancel")

    choice = input("\nEnter your choice (1-3): ").strip()

    if choice == "1":
        # Delete corrupted users
        for user_id, username, _ in corrupted_users:
            c.execute("DELETE FROM users WHERE id = ?", (user_id,))
            print(f"🗑️  Deleted user: {username}")
        conn.commit()
        print(f"\n✅ Deleted {len(corrupted_users)} corrupted user(s)")

    elif choice == "2":
        # Reset passwords
        new_password = "password123"
        hashed = pwd_context.hash(new_password)

        for user_id, username, _ in corrupted_users:
            c.execute("UPDATE users SET hashed_password = ? WHERE id = ?", (hashed, user_id))
            print(f"🔑 Reset password for: {username}")

        conn.commit()
        print(f"\n✅ Reset passwords for {len(corrupted_users)} user(s)")
        print(f"📝 New password for all: {new_password}")
        print("⚠️  Please change these passwords after logging in!")

    else:
        print("\n❌ Cancelled. No changes made.")

    conn.close()

def create_new_user(username, password):
    """Create a new user with proper password hash"""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Check if user exists
    existing = c.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if existing:
        print(f"❌ User '{username}' already exists")
        conn.close()
        return

    # Create user
    hashed_password = pwd_context.hash(password)
    c.execute(
        "INSERT INTO users (username, hashed_password) VALUES (?, ?)",
        (username, hashed_password)
    )
    conn.commit()
    user_id = c.lastrowid

    print(f"✅ Created user: {username} (ID: {user_id})")
    conn.close()

def main():
    """Main entry point"""
    print("=" * 60)
    print("User Password Hash Repair Tool")
    print("=" * 60)
    print()

    print("What would you like to do?")
    print("1. Check and fix corrupted password hashes")
    print("2. Create a new user")
    print("3. List all users")
    print("4. Exit")
    print()

    choice = input("Enter your choice (1-4): ").strip()
    print()

    if choice == "1":
        fix_corrupted_users()

    elif choice == "2":
        username = input("Enter username: ").strip()
        password = input("Enter password: ").strip()
        if username and password:
            create_new_user(username, password)
        else:
            print("❌ Username and password required")

    elif choice == "3":
        db_path = get_db_path()
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        users = c.execute("SELECT id, username, hashed_password FROM users").fetchall()

        print("Users in database:")
        print("-" * 60)
        for user_id, username, hashed_password in users:
            hash_status = "✅ Valid" if check_hash_validity(hashed_password) else "❌ Invalid"
            print(f"ID: {user_id:3d} | Username: {username:20s} | Hash: {hash_status}")
        print("-" * 60)
        print(f"Total: {len(users)} users")
        conn.close()

    elif choice == "4":
        print("👋 Goodbye!")

    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
