"""
Database connection management and utilities for Second Brain.

This module provides centralized database connection handling with:
- Connection pooling and thread-safety
- Proper error handling and recovery
- Database initialization and migration support
- Connection health monitoring
"""

import sqlite3
import threading
import time
import logging
import os
from typing import Optional, Dict, Any
from contextlib import contextmanager
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Centralized database connection manager with pooling and health monitoring."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(settings.db_path)
        self._connections = {}
        self._lock = threading.RLock()
        self._health_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'failed_connections': 0,
            'last_health_check': None
        }
        
    def get_connection(self, check_same_thread: bool = True) -> sqlite3.Connection:
        """Get a database connection with proper error handling."""
        thread_id = threading.get_ident()
        
        with self._lock:
            try:
                # Check if we have a connection for this thread
                if thread_id in self._connections:
                    conn = self._connections[thread_id]
                    # Verify connection is alive
                    try:
                        conn.execute("SELECT 1")
                        return conn
                    except sqlite3.Error:
                        # Connection is dead, remove it
                        del self._connections[thread_id]
                        self._health_stats['active_connections'] -= 1
                
                # Create new connection
                conn = sqlite3.connect(
                    self.db_path, 
                    check_same_thread=check_same_thread,
                    timeout=30.0  # 30 second timeout
                )
                
                # Configure connection
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA temp_store=memory")
                conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory map
                
                # Store connection
                self._connections[thread_id] = conn
                self._health_stats['total_connections'] += 1
                self._health_stats['active_connections'] += 1
                
                logger.debug(f"Created new database connection for thread {thread_id}")
                return conn
                
            except sqlite3.Error as e:
                self._health_stats['failed_connections'] += 1
                logger.error(f"Failed to create database connection: {e}")
                raise
    
    @contextmanager
    def get_db_context(self):
        """Context manager for database connections with automatic cleanup."""
        conn = None
        try:
            conn = self.get_connection()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if conn:
                try:
                    conn.commit()
                except sqlite3.Error:
                    pass
    
    def close_connection(self, thread_id: int = None):
        """Close connection for specific thread."""
        if thread_id is None:
            thread_id = threading.get_ident()
            
        with self._lock:
            if thread_id in self._connections:
                try:
                    self._connections[thread_id].close()
                    del self._connections[thread_id]
                    self._health_stats['active_connections'] -= 1
                    logger.debug(f"Closed database connection for thread {thread_id}")
                except sqlite3.Error as e:
                    logger.error(f"Error closing database connection: {e}")
    
    def close_all_connections(self):
        """Close all active connections."""
        with self._lock:
            for thread_id, conn in list(self._connections.items()):
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
            self._connections.clear()
            self._health_stats['active_connections'] = 0
            logger.info("Closed all database connections")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check and return statistics."""
        health_info = {
            'database_path': self.db_path,
            'database_exists': os.path.exists(self.db_path),
            'database_size_mb': 0,
            'writable': False,
            'connection_test': False,
            'stats': self._health_stats.copy()
        }
        
        try:
            # Check database file
            if health_info['database_exists']:
                health_info['database_size_mb'] = round(
                    os.path.getsize(self.db_path) / (1024 * 1024), 2
                )
                health_info['writable'] = os.access(self.db_path, os.W_OK)
            
            # Test connection
            with self.get_db_context() as conn:
                conn.execute("SELECT 1")
                health_info['connection_test'] = True
                
            self._health_stats['last_health_check'] = time.time()
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            health_info['error'] = str(e)
        
        return health_info
    
    def initialize_database(self):
        """Initialize database with basic schema if needed."""
        try:
            with self.get_db_context() as conn:
                # Check if basic tables exist
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name IN ('notes', 'users')
                """)
                
                existing_tables = [row[0] for row in cursor.fetchall()]
                
                if 'notes' not in existing_tables:
                    logger.info("Creating basic database schema")
                    # Create basic schema - this should ideally use migrations
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS notes (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            title TEXT,
                            content TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                if 'users' not in existing_tables:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            email TEXT UNIQUE,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                logger.info("Database initialization completed")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise


# Global database manager instance
_db_manager = None
_manager_lock = threading.Lock()

def get_db_manager() -> DatabaseManager:
    """Get global database manager instance (singleton)."""
    global _db_manager
    if _db_manager is None:
        with _manager_lock:
            if _db_manager is None:
                _db_manager = DatabaseManager()
    return _db_manager

def get_db_connection() -> sqlite3.Connection:
    """Get database connection - compatible with existing code."""
    return get_db_manager().get_connection()

def get_conn() -> sqlite3.Connection:
    """Legacy alias for get_db_connection."""
    return get_db_connection()

@contextmanager
def db_context():
    """Context manager for database operations."""
    with get_db_manager().get_db_context() as conn:
        yield conn

def close_db_connections():
    """Close all database connections - for cleanup."""
    if _db_manager:
        _db_manager.close_all_connections()

def db_health_check() -> Dict[str, Any]:
    """Get database health information."""
    return get_db_manager().health_check()

def initialize_database():
    """Initialize database if needed."""
    return get_db_manager().initialize_database()

# Test utilities
def create_test_db(test_db_path: str) -> DatabaseManager:
    """Create a test database manager for testing."""
    # Ensure parent directory exists
    Path(test_db_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing test db
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    test_manager = DatabaseManager(test_db_path)
    test_manager.initialize_database()
    return test_manager

# Export commonly used functions
__all__ = [
    'DatabaseManager',
    'get_db_manager',
    'get_db_connection', 
    'get_conn',
    'db_context',
    'close_db_connections',
    'db_health_check',
    'initialize_database',
    'create_test_db'
]