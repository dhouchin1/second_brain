"""
Database migration utilities for Autom8 SQLite storage.

Handles schema changes, data migrations, and vector database setup.
"""

import asyncio
import aiosqlite
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class Migration:
    """Represents a single database migration."""
    
    def __init__(self, version: str, description: str, up_sql: str, down_sql: Optional[str] = None):
        self.version = version
        self.description = description
        self.up_sql = up_sql
        self.down_sql = down_sql
    
    async def apply(self, conn: aiosqlite.Connection) -> bool:
        """Apply the migration."""
        try:
            # Execute migration SQL
            await conn.executescript(self.up_sql)
            
            # Record migration in database
            await conn.execute("""
                INSERT INTO migrations (version, description, applied_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (self.version, self.description))
            
            await conn.commit()
            logger.info(f"Applied migration {self.version}: {self.description}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply migration {self.version}: {e}")
            await conn.rollback()
            return False
    
    async def rollback(self, conn: aiosqlite.Connection) -> bool:
        """Rollback the migration."""
        if not self.down_sql:
            logger.error(f"No rollback SQL for migration {self.version}")
            return False
        
        try:
            await conn.executescript(self.down_sql)
            
            # Remove migration record
            await conn.execute("""
                DELETE FROM migrations WHERE version = ?
            """, (self.version,))
            
            await conn.commit()
            logger.info(f"Rolled back migration {self.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback migration {self.version}: {e}")
            await conn.rollback()
            return False


class MigrationManager:
    """Manages database migrations for Autom8."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.migrations: List[Migration] = []
        self._register_migrations()
    
    def _register_migrations(self):
        """Register all available migrations."""
        
        # Migration 001: Initial schema
        self.migrations.append(Migration(
            version="001",
            description="Initial database schema with context registry and usage tracking",
            up_sql="""
                -- Create migrations table
                CREATE TABLE IF NOT EXISTS migrations (
                    version TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Context registry for reusable snippets
                CREATE TABLE IF NOT EXISTS context_registry (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    summary TEXT,
                    topic TEXT,
                    priority INTEGER DEFAULT 0,
                    pinned BOOLEAN DEFAULT FALSE,
                    expires_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    token_count INTEGER DEFAULT 0,
                    content_hash TEXT,
                    source_type TEXT DEFAULT 'unknown',
                    metadata TEXT DEFAULT '{}'
                );
                
                -- Agent decisions and actions
                CREATE TABLE IF NOT EXISTS agent_decisions (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    decision_type TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    full_content TEXT,
                    complexity_score REAL,
                    model_used TEXT,
                    success BOOLEAN DEFAULT TRUE,
                    cost REAL DEFAULT 0.0,
                    latency_ms INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    affects TEXT DEFAULT '[]',
                    tags TEXT DEFAULT '[]'
                );
                
                -- Usage tracking for optimization
                CREATE TABLE IF NOT EXISTS usage_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    query_hash TEXT,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    estimated_cost REAL DEFAULT 0.0,
                    actual_cost REAL DEFAULT 0.0,
                    complexity_score REAL,
                    success BOOLEAN DEFAULT TRUE,
                    latency_ms INTEGER DEFAULT 0,
                    quality_score REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Model performance statistics
                CREATE TABLE IF NOT EXISTS model_performance (
                    model_name TEXT PRIMARY KEY,
                    total_requests INTEGER DEFAULT 0,
                    successful_requests INTEGER DEFAULT 0,
                    failed_requests INTEGER DEFAULT 0,
                    avg_latency_ms REAL DEFAULT 0.0,
                    avg_input_tokens REAL DEFAULT 0.0,
                    avg_output_tokens REAL DEFAULT 0.0,
                    total_cost REAL DEFAULT 0.0,
                    avg_quality_score REAL DEFAULT 0.0,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Summaries for compressed knowledge
                CREATE TABLE IF NOT EXISTS summaries (
                    id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    title TEXT,
                    content TEXT NOT NULL,
                    source_ids TEXT DEFAULT '[]',
                    token_count INTEGER DEFAULT 0,
                    compression_ratio REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                );
                
                -- Create indexes for performance
                CREATE INDEX IF NOT EXISTS idx_context_topic ON context_registry(topic);
                CREATE INDEX IF NOT EXISTS idx_context_priority ON context_registry(priority DESC);
                CREATE INDEX IF NOT EXISTS idx_context_created ON context_registry(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_decisions_agent ON agent_decisions(agent_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_decisions_type ON agent_decisions(decision_type);
                CREATE INDEX IF NOT EXISTS idx_usage_model ON usage_ledger(model, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_usage_agent ON usage_ledger(agent_id, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_summaries_scope ON summaries(scope, created_at DESC);
            """,
            down_sql="""
                DROP TABLE IF EXISTS summaries;
                DROP TABLE IF EXISTS model_performance;
                DROP TABLE IF EXISTS usage_ledger;
                DROP TABLE IF EXISTS agent_decisions;
                DROP TABLE IF EXISTS context_registry;
                DROP TABLE IF EXISTS migrations;
            """
        ))
        
        # Migration 002: Vector embeddings support
        self.migrations.append(Migration(
            version="002",
            description="Add vector embeddings support with sqlite-vec integration",
            up_sql="""
                -- Fallback embeddings table (used when sqlite-vec is not available)
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    content_id TEXT NOT NULL,
                    embedding_model TEXT DEFAULT 'unknown',
                    embedding_data BLOB,
                    dimensions INTEGER DEFAULT 384,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (content_id) REFERENCES context_registry (id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_embeddings_content ON embeddings(content_id);
                CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(embedding_model);
            """,
            down_sql="""
                DROP TABLE IF EXISTS embeddings;
            """
        ))
        
        # Migration 003: Enhanced context metadata
        self.migrations.append(Migration(
            version="003",
            description="Add enhanced metadata and search capabilities to context registry",
            up_sql="""
                -- Add new columns to context_registry if they don't exist
                ALTER TABLE context_registry ADD COLUMN embedding_model TEXT DEFAULT 'none';
                ALTER TABLE context_registry ADD COLUMN similarity_threshold REAL DEFAULT 0.5;
                ALTER TABLE context_registry ADD COLUMN access_count INTEGER DEFAULT 0;
                ALTER TABLE context_registry ADD COLUMN last_accessed TIMESTAMP;
                
                -- Update existing records
                UPDATE context_registry SET embedding_model = 'none' WHERE embedding_model IS NULL;
                UPDATE context_registry SET similarity_threshold = 0.5 WHERE similarity_threshold IS NULL;
                UPDATE context_registry SET access_count = 0 WHERE access_count IS NULL;
                
                -- Add indexes for new columns
                CREATE INDEX IF NOT EXISTS idx_context_embedding_model ON context_registry(embedding_model);
                CREATE INDEX IF NOT EXISTS idx_context_access_count ON context_registry(access_count DESC);
                CREATE INDEX IF NOT EXISTS idx_context_last_accessed ON context_registry(last_accessed DESC);
            """,
            down_sql="""
                -- SQLite doesn't support DROP COLUMN, so we'll create a new table without the columns
                CREATE TABLE context_registry_backup AS 
                SELECT id, content, summary, topic, priority, pinned, expires_at, 
                       created_at, updated_at, token_count, content_hash, source_type, metadata
                FROM context_registry;
                
                DROP TABLE context_registry;
                
                ALTER TABLE context_registry_backup RENAME TO context_registry;
                
                -- Recreate original indexes
                CREATE INDEX IF NOT EXISTS idx_context_topic ON context_registry(topic);
                CREATE INDEX IF NOT EXISTS idx_context_priority ON context_registry(priority DESC);
                CREATE INDEX IF NOT EXISTS idx_context_created ON context_registry(created_at DESC);
            """
        ))
        
        # Migration 004: Vector search optimization
        self.migrations.append(Migration(
            version="004",
            description="Optimize vector search with additional indexes and cleanup procedures",
            up_sql="""
                -- Add performance tracking for vector operations
                CREATE TABLE IF NOT EXISTS vector_search_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT,
                    search_type TEXT DEFAULT 'semantic',
                    results_count INTEGER DEFAULT 0,
                    search_time_ms REAL DEFAULT 0.0,
                    threshold_used REAL DEFAULT 0.5,
                    model_used TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_vector_stats_timestamp ON vector_search_stats(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_vector_stats_search_type ON vector_search_stats(search_type);
                CREATE INDEX IF NOT EXISTS idx_vector_stats_model ON vector_search_stats(model_used);
                
                -- Add vector database configuration
                CREATE TABLE IF NOT EXISTS vector_config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Insert default configuration
                INSERT OR IGNORE INTO vector_config (key, value) VALUES 
                    ('default_embedding_model', 'all-MiniLM-L6-v2'),
                    ('default_similarity_threshold', '0.5'),
                    ('max_search_results', '50'),
                    ('embedding_dimension', '384'),
                    ('auto_cleanup_days', '30');
            """,
            down_sql="""
                DROP TABLE IF EXISTS vector_config;
                DROP TABLE IF EXISTS vector_search_stats;
            """
        ))
    
    async def get_connection(self) -> aiosqlite.Connection:
        """Get database connection."""
        conn = await aiosqlite.connect(self.db_path, timeout=30.0)
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        return conn
    
    async def create_migrations_table(self, conn: aiosqlite.Connection):
        """Ensure migrations table exists."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                version TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await conn.commit()
    
    async def get_applied_migrations(self, conn: aiosqlite.Connection) -> List[str]:
        """Get list of applied migration versions."""
        try:
            await self.create_migrations_table(conn)
            
            async with conn.execute("SELECT version FROM migrations ORDER BY version") as cursor:
                rows = await cursor.fetchall()
                return [row[0] for row in rows]
        except Exception as e:
            logger.error(f"Failed to get applied migrations: {e}")
            return []
    
    async def get_pending_migrations(self, conn: aiosqlite.Connection) -> List[Migration]:
        """Get list of pending migrations."""
        applied = await self.get_applied_migrations(conn)
        return [m for m in self.migrations if m.version not in applied]
    
    async def apply_migrations(self, target_version: Optional[str] = None) -> Tuple[int, int]:
        """
        Apply pending migrations.
        
        Args:
            target_version: Apply migrations up to this version (all if None)
            
        Returns:
            Tuple of (applied_count, failed_count)
        """
        conn = await self.get_connection()
        applied_count = 0
        failed_count = 0
        
        try:
            pending = await self.get_pending_migrations(conn)
            
            for migration in pending:
                if target_version and migration.version > target_version:
                    break
                
                logger.info(f"Applying migration {migration.version}: {migration.description}")
                
                if await migration.apply(conn):
                    applied_count += 1
                else:
                    failed_count += 1
                    break  # Stop on first failure
            
            return applied_count, failed_count
            
        except Exception as e:
            logger.error(f"Migration process failed: {e}")
            return applied_count, failed_count + 1
        
        finally:
            await conn.close()
    
    async def rollback_migration(self, version: str) -> bool:
        """Rollback a specific migration."""
        migration = next((m for m in self.migrations if m.version == version), None)
        if not migration:
            logger.error(f"Migration {version} not found")
            return False
        
        conn = await self.get_connection()
        
        try:
            applied = await self.get_applied_migrations(conn)
            if version not in applied:
                logger.error(f"Migration {version} not applied")
                return False
            
            logger.info(f"Rolling back migration {version}: {migration.description}")
            return await migration.rollback(conn)
            
        finally:
            await conn.close()
    
    async def get_migration_status(self) -> Dict:
        """Get current migration status."""
        conn = await self.get_connection()
        
        try:
            applied = await self.get_applied_migrations(conn)
            pending = await self.get_pending_migrations(conn)
            
            status = {
                "total_migrations": len(self.migrations),
                "applied_count": len(applied),
                "pending_count": len(pending),
                "applied_versions": applied,
                "pending_versions": [m.version for m in pending],
                "up_to_date": len(pending) == 0
            }
            
            # Get last applied migration info
            if applied:
                last_version = applied[-1]
                async with conn.execute("""
                    SELECT description, applied_at FROM migrations 
                    WHERE version = ? ORDER BY version DESC LIMIT 1
                """, (last_version,)) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        status["last_applied"] = {
                            "version": last_version,
                            "description": row[0],
                            "applied_at": row[1]
                        }
            
            return status
            
        finally:
            await conn.close()
    
    async def reset_database(self) -> bool:
        """Reset database by rolling back all migrations."""
        conn = await self.get_connection()
        
        try:
            applied = await self.get_applied_migrations(conn)
            
            # Rollback in reverse order
            for version in reversed(applied):
                migration = next((m for m in self.migrations if m.version == version), None)
                if migration and migration.down_sql:
                    if not await migration.rollback(conn):
                        logger.error(f"Failed to rollback migration {version}")
                        return False
            
            logger.info("Database reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database reset failed: {e}")
            return False
        
        finally:
            await conn.close()
    
    async def setup_vector_search(self) -> bool:
        """
        Set up vector search capabilities.
        
        This tries to enable sqlite-vec if available, otherwise sets up fallback.
        """
        conn = await self.get_connection()
        
        try:
            # Try to load sqlite-vec extension
            vec_available = False
            try:
                await conn.enable_load_extension(True)
                await conn.load_extension("sqlite_vec")
                await conn.enable_load_extension(False)
                
                # Test if vec0 virtual table is available
                await conn.execute("SELECT vec_version()")
                vec_available = True
                logger.info("sqlite-vec extension loaded successfully")
                
            except Exception as e:
                logger.warning(f"sqlite-vec extension not available: {e}")
                vec_available = False
            
            if vec_available:
                # Create vector table using sqlite-vec
                await conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
                        id TEXT PRIMARY KEY,
                        content_id TEXT,
                        embedding_model TEXT,
                        embedding FLOAT[384]
                    )
                """)
                
                logger.info("Vector search tables created with sqlite-vec")
                
                # Update configuration
                await conn.execute("""
                    INSERT OR REPLACE INTO vector_config (key, value, updated_at)
                    VALUES ('vec_extension_available', 'true', CURRENT_TIMESTAMP)
                """)
            else:
                # Update configuration to indicate fallback mode
                await conn.execute("""
                    INSERT OR REPLACE INTO vector_config (key, value, updated_at)
                    VALUES ('vec_extension_available', 'false', CURRENT_TIMESTAMP)
                """)
                
                logger.info("Using fallback vector search mode")
            
            await conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup vector search: {e}")
            return False
        
        finally:
            await conn.close()


async def run_migrations(db_path: str, target_version: Optional[str] = None) -> bool:
    """
    Run database migrations.
    
    Args:
        db_path: Path to SQLite database
        target_version: Target migration version (latest if None)
        
    Returns:
        True if successful, False otherwise
    """
    manager = MigrationManager(db_path)
    
    # Check current status
    status = await manager.get_migration_status()
    logger.info(f"Migration status: {status['applied_count']}/{status['total_migrations']} applied")
    
    if status['up_to_date']:
        logger.info("Database is up to date")
        return True
    
    # Apply migrations
    applied, failed = await manager.apply_migrations(target_version)
    
    if failed > 0:
        logger.error(f"Migration failed: {applied} applied, {failed} failed")
        return False
    
    logger.info(f"Successfully applied {applied} migrations")
    
    # Set up vector search
    if await manager.setup_vector_search():
        logger.info("Vector search setup completed")
    else:
        logger.warning("Vector search setup had issues")
    
    return True


async def get_migration_status(db_path: str) -> Dict:
    """Get migration status for a database."""
    manager = MigrationManager(db_path)
    return await manager.get_migration_status()


async def reset_database(db_path: str) -> bool:
    """Reset database by removing all tables."""
    manager = MigrationManager(db_path)
    return await manager.reset_database()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python migrations.py <db_path> [target_version]")
        sys.exit(1)
    
    db_path = sys.argv[1]
    target_version = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = asyncio.run(run_migrations(db_path, target_version))
    sys.exit(0 if success else 1)