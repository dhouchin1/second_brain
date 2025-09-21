"""
Unified Database Setup and Initialization for Autom8.

This module provides a comprehensive database setup system that initializes
the SQLite database with all necessary tables, indexes, and vector search
capabilities. It serves as the main entry point for database initialization
in the Autom8 system.

Key Features:
- Automated schema creation and migration
- sqlite-vec extension detection and setup
- Configuration-driven initialization
- Health checks and validation
- Integration with existing SQLiteManager and VectorManager
"""

import asyncio
import aiosqlite
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from autom8.storage.sqlite.manager import SQLiteManager, get_sqlite_manager
from autom8.storage.sqlite.vector_manager import SQLiteVectorManager, VectorSearchConfig
from autom8.storage.sqlite.migrations import MigrationManager, run_migrations
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class DatabaseSetupConfig:
    """Configuration for database setup."""

    def __init__(
        self,
        db_path: str = "./autom8.db",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dimensions: int = 384,
        enable_wal: bool = True,
        enable_foreign_keys: bool = True,
        cache_size: int = 10000,
        similarity_threshold: float = 0.5,
        auto_vacuum: bool = True,
        page_size: int = 4096
    ):
        self.db_path = Path(db_path)
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.enable_wal = enable_wal
        self.enable_foreign_keys = enable_foreign_keys
        self.cache_size = cache_size
        self.similarity_threshold = similarity_threshold
        self.auto_vacuum = auto_vacuum
        self.page_size = page_size


class DatabaseHealthCheck:
    """Database health check and validation."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)

    async def check_database_connectivity(self) -> bool:
        """Test basic database connectivity."""
        try:
            async with aiosqlite.connect(self.db_path, timeout=5.0) as conn:
                await conn.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database connectivity check failed: {e}")
            return False

    async def check_schema_integrity(self) -> Dict[str, bool]:
        """Check that all required tables exist."""
        required_tables = [
            'context_registry',
            'agent_decisions',
            'usage_ledger',
            'model_performance',
            'summaries',
            'embeddings',
            'migrations'
        ]

        results = {}

        try:
            async with aiosqlite.connect(self.db_path) as conn:
                async with conn.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """) as cursor:
                    existing_tables = {row[0] for row in await cursor.fetchall()}

                for table in required_tables:
                    results[table] = table in existing_tables

        except Exception as e:
            logger.error(f"Schema integrity check failed: {e}")
            for table in required_tables:
                results[table] = False

        return results

    async def check_vector_capabilities(self) -> Dict[str, Any]:
        """Check vector search capabilities."""
        result = {
            'vec_extension_available': False,
            'vec_table_exists': False,
            'fallback_table_exists': False,
            'total_embeddings': 0,
            'last_embedding_created': None
        }

        try:
            async with aiosqlite.connect(self.db_path) as conn:
                # Check for sqlite-vec extension
                try:
                    await conn.enable_load_extension(True)
                    await conn.load_extension("sqlite_vec")
                    await conn.execute("SELECT vec_version()")
                    result['vec_extension_available'] = True
                    await conn.enable_load_extension(False)
                except:
                    pass

                # Check for vec_embeddings table (virtual table)
                try:
                    await conn.execute("SELECT COUNT(*) FROM vec_embeddings")
                    result['vec_table_exists'] = True

                    async with conn.execute("SELECT COUNT(*) FROM vec_embeddings") as cursor:
                        count = await cursor.fetchone()
                        result['total_embeddings'] = count[0] if count else 0
                except:
                    pass

                # Check for fallback embeddings table
                try:
                    await conn.execute("SELECT COUNT(*) FROM embeddings")
                    result['fallback_table_exists'] = True

                    if not result['vec_table_exists']:
                        async with conn.execute("SELECT COUNT(*) FROM embeddings") as cursor:
                            count = await cursor.fetchone()
                            result['total_embeddings'] = count[0] if count else 0

                        # Get last embedding creation time
                        async with conn.execute("""
                            SELECT MAX(created_at) FROM embeddings
                        """) as cursor:
                            last_created = await cursor.fetchone()
                            if last_created and last_created[0]:
                                result['last_embedding_created'] = last_created[0]
                except:
                    pass

        except Exception as e:
            logger.error(f"Vector capabilities check failed: {e}")

        return result

    async def check_database_performance(self) -> Dict[str, Any]:
        """Basic performance checks."""
        result = {
            'db_size_mb': 0,
            'page_count': 0,
            'page_size': 0,
            'fragmentation_ratio': 0.0,
            'wal_enabled': False,
            'foreign_keys_enabled': False
        }

        try:
            # Get file size
            if self.db_path.exists():
                result['db_size_mb'] = round(self.db_path.stat().st_size / (1024 * 1024), 2)

            async with aiosqlite.connect(self.db_path) as conn:
                # Get database info
                async with conn.execute("PRAGMA page_count") as cursor:
                    row = await cursor.fetchone()
                    result['page_count'] = row[0] if row else 0

                async with conn.execute("PRAGMA page_size") as cursor:
                    row = await cursor.fetchone()
                    result['page_size'] = row[0] if row else 0

                async with conn.execute("PRAGMA freelist_count") as cursor:
                    row = await cursor.fetchone()
                    freelist_count = row[0] if row else 0
                    if result['page_count'] > 0:
                        result['fragmentation_ratio'] = freelist_count / result['page_count']

                # Check settings
                async with conn.execute("PRAGMA journal_mode") as cursor:
                    row = await cursor.fetchone()
                    result['wal_enabled'] = row[0].upper() == 'WAL' if row else False

                async with conn.execute("PRAGMA foreign_keys") as cursor:
                    row = await cursor.fetchone()
                    result['foreign_keys_enabled'] = bool(row[0]) if row else False

        except Exception as e:
            logger.error(f"Performance check failed: {e}")

        return result

    async def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive report."""
        logger.info("Running comprehensive database health check...")

        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'database_path': str(self.db_path),
            'connectivity': await self.check_database_connectivity(),
            'schema': await self.check_schema_integrity(),
            'vector_capabilities': await self.check_vector_capabilities(),
            'performance': await self.check_database_performance(),
            'overall_health': 'unknown'
        }

        # Determine overall health
        connectivity_ok = report['connectivity']
        schema_ok = all(report['schema'].values())
        vector_ok = report['vector_capabilities']['vec_extension_available'] or report['vector_capabilities']['fallback_table_exists']

        if connectivity_ok and schema_ok and vector_ok:
            report['overall_health'] = 'healthy'
        elif connectivity_ok and schema_ok:
            report['overall_health'] = 'degraded'
        else:
            report['overall_health'] = 'unhealthy'

        logger.info(f"Database health check completed. Status: {report['overall_health']}")
        return report


class DatabaseSetup:
    """Main database setup and initialization class."""

    def __init__(self, config: DatabaseSetupConfig = None):
        self.config = config or DatabaseSetupConfig()
        self.db_path = self.config.db_path

        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Database setup initialized for {self.db_path}")

    async def configure_database(self, conn: aiosqlite.Connection):
        """Configure database settings for optimal performance."""
        logger.debug("Configuring database settings...")

        try:
            # Enable WAL mode for better concurrency
            if self.config.enable_wal:
                await conn.execute("PRAGMA journal_mode=WAL")
                logger.debug("Enabled WAL journal mode")

            # Enable foreign key constraints
            if self.config.enable_foreign_keys:
                await conn.execute("PRAGMA foreign_keys=ON")
                logger.debug("Enabled foreign key constraints")

            # Set cache size
            await conn.execute(f"PRAGMA cache_size={self.config.cache_size}")

            # Set page size (only works on new databases)
            await conn.execute(f"PRAGMA page_size={self.config.page_size}")

            # Configure auto vacuum
            if self.config.auto_vacuum:
                await conn.execute("PRAGMA auto_vacuum=INCREMENTAL")

            # Set synchronous mode for performance/safety balance
            await conn.execute("PRAGMA synchronous=NORMAL")

            # Use memory for temporary tables
            await conn.execute("PRAGMA temp_store=memory")

            await conn.commit()
            logger.debug("Database configuration completed")

        except Exception as e:
            logger.error(f"Failed to configure database: {e}")
            raise

    async def run_initial_setup(self) -> bool:
        """Run initial database setup including migrations."""
        logger.info("Starting initial database setup...")

        try:
            # Run migrations to create schema
            logger.info("Running database migrations...")
            if not await run_migrations(str(self.db_path)):
                logger.error("Database migrations failed")
                return False

            # Configure database settings
            async with aiosqlite.connect(self.db_path, timeout=30.0) as conn:
                await self.configure_database(conn)

            logger.info("Initial database setup completed successfully")
            return True

        except Exception as e:
            logger.error(f"Initial database setup failed: {e}")
            return False

    async def initialize_vector_search(self) -> bool:
        """Initialize vector search capabilities."""
        logger.info("Initializing vector search capabilities...")

        try:
            # Create vector manager with our configuration
            vector_config = VectorSearchConfig(
                embedding_model=self.config.embedding_model,
                embedding_dimensions=self.config.embedding_dimensions,
                similarity_threshold=self.config.similarity_threshold
            )

            vector_manager = SQLiteVectorManager(
                db_path=str(self.db_path),
                config=vector_config
            )

            # Initialize vector manager
            if await vector_manager.initialize():
                logger.info("Vector search initialization completed")
                await vector_manager.close()
                return True
            else:
                logger.error("Vector search initialization failed")
                return False

        except Exception as e:
            logger.error(f"Vector search initialization failed: {e}")
            return False

    async def populate_initial_data(self) -> bool:
        """Populate database with initial configuration data."""
        logger.info("Populating initial configuration data...")

        try:
            sqlite_manager = await get_sqlite_manager()

            # Store initial system configuration
            config_data = {
                'embedding_model': self.config.embedding_model,
                'embedding_dimensions': self.config.embedding_dimensions,
                'similarity_threshold': self.config.similarity_threshold,
                'setup_timestamp': datetime.utcnow().isoformat(),
                'version': '3.0'
            }

            success = await sqlite_manager.store_context(
                context_id='system_config',
                content=json.dumps(config_data, indent=2),
                summary='Autom8 system configuration and setup information',
                topic='system',
                priority=100,
                pinned=True,
                source_type='system',
                metadata={
                    'type': 'configuration',
                    'created_by': 'database_setup',
                    'version': '3.0'
                }
            )

            if success:
                logger.debug("Initial configuration data stored")
            else:
                logger.warning("Failed to store initial configuration data")

            # Store example pinned context
            example_context = """
            Autom8 v3.0 Context Guidelines:

            1. Always prioritize context transparency - users should know what's being sent to LLMs
            2. Prefer local models when quality threshold is met (default: 85%)
            3. Use semantic search to find relevant context efficiently
            4. Keep context under token limits through summarization when necessary
            5. Track model performance and usage for optimization
            """

            await sqlite_manager.store_context(
                context_id='context_guidelines',
                content=example_context.strip(),
                summary='Guidelines for context management in Autom8',
                topic='guidelines',
                priority=90,
                pinned=True,
                source_type='documentation',
                metadata={
                    'type': 'guidelines',
                    'category': 'context_management'
                }
            )

            logger.info("Initial data population completed")
            return True

        except Exception as e:
            logger.error(f"Failed to populate initial data: {e}")
            return False

    async def validate_setup(self) -> bool:
        """Validate that the database setup is correct."""
        logger.info("Validating database setup...")

        try:
            health_check = DatabaseHealthCheck(str(self.db_path))
            report = await health_check.run_comprehensive_check()

            if report['overall_health'] == 'healthy':
                logger.info("Database setup validation passed")
                return True
            else:
                logger.error(f"Database setup validation failed: {report['overall_health']}")
                logger.error(f"Health report: {json.dumps(report, indent=2)}")
                return False

        except Exception as e:
            logger.error(f"Setup validation failed: {e}")
            return False

    async def setup_complete_database(self) -> bool:
        """Run complete database setup process."""
        logger.info("=== Starting Complete Database Setup ===")

        try:
            # Step 1: Initial setup and migrations
            if not await self.run_initial_setup():
                return False

            # Step 2: Initialize vector search
            if not await self.initialize_vector_search():
                return False

            # Step 3: Populate initial data
            if not await self.populate_initial_data():
                return False

            # Step 4: Validate setup
            if not await self.validate_setup():
                return False

            logger.info("=== Database Setup Completed Successfully ===")
            return True

        except Exception as e:
            logger.error(f"Complete database setup failed: {e}")
            return False

    async def reset_database(self) -> bool:
        """Reset database to clean state."""
        logger.warning("Resetting database to clean state...")

        try:
            # Close any existing connections
            if self.db_path.exists():
                # Try to close cleanly first
                async with aiosqlite.connect(self.db_path) as conn:
                    await conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

                # Remove database files
                self.db_path.unlink()

                # Remove WAL and SHM files if they exist
                wal_file = self.db_path.with_suffix('.db-wal')
                if wal_file.exists():
                    wal_file.unlink()

                shm_file = self.db_path.with_suffix('.db-shm')
                if shm_file.exists():
                    shm_file.unlink()

            logger.info("Database reset completed")
            return True

        except Exception as e:
            logger.error(f"Database reset failed: {e}")
            return False


async def setup_database(
    db_path: str = "./autom8.db",
    config: DatabaseSetupConfig = None,
    reset_if_exists: bool = False
) -> bool:
    """
    Convenience function to set up Autom8 database.

    Args:
        db_path: Path to database file
        config: Setup configuration
        reset_if_exists: Reset database if it already exists

    Returns:
        True if setup successful, False otherwise
    """
    if config is None:
        config = DatabaseSetupConfig(db_path=db_path)

    setup = DatabaseSetup(config)

    # Reset if requested
    if reset_if_exists and Path(db_path).exists():
        if not await setup.reset_database():
            return False

    # Run complete setup
    return await setup.setup_complete_database()


async def check_database_health(db_path: str = "./autom8.db") -> Dict[str, Any]:
    """
    Check database health and return report.

    Args:
        db_path: Path to database file

    Returns:
        Health check report
    """
    health_check = DatabaseHealthCheck(db_path)
    return await health_check.run_comprehensive_check()


async def main():
    """Main function for standalone execution."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Autom8 Database Setup")
    parser.add_argument("--db-path", default="./autom8.db", help="Database path")
    parser.add_argument("--reset", action="store_true", help="Reset database")
    parser.add_argument("--check-health", action="store_true", help="Check database health")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model")

    args = parser.parse_args()

    if args.check_health:
        report = await check_database_health(args.db_path)
        print(json.dumps(report, indent=2))
        return

    config = DatabaseSetupConfig(
        db_path=args.db_path,
        embedding_model=args.embedding_model
    )

    success = await setup_database(
        db_path=args.db_path,
        config=config,
        reset_if_exists=args.reset
    )

    if success:
        print(f"Database setup completed successfully: {args.db_path}")
        sys.exit(0)
    else:
        print(f"Database setup failed: {args.db_path}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())