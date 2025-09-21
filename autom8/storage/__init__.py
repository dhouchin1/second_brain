"""
Storage Layer Module

Provides comprehensive storage capabilities for Autom8 including:
- SQLite database with sqlite-vec extension for semantic search
- Redis-based shared memory for agent coordination
- Integrated storage management with automatic embedding generation
- Connection pooling and health monitoring
- Database migration and setup utilities
- Comprehensive testing and validation tools

This module serves as the foundation for Autom8's context transparency
and intelligent model routing by providing reliable, efficient storage
with semantic search capabilities.
"""

# Core managers
from autom8.storage.sqlite.manager import SQLiteManager, get_sqlite_manager
from autom8.storage.sqlite.vector_manager import SQLiteVectorManager, VectorSearchConfig
from autom8.storage.sqlite.integrated_manager import IntegratedSQLiteManager, get_integrated_manager

# Setup and configuration
from autom8.storage.sqlite.database_setup import (
    DatabaseSetup,
    DatabaseSetupConfig,
    DatabaseHealthCheck,
    setup_database,
    check_database_health
)

# Connection management
from autom8.storage.sqlite.connection_manager import (
    ConnectionManager,
    ConnectionState,
    RetryConfig,
    get_connection_manager
)

# Migration system
from autom8.storage.sqlite.migrations import (
    MigrationManager,
    Migration,
    run_migrations,
    get_migration_status,
    reset_database
)

# Testing utilities
from autom8.storage.sqlite.test_utilities import (
    DatabaseTester,
    MockDataGenerator,
    run_database_tests
)

__all__ = [
    # Core managers
    "SQLiteManager",
    "get_sqlite_manager",
    "SQLiteVectorManager",
    "VectorSearchConfig",
    "IntegratedSQLiteManager",
    "get_integrated_manager",

    # Setup and configuration
    "DatabaseSetup",
    "DatabaseSetupConfig",
    "DatabaseHealthCheck",
    "setup_database",
    "check_database_health",

    # Connection management
    "ConnectionManager",
    "ConnectionState",
    "RetryConfig",
    "get_connection_manager",

    # Migration system
    "MigrationManager",
    "Migration",
    "run_migrations",
    "get_migration_status",
    "reset_database",

    # Testing utilities
    "DatabaseTester",
    "MockDataGenerator",
    "run_database_tests"
]