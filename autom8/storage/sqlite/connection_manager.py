"""
Enhanced Database Connection Management for Autom8.

This module provides robust connection management with automatic retry,
health monitoring, connection pooling, and error recovery for SQLite
database operations. It's designed to ensure reliable database access
even under adverse conditions.

Key Features:
- Automatic connection retry with exponential backoff
- Connection health monitoring and recovery
- Transaction-safe operations with automatic rollback
- Connection pooling for high-concurrency scenarios
- Comprehensive error handling and logging
- Database corruption detection and recovery
"""

import asyncio
import aiosqlite
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, AsyncContextManager, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class ConnectionState(Enum):
    """Database connection states."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CORRUPTED = "corrupted"
    INACCESSIBLE = "inaccessible"


@dataclass
class ConnectionStats:
    """Connection statistics and metrics."""
    total_connections: int = 0
    active_connections: int = 0
    failed_connections: int = 0
    total_operations: int = 0
    failed_operations: int = 0
    avg_connection_time_ms: float = 0.0
    avg_operation_time_ms: float = 0.0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    uptime_start: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RetryConfig:
    """Configuration for retry operations."""
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True


class DatabaseError(Exception):
    """Base exception for database operations."""
    pass


class ConnectionError(DatabaseError):
    """Database connection-specific errors."""
    pass


class TransactionError(DatabaseError):
    """Transaction-specific errors."""
    pass


class CorruptionError(DatabaseError):
    """Database corruption errors."""
    pass


class ConnectionHealthChecker:
    """Monitors database connection health."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.last_check = None
        self.last_state = ConnectionState.UNKNOWN

    async def check_basic_connectivity(self) -> Tuple[bool, str]:
        """Test basic database connectivity."""
        try:
            async with aiosqlite.connect(self.db_path, timeout=5.0) as conn:
                await conn.execute("SELECT 1")
                return True, "OK"
        except Exception as e:
            return False, str(e)

    async def check_database_integrity(self) -> Tuple[bool, str]:
        """Check database integrity."""
        try:
            async with aiosqlite.connect(self.db_path, timeout=10.0) as conn:
                # Check database integrity
                async with conn.execute("PRAGMA integrity_check") as cursor:
                    result = await cursor.fetchone()
                    if result and result[0] == "ok":
                        return True, "Integrity OK"
                    else:
                        return False, f"Integrity check failed: {result[0] if result else 'Unknown'}"
        except Exception as e:
            return False, f"Integrity check error: {e}"

    async def check_schema_validity(self) -> Tuple[bool, str]:
        """Check that required tables exist and are accessible."""
        required_tables = [
            'context_registry',
            'agent_decisions',
            'usage_ledger',
            'model_performance'
        ]

        try:
            async with aiosqlite.connect(self.db_path, timeout=5.0) as conn:
                for table in required_tables:
                    try:
                        await conn.execute(f"SELECT COUNT(*) FROM {table} LIMIT 1")
                    except Exception as e:
                        return False, f"Table {table} error: {e}"

                return True, "Schema valid"

        except Exception as e:
            return False, f"Schema check error: {e}"

    async def check_write_permissions(self) -> Tuple[bool, str]:
        """Test write permissions and basic operations."""
        try:
            async with aiosqlite.connect(self.db_path, timeout=5.0) as conn:
                # Try to create a temporary table and insert data
                test_table = f"test_write_{int(time.time() * 1000)}"

                await conn.execute(f"""
                    CREATE TEMPORARY TABLE {test_table} (
                        id INTEGER PRIMARY KEY,
                        test_data TEXT
                    )
                """)

                await conn.execute(f"""
                    INSERT INTO {test_table} (test_data) VALUES (?)
                """, ("test_write_check",))

                await conn.execute(f"DROP TABLE {test_table}")
                await conn.commit()

                return True, "Write permissions OK"

        except Exception as e:
            return False, f"Write permission error: {e}"

    async def comprehensive_health_check(self) -> Tuple[ConnectionState, Dict[str, Any]]:
        """Perform comprehensive health check."""
        self.last_check = datetime.utcnow()

        checks = {
            'connectivity': await self.check_basic_connectivity(),
            'integrity': await self.check_database_integrity(),
            'schema': await self.check_schema_validity(),
            'write_permissions': await self.check_write_permissions()
        }

        # Determine overall state
        all_passed = all(check[0] for check in checks.values())
        critical_passed = checks['connectivity'][0] and checks['schema'][0]

        if all_passed:
            state = ConnectionState.HEALTHY
        elif critical_passed:
            state = ConnectionState.DEGRADED
        elif not checks['integrity'][0]:
            state = ConnectionState.CORRUPTED
        else:
            state = ConnectionState.INACCESSIBLE

        self.last_state = state

        report = {
            'state': state.value,
            'timestamp': self.last_check.isoformat(),
            'checks': {name: {'passed': result[0], 'message': result[1]}
                      for name, result in checks.items()},
            'database_exists': self.db_path.exists(),
            'database_size_mb': round(self.db_path.stat().st_size / (1024 * 1024), 2) if self.db_path.exists() else 0
        }

        return state, report


class ConnectionManager:
    """Enhanced database connection manager with retry and health monitoring."""

    def __init__(
        self,
        db_path: str,
        retry_config: RetryConfig = None,
        health_check_interval: float = 300.0,  # 5 minutes
        max_connection_age: float = 3600.0,   # 1 hour
        enable_wal: bool = True,
        foreign_keys: bool = True
    ):
        """
        Initialize connection manager.

        Args:
            db_path: Path to SQLite database
            retry_config: Retry configuration
            health_check_interval: Seconds between health checks
            max_connection_age: Maximum connection age before refresh
            enable_wal: Enable WAL journal mode
            foreign_keys: Enable foreign key constraints
        """
        self.db_path = Path(db_path)
        self.retry_config = retry_config or RetryConfig()
        self.health_check_interval = health_check_interval
        self.max_connection_age = max_connection_age
        self.enable_wal = enable_wal
        self.foreign_keys = foreign_keys

        # State management
        self.health_checker = ConnectionHealthChecker(str(self.db_path))
        self.stats = ConnectionStats()
        self._connection_cache = {}
        self._last_health_check = None
        self._health_check_task = None

        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Connection manager initialized for {self.db_path}")

    async def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter."""
        base_delay = min(
            self.retry_config.base_delay_seconds * (self.retry_config.exponential_base ** attempt),
            self.retry_config.max_delay_seconds
        )

        if self.retry_config.jitter:
            import random
            jitter = random.uniform(0.1, 0.3) * base_delay
            return base_delay + jitter

        return base_delay

    async def _configure_connection(self, conn: aiosqlite.Connection):
        """Configure connection with optimal settings."""
        try:
            # Enable WAL mode for better concurrency
            if self.enable_wal:
                await conn.execute("PRAGMA journal_mode=WAL")

            # Enable foreign key constraints
            if self.foreign_keys:
                await conn.execute("PRAGMA foreign_keys=ON")

            # Optimize for performance
            await conn.execute("PRAGMA synchronous=NORMAL")
            await conn.execute("PRAGMA cache_size=10000")
            await conn.execute("PRAGMA temp_store=memory")
            await conn.execute("PRAGMA mmap_size=268435456")  # 256MB

            # Set busy timeout for better handling of concurrent access
            await conn.execute("PRAGMA busy_timeout=30000")  # 30 seconds

        except Exception as e:
            logger.warning(f"Failed to configure connection: {e}")

    async def _create_connection(self) -> aiosqlite.Connection:
        """Create a new database connection with proper configuration."""
        start_time = time.perf_counter()

        try:
            conn = await aiosqlite.connect(
                self.db_path,
                timeout=30.0,
                check_same_thread=False
            )

            await self._configure_connection(conn)

            # Record stats
            connection_time = (time.perf_counter() - start_time) * 1000
            self.stats.total_connections += 1
            self.stats.avg_connection_time_ms = (
                (self.stats.avg_connection_time_ms * (self.stats.total_connections - 1) + connection_time) /
                self.stats.total_connections
            )

            logger.debug(f"Created database connection in {connection_time:.2f}ms")
            return conn

        except Exception as e:
            self.stats.failed_connections += 1
            self.stats.last_error = str(e)
            self.stats.last_error_time = datetime.utcnow()
            logger.error(f"Failed to create database connection: {e}")
            raise ConnectionError(f"Could not create database connection: {e}")

    @asynccontextmanager
    async def get_connection(self) -> AsyncContextManager[aiosqlite.Connection]:
        """Get a managed database connection with automatic cleanup."""
        conn = None

        try:
            for attempt in range(self.retry_config.max_attempts):
                try:
                    conn = await self._create_connection()
                    self.stats.active_connections += 1

                    # Yield connection for use
                    yield conn

                    # If we get here, operation was successful
                    break

                except Exception as e:
                    self.stats.failed_operations += 1

                    if conn:
                        try:
                            await conn.close()
                        except:
                            pass
                        conn = None

                    # If this is the last attempt, re-raise the exception
                    if attempt == self.retry_config.max_attempts - 1:
                        logger.error(f"Connection failed after {self.retry_config.max_attempts} attempts: {e}")
                        raise

                    # Wait before retrying
                    delay = await self._calculate_retry_delay(attempt)
                    logger.warning(f"Connection attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)

        finally:
            # Cleanup
            if conn:
                try:
                    await conn.close()
                    self.stats.active_connections = max(0, self.stats.active_connections - 1)
                except Exception as e:
                    logger.warning(f"Error closing connection: {e}")

    @asynccontextmanager
    async def transaction(self) -> AsyncContextManager[aiosqlite.Connection]:
        """Get a connection with automatic transaction management."""
        async with self.get_connection() as conn:
            try:
                await conn.execute("BEGIN")
                yield conn
                await conn.commit()

            except Exception as e:
                try:
                    await conn.rollback()
                except Exception as rollback_error:
                    logger.error(f"Rollback failed: {rollback_error}")

                logger.error(f"Transaction failed, rolled back: {e}")
                raise TransactionError(f"Transaction failed: {e}")

    async def execute_with_retry(
        self,
        query: str,
        parameters: Tuple = (),
        fetch_method: str = "none"  # "none", "one", "all"
    ) -> Any:
        """Execute a query with automatic retry logic."""
        for attempt in range(self.retry_config.max_attempts):
            try:
                async with self.get_connection() as conn:
                    start_time = time.perf_counter()

                    cursor = await conn.execute(query, parameters)

                    if fetch_method == "one":
                        result = await cursor.fetchone()
                    elif fetch_method == "all":
                        result = await cursor.fetchall()
                    else:
                        result = cursor.rowcount

                    await conn.commit()

                    # Record operation stats
                    operation_time = (time.perf_counter() - start_time) * 1000
                    self.stats.total_operations += 1
                    self.stats.avg_operation_time_ms = (
                        (self.stats.avg_operation_time_ms * (self.stats.total_operations - 1) + operation_time) /
                        self.stats.total_operations
                    )

                    return result

            except Exception as e:
                self.stats.failed_operations += 1
                self.stats.last_error = str(e)
                self.stats.last_error_time = datetime.utcnow()

                if attempt == self.retry_config.max_attempts - 1:
                    logger.error(f"Query failed after {self.retry_config.max_attempts} attempts: {e}")
                    raise

                delay = await self._calculate_retry_delay(attempt)
                logger.warning(f"Query attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                await asyncio.sleep(delay)

    async def check_health(self) -> Tuple[ConnectionState, Dict[str, Any]]:
        """Check connection health and return status."""
        return await self.health_checker.comprehensive_health_check()

    async def recover_from_corruption(self) -> bool:
        """Attempt to recover from database corruption."""
        logger.warning("Attempting database corruption recovery...")

        try:
            # First, try to backup any recoverable data
            backup_path = self.db_path.with_suffix('.db.backup')

            if self.db_path.exists():
                # Try to dump recoverable data
                try:
                    async with aiosqlite.connect(self.db_path) as conn:
                        await conn.execute(f"VACUUM INTO '{backup_path}'")

                    logger.info(f"Created backup before recovery: {backup_path}")

                except Exception as e:
                    logger.warning(f"Could not create backup: {e}")

            # Try integrity check and repair
            async with aiosqlite.connect(self.db_path) as conn:
                # Try REINDEX
                await conn.execute("REINDEX")

                # Try VACUUM
                await conn.execute("VACUUM")

                # Check integrity again
                async with conn.execute("PRAGMA integrity_check") as cursor:
                    result = await cursor.fetchone()
                    if result and result[0] == "ok":
                        logger.info("Database corruption recovery successful")
                        return True

            logger.error("Database corruption recovery failed")
            return False

        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            return False

    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection manager statistics."""
        uptime = datetime.utcnow() - self.stats.uptime_start

        return {
            'uptime_seconds': uptime.total_seconds(),
            'total_connections': self.stats.total_connections,
            'active_connections': self.stats.active_connections,
            'failed_connections': self.stats.failed_connections,
            'total_operations': self.stats.total_operations,
            'failed_operations': self.stats.failed_operations,
            'success_rate': (
                (self.stats.total_operations - self.stats.failed_operations) /
                max(1, self.stats.total_operations)
            ) * 100,
            'avg_connection_time_ms': self.stats.avg_connection_time_ms,
            'avg_operation_time_ms': self.stats.avg_operation_time_ms,
            'last_error': self.stats.last_error,
            'last_error_time': self.stats.last_error_time.isoformat() if self.stats.last_error_time else None,
            'last_health_check': self.health_checker.last_check.isoformat() if self.health_checker.last_check else None,
            'last_health_state': self.health_checker.last_state.value,
            'database_path': str(self.db_path),
            'database_exists': self.db_path.exists(),
            'database_size_mb': round(self.db_path.stat().st_size / (1024 * 1024), 2) if self.db_path.exists() else 0
        }

    async def start_background_monitoring(self):
        """Start background health monitoring."""
        if self._health_check_task and not self._health_check_task.done():
            return

        async def monitor():
            while True:
                try:
                    await asyncio.sleep(self.health_check_interval)
                    state, report = await self.check_health()

                    if state in [ConnectionState.CORRUPTED, ConnectionState.INACCESSIBLE]:
                        logger.warning(f"Database health issue detected: {state.value}")

                        if state == ConnectionState.CORRUPTED:
                            await self.recover_from_corruption()

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")

        self._health_check_task = asyncio.create_task(monitor())
        logger.info("Background health monitoring started")

    async def stop_background_monitoring(self):
        """Stop background health monitoring."""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        logger.info("Background health monitoring stopped")

    async def close(self):
        """Clean up connection manager resources."""
        await self.stop_background_monitoring()

        # Close any cached connections
        for conn in self._connection_cache.values():
            try:
                await conn.close()
            except:
                pass

        self._connection_cache.clear()
        logger.info("Connection manager closed")


# Global connection manager instance
_connection_manager = None


async def get_connection_manager(
    db_path: str = "./autom8.db",
    retry_config: RetryConfig = None,
    reset_instance: bool = False
) -> ConnectionManager:
    """Get global connection manager instance."""
    global _connection_manager

    if _connection_manager is None or reset_instance:
        _connection_manager = ConnectionManager(
            db_path=db_path,
            retry_config=retry_config
        )
        await _connection_manager.start_background_monitoring()

    return _connection_manager