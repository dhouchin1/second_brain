"""
Comprehensive Database Testing Utilities for Autom8.

This module provides extensive testing capabilities for the SQLite database
system, including unit tests, integration tests, performance benchmarks,
and validation utilities. It's designed to ensure the reliability and
performance of the Autom8 database layer.

Key Features:
- Automated database schema validation
- Performance benchmarking for operations
- Data integrity testing
- Vector search accuracy testing
- Stress testing for concurrent operations
- Mock data generation for testing
- Database migration testing
"""

import asyncio
import json
import random
import string
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from contextlib import asynccontextmanager

import numpy as np

from autom8.storage.sqlite.manager import SQLiteManager
from autom8.storage.sqlite.vector_manager import SQLiteVectorManager, VectorSearchConfig
from autom8.storage.sqlite.integrated_manager import IntegratedSQLiteManager
from autom8.storage.sqlite.connection_manager import ConnectionManager
from autom8.storage.sqlite.database_setup import DatabaseSetup, DatabaseSetupConfig
from autom8.storage.sqlite.migrations import MigrationManager
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TestResult:
    """Test result with timing and status information."""
    test_name: str
    success: bool
    duration_ms: float
    error: Optional[str] = None
    details: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'success': self.success,
            'duration_ms': self.duration_ms,
            'error': self.error,
            'details': self.details or {}
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for database operations."""
    operation: str
    total_operations: int
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    success_rate: float
    operations_per_second: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation': self.operation,
            'total_operations': self.total_operations,
            'avg_time_ms': self.avg_time_ms,
            'min_time_ms': self.min_time_ms,
            'max_time_ms': self.max_time_ms,
            'success_rate': self.success_rate,
            'operations_per_second': self.operations_per_second
        }


class MockDataGenerator:
    """Generates realistic mock data for testing."""

    def __init__(self, seed: int = 42):
        """Initialize with optional random seed for reproducible tests."""
        random.seed(seed)
        np.random.seed(seed)

    def generate_content(self, length: int = None) -> str:
        """Generate random content of specified length."""
        if length is None:
            length = random.randint(50, 500)

        words = [
            'autom8', 'context', 'intelligent', 'model', 'routing', 'semantic', 'search',
            'vector', 'embedding', 'database', 'sqlite', 'performance', 'optimization',
            'agent', 'decision', 'complexity', 'analysis', 'memory', 'shared', 'redis',
            'transparency', 'user', 'preview', 'token', 'limit', 'efficiency', 'cost',
            'quality', 'local', 'cloud', 'api', 'configuration', 'settings', 'pipeline'
        ]

        content_words = []
        current_length = 0

        while current_length < length:
            word = random.choice(words)
            content_words.append(word)
            current_length += len(word) + 1  # +1 for space

        return ' '.join(content_words)[:length]

    def generate_context_item(self, content_id: str = None) -> Dict[str, Any]:
        """Generate a complete context item for testing."""
        if content_id is None:
            content_id = f"test_{uuid.uuid4().hex[:8]}"

        topics = ['authentication', 'routing', 'context', 'performance', 'testing', 'configuration']
        source_types = ['documentation', 'code', 'decision', 'configuration', 'example']

        content = self.generate_content()
        summary = self.generate_content(50) if random.random() > 0.3 else None

        return {
            'content_id': content_id,
            'content': content,
            'summary': summary,
            'topic': random.choice(topics),
            'priority': random.randint(0, 100),
            'pinned': random.random() > 0.8,
            'source_type': random.choice(source_types),
            'metadata': {
                'generated_at': datetime.utcnow().isoformat(),
                'test_data': True,
                'complexity': random.uniform(0.1, 1.0),
                'tags': random.sample(['test', 'mock', 'sample', 'data', 'example'], k=random.randint(1, 3))
            }
        }

    def generate_test_dataset(self, size: int = 100) -> List[Dict[str, Any]]:
        """Generate a dataset of test items."""
        return [self.generate_context_item() for _ in range(size)]

    def generate_embedding(self, dimensions: int = 384) -> np.ndarray:
        """Generate a random embedding vector."""
        # Generate random normalized vector
        vector = np.random.normal(0, 1, dimensions)
        return vector / np.linalg.norm(vector)


class DatabaseTester:
    """Comprehensive database testing framework."""

    def __init__(self, test_db_path: str = None):
        """Initialize with test database path."""
        if test_db_path is None:
            test_db_path = f"./test_autom8_{int(time.time())}.db"

        self.test_db_path = Path(test_db_path)
        self.mock_data = MockDataGenerator()
        self.test_results = []

        # Ensure test database directory exists
        self.test_db_path.parent.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def get_test_manager(self, manager_type: str = "integrated"):
        """Get a test database manager."""
        try:
            if manager_type == "sqlite":
                manager = SQLiteManager(str(self.test_db_path))
                await manager.initialize()
            elif manager_type == "vector":
                config = VectorSearchConfig(embedding_model="all-MiniLM-L6-v2")
                manager = SQLiteVectorManager(str(self.test_db_path), config)
                await manager.initialize()
            elif manager_type == "integrated":
                manager = IntegratedSQLiteManager(str(self.test_db_path), auto_embed=False)
                await manager.initialize()
            else:
                raise ValueError(f"Unknown manager type: {manager_type}")

            yield manager

        finally:
            if hasattr(manager, 'close'):
                await manager.close()

    async def run_test(self, test_func, test_name: str, *args, **kwargs) -> TestResult:
        """Run a single test and record results."""
        start_time = time.perf_counter()

        try:
            result = await test_func(*args, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000

            test_result = TestResult(
                test_name=test_name,
                success=True,
                duration_ms=duration_ms,
                details=result if isinstance(result, dict) else None
            )

            logger.info(f"✓ {test_name} passed in {duration_ms:.2f}ms")

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            test_result = TestResult(
                test_name=test_name,
                success=False,
                duration_ms=duration_ms,
                error=str(e)
            )

            logger.error(f"✗ {test_name} failed in {duration_ms:.2f}ms: {e}")

        self.test_results.append(test_result)
        return test_result

    async def test_database_setup(self) -> Dict[str, Any]:
        """Test database setup and initialization."""
        setup = DatabaseSetup(DatabaseSetupConfig(db_path=str(self.test_db_path)))

        # Test complete setup
        setup_success = await setup.setup_complete_database()

        # Test validation
        validation_success = await setup.validate_setup()

        return {
            'setup_success': setup_success,
            'validation_success': validation_success,
            'database_exists': self.test_db_path.exists()
        }

    async def test_basic_crud_operations(self) -> Dict[str, Any]:
        """Test basic CRUD operations."""
        async with self.get_test_manager("sqlite") as manager:
            # Test data
            test_item = self.mock_data.generate_context_item("crud_test_001")

            # CREATE
            create_success = await manager.store_context(**test_item)

            # READ
            retrieved = await manager.get_context(test_item['content_id'])
            read_success = retrieved is not None and retrieved['content'] == test_item['content']

            # UPDATE
            updated_content = "Updated test content"
            update_success = await manager.store_context(
                test_item['content_id'],
                updated_content,
                test_item['summary'],
                test_item['topic']
            )

            # Verify update
            updated_retrieved = await manager.get_context(test_item['content_id'])
            update_verified = updated_retrieved['content'] == updated_content

            # DELETE (using raw SQL since manager doesn't have delete method)
            async with manager._get_connection() as conn:
                cursor = await conn.execute(
                    "DELETE FROM context_registry WHERE id = ?",
                    (test_item['content_id'],)
                )
                delete_success = cursor.rowcount > 0
                await conn.commit()

            return {
                'create_success': create_success,
                'read_success': read_success,
                'update_success': update_success and update_verified,
                'delete_success': delete_success
            }

    async def test_vector_operations(self) -> Dict[str, Any]:
        """Test vector embedding and search operations."""
        async with self.get_test_manager("vector") as manager:
            # Test data
            test_items = [
                {
                    'content_id': 'vector_test_001',
                    'content': 'This is about database performance optimization techniques',
                    'topic': 'performance'
                },
                {
                    'content_id': 'vector_test_002',
                    'content': 'Machine learning model routing and complexity analysis',
                    'topic': 'routing'
                },
                {
                    'content_id': 'vector_test_003',
                    'content': 'Database optimization and performance tuning strategies',
                    'topic': 'performance'
                }
            ]

            # Store content with embeddings
            store_results = []
            for item in test_items:
                success = await manager.store_content_with_embedding(**item)
                store_results.append(success)

            store_success = all(store_results)

            # Test semantic search
            search_results = await manager.semantic_search(
                query="database performance optimization",
                k=3,
                threshold=0.1
            )

            search_success = len(search_results) > 0

            # Test similarity search
            similarity_results = await manager.get_similar_content(
                content_id='vector_test_001',
                k=2,
                threshold=0.1
            )

            similarity_success = len(similarity_results) > 0

            return {
                'store_success': store_success,
                'search_success': search_success,
                'similarity_success': similarity_success,
                'search_results_count': len(search_results),
                'similarity_results_count': len(similarity_results)
            }

    async def test_integrated_operations(self) -> Dict[str, Any]:
        """Test integrated manager operations."""
        async with self.get_test_manager("integrated") as manager:
            # Test data
            test_item = self.mock_data.generate_context_item("integrated_test_001")

            # Store content
            store_result = await manager.store_content(**test_item)

            # Search using different methods
            text_results = await manager.search_content(
                query=test_item['topic'],
                method="text",
                k=5
            )

            hybrid_results = await manager.search_content(
                query=test_item['content'][:50],
                method="hybrid",
                k=5
            )

            # Get stats
            stats = await manager.get_content_stats()

            return {
                'store_success': store_result.success,
                'text_search_count': len(text_results),
                'hybrid_search_count': len(hybrid_results),
                'total_content_items': stats.get('total_content_items', 0),
                'integration_stats': stats.get('integration_stats', {})
            }

    async def test_migration_system(self) -> Dict[str, Any]:
        """Test database migration system."""
        manager = MigrationManager(str(self.test_db_path))

        # Get migration status
        status = await manager.get_migration_status()

        # Apply migrations
        applied, failed = await manager.apply_migrations()

        # Get updated status
        final_status = await manager.get_migration_status()

        return {
            'initial_status': status,
            'migrations_applied': applied,
            'migrations_failed': failed,
            'final_status': final_status,
            'up_to_date': final_status.get('up_to_date', False)
        }

    async def test_connection_manager(self) -> Dict[str, Any]:
        """Test connection manager functionality."""
        conn_manager = ConnectionManager(str(self.test_db_path))

        # Test basic connection
        connection_success = False
        try:
            async with conn_manager.get_connection() as conn:
                await conn.execute("SELECT 1")
                connection_success = True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")

        # Test transaction
        transaction_success = False
        try:
            async with conn_manager.transaction() as conn:
                await conn.execute("CREATE TEMPORARY TABLE test_txn (id INTEGER)")
                await conn.execute("INSERT INTO test_txn (id) VALUES (1)")
                transaction_success = True
        except Exception as e:
            logger.error(f"Transaction test failed: {e}")

        # Get stats
        stats = await conn_manager.get_connection_stats()

        # Health check
        health_state, health_report = await conn_manager.check_health()

        await conn_manager.close()

        return {
            'connection_success': connection_success,
            'transaction_success': transaction_success,
            'health_state': health_state.value,
            'connection_stats': stats,
            'health_report': health_report
        }

    async def benchmark_operations(self, operation_count: int = 100) -> List[PerformanceMetrics]:
        """Benchmark database operations performance."""
        metrics = []

        async with self.get_test_manager("integrated") as manager:
            # Benchmark: Content storage
            storage_times = []
            storage_successes = 0

            for i in range(operation_count):
                test_item = self.mock_data.generate_context_item(f"bench_store_{i}")

                start_time = time.perf_counter()
                result = await manager.store_content(**test_item)
                duration = (time.perf_counter() - start_time) * 1000

                storage_times.append(duration)
                if result.success:
                    storage_successes += 1

            storage_metrics = PerformanceMetrics(
                operation="content_storage",
                total_operations=operation_count,
                avg_time_ms=sum(storage_times) / len(storage_times),
                min_time_ms=min(storage_times),
                max_time_ms=max(storage_times),
                success_rate=(storage_successes / operation_count) * 100,
                operations_per_second=1000 / (sum(storage_times) / len(storage_times))
            )
            metrics.append(storage_metrics)

            # Benchmark: Content retrieval
            retrieval_times = []
            retrieval_successes = 0

            for i in range(min(operation_count, 50)):  # Limit to stored items
                content_id = f"bench_store_{i}"

                start_time = time.perf_counter()
                result = await manager.get_content(content_id)
                duration = (time.perf_counter() - start_time) * 1000

                retrieval_times.append(duration)
                if result is not None:
                    retrieval_successes += 1

            if retrieval_times:
                retrieval_metrics = PerformanceMetrics(
                    operation="content_retrieval",
                    total_operations=len(retrieval_times),
                    avg_time_ms=sum(retrieval_times) / len(retrieval_times),
                    min_time_ms=min(retrieval_times),
                    max_time_ms=max(retrieval_times),
                    success_rate=(retrieval_successes / len(retrieval_times)) * 100,
                    operations_per_second=1000 / (sum(retrieval_times) / len(retrieval_times))
                )
                metrics.append(retrieval_metrics)

            # Benchmark: Search operations
            search_times = []
            search_successes = 0

            search_queries = ["performance", "database", "optimization", "testing", "context"]

            for query in search_queries:
                for _ in range(operation_count // len(search_queries)):
                    start_time = time.perf_counter()
                    results = await manager.search_content(query, method="hybrid", k=5)
                    duration = (time.perf_counter() - start_time) * 1000

                    search_times.append(duration)
                    if len(results) > 0:
                        search_successes += 1

            if search_times:
                search_metrics = PerformanceMetrics(
                    operation="content_search",
                    total_operations=len(search_times),
                    avg_time_ms=sum(search_times) / len(search_times),
                    min_time_ms=min(search_times),
                    max_time_ms=max(search_times),
                    success_rate=(search_successes / len(search_times)) * 100,
                    operations_per_second=1000 / (sum(search_times) / len(search_times))
                )
                metrics.append(search_metrics)

        return metrics

    async def stress_test_concurrent_operations(self, concurrent_tasks: int = 10) -> Dict[str, Any]:
        """Stress test with concurrent database operations."""

        async def worker_task(worker_id: int, operations: int):
            """Worker task for concurrent operations."""
            successes = 0
            errors = []

            async with self.get_test_manager("integrated") as manager:
                for i in range(operations):
                    try:
                        # Store content
                        test_item = self.mock_data.generate_context_item(f"stress_{worker_id}_{i}")
                        result = await manager.store_content(**test_item)

                        if result.success:
                            successes += 1

                        # Search content
                        await manager.search_content(query="test", method="text", k=5)

                    except Exception as e:
                        errors.append(str(e))

            return {
                'worker_id': worker_id,
                'successes': successes,
                'errors': errors,
                'operations': operations
            }

        # Run concurrent workers
        tasks = []
        operations_per_worker = 20

        for worker_id in range(concurrent_tasks):
            task = asyncio.create_task(worker_task(worker_id, operations_per_worker))
            tasks.append(task)

        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_duration = time.perf_counter() - start_time

        # Analyze results
        total_successes = 0
        total_errors = 0
        all_errors = []

        for result in results:
            if isinstance(result, dict):
                total_successes += result['successes']
                total_errors += len(result['errors'])
                all_errors.extend(result['errors'])

        return {
            'concurrent_tasks': concurrent_tasks,
            'operations_per_task': operations_per_worker,
            'total_operations': concurrent_tasks * operations_per_worker,
            'total_successes': total_successes,
            'total_errors': total_errors,
            'success_rate': (total_successes / (concurrent_tasks * operations_per_worker)) * 100,
            'total_duration_seconds': total_duration,
            'operations_per_second': (concurrent_tasks * operations_per_worker) / total_duration,
            'error_samples': all_errors[:5]  # First 5 errors as examples
        }

    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all database tests and return comprehensive report."""
        logger.info("Starting comprehensive database tests...")

        test_suite = [
            (self.test_database_setup, "Database Setup"),
            (self.test_basic_crud_operations, "Basic CRUD Operations"),
            (self.test_vector_operations, "Vector Operations"),
            (self.test_integrated_operations, "Integrated Operations"),
            (self.test_migration_system, "Migration System"),
            (self.test_connection_manager, "Connection Manager")
        ]

        # Run basic tests
        for test_func, test_name in test_suite:
            await self.run_test(test_func, test_name)

        # Run performance benchmarks
        logger.info("Running performance benchmarks...")
        benchmark_result = await self.run_test(
            self.benchmark_operations,
            "Performance Benchmarks",
            50  # operation count
        )

        # Run stress test
        logger.info("Running stress test...")
        stress_result = await self.run_test(
            self.stress_test_concurrent_operations,
            "Concurrent Stress Test",
            5  # concurrent tasks
        )

        # Generate report
        passed_tests = sum(1 for r in self.test_results if r.success)
        total_tests = len(self.test_results)

        report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
                'total_duration_ms': sum(r.duration_ms for r in self.test_results)
            },
            'test_results': [r.to_dict() for r in self.test_results],
            'database_path': str(self.test_db_path),
            'database_size_mb': round(self.test_db_path.stat().st_size / (1024 * 1024), 2) if self.test_db_path.exists() else 0
        }

        logger.info(f"Tests completed: {passed_tests}/{total_tests} passed")
        return report

    async def cleanup(self):
        """Clean up test resources."""
        try:
            if self.test_db_path.exists():
                self.test_db_path.unlink()

            # Clean up WAL and SHM files
            for suffix in ['-wal', '-shm']:
                wal_file = self.test_db_path.with_suffix(f'.db{suffix}')
                if wal_file.exists():
                    wal_file.unlink()

            logger.info("Test database cleanup completed")

        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


async def run_database_tests(
    test_db_path: str = None,
    cleanup_after: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to run comprehensive database tests.

    Args:
        test_db_path: Path for test database
        cleanup_after: Clean up test database after completion
        verbose: Enable verbose logging

    Returns:
        Test report dictionary
    """
    tester = DatabaseTester(test_db_path)

    try:
        if verbose:
            logger.info(f"Running database tests with test database: {tester.test_db_path}")

        report = await tester.run_comprehensive_tests()

        if verbose:
            print("\n" + "="*60)
            print("DATABASE TEST REPORT")
            print("="*60)
            print(f"Total Tests: {report['test_summary']['total_tests']}")
            print(f"Passed: {report['test_summary']['passed_tests']}")
            print(f"Failed: {report['test_summary']['failed_tests']}")
            print(f"Success Rate: {report['test_summary']['success_rate']:.1f}%")
            print(f"Total Duration: {report['test_summary']['total_duration_ms']:.2f}ms")
            print("="*60)

            # Show failed tests
            failed_tests = [r for r in report['test_results'] if not r['success']]
            if failed_tests:
                print("\nFAILED TESTS:")
                for test in failed_tests:
                    print(f"  ✗ {test['test_name']}: {test['error']}")

        return report

    finally:
        if cleanup_after:
            await tester.cleanup()


if __name__ == "__main__":
    import sys

    async def main():
        """Main function for standalone execution."""
        import argparse

        parser = argparse.ArgumentParser(description="Autom8 Database Testing")
        parser.add_argument("--test-db", help="Test database path")
        parser.add_argument("--no-cleanup", action="store_true", help="Don't cleanup test database")
        parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

        args = parser.parse_args()

        report = await run_database_tests(
            test_db_path=args.test_db,
            cleanup_after=not args.no_cleanup,
            verbose=not args.quiet
        )

        # Exit with appropriate code
        success_rate = report['test_summary']['success_rate']
        if success_rate >= 100:
            sys.exit(0)
        elif success_rate >= 80:
            sys.exit(1)
        else:
            sys.exit(2)

    asyncio.run(main())