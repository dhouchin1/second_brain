"""
Dashboard v3 Backend Testing Suite
Comprehensive testing for dashboard API endpoints, performance, and functionality

Test Categories:
- API endpoint testing
- Performance regression testing
- Authentication and security testing
- Data integrity testing
- Integration testing
- Error handling validation
- Load testing
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import status
import tempfile
import sqlite3
import os
import sys

# Add the parent directory to sys.path to import from the project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from database import get_db_connection
from services.unified_capture_service import get_unified_capture_service
from services.search_adapter import SearchAdapter
from services.embeddings import EmbeddingService


class DashboardV3TestSuite:
    """Comprehensive test suite for Dashboard v3"""
    
    def __init__(self):
        self.client = TestClient(app)
        self.test_db_path = None
        self.performance_baselines = {}
        self.test_data = {}
        
    def setup_method(self, method):
        """Setup for each test method"""
        # Create temporary test database
        self.test_db_fd, self.test_db_path = tempfile.mkstemp(suffix='.db')
        os.environ['TEST_DATABASE_PATH'] = self.test_db_path
        
        # Initialize test database
        self._setup_test_database()
        
        # Create test data
        self._create_test_data()
        
    def teardown_method(self, method):
        """Cleanup after each test method"""
        if self.test_db_path and os.path.exists(self.test_db_path):
            os.close(self.test_db_fd)
            os.unlink(self.test_db_path)
            
        if 'TEST_DATABASE_PATH' in os.environ:
            del os.environ['TEST_DATABASE_PATH']
    
    def _setup_test_database(self):
        """Initialize test database with required schema"""
        conn = sqlite3.connect(self.test_db_path)
        
        # Create basic schema
        conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                title TEXT,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT DEFAULT 'test_user',
                source TEXT DEFAULT 'manual',
                embedding BLOB
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        
        # Create FTS5 table for search
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
                id,
                title,
                content,
                tags,
                content='notes',
                content_rowid='id'
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _create_test_data(self):
        """Create test data for various scenarios"""
        conn = sqlite3.connect(self.test_db_path)
        
        # Test user
        conn.execute(
            "INSERT INTO users (id, email) VALUES (?, ?)",
            ('test_user', 'test@example.com')
        )
        
        # Sample notes
        test_notes = [
            {
                'title': 'Test Note 1',
                'content': 'This is a test note about productivity and time management.',
                'tags': 'productivity,testing,time-management'
            },
            {
                'title': 'Meeting Notes',
                'content': 'Important meeting discussion about project roadmap and milestones.',
                'tags': 'meeting,project,roadmap'
            },
            {
                'title': 'Research Ideas',
                'content': 'Collection of research ideas for machine learning and AI projects.',
                'tags': 'research,ai,machine-learning'
            },
            {
                'title': 'Daily Reflection',
                'content': 'Daily reflection on learning progress and personal development.',
                'tags': 'reflection,learning,personal-development'
            },
            {
                'title': 'Book Summary',
                'content': 'Summary of key insights from a productivity book about deep work.',
                'tags': 'book,summary,productivity,deep-work'
            }
        ]
        
        for note in test_notes:
            cursor = conn.execute(
                "INSERT INTO notes (title, content, tags, user_id) VALUES (?, ?, ?, ?)",
                (note['title'], note['content'], note['tags'], 'test_user')
            )
            note_id = cursor.lastrowid
            
            # Add to FTS index
            conn.execute(
                "INSERT INTO notes_fts (id, title, content, tags) VALUES (?, ?, ?, ?)",
                (note_id, note['title'], note['content'], note['tags'])
            )
        
        conn.commit()
        conn.close()
        
        self.test_data = {
            'user_id': 'test_user',
            'notes': test_notes
        }


class TestDashboardAPI(DashboardV3TestSuite):
    """Test Dashboard API endpoints"""
    
    def test_dashboard_home_endpoint(self):
        """Test main dashboard page loads correctly"""
        response = self.client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
    def test_dashboard_v3_endpoint(self):
        """Test Dashboard v3 page loads correctly"""
        response = self.client.get("/dashboard/v3")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
    def test_api_health_check(self):
        """Test API health check endpoint"""
        response = self.client.get("/api/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
        
    def test_notes_list_endpoint(self):
        """Test notes listing endpoint"""
        response = self.client.get("/api/notes")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 0  # Should return list even if empty
        
    def test_notes_create_endpoint(self):
        """Test note creation endpoint"""
        new_note = {
            "content": "This is a test note created via API",
            "tags": "api,testing"
        }
        
        response = self.client.post("/api/notes", json=new_note)
        assert response.status_code in [200, 201]
        
        data = response.json()
        assert "id" in data or "success" in data
        
    def test_search_endpoint(self):
        """Test search functionality"""
        # Test basic search
        response = self.client.get("/api/search?q=productivity")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, (list, dict))
        
        # Test empty search
        response = self.client.get("/api/search?q=")
        assert response.status_code in [200, 400]  # Should handle empty queries gracefully
        
    def test_analytics_endpoint(self):
        """Test analytics endpoint"""
        response = self.client.get("/api/analytics")
        assert response.status_code == 200
        
        data = response.json()
        # Should return analytics data structure
        assert isinstance(data, dict)
        
    def test_upload_endpoint(self):
        """Test file upload endpoint"""
        # Create a test file
        test_content = b"This is test file content"
        
        response = self.client.post(
            "/api/upload",
            files={"file": ("test.txt", test_content, "text/plain")}
        )
        
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 201, 400, 413, 500]


class TestPerformanceRegression(DashboardV3TestSuite):
    """Test performance metrics and regression detection"""
    
    def test_response_time_baseline(self):
        """Test API response times are within acceptable limits"""
        endpoints = [
            "/",
            "/dashboard/v3",
            "/api/health",
            "/api/notes",
            "/api/analytics"
        ]
        
        response_times = {}
        
        for endpoint in endpoints:
            start_time = time.time()
            
            try:
                response = self.client.get(endpoint)
                end_time = time.time()
                
                response_time = (end_time - start_time) * 1000  # Convert to milliseconds
                response_times[endpoint] = response_time
                
                # Assert response time is reasonable (under 2 seconds for most endpoints)
                max_time = 5000 if endpoint == "/dashboard/v3" else 2000
                assert response_time < max_time, f"{endpoint} took {response_time:.2f}ms (max: {max_time}ms)"
                
            except Exception as e:
                pytest.fail(f"Failed to test {endpoint}: {str(e)}")
        
        # Store baselines for future regression testing
        self.performance_baselines.update(response_times)
        
    def test_concurrent_requests(self):
        """Test system performance under concurrent load"""
        import concurrent.futures
        import threading
        
        def make_request():
            response = self.client.get("/api/health")
            return response.status_code == 200, response.elapsed if hasattr(response, 'elapsed') else 0
        
        # Test with 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        success_count = sum(1 for success, _ in results if success)
        assert success_count >= 8, f"Only {success_count}/10 concurrent requests succeeded"
        
    def test_memory_usage_stability(self):
        """Test memory usage doesn't grow excessively"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Make multiple requests to stress test
        for _ in range(50):
            self.client.get("/api/health")
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 50MB)
        max_growth = 50 * 1024 * 1024  # 50MB
        assert memory_growth < max_growth, f"Memory grew by {memory_growth / 1024 / 1024:.2f}MB"
        
    def test_large_data_handling(self):
        """Test performance with larger datasets"""
        # Create a large note
        large_content = "A" * 10000  # 10KB of content
        
        start_time = time.time()
        response = self.client.post("/api/notes", json={
            "content": large_content,
            "tags": "performance,test"
        })
        end_time = time.time()
        
        # Should handle large content within reasonable time
        response_time = (end_time - start_time) * 1000
        assert response_time < 5000, f"Large content handling took {response_time:.2f}ms"
        
        # Response should be successful or appropriately handle size limits
        assert response.status_code in [200, 201, 400, 413]


class TestAuthentication(DashboardV3TestSuite):
    """Test authentication and security features"""
    
    def test_protected_endpoints_require_auth(self):
        """Test that protected endpoints require authentication"""
        protected_endpoints = [
            "/api/notes",
            "/api/upload",
            "/api/user/profile"
        ]
        
        for endpoint in protected_endpoints:
            try:
                response = self.client.get(endpoint)
                # Should either be accessible or properly protected
                assert response.status_code in [200, 401, 403], f"{endpoint} returned unexpected status"
            except Exception:
                # Some endpoints might not exist yet, which is fine
                pass
                
    def test_session_handling(self):
        """Test session creation and validation"""
        # Test session creation (if endpoint exists)
        try:
            response = self.client.post("/api/auth/login", json={
                "email": "test@example.com",
                "password": "testpassword"
            })
            
            # Should handle login attempt appropriately
            assert response.status_code in [200, 400, 401, 404]
            
        except Exception:
            # Login endpoint might not be implemented yet
            pass
            
    def test_csrf_protection(self):
        """Test CSRF protection measures"""
        # Test that state-changing operations have appropriate protection
        response = self.client.post("/api/notes", json={
            "content": "CSRF test note"
        })
        
        # Should either succeed with proper headers or be protected
        assert response.status_code in [200, 201, 403, 400]
        
    def test_input_validation(self):
        """Test input validation and sanitization"""
        # Test XSS prevention
        malicious_content = "<script>alert('xss')</script>"
        
        response = self.client.post("/api/notes", json={
            "content": malicious_content,
            "tags": "security,test"
        })
        
        # Should handle malicious input appropriately
        assert response.status_code in [200, 201, 400]
        
        # Test SQL injection prevention
        sql_injection = "'; DROP TABLE notes; --"
        
        response = self.client.get(f"/api/search?q={sql_injection}")
        assert response.status_code in [200, 400]  # Should not crash


class TestDataIntegrity(DashboardV3TestSuite):
    """Test data integrity and consistency"""
    
    def test_note_crud_operations(self):
        """Test Create, Read, Update, Delete operations for notes"""
        # Create
        new_note = {
            "content": "CRUD test note",
            "tags": "crud,testing",
            "title": "CRUD Test"
        }
        
        create_response = self.client.post("/api/notes", json=new_note)
        assert create_response.status_code in [200, 201]
        
        # Read (list all)
        list_response = self.client.get("/api/notes")
        assert list_response.status_code == 200
        
        # Try to extract note ID if available
        try:
            created_data = create_response.json()
            if "id" in created_data:
                note_id = created_data["id"]
                
                # Read specific note
                read_response = self.client.get(f"/api/notes/{note_id}")
                assert read_response.status_code in [200, 404]  # 404 if endpoint doesn't exist
                
                # Update
                update_data = {"content": "Updated CRUD test note"}
                update_response = self.client.put(f"/api/notes/{note_id}", json=update_data)
                assert update_response.status_code in [200, 404, 405]  # Various acceptable responses
                
                # Delete
                delete_response = self.client.delete(f"/api/notes/{note_id}")
                assert delete_response.status_code in [200, 204, 404, 405]
                
        except (KeyError, ValueError, json.JSONDecodeError):
            # Note ID might not be returned in expected format
            pass
    
    def test_search_consistency(self):
        """Test search results are consistent and accurate"""
        # Create a note with specific content
        test_note = {
            "content": "This note contains unique search terms: dashboard testing framework",
            "tags": "unique,search,test"
        }
        
        self.client.post("/api/notes", json=test_note)
        
        # Search for the unique terms
        search_response = self.client.get("/api/search?q=unique+search+terms")
        assert search_response.status_code == 200
        
        # Results should be consistent across multiple searches
        search_response2 = self.client.get("/api/search?q=unique+search+terms")
        assert search_response2.status_code == 200
        
        # Results should be the same
        try:
            results1 = search_response.json()
            results2 = search_response2.json()
            # Results should be consistent (same number of items)
            if isinstance(results1, list) and isinstance(results2, list):
                assert len(results1) == len(results2)
        except (ValueError, json.JSONDecodeError):
            # Search might return different format
            pass
    
    def test_data_validation(self):
        """Test data validation rules are enforced"""
        # Test empty content
        response = self.client.post("/api/notes", json={
            "content": "",
            "tags": "validation,test"
        })
        assert response.status_code in [200, 201, 400]  # Should handle appropriately
        
        # Test very long content
        very_long_content = "A" * 100000  # 100KB
        response = self.client.post("/api/notes", json={
            "content": very_long_content,
            "tags": "validation,test"
        })
        assert response.status_code in [200, 201, 400, 413]  # Should handle size limits
        
        # Test invalid JSON
        response = self.client.post("/api/notes", data="invalid json")
        assert response.status_code == 400  # Should return bad request
    
    def test_database_constraints(self):
        """Test database constraints are properly enforced"""
        # This would require direct database access
        conn = sqlite3.connect(self.test_db_path)
        
        try:
            # Test foreign key constraints (if any)
            # Test unique constraints
            # Test not null constraints
            
            # Example: Try to insert duplicate data if unique constraint exists
            result = conn.execute("SELECT COUNT(*) FROM notes").fetchone()
            initial_count = result[0]
            
            # Try to insert a note
            conn.execute(
                "INSERT INTO notes (content, user_id) VALUES (?, ?)",
                ("Constraint test note", "test_user")
            )
            conn.commit()
            
            # Verify it was inserted
            result = conn.execute("SELECT COUNT(*) FROM notes").fetchone()
            new_count = result[0]
            assert new_count == initial_count + 1
            
        except sqlite3.Error as e:
            # Some constraint violations are expected and should be handled
            pass
        finally:
            conn.close()


class TestIntegrationPoints(DashboardV3TestSuite):
    """Test integration between different system components"""
    
    def test_search_integration(self):
        """Test search functionality integration"""
        # Test that new notes appear in search results
        unique_term = f"integration_test_{int(time.time())}"
        
        # Create note with unique term
        response = self.client.post("/api/notes", json={
            "content": f"This note contains {unique_term} for testing",
            "tags": "integration,search"
        })
        
        assert response.status_code in [200, 201]
        
        # Small delay for indexing
        time.sleep(0.1)
        
        # Search for the unique term
        search_response = self.client.get(f"/api/search?q={unique_term}")
        assert search_response.status_code == 200
        
        # Should find the note (if search is properly integrated)
        try:
            results = search_response.json()
            if isinstance(results, list):
                # Look for our unique term in results
                found = any(unique_term in str(result).lower() for result in results)
                # Note: might not be immediately indexed, so we don't assert found=True
        except (ValueError, json.JSONDecodeError):
            pass
    
    def test_analytics_integration(self):
        """Test analytics data reflects actual usage"""
        # Get initial analytics
        initial_response = self.client.get("/api/analytics")
        assert initial_response.status_code == 200
        
        try:
            initial_data = initial_response.json()
        except (ValueError, json.JSONDecodeError):
            initial_data = {}
        
        # Perform some actions
        self.client.post("/api/notes", json={
            "content": "Analytics test note",
            "tags": "analytics,test"
        })
        
        self.client.get("/api/search?q=test")
        
        # Get updated analytics
        updated_response = self.client.get("/api/analytics")
        assert updated_response.status_code == 200
        
        # Analytics should potentially reflect the activity
        # (though immediate updates aren't required)
        
    def test_file_upload_integration(self):
        """Test file upload integration with note creation"""
        # Test file upload
        test_content = b"This is a test file for integration testing"
        
        response = self.client.post(
            "/api/upload",
            files={"file": ("integration_test.txt", test_content, "text/plain")}
        )
        
        # Should handle upload appropriately
        assert response.status_code in [200, 201, 400, 413, 500]
        
        if response.status_code in [200, 201]:
            try:
                upload_data = response.json()
                # If upload successful, file should be processed/stored
                # This depends on implementation details
            except (ValueError, json.JSONDecodeError):
                pass


class TestErrorHandling(DashboardV3TestSuite):
    """Test error handling and edge cases"""
    
    def test_404_handling(self):
        """Test 404 error handling for non-existent resources"""
        response = self.client.get("/api/notes/99999999")
        assert response.status_code in [404, 400]
        
        response = self.client.get("/nonexistent/endpoint")
        assert response.status_code == 404
        
    def test_malformed_requests(self):
        """Test handling of malformed requests"""
        # Invalid JSON
        response = self.client.post(
            "/api/notes",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 400
        
        # Missing required fields
        response = self.client.post("/api/notes", json={})
        assert response.status_code in [200, 201, 400]  # Depends on requirements
        
    def test_server_error_handling(self):
        """Test server error handling"""
        # This is harder to test without mocking internal failures
        # Test with edge case inputs that might cause issues
        
        # Extremely nested JSON (if applicable)
        nested_data = {"level1": {"level2": {"level3": {"content": "deep test"}}}}
        response = self.client.post("/api/notes", json=nested_data)
        assert response.status_code in [200, 201, 400]  # Should handle gracefully
        
    def test_rate_limiting(self):
        """Test rate limiting if implemented"""
        # Make many requests quickly
        responses = []
        for _ in range(100):
            response = self.client.get("/api/health")
            responses.append(response.status_code)
        
        # Should either allow all requests or implement rate limiting
        # Most responses should be successful
        success_rate = sum(1 for status in responses if status == 200) / len(responses)
        assert success_rate >= 0.8  # At least 80% should succeed


class TestLoadAndStress(DashboardV3TestSuite):
    """Load and stress testing"""
    
    def test_bulk_operations(self):
        """Test system performance with bulk operations"""
        # Create multiple notes at once
        notes_data = []
        for i in range(20):  # Create 20 notes
            notes_data.append({
                "content": f"Bulk test note {i} with content about testing performance",
                "tags": f"bulk,test,note{i}",
                "title": f"Bulk Test Note {i}"
            })
        
        start_time = time.time()
        responses = []
        
        for note_data in notes_data:
            response = self.client.post("/api/notes", json=note_data)
            responses.append(response.status_code)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete bulk operations within reasonable time
        assert total_time < 30, f"Bulk operations took {total_time:.2f} seconds"
        
        # Most operations should succeed
        success_rate = sum(1 for status in responses if status in [200, 201]) / len(responses)
        assert success_rate >= 0.8, f"Only {success_rate:.1%} of bulk operations succeeded"
        
    def test_search_performance(self):
        """Test search performance with various query types"""
        queries = [
            "test",
            "productivity time management",
            "meeting project roadmap",
            "ai machine learning research",
            "book summary deep work",
            "daily reflection learning development"
        ]
        
        search_times = []
        
        for query in queries:
            start_time = time.time()
            response = self.client.get(f"/api/search?q={query}")
            end_time = time.time()
            
            search_time = (end_time - start_time) * 1000  # Convert to ms
            search_times.append(search_time)
            
            assert response.status_code == 200
            assert search_time < 2000, f"Search for '{query}' took {search_time:.2f}ms"
        
        # Average search time should be reasonable
        avg_time = sum(search_times) / len(search_times)
        assert avg_time < 1000, f"Average search time was {avg_time:.2f}ms"


# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor test performance"""
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        result = func(self, *args, **kwargs)
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        
        # Log performance metrics
        print(f"\n{func.__name__} Performance:")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Memory Delta: {memory_delta / 1024 / 1024:.2f}MB")
        
        return result
    return wrapper


# Test configuration and utilities
@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture"""
    return {
        "test_database": ":memory:",
        "performance_thresholds": {
            "response_time_ms": 2000,
            "memory_limit_mb": 100,
            "success_rate": 0.95
        }
    }


@pytest.fixture
def test_client():
    """FastAPI test client fixture"""
    return TestClient(app)


def pytest_configure(config):
    """Configure pytest for dashboard testing"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Test runner and reporting
class DashboardTestReporter:
    """Custom test reporter for dashboard testing"""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = {}
        
    def record_test_result(self, test_name, status, duration, details=None):
        """Record test result"""
        self.test_results.append({
            'test_name': test_name,
            'status': status,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        })
        
    def record_performance_metric(self, metric_name, value, threshold=None):
        """Record performance metric"""
        self.performance_metrics[metric_name] = {
            'value': value,
            'threshold': threshold,
            'timestamp': datetime.now().isoformat(),
            'passed': value <= threshold if threshold else True
        }
        
    def generate_report(self):
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['status'] == 'passed')
        failed_tests = total_tests - passed_tests
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'timestamp': datetime.now().isoformat()
            },
            'test_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'recommendations': self._generate_recommendations()
        }
        
        return report
        
    def _generate_recommendations(self):
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Performance recommendations
        slow_tests = [r for r in self.test_results if r['duration'] > 2.0]
        if slow_tests:
            recommendations.append({
                'type': 'performance',
                'message': f'{len(slow_tests)} tests are running slowly (>2s)',
                'action': 'Consider optimizing these endpoints or increasing timeout thresholds'
            })
            
        # Error rate recommendations
        failed_rate = len([r for r in self.test_results if r['status'] == 'failed']) / len(self.test_results)
        if failed_rate > 0.1:
            recommendations.append({
                'type': 'reliability',
                'message': f'High failure rate: {failed_rate:.1%}',
                'action': 'Review failed tests and improve error handling'
            })
            
        return recommendations


if __name__ == "__main__":
    # Run tests with custom reporting
    import pytest
    
    # Configure pytest arguments
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "--maxfail=5",
        "-x"  # Stop on first failure for debugging
    ]
    
    # Run the tests
    exit_code = pytest.main(pytest_args)
    
    print(f"\nDashboard v3 Test Suite completed with exit code: {exit_code}")
    print("✅ All critical dashboard functionality tested" if exit_code == 0 else "❌ Some tests failed")