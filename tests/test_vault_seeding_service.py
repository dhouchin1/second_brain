# ──────────────────────────────────────────────────────────────────────────────
# File: tests/test_vault_seeding_service.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Test suite for the Vault Seeding Service.

Tests vault seeding functionality, content management, and seeding operations.
"""

import pytest
import sqlite3
import tempfile
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path

from services.vault_seeding_service import (
    VaultSeedingService,
    SeedingResult,
    SeedingOptions,
    get_seeding_service
)


class TestSeedingResult:
    """Test SeedingResult data class."""
    
    def test_successful_result_creation(self):
        """Test creating a successful seeding result."""
        result = SeedingResult(
            success=True,
            message="Seeding completed",
            notes_created=10,
            embeddings_created=10,
            files_written=5
        )
        
        assert result.success == True
        assert result.message == "Seeding completed"
        assert result.notes_created == 10
        assert result.embeddings_created == 10
        assert result.files_written == 5
        assert result.error is None
    
    def test_failed_result_creation(self):
        """Test creating a failed seeding result."""
        result = SeedingResult(
            success=False,
            message="Seeding failed",
            error="Database connection error"
        )
        
        assert result.success == False
        assert result.message == "Seeding failed"
        assert result.notes_created == 0
        assert result.embeddings_created == 0
        assert result.files_written == 0
        assert result.error == "Database connection error"


class TestSeedingOptions:
    """Test SeedingOptions configuration."""
    
    def test_default_options(self):
        """Test default seeding options."""
        options = SeedingOptions()
        
        assert options.namespace == ".seed_samples"
        assert options.force_overwrite == False
        assert options.include_embeddings == True
        assert options.embed_model == "nomic-embed-text"
        assert options.ollama_url == "http://localhost:11434"
    
    def test_custom_options(self):
        """Test custom seeding options."""
        options = SeedingOptions(
            namespace=".custom_seeds",
            force_overwrite=True,
            include_embeddings=False,
            embed_model="custom-model",
            ollama_url="http://custom:8080"
        )
        
        assert options.namespace == ".custom_seeds"
        assert options.force_overwrite == True
        assert options.include_embeddings == False
        assert options.embed_model == "custom-model"
        assert options.ollama_url == "http://custom:8080"


class TestVaultSeedingService:
    """Test the main vault seeding service."""
    
    @pytest.fixture
    def mock_db_conn(self):
        """Mock database connection."""
        conn = Mock(spec=sqlite3.Connection)
        cursor = Mock(spec=sqlite3.Cursor)
        conn.cursor.return_value = cursor
        cursor.fetchall.return_value = []
        return conn
    
    @pytest.fixture
    def mock_get_conn(self, mock_db_conn):
        """Mock database connection function."""
        return lambda: mock_db_conn
    
    @pytest.fixture
    def service(self, mock_get_conn):
        """Create vault seeding service instance."""
        return VaultSeedingService(mock_get_conn)
    
    @pytest.fixture
    def seeding_options(self):
        """Default seeding options for tests."""
        return SeedingOptions(
            namespace=".test_seeds",
            include_embeddings=False  # Disable embeddings for faster tests
        )
    
    def test_service_initialization(self, service):
        """Test service initializes correctly."""
        assert service is not None
        assert hasattr(service, 'get_conn')
    
    def test_is_seeding_available_with_import(self, service):
        """Test seeding availability when import is successful."""
        with patch('services.vault_seeding_service.seed_active_vault', 'mock_function'):
            available = service.is_seeding_available()
            
            assert available == True
    
    def test_is_seeding_available_without_import(self, mock_get_conn):
        """Test seeding availability when import fails."""
        with patch('services.vault_seeding_service.seed_active_vault', None):
            service = VaultSeedingService(mock_get_conn)
            available = service.is_seeding_available()
            
            assert available == False
    
    def test_get_available_seed_content_types(self, service):
        """Test getting available seed content types."""
        with patch('services.vault_seeding_service.SEED_NOTES', [{"title": "Note 1"}]), \
             patch('services.vault_seeding_service.SEED_BOOKMARKS', [{"title": "Bookmark 1"}]):
            
            content_types = service.get_available_seed_content()
            
            assert "notes" in content_types
            assert "bookmarks" in content_types
            assert content_types["notes"]["count"] == 1
            assert content_types["bookmarks"]["count"] == 1
    
    def test_get_available_seed_content_unavailable(self, mock_get_conn):
        """Test getting seed content when seeding is unavailable."""
        with patch('services.vault_seeding_service.seed_active_vault', None):
            service = VaultSeedingService(mock_get_conn)
            
            content_types = service.get_available_seed_content()
            
            assert content_types == {}
    
    def test_seed_vault_with_content_success(self, service, seeding_options):
        """Test successful vault seeding."""
        # Mock the seeding function
        mock_seed_function = Mock()
        mock_seed_function.return_value = None  # Success (no exception)
        
        with patch('services.vault_seeding_service.seed_active_vault', mock_seed_function), \
             patch.object(service, '_count_seeded_notes', return_value=15), \
             patch.object(service, '_count_seeded_embeddings', return_value=15), \
             patch.object(service, '_log_seeding_operation') as mock_log:
            
            result = service.seed_vault_with_content(seeding_options)
            
            assert result.success == True
            assert result.notes_created == 15
            assert result.embeddings_created == 15
            assert "successfully seeded" in result.message.lower()
            
            # Verify seeding function was called with correct parameters
            mock_seed_function.assert_called_once()
            args, kwargs = mock_seed_function.call_args
            assert args[0] == service.get_conn  # Database connection function
    
    def test_seed_vault_with_content_failure(self, service, seeding_options):
        """Test vault seeding failure handling."""
        # Mock the seeding function to raise an exception
        mock_seed_function = Mock()
        mock_seed_function.side_effect = Exception("Seeding error")
        
        with patch('services.vault_seeding_service.seed_active_vault', mock_seed_function), \
             patch.object(service, '_log_seeding_operation') as mock_log:
            
            result = service.seed_vault_with_content(seeding_options)
            
            assert result.success == False
            assert "error" in result.message.lower()
            assert result.error == "Seeding error"
            assert result.notes_created == 0
            assert result.embeddings_created == 0
    
    def test_seed_vault_unavailable(self, mock_get_conn, seeding_options):
        """Test vault seeding when functionality is unavailable."""
        with patch('services.vault_seeding_service.seed_active_vault', None):
            service = VaultSeedingService(mock_get_conn)
            
            result = service.seed_vault_with_content(seeding_options)
            
            assert result.success == False
            assert "not available" in result.message.lower()
    
    def test_get_seeding_history_empty(self, service, mock_db_conn):
        """Test getting seeding history when none exists."""
        cursor = mock_db_conn.cursor.return_value
        cursor.fetchall.return_value = []
        
        history = service.get_seeding_history()
        
        assert history == []
    
    def test_get_seeding_history_with_records(self, service, mock_db_conn):
        """Test getting seeding history with existing records."""
        cursor = mock_db_conn.cursor.return_value
        cursor.fetchall.return_value = [
            (1, datetime.now().isoformat(), True, 10, 10, "Success"),
            (2, datetime.now().isoformat(), False, 0, 0, "Failed")
        ]
        
        history = service.get_seeding_history()
        
        assert len(history) == 2
        assert history[0]["success"] == True
        assert history[0]["notes_created"] == 10
        assert history[1]["success"] == False
        assert history[1]["notes_created"] == 0
    
    def test_get_seeding_stats(self, service, mock_db_conn):
        """Test getting seeding statistics."""
        cursor = mock_db_conn.cursor.return_value
        # Mock aggregate query results
        cursor.fetchone.side_effect = [
            (2,),  # total_operations
            (1,),  # successful_operations
            (25,), # total_notes_created
            (20,), # total_embeddings_created
        ]
        
        stats = service.get_seeding_stats()
        
        assert stats["total_operations"] == 2
        assert stats["successful_operations"] == 1
        assert stats["success_rate"] == 50.0  # 1/2 * 100
        assert stats["total_notes_created"] == 25
        assert stats["total_embeddings_created"] == 20
    
    def test_get_seeding_stats_no_history(self, service, mock_db_conn):
        """Test getting stats when no seeding history exists."""
        cursor = mock_db_conn.cursor.return_value
        cursor.fetchone.side_effect = [
            (0,), (0,), (0,), (0,)  # All zero counts
        ]
        
        stats = service.get_seeding_stats()
        
        assert stats["total_operations"] == 0
        assert stats["successful_operations"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["total_notes_created"] == 0
        assert stats["total_embeddings_created"] == 0
    
    def test_count_seeded_notes(self, service, mock_db_conn):
        """Test counting seeded notes."""
        cursor = mock_db_conn.cursor.return_value
        cursor.fetchone.return_value = (42,)
        
        count = service._count_seeded_notes(".test_seeds")
        
        assert count == 42
        
        # Verify query includes namespace filter
        query_call = cursor.execute.call_args[0][0]
        assert ".test_seeds" in str(cursor.execute.call_args)
    
    def test_count_seeded_embeddings(self, service, mock_db_conn):
        """Test counting seeded embeddings."""
        cursor = mock_db_conn.cursor.return_value
        cursor.fetchone.return_value = (24,)
        
        count = service._count_seeded_embeddings(".test_seeds")
        
        assert count == 24
    
    def test_log_seeding_operation_success(self, service, mock_db_conn):
        """Test logging successful seeding operation."""
        result = SeedingResult(
            success=True,
            message="Test seeding",
            notes_created=10,
            embeddings_created=8
        )
        
        service._log_seeding_operation(result)
        
        cursor = mock_db_conn.cursor.return_value
        cursor.execute.assert_called()
        
        # Verify the INSERT statement includes correct values
        insert_call = cursor.execute.call_args[0]
        assert "INSERT INTO" in insert_call[0]
        assert insert_call[1][1] == True  # success
        assert insert_call[1][2] == 10    # notes_created
        assert insert_call[1][3] == 8     # embeddings_created
        assert insert_call[1][4] == "Test seeding"  # message
    
    def test_log_seeding_operation_failure(self, service, mock_db_conn):
        """Test logging failed seeding operation."""
        result = SeedingResult(
            success=False,
            message="Test failure",
            error="Database error"
        )
        
        service._log_seeding_operation(result)
        
        cursor = mock_db_conn.cursor.return_value
        insert_call = cursor.execute.call_args[0]
        assert insert_call[1][1] == False  # success
        assert insert_call[1][2] == 0      # notes_created
        assert insert_call[1][3] == 0      # embeddings_created
        assert "Test failure" in insert_call[1][4]  # message includes error


class TestVaultSeedingIntegration:
    """Integration tests for vault seeding service."""
    
    def test_get_seeding_service_factory(self):
        """Test the factory function for getting service instance."""
        mock_get_conn = Mock()
        
        service = get_seeding_service(mock_get_conn)
        
        assert isinstance(service, VaultSeedingService)
        assert service.get_conn == mock_get_conn
    
    def test_full_seeding_workflow_with_database(self):
        """Test complete seeding workflow with real database."""
        # Create a temporary database
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            db_path = tmp_file.name
        
        try:
            def get_conn():
                return sqlite3.connect(db_path)
            
            # Initialize basic schema
            conn = get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY,
                    title TEXT,
                    body TEXT,
                    tags TEXT,
                    created_at TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS seeding_log (
                    id INTEGER PRIMARY KEY,
                    created_at TEXT,
                    success BOOLEAN,
                    notes_created INTEGER,
                    embeddings_created INTEGER,
                    message TEXT
                )
            """)
            conn.commit()
            conn.close()
            
            # Test the service
            service = VaultSeedingService(get_conn)
            
            # Check initial state
            assert service.get_seeding_stats()["total_operations"] == 0
            
            # Mock successful seeding operation
            options = SeedingOptions(namespace=".integration_test", include_embeddings=False)
            
            with patch('services.vault_seeding_service.seed_active_vault') as mock_seed, \
                 patch.object(service, '_count_seeded_notes', return_value=5), \
                 patch.object(service, '_count_seeded_embeddings', return_value=0):
                
                result = service.seed_vault_with_content(options)
                
                assert result.success == True
                assert result.notes_created == 5
                
                # Check that history was recorded
                history = service.get_seeding_history()
                assert len(history) == 1
                assert history[0]["success"] == True
        
        finally:
            # Clean up
            Path(db_path).unlink(missing_ok=True)


class TestVaultSeedingEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def service(self):
        """Service with mocked dependencies."""
        mock_get_conn = Mock()
        return VaultSeedingService(mock_get_conn)
    
    def test_database_connection_failure_during_seeding(self, service):
        """Test handling of database connection failures during seeding."""
        service.get_conn.side_effect = sqlite3.Error("Connection failed")
        options = SeedingOptions()
        
        result = service.seed_vault_with_content(options)
        
        assert result.success == False
        assert "connection" in result.error.lower()
    
    def test_database_query_failure_in_stats(self, service):
        """Test handling of database query failures in statistics."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = sqlite3.Error("Query failed")
        mock_conn.cursor.return_value = mock_cursor
        service.get_conn.return_value = mock_conn
        
        stats = service.get_seeding_stats()
        
        # Should return default stats on error
        assert stats["total_operations"] == 0
        assert stats["success_rate"] == 0.0
    
    def test_corrupted_seeding_history_data(self, service):
        """Test handling of corrupted history data."""
        mock_conn = Mock()
        mock_cursor = Mock()
        # Return malformed data
        mock_cursor.fetchall.return_value = [
            ("invalid", "data", "format")
        ]
        mock_conn.cursor.return_value = mock_cursor
        service.get_conn.return_value = mock_conn
        
        history = service.get_seeding_history()
        
        # Should handle gracefully and return empty list
        assert history == []
    
    def test_seeding_with_invalid_namespace(self, service):
        """Test seeding with invalid namespace characters."""
        options = SeedingOptions(namespace="invalid/namespace")
        
        with patch('services.vault_seeding_service.seed_active_vault') as mock_seed:
            # Should still attempt seeding (validation happens in seeding function)
            service.seed_vault_with_content(options)
            
            mock_seed.assert_called_once()
    
    def test_missing_seeding_log_table(self, service):
        """Test graceful handling when seeding log table doesn't exist."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.execute.side_effect = sqlite3.OperationalError("no such table: seeding_log")
        mock_conn.cursor.return_value = mock_cursor
        service.get_conn.return_value = mock_conn
        
        # Should handle table not existing gracefully
        stats = service.get_seeding_stats()
        history = service.get_seeding_history()
        
        assert stats["total_operations"] == 0
        assert history == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])