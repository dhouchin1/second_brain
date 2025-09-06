# ──────────────────────────────────────────────────────────────────────────────
# File: tests/test_auto_seeding_service.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Test suite for the Auto-Seeding Service.

Tests automatic content seeding for new users, configuration management,
and intelligent seeding decision logic.
"""

import pytest
import sqlite3
import tempfile
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path

from services.auto_seeding_service import (
    AutoSeedingService, 
    AutoSeedingConfig,
    get_auto_seeding_service
)
from services.vault_seeding_service import SeedingResult, SeedingOptions


class TestAutoSeedingConfig:
    """Test configuration management for auto-seeding."""
    
    def test_default_config_values(self):
        """Test default configuration values."""
        config = AutoSeedingConfig()
        
        assert config.enabled == True
        assert config.namespace == ".starter_content"
        assert config.include_embeddings == True
        assert config.embed_model == "nomic-embed-text"
        assert config.ollama_url == "http://localhost:11434"
        assert config.skip_if_content_exists == True
        assert config.min_notes_threshold == 5
    
    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = AutoSeedingConfig(
            enabled=False,
            namespace=".custom_seeds",
            include_embeddings=False,
            embed_model="custom-model",
            ollama_url="http://custom:8080",
            skip_if_content_exists=False,
            min_notes_threshold=10
        )
        
        assert config.enabled == False
        assert config.namespace == ".custom_seeds"
        assert config.include_embeddings == False
        assert config.embed_model == "custom-model"
        assert config.ollama_url == "http://custom:8080"
        assert config.skip_if_content_exists == False
        assert config.min_notes_threshold == 10


class TestAutoSeedingService:
    """Test the main auto-seeding service functionality."""
    
    @pytest.fixture
    def mock_db_conn(self):
        """Mock database connection."""
        conn = Mock(spec=sqlite3.Connection)
        cursor = Mock(spec=sqlite3.Cursor)
        conn.cursor.return_value = cursor
        cursor.fetchone.return_value = None  # No existing notes by default
        cursor.fetchall.return_value = []
        return conn
    
    @pytest.fixture
    def mock_get_conn(self, mock_db_conn):
        """Mock database connection function."""
        return lambda: mock_db_conn
    
    @pytest.fixture
    def service(self, mock_get_conn):
        """Create auto-seeding service instance."""
        with patch('services.auto_seeding_service.settings') as mock_settings:
            # Mock settings to avoid dependency on actual config
            mock_settings.auto_seeding_enabled = True
            mock_settings.auto_seeding_namespace = ".test_seeds"
            
            return AutoSeedingService(mock_get_conn)
    
    def test_service_initialization(self, service):
        """Test service initializes correctly."""
        assert service is not None
        assert isinstance(service.config, AutoSeedingConfig)
        assert service.config.namespace == ".test_seeds"
    
    def test_config_loading_from_settings(self, mock_get_conn):
        """Test configuration loading from settings."""
        with patch('services.auto_seeding_service.settings') as mock_settings:
            mock_settings.auto_seeding_enabled = False
            mock_settings.auto_seeding_namespace = ".custom"
            mock_settings.auto_seeding_embeddings = False
            mock_settings.auto_seeding_embed_model = "test-model"
            mock_settings.ollama_api_url = "http://test:9999/api/generate"
            mock_settings.auto_seeding_skip_if_content = False
            mock_settings.auto_seeding_min_notes = 3
            
            service = AutoSeedingService(mock_get_conn)
            
            assert service.config.enabled == False
            assert service.config.namespace == ".custom"
            assert service.config.include_embeddings == False
            assert service.config.embed_model == "test-model"
            assert service.config.ollama_url == "http://test:9999"  # Should strip /api/generate
            assert service.config.skip_if_content_exists == False
            assert service.config.min_notes_threshold == 3
    
    def test_should_auto_seed_new_database(self, service, mock_db_conn):
        """Test seeding decision for new database."""
        # Mock empty database
        cursor = mock_db_conn.cursor.return_value
        cursor.fetchone.return_value = (0,)  # No notes
        
        should_seed, reason = service.should_auto_seed()
        
        assert should_seed == True
        assert "empty database" in reason.lower()
    
    def test_should_auto_seed_with_existing_content(self, service, mock_db_conn):
        """Test seeding decision when content already exists."""
        # Mock database with existing notes
        cursor = mock_db_conn.cursor.return_value
        cursor.fetchone.return_value = (10,)  # 10 notes exist
        
        should_seed, reason = service.should_auto_seed()
        
        assert should_seed == False
        assert "content exists" in reason.lower()
    
    def test_should_auto_seed_below_threshold(self, service, mock_db_conn):
        """Test seeding when note count is below threshold."""
        # Mock database with few notes (below threshold)
        cursor = mock_db_conn.cursor.return_value
        cursor.fetchone.return_value = (3,)  # Only 3 notes (below threshold of 5)
        
        should_seed, reason = service.should_auto_seed()
        
        assert should_seed == True
        assert "below threshold" in reason.lower()
    
    def test_should_auto_seed_disabled(self, mock_get_conn, mock_db_conn):
        """Test seeding when auto-seeding is disabled."""
        with patch('services.auto_seeding_service.settings') as mock_settings:
            mock_settings.auto_seeding_enabled = False
            
            service = AutoSeedingService(mock_get_conn)
            
            should_seed, reason = service.should_auto_seed()
            
            assert should_seed == False
            assert "disabled" in reason.lower()
    
    def test_should_auto_seed_skip_if_content_disabled(self, mock_get_conn, mock_db_conn):
        """Test seeding when skip_if_content_exists is disabled."""
        with patch('services.auto_seeding_service.settings') as mock_settings:
            mock_settings.auto_seeding_enabled = True
            mock_settings.auto_seeding_skip_if_content = False
            
            service = AutoSeedingService(mock_get_conn)
            
            # Mock database with existing notes
            cursor = mock_db_conn.cursor.return_value
            cursor.fetchone.return_value = (10,)  # 10 notes exist
            
            should_seed, reason = service.should_auto_seed()
            
            assert should_seed == True  # Should seed even with existing content
            assert "force seeding enabled" in reason.lower()
    
    @pytest.mark.asyncio
    async def test_perform_auto_seed_success(self, service):
        """Test successful auto-seeding operation."""
        mock_seeding_result = SeedingResult(
            success=True,
            message="Seeding completed successfully",
            notes_created=25,
            embeddings_created=25,
            files_written=0
        )
        
        with patch.object(service, '_get_vault_seeding_service') as mock_get_seeding, \
             patch.object(service, 'should_auto_seed') as mock_should_seed:
            
            mock_should_seed.return_value = (True, "Database is empty")
            
            mock_seeding_service = Mock()
            mock_seeding_service.seed_vault_with_content.return_value = mock_seeding_result
            mock_get_seeding.return_value = mock_seeding_service
            
            result = await service.perform_auto_seed()
            
            assert result.success == True
            assert result.notes_created == 25
            assert result.embeddings_created == 25
            assert "Seeding completed successfully" in result.message
    
    @pytest.mark.asyncio
    async def test_perform_auto_seed_skip(self, service):
        """Test auto-seeding when it should be skipped."""
        with patch.object(service, 'should_auto_seed') as mock_should_seed:
            mock_should_seed.return_value = (False, "Content already exists")
            
            result = await service.perform_auto_seed()
            
            assert result.success == True
            assert result.notes_created == 0
            assert "skipped" in result.message.lower()
            assert "Content already exists" in result.message
    
    @pytest.mark.asyncio
    async def test_perform_auto_seed_failure(self, service):
        """Test auto-seeding failure handling."""
        mock_seeding_result = SeedingResult(
            success=False,
            message="Seeding failed",
            error="Database connection error"
        )
        
        with patch.object(service, '_get_vault_seeding_service') as mock_get_seeding, \
             patch.object(service, 'should_auto_seed') as mock_should_seed:
            
            mock_should_seed.return_value = (True, "Database is empty")
            
            mock_seeding_service = Mock()
            mock_seeding_service.seed_vault_with_content.return_value = mock_seeding_result
            mock_get_seeding.return_value = mock_seeding_service
            
            result = await service.perform_auto_seed()
            
            assert result.success == False
            assert "Seeding failed" in result.message
            assert result.error == "Database connection error"
    
    @pytest.mark.asyncio
    async def test_perform_auto_seed_exception_handling(self, service):
        """Test exception handling during auto-seeding."""
        with patch.object(service, 'should_auto_seed') as mock_should_seed, \
             patch.object(service, '_get_vault_seeding_service') as mock_get_seeding:
            
            mock_should_seed.return_value = (True, "Database is empty")
            mock_get_seeding.side_effect = Exception("Unexpected error")
            
            result = await service.perform_auto_seed()
            
            assert result.success == False
            assert "error" in result.error.lower()
            assert "Unexpected error" in result.error
    
    def test_get_seeding_status_never_run(self, service, mock_db_conn):
        """Test status when auto-seeding has never been run."""
        cursor = mock_db_conn.cursor.return_value
        cursor.fetchone.return_value = None  # No seeding record
        
        status = service.get_seeding_status()
        
        assert status["has_been_seeded"] == False
        assert status["last_seeded"] is None
        assert status["total_notes_created"] == 0
        assert "never been run" in status["status"].lower()
    
    def test_get_seeding_status_completed(self, service, mock_db_conn):
        """Test status when seeding has been completed."""
        cursor = mock_db_conn.cursor.return_value
        cursor.fetchone.return_value = (
            1,  # id
            datetime.now().isoformat(),  # created_at
            True,  # success
            25,  # notes_created
            25,  # embeddings_created
            "Seeding completed successfully"  # message
        )
        
        status = service.get_seeding_status()
        
        assert status["has_been_seeded"] == True
        assert status["last_seeded"] is not None
        assert status["total_notes_created"] == 25
        assert status["total_embeddings_created"] == 25
        assert "completed successfully" in status["status"].lower()
    
    def test_get_current_note_count(self, service, mock_db_conn):
        """Test getting current note count."""
        cursor = mock_db_conn.cursor.return_value
        cursor.fetchone.return_value = (42,)
        
        count = service.get_current_note_count()
        
        assert count == 42
    
    def test_get_current_note_count_empty_db(self, service, mock_db_conn):
        """Test getting note count from empty database."""
        cursor = mock_db_conn.cursor.return_value
        cursor.fetchone.return_value = (0,)
        
        count = service.get_current_note_count()
        
        assert count == 0
    
    def test_create_seeding_options_from_config(self, service):
        """Test creating seeding options from service config."""
        service.config.namespace = ".test_namespace"
        service.config.include_embeddings = False
        service.config.embed_model = "test-embed-model"
        service.config.ollama_url = "http://test:8080"
        
        options = service._create_seeding_options()
        
        assert isinstance(options, SeedingOptions)
        assert options.namespace == ".test_namespace"
        assert options.include_embeddings == False
        assert options.embed_model == "test-embed-model"
        assert options.ollama_url == "http://test:8080"
        assert options.force_overwrite == False  # Default value


class TestAutoSeedingIntegration:
    """Integration tests for auto-seeding service."""
    
    def test_get_auto_seeding_service_factory(self):
        """Test the factory function for getting service instance."""
        mock_get_conn = Mock()
        
        with patch('services.auto_seeding_service.settings'):
            service = get_auto_seeding_service(mock_get_conn)
            
            assert isinstance(service, AutoSeedingService)
            assert service.get_conn == mock_get_conn
    
    @pytest.mark.asyncio
    async def test_full_seeding_workflow(self):
        """Test complete auto-seeding workflow."""
        # Create a temporary database
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            db_path = tmp_file.name
        
        try:
            def get_conn():
                return sqlite3.connect(db_path)
            
            # Initialize database schema
            conn = get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY,
                    title TEXT,
                    body TEXT,
                    created_at TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS auto_seeding_log (
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
            with patch('services.auto_seeding_service.settings') as mock_settings:
                mock_settings.auto_seeding_enabled = True
                mock_settings.auto_seeding_namespace = ".integration_test"
                
                service = AutoSeedingService(get_conn)
                
                # Check initial state
                should_seed, reason = service.should_auto_seed()
                assert should_seed == True
                assert service.get_current_note_count() == 0
                
                # Mock successful seeding
                mock_result = SeedingResult(
                    success=True,
                    message="Integration test seeding",
                    notes_created=5,
                    embeddings_created=5
                )
                
                with patch.object(service, '_get_vault_seeding_service') as mock_get_seeding:
                    mock_seeding_service = Mock()
                    mock_seeding_service.seed_vault_with_content.return_value = mock_result
                    mock_get_seeding.return_value = mock_seeding_service
                    
                    result = await service.perform_auto_seed()
                    
                    assert result.success == True
                    assert result.notes_created == 5
        
        finally:
            # Clean up
            Path(db_path).unlink(missing_ok=True)


class TestAutoSeedingEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def service(self):
        """Service with mocked dependencies."""
        mock_get_conn = Mock()
        
        with patch('services.auto_seeding_service.settings'):
            return AutoSeedingService(mock_get_conn)
    
    def test_database_connection_failure(self, service):
        """Test handling of database connection failures."""
        service.get_conn.side_effect = sqlite3.Error("Connection failed")
        
        count = service.get_current_note_count()
        
        assert count == 0  # Should return 0 on error
    
    def test_should_auto_seed_database_error(self, service):
        """Test seeding decision when database query fails."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = sqlite3.Error("Query failed")
        mock_conn.cursor.return_value = mock_cursor
        service.get_conn.return_value = mock_conn
        
        should_seed, reason = service.should_auto_seed()
        
        assert should_seed == False
        assert "error" in reason.lower()
    
    def test_get_seeding_status_database_error(self, service):
        """Test status retrieval when database query fails."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = sqlite3.Error("Query failed")
        mock_conn.cursor.return_value = mock_cursor
        service.get_conn.return_value = mock_conn
        
        status = service.get_seeding_status()
        
        assert status["has_been_seeded"] == False
        assert "error" in status["status"].lower()
    
    def test_config_loading_with_missing_settings(self):
        """Test config loading when some settings are missing."""
        mock_get_conn = Mock()
        
        with patch('services.auto_seeding_service.settings') as mock_settings:
            # Only set some attributes, leave others missing
            mock_settings.auto_seeding_enabled = False
            # Don't set other attributes to test defaults
            del mock_settings.auto_seeding_namespace
            
            service = AutoSeedingService(mock_get_conn)
            
            # Should use defaults for missing settings
            assert service.config.enabled == False  # From setting
            assert service.config.namespace == ".starter_content"  # Default
            assert service.config.include_embeddings == True  # Default
    
    def test_seeding_options_with_malformed_ollama_url(self, service):
        """Test seeding options creation with malformed Ollama URL."""
        service.config.ollama_url = "not-a-url"
        
        options = service._create_seeding_options()
        
        # Should still create options (URL validation happens elsewhere)
        assert options.ollama_url == "not-a-url"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])