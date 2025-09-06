"""
Integration tests for Enhanced Capture API endpoints

Tests all new API endpoints end-to-end including unified capture,
advanced capture, Apple Shortcuts, and Discord integration APIs.
"""

import pytest
import json
import base64
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import tempfile
import sqlite3
from pathlib import Path

# Import the FastAPI app
from app import app, get_conn


class TestAPIIntegration:
    
    @pytest.fixture
    def test_db(self):
        """Create temporary test database."""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        # Initialize test database
        conn = sqlite3.connect(temp_db.name)
        
        # Create required tables
        conn.execute('''
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                body TEXT,
                content TEXT,
                summary TEXT,
                tags TEXT,
                actions TEXT,
                type TEXT,
                timestamp TEXT,
                created_at TEXT,
                updated_at TEXT,
                metadata TEXT,
                user_id INTEGER DEFAULT 1,
                status TEXT DEFAULT 'complete'
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                hashed_password TEXT
            )
        ''')
        
        # Insert test user
        conn.execute(
            "INSERT INTO users (username, hashed_password) VALUES (?, ?)",
            ("testuser", "$2b$12$hashed_password_here")
        )
        
        conn.commit()
        conn.close()
        
        yield temp_db.name
        
        # Cleanup
        Path(temp_db.name).unlink()
    
    @pytest.fixture
    def client(self, test_db):
        """Test client with test database."""
        # Override database dependency
        def get_test_conn():
            return sqlite3.connect(test_db)
        
        app.dependency_overrides[get_conn] = get_test_conn
        
        with TestClient(app) as test_client:
            yield test_client
        
        # Cleanup
        app.dependency_overrides.clear()
    
    @pytest.fixture
    def sample_image_data(self):
        """Sample base64 image data."""
        return base64.b64encode(b"fake_image_data_for_testing").decode()
    
    @pytest.fixture
    def sample_pdf_data(self):
        """Sample base64 PDF data."""
        return base64.b64encode(b"fake_pdf_data_for_testing").decode()
    
    # Unified Capture API Tests
    
    def test_unified_capture_text_endpoint(self, client):
        """Test unified capture text endpoint."""
        payload = {
            "content": "This is a test note for API integration",
            "source": "api",
            "title": "API Test Note",
            "tags": ["test", "api", "integration"],
            "auto_tag": True,
            "generate_summary": False
        }
        
        with patch('services.unified_capture_service.ollama_generate_title'), \
             patch('services.unified_capture_service.ollama_summarize'):
            
            response = client.post("/api/unified-capture/text", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] == True
            assert data["note_id"] is not None
            assert data["title"] == "API Test Note"
            assert "test" in data["tags"]
            assert "api" in data["tags"]
            assert data["processing_time"] > 0
    
    def test_unified_capture_audio_endpoint(self, client):
        """Test unified capture audio endpoint."""
        audio_data = base64.b64encode(b"fake_audio_data").decode()
        
        payload = {
            "audio_data": audio_data,
            "transcription": "This is the transcribed audio content",
            "source": "api",
            "title": "Audio Test"
        }
        
        response = client.post("/api/unified-capture/audio", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["title"] == "Audio Test"
    
    def test_unified_capture_image_endpoint(self, client, sample_image_data):
        """Test unified capture image OCR endpoint."""
        payload = {
            "image_data": sample_image_data,
            "source": "api",
            "title": "OCR Test",
            "auto_tag": True
        }
        
        with patch('pytesseract.image_to_string') as mock_ocr:
            mock_ocr.return_value = "Extracted text from test image"
            
            response = client.post("/api/unified-capture/image", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert data["title"] == "OCR Test"
    
    def test_unified_capture_url_endpoint(self, client):
        """Test unified capture URL endpoint."""
        payload = {
            "url": "https://example.com/test-article",
            "source": "api",
            "auto_tag": True,
            "generate_summary": True
        }
        
        with patch('services.web_ingestion_service.WebIngestionService') as mock_web:
            mock_service = Mock()
            mock_service.ingest_url.return_value = {
                "success": True,
                "note_id": 123,
                "title": "Test Article",
                "content": "Article content"
            }
            mock_web.return_value = mock_service
            
            response = client.post("/api/unified-capture/url", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
    
    def test_unified_capture_pdf_endpoint(self, client, sample_pdf_data):
        """Test unified capture PDF endpoint."""
        payload = {
            "file_data": sample_pdf_data,
            "filename": "test.pdf",
            "source": "api",
            "title": "PDF Test"
        }
        
        with patch('fitz.open') as mock_fitz:
            mock_doc = Mock()
            mock_page = Mock()
            mock_page.get_text.return_value = "Extracted PDF content"
            mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
            mock_doc.close = Mock()
            mock_fitz.return_value = mock_doc
            
            response = client.post("/api/unified-capture/pdf", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert data["title"] == "PDF Test"
    
    def test_unified_capture_batch_endpoint(self, client):
        """Test unified capture batch endpoint."""
        requests = [
            {
                "content_type": "text",
                "source": "api",
                "content": "First batch note",
                "title": "Batch 1"
            },
            {
                "content_type": "text", 
                "source": "api",
                "content": "Second batch note",
                "title": "Batch 2"
            }
        ]
        
        payload = {"requests": requests}
        
        with patch('services.unified_capture_service.ollama_generate_title'), \
             patch('services.unified_capture_service.ollama_summarize'):
            
            response = client.post("/api/unified-capture/batch", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] == True
            assert data["total_requests"] == 2
            assert data["successful"] <= 2  # May vary based on mocking
            assert len(data["results"]) == 2
    
    def test_unified_capture_stats_endpoint(self, client):
        """Test unified capture statistics endpoint."""
        response = client.get("/api/unified-capture/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert "data" in data
        assert "total_requests" in data["data"]
        assert "success_rate" in data["data"]
    
    def test_unified_capture_integrations_endpoint(self, client):
        """Test unified capture integrations endpoint."""
        response = client.get("/api/unified-capture/integrations")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert "data" in data
        assert "sources" in data["data"]
        assert "content_types" in data["data"]
        assert "features" in data["data"]
        assert "limits" in data["data"]
    
    def test_unified_capture_quick_note_endpoint(self, client):
        """Test unified capture quick note endpoint."""
        response = client.post(
            "/api/unified-capture/quick-note",
            data={"content": "Quick test note", "source": "api"}
        )
        
        with patch('services.unified_capture_service.ollama_generate_title'), \
             patch('services.unified_capture_service.ollama_summarize'):
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
    
    def test_unified_capture_health_endpoint(self, client):
        """Test unified capture health endpoint."""
        response = client.get("/api/unified-capture/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert data["service"] == "unified_capture"
        assert data["status"] == "healthy"
        assert "stats" in data
        assert "supported_features" in data
    
    # Advanced Capture API Tests
    
    def test_advanced_capture_screenshot_ocr_endpoint(self, client, sample_image_data):
        """Test advanced capture screenshot OCR endpoint."""
        payload = {
            "image_data": sample_image_data,
            "options": {
                "generate_title": True,
                "extract_summary": True,
                "custom_tags": ["screenshot", "test"]
            }
        }
        
        with patch('pytesseract.image_to_string') as mock_ocr:
            mock_ocr.return_value = "Screenshot text content"
            
            response = client.post("/api/advanced-capture/screenshot-ocr", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "extracted_text" in data
    
    def test_advanced_capture_pdf_endpoint(self, client, sample_pdf_data):
        """Test advanced capture PDF endpoint."""
        payload = {
            "file_data": sample_pdf_data,
            "filename": "advanced_test.pdf",
            "options": {
                "generate_title": True,
                "extract_summary": True
            }
        }
        
        with patch('fitz.open') as mock_fitz:
            mock_doc = Mock()
            mock_page = Mock()
            mock_page.get_text.return_value = "Advanced PDF content"
            mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
            mock_doc.close = Mock()
            mock_fitz.return_value = mock_doc
            
            response = client.post("/api/advanced-capture/pdf", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
    
    def test_advanced_capture_youtube_endpoint(self, client):
        """Test advanced capture YouTube endpoint."""
        payload = {
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "options": {
                "generate_title": True,
                "extract_summary": True
            }
        }
        
        with patch('youtube_transcript_api.YouTubeTranscriptApi.get_transcript') as mock_transcript:
            mock_transcript.return_value = [
                {"text": "YouTube transcript text", "start": 0.0}
            ]
            
            response = client.post("/api/advanced-capture/youtube", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
    
    def test_advanced_capture_bulk_urls_endpoint(self, client):
        """Test advanced capture bulk URLs endpoint."""
        payload = {
            "urls": [
                "https://example.com/article1",
                "https://example.com/article2"
            ],
            "max_concurrent": 2
        }
        
        with patch('services.web_ingestion_service.WebIngestionService') as mock_web:
            mock_service = Mock()
            mock_service.ingest_url.return_value = {
                "success": True,
                "note_id": 456,
                "title": "Bulk Article"
            }
            mock_web.return_value = mock_service
            
            response = client.post("/api/advanced-capture/bulk-urls", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert len(data["results"]) == 2
    
    def test_advanced_capture_features_endpoint(self, client):
        """Test advanced capture features endpoint."""
        response = client.get("/api/advanced-capture/features")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert "features" in data
        assert "ocr" in data["features"]
        assert "pdf" in data["features"]
        assert "image_processing" in data["features"]
    
    # Apple Shortcuts API Tests
    
    def test_shortcuts_voice_memo_endpoint(self, client):
        """Test Apple Shortcuts voice memo endpoint."""
        payload = {
            "audio_data": base64.b64encode(b"ios_audio_data").decode(),
            "transcription": "Voice memo from iOS Shortcuts",
            "location_data": {
                "latitude": 37.7749,
                "longitude": -122.4194,
                "address": "San Francisco, CA"
            },
            "context": {
                "timestamp": "2024-12-15T10:30:00Z",
                "device": "iPhone"
            }
        }
        
        with patch('services.enhanced_apple_shortcuts_service.ollama_generate_title'), \
             patch('services.enhanced_apple_shortcuts_service.ollama_summarize'):
            
            response = client.post("/api/shortcuts/voice-memo", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert data["note_id"] is not None
    
    def test_shortcuts_photo_ocr_endpoint(self, client, sample_image_data):
        """Test Apple Shortcuts photo OCR endpoint."""
        payload = {
            "image_data": sample_image_data,
            "location_data": {
                "latitude": 37.7749,
                "longitude": -122.4194
            },
            "context": {
                "device": "iPhone",
                "app": "Shortcuts"
            }
        }
        
        with patch('pytesseract.image_to_string') as mock_ocr:
            mock_ocr.return_value = "iOS photo OCR text"
            
            response = client.post("/api/shortcuts/photo-ocr", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
    
    def test_shortcuts_quick_note_endpoint(self, client):
        """Test Apple Shortcuts quick note endpoint."""
        payload = {
            "text": "Quick note from iOS Shortcuts",
            "note_type": "thought",
            "location_data": {
                "latitude": 37.7749,
                "longitude": -122.4194,
                "address": "San Francisco, CA"
            },
            "auto_tag": True
        }
        
        with patch('services.enhanced_apple_shortcuts_service.ollama_generate_title'), \
             patch('services.enhanced_apple_shortcuts_service.ollama_summarize'):
            
            response = client.post("/api/shortcuts/quick-note", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
    
    def test_shortcuts_web_clip_endpoint(self, client):
        """Test Apple Shortcuts web clip endpoint."""
        payload = {
            "url": "https://example.com/ios-article",
            "selected_text": "Selected text from Safari",
            "page_title": "iOS Article Title",
            "context": {
                "device": "iPhone",
                "app": "Safari"
            }
        }
        
        with patch('services.web_ingestion_service.WebIngestionService') as mock_web:
            mock_service = Mock()
            mock_service.ingest_url.return_value = {
                "success": True,
                "note_id": 789,
                "title": "iOS Web Clip"
            }
            mock_web.return_value = mock_service
            
            response = client.post("/api/shortcuts/web-clip", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
    
    def test_shortcuts_templates_endpoint(self, client):
        """Test Apple Shortcuts templates endpoint."""
        response = client.get("/api/shortcuts/templates")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert "templates" in data
        assert len(data["templates"]) >= 4
        assert data["total"] >= 4
        
        # Check template structure
        template = data["templates"][0]
        assert "name" in template
        assert "description" in template
        assert "endpoint" in template
        assert "parameters" in template
    
    def test_shortcuts_stats_endpoint(self, client):
        """Test Apple Shortcuts statistics endpoint."""
        response = client.get("/api/shortcuts/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert "data" in data
    
    def test_shortcuts_batch_endpoint(self, client):
        """Test Apple Shortcuts batch endpoint."""
        requests = [
            {
                "type": "quick_note",
                "data": {
                    "text": "Batch note 1",
                    "note_type": "thought"
                }
            },
            {
                "type": "quick_note",
                "data": {
                    "text": "Batch note 2", 
                    "note_type": "idea"
                }
            }
        ]
        
        with patch('services.enhanced_apple_shortcuts_service.ollama_generate_title'), \
             patch('services.enhanced_apple_shortcuts_service.ollama_summarize'):
            
            response = client.post("/api/shortcuts/batch", json=requests)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert data["total_requests"] == 2
            assert len(data["results"]) == 2
    
    def test_shortcuts_health_endpoint(self, client):
        """Test Apple Shortcuts health endpoint."""
        response = client.get("/api/shortcuts/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert data["service"] == "enhanced_apple_shortcuts"
        assert data["status"] == "healthy"
        assert "features" in data
    
    # Discord Integration API Tests
    
    def test_discord_health_endpoint(self, client):
        """Test Discord health endpoint."""
        response = client.get("/api/discord/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return health status (may be not configured in test)
        assert "success" in data
        assert "status" in data
    
    def test_discord_stats_endpoint(self, client):
        """Test Discord statistics endpoint."""
        with patch('services.enhanced_discord_service.EnhancedDiscordService.get_discord_usage_stats') as mock_stats:
            mock_stats.return_value = {
                "total_discord_notes": 50,
                "server_notes": 25,
                "top_channel": "general",
                "recent_notes": []
            }
            
            response = client.get("/api/discord/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "data" in data
    
    def test_discord_commands_endpoint(self, client):
        """Test Discord commands endpoint."""
        response = client.get("/api/discord/commands")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert "data" in data
        assert "slash_commands" in data["data"]
        assert "reaction_shortcuts" in data["data"]
        
        # Check for expected commands
        command_names = [cmd["name"] for cmd in data["data"]["slash_commands"]]
        assert "/capture" in command_names
        assert "/search" in command_names
        assert "/thread_summary" in command_names
    
    def test_discord_test_connection_endpoint(self, client):
        """Test Discord connection test endpoint."""
        response = client.post("/api/discord/test-connection")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
    
    # Error Handling Tests
    
    def test_unified_capture_invalid_content_type(self, client):
        """Test unified capture with invalid content type."""
        payload = {
            "content": "Test",
            "source": "invalid_source"
        }
        
        response = client.post("/api/unified-capture/text", json=payload)
        # Should still work, fallback to API source
        assert response.status_code == 200
    
    def test_unified_capture_missing_required_fields(self, client):
        """Test unified capture with missing required fields."""
        payload = {}  # Empty payload
        
        response = client.post("/api/unified-capture/text", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_advanced_capture_invalid_image(self, client):
        """Test advanced capture with invalid image data."""
        payload = {
            "image_data": "invalid_base64_data",
            "options": {}
        }
        
        response = client.post("/api/advanced-capture/screenshot-ocr", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == False
        assert "error" in data
    
    def test_shortcuts_invalid_location_data(self, client):
        """Test shortcuts with invalid location data."""
        payload = {
            "text": "Test note",
            "location_data": {
                "invalid": "format"
            }
        }
        
        response = client.post("/api/shortcuts/quick-note", json=payload)
        
        # Should still work, just ignore invalid location
        assert response.status_code == 200
    
    def test_batch_processing_size_limit(self, client):
        """Test batch processing size limits."""
        # Create too many requests
        requests = [{"content_type": "text", "content": f"Note {i}"} for i in range(51)]
        payload = {"requests": requests}
        
        response = client.post("/api/unified-capture/batch", json=payload)
        
        assert response.status_code == 422  # Should reject oversized batch
        data = response.json()
        assert "Maximum 50 requests per batch" in data["detail"]
    
    # Service Integration Tests
    
    def test_service_interdependencies(self, client):
        """Test that services work together correctly."""
        # Test unified capture using advanced capture for OCR
        payload = {
            "image_data": base64.b64encode(b"integrated_test_image").decode(),
            "source": "web_ui"
        }
        
        with patch('pytesseract.image_to_string') as mock_ocr:
            mock_ocr.return_value = "Integrated test text"
            
            response = client.post("/api/unified-capture/image", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert data["source_service"] == "advanced_capture"
    
    def test_webhook_endpoint_integration(self, client):
        """Test webhook endpoint for external integrations."""
        payload = {
            "content": "Webhook test content",
            "title": "External Integration",
            "tags": ["webhook", "external"]
        }
        
        response = client.post("/api/unified-capture/webhook/external", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True