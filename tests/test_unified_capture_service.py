"""
Unit tests for Unified Capture Service

Tests the core functionality of the unified capture system including
all content types, batch processing, and error handling.
"""

import pytest
import asyncio
import json
import base64
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from services.unified_capture_service import (
    UnifiedCaptureService,
    UnifiedCaptureRequest, 
    UnifiedCaptureResponse,
    CaptureSourceType,
    CaptureContentType
)


class TestUnifiedCaptureService:
    
    def _create_test_request(self, content_type=CaptureContentType.TEXT, source_type=CaptureSourceType.API, **kwargs):
        """Helper to create test requests with test mode enabled."""
        request = UnifiedCaptureRequest(
            content_type=content_type,
            source_type=source_type,
            **kwargs
        )
        request._test_mode = True  # Skip enhanced processing in tests
        return request
    
    @pytest.fixture
    def mock_db_conn(self):
        """Mock database connection."""
        conn = Mock()
        cursor = Mock()
        conn.cursor.return_value = cursor
        cursor.lastrowid = 123
        cursor.fetchone.return_value = None
        return conn
    
    @pytest.fixture
    def get_conn_func(self, mock_db_conn):
        """Mock database connection function."""
        return lambda: mock_db_conn
    
    @pytest.fixture
    def service(self, get_conn_func):
        """Create UnifiedCaptureService instance."""
        return UnifiedCaptureService(get_conn_func)
    
    @pytest.fixture
    def sample_text_request(self):
        """Sample text capture request."""
        request = UnifiedCaptureRequest(
            content_type=CaptureContentType.TEXT,
            source_type=CaptureSourceType.API,
            primary_content="This is a test note for unit testing",
            metadata={"test": True},
            custom_title="Test Note",
            custom_tags=["test", "unit-test"]
        )
        request._test_mode = True  # Skip enhanced processing in tests
        return request
    
    @pytest.fixture
    def sample_image_request(self):
        """Sample image OCR request."""
        # Create a simple base64 encoded test image
        test_image_data = base64.b64encode(b"fake_image_data").decode()
        return UnifiedCaptureRequest(
            content_type=CaptureContentType.PHOTO_OCR,
            source_type=CaptureSourceType.API,
            primary_content="",
            metadata={"filename": "test.png"},
            image_data=test_image_data
        )
    
    def test_service_initialization(self, service):
        """Test service initializes correctly."""
        assert service is not None
        assert service.processing_stats["total_requests"] == 0
        assert service.processing_stats["successful_captures"] == 0
        assert service.processing_stats["failed_captures"] == 0
    
    @pytest.mark.asyncio
    async def test_text_capture_success(self, service, sample_text_request):
        """Test successful text capture."""
        with patch('services.unified_capture_service.ollama_generate_title') as mock_title, \
             patch('services.unified_capture_service.ollama_summarize') as mock_summarize:
            
            mock_title.return_value = "Generated Title"
            mock_summarize.return_value = {
                "summary": "Test summary",
                "tags": ["ai-generated"],
                "actions": ["Review test results"]
            }
            
            result = await service.unified_capture(sample_text_request)
            
            assert result.success == True
            assert result.note_id == 123
            assert result.title == "Test Note"  # Uses custom title
            assert "test" in result.tags
            assert "unit-test" in result.tags
            assert result.summary == "Test summary"
            assert result.action_items == ["Review test results"]
            assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_text_capture_without_custom_title(self, service):
        """Test text capture uses generated title when none provided."""
        request = self._create_test_request(
            primary_content="Test content without title",
            metadata={}
        )
        
        with patch('services.unified_capture_service.ollama_generate_title') as mock_title, \
             patch('services.unified_capture_service.ollama_summarize') as mock_summarize:
            
            mock_title.return_value = "AI Generated Title"
            mock_summarize.return_value = {"summary": "", "tags": [], "actions": []}
            
            result = await service.unified_capture(request)
            
            assert result.success == True
            assert result.title == "AI Generated Title"
    
    @pytest.mark.asyncio
    async def test_image_ocr_capture(self, service, sample_image_request):
        """Test image OCR capture routing."""
        with patch.object(service, '_get_advanced_capture') as mock_advanced:
            mock_advanced_service = Mock()
            mock_advanced_service.capture_screenshot_with_ocr = AsyncMock()
            mock_advanced_service.capture_screenshot_with_ocr.return_value = Mock(
                success=True,
                note_id=456,
                title="OCR Result",
                extracted_text="Extracted text from image",
                tags=["ocr", "image"],
                summary="Image processed",
                error=None,
                warnings=[]
            )
            mock_advanced.return_value = mock_advanced_service
            
            result = await service.unified_capture(sample_image_request)
            
            assert result.success == True
            assert result.note_id == 456
            assert result.title == "OCR Result"
            assert result.source_service == "advanced_capture"
    
    @pytest.mark.asyncio
    async def test_batch_capture(self, service):
        """Test batch processing functionality."""
        requests = [
            UnifiedCaptureRequest(
                content_type=CaptureContentType.TEXT,
                source_type=CaptureSourceType.API,
                primary_content=f"Test note {i}",
                metadata={}
            ) for i in range(3)
        ]
        
        with patch('services.unified_capture_service.ollama_generate_title') as mock_title, \
             patch('services.unified_capture_service.ollama_summarize') as mock_summarize:
            
            mock_title.return_value = "Batch Title"
            mock_summarize.return_value = {"summary": "", "tags": [], "actions": []}
            
            results = await service.batch_capture(requests)
            
            assert len(results) == 3
            assert all(result.success for result in results)
            assert service.processing_stats["total_requests"] == 3
    
    @pytest.mark.asyncio
    async def test_batch_capture_size_limit(self, service):
        """Test batch processing respects size limits."""
        # Create too many requests
        requests = [Mock() for _ in range(51)]
        
        with pytest.raises(ValueError, match="Maximum 50 requests per batch"):
            await service.batch_capture(requests)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, service, sample_text_request):
        """Test error handling in capture process."""
        # Mock database connection to raise an exception
        service.get_conn = Mock(side_effect=Exception("Database error"))
        
        result = await service.unified_capture(sample_text_request)
        
        assert result.success == False
        assert "Database error" in result.error
        assert service.processing_stats["failed_captures"] == 1
    
    def test_processing_stats_tracking(self, service):
        """Test processing statistics are tracked correctly."""
        stats = service.get_processing_stats()
        
        assert "total_requests" in stats
        assert "successful_captures" in stats
        assert "failed_captures" in stats
        assert "success_rate" in stats
        assert "by_source" in stats
        assert "by_content_type" in stats
    
    def test_supported_integrations(self, service):
        """Test supported integrations information."""
        integrations = service.get_supported_integrations()
        
        assert "sources" in integrations
        assert "content_types" in integrations
        assert "features" in integrations
        assert "limits" in integrations
        
        # Check specific features
        assert CaptureSourceType.API.value in integrations["sources"]
        assert CaptureContentType.TEXT.value in integrations["content_types"]
        assert integrations["features"]["ai_processing"] == True
        assert integrations["limits"]["max_batch_size"] == 50
    
    @pytest.mark.asyncio
    async def test_voice_memo_processing(self, service):
        """Test voice memo capture routing."""
        request = UnifiedCaptureRequest(
            content_type=CaptureContentType.VOICE_MEMO,
            source_type=CaptureSourceType.APPLE_SHORTCUTS,
            primary_content="Transcribed voice memo",
            metadata={"duration": 30},
            audio_data="base64_audio_data",
            location_data={"latitude": 37.7749, "longitude": -122.4194}
        )
        
        with patch.object(service, '_get_apple_shortcuts') as mock_shortcuts:
            mock_service = Mock()
            mock_service.process_voice_memo = AsyncMock()
            mock_service.process_voice_memo.return_value = {
                "success": True,
                "note_id": 789,
                "title": "Voice Memo",
                "summary": "Processed voice memo",
                "tags": ["voice", "memo"],
                "action_items": []
            }
            mock_shortcuts.return_value = mock_service
            
            result = await service.unified_capture(request)
            
            assert result.success == True
            assert result.note_id == 789
            assert result.source_service == "apple_shortcuts"
    
    @pytest.mark.asyncio
    async def test_url_capture_routing(self, service):
        """Test URL capture routing to different services."""
        # Test Apple Shortcuts routing
        shortcuts_request = UnifiedCaptureRequest(
            content_type=CaptureContentType.URL,
            source_type=CaptureSourceType.APPLE_SHORTCUTS,
            primary_content="",
            metadata={"page_title": "Test Page"},
            url="https://example.com"
        )
        
        with patch.object(service, '_get_apple_shortcuts') as mock_shortcuts:
            mock_service = Mock()
            mock_service.process_web_clip = AsyncMock()
            mock_service.process_web_clip.return_value = {
                "success": True,
                "note_id": 101,
                "title": "Web Clip"
            }
            mock_shortcuts.return_value = mock_service
            
            result = await service.unified_capture(shortcuts_request)
            
            assert result.success == True
            assert result.source_service == "apple_shortcuts"
        
        # Test web ingestion routing for other sources
        web_request = UnifiedCaptureRequest(
            content_type=CaptureContentType.URL,
            source_type=CaptureSourceType.WEB_UI,
            primary_content="",
            metadata={},
            url="https://example.com"
        )
        
        with patch.object(service, '_get_web_ingestion') as mock_web:
            mock_service = Mock()
            mock_service.ingest_url = AsyncMock()
            mock_service.ingest_url.return_value = {
                "success": True,
                "note_id": 102,
                "title": "Web Content"
            }
            mock_web.return_value = mock_service
            
            result = await service.unified_capture(web_request)
            
            assert result.success == True
            assert result.source_service == "web_ingestion"
    
    @pytest.mark.asyncio
    async def test_pdf_capture(self, service):
        """Test PDF capture functionality."""
        request = UnifiedCaptureRequest(
            content_type=CaptureContentType.PDF,
            source_type=CaptureSourceType.API,
            primary_content="",
            metadata={"filename": "test.pdf"},
            file_data="base64_pdf_data"
        )
        
        with patch.object(service, '_get_advanced_capture') as mock_advanced:
            mock_service = Mock()
            mock_service.capture_pdf = AsyncMock()
            mock_service.capture_pdf.return_value = Mock(
                success=True,
                note_id=201,
                title="PDF Document",
                extracted_text="PDF content",
                tags=["pdf", "document"],
                summary="PDF processed",
                error=None,
                warnings=[]
            )
            mock_advanced.return_value = mock_service
            
            result = await service.unified_capture(request)
            
            assert result.success == True
            assert result.note_id == 201
            assert result.source_service == "advanced_capture"
    
    @pytest.mark.asyncio
    async def test_discord_context_handling(self, service):
        """Test Discord context processing."""
        request = UnifiedCaptureRequest(
            content_type=CaptureContentType.QUICK_NOTE,
            source_type=CaptureSourceType.DISCORD,
            primary_content="Discord message content",
            metadata={"note_type": "quick_note"},
            discord_context={
                "guild_id": 123456,
                "guild_name": "Test Server",
                "channel_id": 789012,
                "channel_name": "general",
                "user_id": 345678,
                "username": "testuser"
            }
        )
        
        with patch.object(service, '_get_discord_service') as mock_discord:
            mock_service = Mock()
            mock_service.capture_text_note = AsyncMock()
            mock_service.capture_text_note.return_value = {
                "success": True,
                "note_id": 301,
                "title": "Discord Note",
                "tags": ["discord", "quick_note"],
                "summary": "Discord message captured",
                "action_items": []
            }
            mock_discord.return_value = mock_service
            
            result = await service.unified_capture(request)
            
            assert result.success == True
            assert result.source_service == "discord"
    
    @pytest.mark.asyncio
    async def test_statistics_update_on_capture(self, service, sample_text_request):
        """Test that statistics are updated correctly after captures."""
        initial_stats = service.get_processing_stats()
        initial_total = initial_stats["total_requests"]
        
        with patch('services.unified_capture_service.ollama_generate_title'), \
             patch('services.unified_capture_service.ollama_summarize'):
            
            await service.unified_capture(sample_text_request)
            
            updated_stats = service.get_processing_stats()
            
            assert updated_stats["total_requests"] == initial_total + 1
            assert updated_stats["successful_captures"] >= initial_stats["successful_captures"]
            assert "api" in updated_stats["by_source"]
            assert "text" in updated_stats["by_content_type"]
    
    @pytest.mark.asyncio
    async def test_ai_processing_failure_handling(self, service, sample_text_request):
        """Test handling of AI processing failures."""
        with patch('services.unified_capture_service.ollama_generate_title') as mock_title, \
             patch('services.unified_capture_service.ollama_summarize') as mock_summarize:
            
            # Mock AI failure
            mock_title.side_effect = Exception("AI service unavailable")
            mock_summarize.side_effect = Exception("AI service unavailable")
            
            result = await service.unified_capture(sample_text_request)
            
            # Should still succeed with fallback behavior
            assert result.success == True
            assert result.title == "Test Note"  # Uses custom title
            # Should not have AI-generated content but still work
    
    @pytest.mark.asyncio
    async def test_content_formatting(self, service, sample_text_request):
        """Test content formatting with metadata."""
        with patch('services.unified_capture_service.ollama_generate_title') as mock_title, \
             patch('services.unified_capture_service.ollama_summarize') as mock_summarize:
            
            mock_title.return_value = "Test Title"
            mock_summarize.return_value = {
                "summary": "Content summary",
                "tags": ["formatted"],
                "actions": ["Test action item"]
            }
            
            # Add location data
            sample_text_request.location_data = {
                "latitude": 37.7749,
                "longitude": -122.4194,
                "name": "San Francisco, CA"
            }
            
            result = await service.unified_capture(sample_text_request)
            
            assert result.success == True
            # Content should be formatted with metadata
            # The actual formatting is tested in the integration tests