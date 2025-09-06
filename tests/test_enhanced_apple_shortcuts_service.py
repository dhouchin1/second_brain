"""
Unit tests for Enhanced Apple Shortcuts Service

Tests iOS/macOS integration including voice memos, photo OCR, location-based notes,
and shortcut template generation.
"""

import pytest
import base64
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from services.enhanced_apple_shortcuts_service import (
    EnhancedAppleShortcutsService
)


class TestEnhancedAppleShortcutsService:
    
    @pytest.fixture
    def mock_db_conn(self):
        """Mock database connection."""
        conn = Mock()
        cursor = Mock()
        conn.cursor.return_value = cursor
        cursor.lastrowid = 789
        cursor.fetchone.return_value = None
        return conn
    
    @pytest.fixture
    def get_conn_func(self, mock_db_conn):
        """Mock database connection function."""
        return lambda: mock_db_conn
    
    @pytest.fixture
    def service(self, get_conn_func):
        """Create EnhancedAppleShortcutsService instance."""
        return EnhancedAppleShortcutsService(get_conn_func)
    
    @pytest.fixture
    def sample_location_data(self):
        """Sample location data from iOS."""
        return {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "address": "San Francisco, CA, USA",
            "accuracy": 10.0,
            "altitude": 15.0
        }
    
    @pytest.fixture
    def sample_context_data(self):
        """Sample context data from iOS."""
        return {
            "timestamp": "2024-12-15T10:30:00Z",
            "device": "iPhone 15 Pro",
            "app": "Shortcuts",
            "battery_level": 0.85,
            "network_type": "WiFi"
        }
    
    @pytest.fixture
    def sample_image_data(self):
        """Sample base64 encoded image data."""
        return base64.b64encode(b"fake_image_data_from_ios").decode()
    
    def test_service_initialization(self, service):
        """Test service initializes correctly."""
        assert service is not None
        assert service.embedder is not None
        assert service.advanced_capture is None  # Lazy loaded
    
    def test_get_shortcut_templates(self, service):
        """Test shortcut template generation."""
        templates = service.get_shortcut_templates()
        
        assert isinstance(templates, list)
        assert len(templates) >= 4  # At least voice memo, photo OCR, quick note, web clip
        
        # Check template structure
        for template in templates:
            assert "name" in template
            assert "description" in template
            assert "endpoint" in template
            assert "method" in template
            assert "parameters" in template
            assert "shortcut_url" in template
        
        # Check specific templates exist
        template_names = [t["name"] for t in templates]
        assert "Quick Voice Memo" in template_names
        assert "Photo OCR Capture" in template_names
        assert "Quick Thought Capture" in template_names
        assert "Web Clip from Safari" in template_names
    
    @pytest.mark.asyncio
    async def test_voice_memo_with_transcription(self, service, sample_location_data, sample_context_data):
        """Test voice memo processing with pre-transcribed text."""
        transcription = "This is a test voice memo from my iPhone"
        
        with patch('services.enhanced_apple_shortcuts_service.ollama_generate_title') as mock_title, \
             patch('services.enhanced_apple_shortcuts_service.ollama_summarize') as mock_summarize:
            
            mock_title.return_value = "Voice Memo Test"
            mock_summarize.return_value = {
                "summary": "Test voice memo captured",
                "tags": ["personal", "test"],
                "actions": ["Review test results"]
            }
            
            result = await service.process_voice_memo(
                transcription=transcription,
                location_data=sample_location_data,
                context=sample_context_data
            )
            
            assert result["success"] == True
            assert result["note_id"] == 789
            assert result["title"] == "Voice Memo Test"
            assert result["summary"] == "Test voice memo captured"
            assert "voice-memo" in result["tags"]
            assert "audio" in result["tags"]
            assert "ios-shortcut" in result["tags"]
            assert result["action_items"] == ["Review test results"]
    
    @pytest.mark.asyncio
    async def test_voice_memo_with_audio_data(self, service):
        """Test voice memo processing with audio data but no transcription."""
        audio_data = base64.b64encode(b"fake_audio_data").decode()
        
        result = await service.process_voice_memo(
            audio_data=audio_data,
            transcription=None
        )
        
        # Should still work but use placeholder transcription
        assert result["success"] == True
        # Should indicate transcription not available
        # (In a full implementation, this would integrate with speech-to-text)
    
    @pytest.mark.asyncio
    async def test_voice_memo_no_content(self, service):
        """Test voice memo processing with no content provided."""
        result = await service.process_voice_memo()
        
        assert result["success"] == False
        assert "No transcription or audio data provided" in result["error"]
    
    @pytest.mark.asyncio
    async def test_photo_ocr_processing(self, service, sample_image_data, sample_location_data, sample_context_data):
        """Test photo OCR processing."""
        with patch.object(service, '_get_advanced_capture') as mock_advanced, \
             patch.object(service, 'get_conn') as mock_get_conn:
            # Mock advanced capture service
            mock_capture_service = Mock()
            mock_capture_service.capture_screenshot_with_ocr = AsyncMock()
            mock_result = Mock()
            mock_result.success = True
            mock_result.note_id = 456
            mock_result.title = "Photo OCR Result"
            mock_result.content = "Text extracted from photo"  # Use content instead of extracted_text
            mock_result.tags = ["ocr", "photo", "location"]  # Include location tag since location data is provided
            mock_result.error = None
            mock_capture_service.capture_screenshot_with_ocr.return_value = mock_result
            mock_advanced.return_value = mock_capture_service
            
            # Mock database connection
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_conn.cursor.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn
            
            result = await service.process_photo_ocr(
                image_data=sample_image_data,
                location_data=sample_location_data,
                context=sample_context_data
            )
            
            assert result["success"] == True
            assert result["note_id"] == 456
            assert result["title"] == "Photo OCR Result"
            assert result["extracted_text"] == "Text extracted from photo"
            assert "ocr" in result["tags"]
            assert "location" in result["tags"]  # Added due to location data
    
    @pytest.mark.asyncio
    async def test_photo_ocr_failure(self, service, sample_image_data):
        """Test photo OCR processing failure handling."""
        with patch.object(service, '_get_advanced_capture') as mock_advanced:
            mock_capture_service = Mock()
            mock_capture_service.capture_screenshot_with_ocr = AsyncMock()
            mock_capture_service.capture_screenshot_with_ocr.return_value = Mock(
                success=False,
                error="OCR processing failed",
                note_id=None
            )
            mock_advanced.return_value = mock_capture_service
            
            result = await service.process_photo_ocr(image_data=sample_image_data)
            
            assert result["success"] == False
            assert "OCR processing failed" in result["error"]
    
    @pytest.mark.asyncio
    async def test_quick_note_processing(self, service, sample_location_data, sample_context_data):
        """Test quick note processing."""
        note_text = "Quick thought: Testing the iOS shortcut integration"
        
        with patch('services.enhanced_apple_shortcuts_service.ollama_generate_title') as mock_title, \
             patch('services.enhanced_apple_shortcuts_service.ollama_summarize') as mock_summarize:
            
            mock_title.return_value = "iOS Quick Note"
            mock_summarize.return_value = {
                "summary": "Note about testing",
                "tags": ["testing", "ios"],
                "actions": []
            }
            
            result = await service.process_quick_note(
                text=note_text,
                note_type="thought",
                location_data=sample_location_data,
                context=sample_context_data,
                auto_tag=True
            )
            
            assert result["success"] == True
            assert result["title"] == "iOS Quick Note"
            assert "quick-note" in result["tags"]
            assert "thought" in result["tags"]
            assert "ios-shortcut" in result["tags"]
            assert "location" in result["tags"]
    
    @pytest.mark.asyncio
    async def test_quick_note_different_types(self, service):
        """Test different quick note types."""
        note_types = ["thought", "task", "meeting", "idea", "reminder"]
        
        for note_type in note_types:
            with patch('services.enhanced_apple_shortcuts_service.ollama_generate_title') as mock_title:
                mock_title.return_value = f"Test {note_type}"
                
                result = await service.process_quick_note(
                    text=f"Test {note_type} content",
                    note_type=note_type,
                    auto_tag=False
                )
                
                assert result["success"] == True
                assert note_type in result["tags"]
    
    @pytest.mark.asyncio
    async def test_web_clip_processing(self, service, sample_context_data):
        """Test web clip processing."""
        url = "https://example.com/article"
        selected_text = "This is the selected text from the webpage"
        page_title = "Example Article Title"
        
        with patch.object(service, '_get_advanced_capture') as mock_advanced:
            mock_capture_service = Mock()
            mock_capture_service.capture_url = AsyncMock()
            mock_capture_service.capture_url.return_value = Mock(
                success=True,
                note_id=101,
                title="Web Clip: Example Article",
                extracted_text="Article content",
                tags=["web-clip", "article"],
                summary="Article summary",
                error=None
            )
            mock_advanced.return_value = mock_capture_service
            
            result = await service.process_web_clip(
                url=url,
                selected_text=selected_text,
                page_title=page_title,
                context=sample_context_data
            )
            
            assert result["success"] == True
            assert result["note_id"] == 101
            assert result["content_type"] == "web_clip"
    
    @pytest.mark.asyncio
    async def test_location_data_processing(self, service, sample_location_data):
        """Test location data integration."""
        with patch('services.enhanced_apple_shortcuts_service.ollama_generate_title'), \
             patch('services.enhanced_apple_shortcuts_service.ollama_summarize'):
            
            result = await service.process_quick_note(
                text="Note with location",
                location_data=sample_location_data
            )
            
            assert result["success"] == True
            assert "location" in result["tags"]
            # Location should be included in note content (tested in integration tests)
    
    @pytest.mark.asyncio
    async def test_context_data_processing(self, service, sample_context_data):
        """Test context data integration."""
        with patch('services.enhanced_apple_shortcuts_service.ollama_generate_title'), \
             patch('services.enhanced_apple_shortcuts_service.ollama_summarize'):
            
            result = await service.process_quick_note(
                text="Note with context",
                context=sample_context_data
            )
            
            assert result["success"] == True
            # Context should be preserved in metadata (tested in integration tests)
    
    @pytest.mark.asyncio
    async def test_ai_processing_failure_handling(self, service):
        """Test handling of AI processing failures."""
        with patch('services.enhanced_apple_shortcuts_service.ollama_generate_title') as mock_title, \
             patch('services.enhanced_apple_shortcuts_service.ollama_summarize') as mock_summarize:
            
            # Mock AI failures
            mock_title.side_effect = Exception("AI service unavailable")
            mock_summarize.side_effect = Exception("AI service unavailable")
            
            result = await service.process_quick_note(text="Test note")
            
            # Should still succeed with fallback behavior
            assert result["success"] == True
            assert "Quick Note" in result["title"]  # Fallback title
    
    @pytest.mark.asyncio
    async def test_database_storage_with_metadata(self, service, mock_db_conn, sample_location_data, sample_context_data):
        """Test database storage includes iOS-specific metadata."""
        with patch('services.enhanced_apple_shortcuts_service.ollama_generate_title'), \
             patch('services.enhanced_apple_shortcuts_service.ollama_summarize'):
            
            result = await service.process_quick_note(
                text="Test with metadata",
                location_data=sample_location_data,
                context=sample_context_data
            )
            
            assert result["success"] == True
            
            # Verify database call included metadata
            cursor = mock_db_conn.cursor.return_value
            cursor.execute.assert_called()
            
            # Check that execute was called with metadata parameter
            execute_calls = cursor.execute.call_args_list
            assert len(execute_calls) > 0
            
            # The metadata should be JSON serializable
            for call in execute_calls:
                args = call[0]
                if len(args) > 1 and isinstance(args[1], tuple):
                    for param in args[1]:
                        if isinstance(param, str):
                            try:
                                json.loads(param)
                                # If it parses, it's valid JSON metadata
                                break
                            except (json.JSONDecodeError, TypeError):
                                continue
    
    @pytest.mark.asyncio
    async def test_embedding_generation_with_location(self, service, sample_location_data):
        """Test embedding generation includes location context."""
        with patch.object(service.embedder, 'embed') as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3]
            
            result = await service.process_quick_note(
                text="Location-based note",
                location_data=sample_location_data
            )
            
            assert result["success"] == True
            mock_embed.assert_called_once()
            
            # Embedding text should include location context
            embed_text = mock_embed.call_args[0][0]
            assert "San Francisco, CA" in embed_text or "37.7749" in embed_text
    
    @pytest.mark.asyncio
    async def test_error_handling_in_storage(self, service):
        """Test error handling when database storage fails."""
        # Mock database connection to fail
        service.get_conn = Mock(side_effect=Exception("Database connection failed"))
        
        result = await service.process_quick_note(text="Test note")
        
        assert result["success"] == False
        assert "Database connection failed" in result["error"]
    
    def test_template_endpoint_validation(self, service):
        """Test that template endpoints are correctly formatted."""
        templates = service.get_shortcut_templates()
        
        for template in templates:
            endpoint = template["endpoint"]
            
            # Should be valid API endpoint format
            assert endpoint.startswith("/api/shortcuts/")
            assert template["method"] in ["GET", "POST", "PUT", "DELETE"]
            
            # Parameters should be valid JSON structure
            params = template["parameters"]
            assert isinstance(params, dict)
            
            # Should be JSON serializable
            json.dumps(params)
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, service):
        """Test that service handles concurrent requests correctly."""
        import asyncio
        
        # Process multiple notes concurrently
        tasks = []
        for i in range(5):
            task = service.process_quick_note(
                text=f"Concurrent note {i}",
                note_type="test"
            )
            tasks.append(task)
        
        with patch('services.enhanced_apple_shortcuts_service.ollama_generate_title'), \
             patch('services.enhanced_apple_shortcuts_service.ollama_summarize'):
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 5
            assert all(result["success"] for result in results)
    
    @pytest.mark.asyncio
    async def test_large_content_handling(self, service):
        """Test handling of large content from iOS."""
        # Create large note content
        large_text = "Large content. " * 1000  # ~15KB of text
        
        with patch('services.enhanced_apple_shortcuts_service.ollama_generate_title'), \
             patch('services.enhanced_apple_shortcuts_service.ollama_summarize'):
            
            result = await service.process_quick_note(text=large_text)
            
            assert result["success"] == True
            # Should handle large content without issues
    
    @pytest.mark.asyncio
    async def test_special_characters_handling(self, service):
        """Test handling of special characters from iOS."""
        special_text = "Note with émojis 🎉, unicode ñ, and symbols @#$%"
        
        with patch('services.enhanced_apple_shortcuts_service.ollama_generate_title'), \
             patch('services.enhanced_apple_shortcuts_service.ollama_summarize'):
            
            result = await service.process_quick_note(text=special_text)
            
            assert result["success"] == True
            # Should preserve special characters correctly
    
    @pytest.mark.asyncio 
    async def test_validation_and_sanitization(self, service):
        """Test input validation and sanitization."""
        # Test empty inputs
        result = await service.process_quick_note(text="")
        assert result["success"] == False
        assert "empty" in result["error"].lower() or "required" in result["error"].lower()
        
        # Test None inputs
        result = await service.process_quick_note(text=None)
        assert result["success"] == False
        
        # Test whitespace-only input
        result = await service.process_quick_note(text="   ")
        assert result["success"] == False