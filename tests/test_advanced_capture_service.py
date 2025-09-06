"""
Unit tests for Advanced Capture Service

Tests OCR processing, PDF extraction, YouTube transcripts, and bulk operations.
"""

import pytest
import base64
import json
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
from PIL import Image
import io

from services.advanced_capture_service import (
    AdvancedCaptureService,
    CaptureOptions,
    CaptureResult
)


class TestAdvancedCaptureService:
    
    @pytest.fixture
    def mock_db_conn(self):
        """Mock database connection."""
        conn = Mock()
        cursor = Mock()
        conn.cursor.return_value = cursor
        cursor.lastrowid = 456
        cursor.fetchone.return_value = None
        return conn
    
    @pytest.fixture
    def get_conn_func(self, mock_db_conn):
        """Mock database connection function.""" 
        return lambda: mock_db_conn
    
    @pytest.fixture
    def service(self, get_conn_func):
        """Create AdvancedCaptureService instance."""
        return AdvancedCaptureService(get_conn_func)
    
    @pytest.fixture
    def sample_image_data(self):
        """Create sample base64 image data."""
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='white')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode()
    
    @pytest.fixture
    def capture_options(self):
        """Sample capture options."""
        return CaptureOptions(
            enable_ai_processing=True,
            enable_ocr=True,
            custom_tags=["test", "ocr"],
            quality="high"
        )
    
    def test_service_initialization(self, service):
        """Test service initializes correctly."""
        assert service is not None
        features = service.get_feature_availability()
        assert "ocr" in features
        assert "image_processing" in features
    
    def test_feature_availability_checking(self, service):
        """Test feature availability detection."""
        features = service.get_feature_availability()
        
        # These should always be available
        assert features["image_processing"] == True
        
        # OCR availability depends on pytesseract
        assert isinstance(features["ocr"], bool)
        
        # PDF and YouTube availability depend on optional packages
        assert isinstance(features["pdf"], bool)
        assert isinstance(features["youtube"], bool)
    
    @pytest.mark.asyncio
    async def test_screenshot_ocr_success(self, service, sample_image_data, capture_options):
        """Test successful OCR processing."""
        with patch('pytesseract.image_to_string') as mock_ocr, \
             patch('services.advanced_capture_service.ollama_generate_title') as mock_title, \
             patch('services.advanced_capture_service.ollama_summarize') as mock_summarize, \
             patch.object(service, '_save_note') as mock_save:
            
            # Mock OCR extraction
            mock_ocr.return_value = "Extracted text from image"
            mock_title.return_value = "OCR Result"
            mock_summarize.return_value = {
                "summary": "Image contains extracted text",
                "tags": ["extracted", "text"],
                "actions": []
            }
            mock_save.return_value = 456  # Mock note ID
            
            result = await service.capture_screenshot_with_ocr(sample_image_data, capture_options)
            
            assert result.success == True
            assert result.note_id == 456
            assert result.title == "OCR Result"
            assert "Extracted text from image" in result.content
            assert "test" in result.tags
            assert "ocr" in result.tags
            # Verify processing completed successfully
            assert result.processing_time >= 0
    
    @pytest.mark.asyncio
    async def test_screenshot_ocr_no_tesseract(self, service, sample_image_data, capture_options):
        """Test OCR when pytesseract is not available."""
        with patch('pytesseract.image_to_string', side_effect=ImportError("pytesseract not available")):
            
            result = await service.capture_screenshot_with_ocr(sample_image_data, capture_options)
            
            assert result.success == False
            assert "pytesseract not available" in result.error
    
    @pytest.mark.asyncio
    async def test_screenshot_ocr_invalid_image(self, service, capture_options):
        """Test OCR with invalid image data."""
        invalid_image_data = "invalid_base64_data"
        
        result = await service.capture_screenshot_with_ocr(invalid_image_data, capture_options)
        
        assert result.success == False
        assert "Invalid base64-encoded string" in result.error or "Invalid image data" in result.error or "Failed to decode" in result.error
    
    @pytest.mark.asyncio
    async def test_pdf_capture_dependency_check(self, service, capture_options):
        """Test PDF processing dependency availability check."""
        # Test that the service properly checks for PDF dependency
        # Since PyMuPDF is not installed in the test environment, this should fail gracefully
        pdf_data_bytes = b"fake_pdf_content"
        
        result = await service.capture_pdf_content(pdf_data=pdf_data_bytes, options=capture_options)
        
        # Should fail gracefully when dependency is not available
        assert result.success == False
        assert "PDF processing not available" in result.error or "PyMuPDF" in result.error
    
    @pytest.mark.asyncio
    async def test_pdf_capture_no_pymupdf(self, service, capture_options):
        """Test PDF processing when PyMuPDF is not available."""
        pdf_data = base64.b64encode(b"fake_pdf_content").decode()
        
        with patch('fitz.open', side_effect=ImportError("PyMuPDF not available")):
            
            result = await service.capture_pdf(pdf_data, "test.pdf", capture_options)
            
            assert result.success == False
            assert "PDF processing not available" in result.error
    
    @pytest.mark.asyncio
    async def test_youtube_transcript_success(self, service, capture_options):
        """Test successful YouTube transcript extraction."""
        youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        
        with patch('youtube_transcript_api.YouTubeTranscriptApi.get_transcript') as mock_transcript, \
             patch('services.advanced_capture_service.ollama_generate_title') as mock_title, \
             patch('services.advanced_capture_service.ollama_summarize') as mock_summarize:
            
            # Mock transcript data
            mock_transcript.return_value = [
                {"text": "Hello and welcome", "start": 0.0},
                {"text": "to this video", "start": 2.5},
                {"text": "about testing", "start": 5.0}
            ]
            
            mock_title.return_value = "YouTube Video Transcript"
            mock_summarize.return_value = {
                "summary": "Video about testing",
                "tags": ["video", "transcript"],
                "actions": []
            }
            
            result = await service.capture_youtube_transcript(youtube_url, capture_options)
            
            assert result.success == True
            assert result.title == "YouTube Video Transcript"
            assert "Hello and welcome to this video about testing" in result.extracted_text
            assert "video" in result.tags
    
    @pytest.mark.asyncio
    async def test_youtube_transcript_invalid_url(self, service, capture_options):
        """Test YouTube transcript with invalid URL."""
        invalid_url = "https://not-youtube.com/watch?v=invalid"
        
        result = await service.capture_youtube_transcript(invalid_url, capture_options)
        
        assert result.success == False
        assert "Invalid YouTube URL" in result.error
    
    @pytest.mark.asyncio
    async def test_youtube_transcript_no_api(self, service, capture_options):
        """Test YouTube transcript when API is not available."""
        youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        
        with patch('youtube_transcript_api.YouTubeTranscriptApi.get_transcript', 
                  side_effect=ImportError("youtube-transcript-api not available")):
            
            result = await service.capture_youtube_transcript(youtube_url, capture_options)
            
            assert result.success == False
            assert "YouTube transcript extraction not available" in result.error
    
    @pytest.mark.asyncio
    async def test_bulk_url_processing(self, service):
        """Test bulk URL processing."""
        urls = [
            "https://example.com/page1",
            "https://example.com/page2", 
            "https://example.com/page3"
        ]
        
        with patch('services.web_ingestion_service.WebIngestionService') as mock_web_service:
            mock_service_instance = Mock()
            mock_service_instance.ingest_url = AsyncMock()
            mock_service_instance.ingest_url.side_effect = [
                {"success": True, "note_id": 101, "title": "Page 1"},
                {"success": True, "note_id": 102, "title": "Page 2"},
                {"success": False, "error": "Failed to process Page 3"}
            ]
            mock_web_service.return_value = mock_service_instance
            
            results = await service.process_bulk_urls(urls, max_concurrent=2)
            
            assert len(results) == 3
            assert results[0]["success"] == True
            assert results[0]["note_id"] == 101
            assert results[1]["success"] == True
            assert results[2]["success"] == False
            assert "Failed to process" in results[2]["error"]
    
    @pytest.mark.asyncio
    async def test_bulk_url_processing_limit(self, service):
        """Test bulk URL processing respects limits."""
        # Create too many URLs
        urls = [f"https://example.com/page{i}" for i in range(101)]
        
        with pytest.raises(ValueError, match="Maximum 100 URLs per batch"):
            await service.process_bulk_urls(urls)
    
    @pytest.mark.asyncio
    async def test_capture_options_handling(self, service, sample_image_data):
        """Test different capture options are handled correctly."""
        # Test with minimal options
        minimal_options = CaptureOptions(
            generate_title=False,
            extract_summary=False
        )
        
        with patch('pytesseract.image_to_string') as mock_ocr:
            mock_ocr.return_value = "Test text"
            
            result = await service.capture_screenshot_with_ocr(sample_image_data, minimal_options)
            
            assert result.success == True
            assert result.title == "Screenshot OCR Result"  # Default title
            assert result.summary == ""  # No summary generated
    
    @pytest.mark.asyncio
    async def test_error_handling_in_ai_processing(self, service, sample_image_data, capture_options):
        """Test error handling when AI processing fails."""
        with patch('pytesseract.image_to_string') as mock_ocr, \
             patch('services.advanced_capture_service.ollama_generate_title') as mock_title, \
             patch('services.advanced_capture_service.ollama_summarize') as mock_summarize:
            
            mock_ocr.return_value = "Extracted text"
            mock_title.side_effect = Exception("AI service error")
            mock_summarize.side_effect = Exception("AI service error")
            
            result = await service.capture_screenshot_with_ocr(sample_image_data, capture_options)
            
            # Should still succeed with fallback behavior
            assert result.success == True
            assert result.extracted_text == "Extracted text"
            assert "Screenshot OCR Result" in result.title  # Fallback title
            assert len(result.warnings) > 0  # Should have warnings about AI failure
    
    @pytest.mark.asyncio
    async def test_database_storage(self, service, sample_image_data, capture_options, mock_db_conn):
        """Test database storage functionality."""
        with patch('pytesseract.image_to_string') as mock_ocr:
            mock_ocr.return_value = "Test OCR content"
            
            result = await service.capture_screenshot_with_ocr(sample_image_data, capture_options)
            
            assert result.success == True
            
            # Verify database interaction
            mock_db_conn.cursor.assert_called()
            cursor = mock_db_conn.cursor.return_value
            cursor.execute.assert_called()
            mock_db_conn.commit.assert_called()
            mock_db_conn.close.assert_called()
    
    @pytest.mark.asyncio
    async def test_embedding_generation(self, service, sample_image_data, capture_options):
        """Test vector embedding generation."""
        with patch('pytesseract.image_to_string') as mock_ocr, \
             patch.object(service.embedder, 'embed') as mock_embed:
            
            mock_ocr.return_value = "Content for embedding"
            mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]  # Mock embedding vector
            
            result = await service.capture_screenshot_with_ocr(sample_image_data, capture_options)
            
            assert result.success == True
            mock_embed.assert_called_once()
            # Should embed title + content
            embed_text = mock_embed.call_args[0][0]
            assert "Content for embedding" in embed_text
    
    def test_url_validation(self, service):
        """Test URL validation for YouTube."""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ"
        ]
        
        invalid_urls = [
            "https://vimeo.com/123456",
            "https://example.com",
            "not-a-url",
            ""
        ]
        
        for url in valid_urls:
            assert service._is_youtube_url(url) == True
        
        for url in invalid_urls:
            assert service._is_youtube_url(url) == False
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, service):
        """Test that bulk operations handle concurrency correctly."""
        urls = [f"https://example.com/page{i}" for i in range(10)]
        
        with patch('services.web_ingestion_service.WebIngestionService') as mock_web_service:
            mock_service_instance = Mock()
            
            # Mock successful processing for all URLs
            mock_service_instance.ingest_url = AsyncMock()
            mock_service_instance.ingest_url.return_value = {
                "success": True, 
                "note_id": 123, 
                "title": "Test Page"
            }
            mock_web_service.return_value = mock_service_instance
            
            results = await service.process_bulk_urls(urls, max_concurrent=5)
            
            assert len(results) == 10
            assert all(result["success"] for result in results)
            
            # Verify all URLs were processed
            assert mock_service_instance.ingest_url.call_count == 10
    
    @pytest.mark.asyncio
    async def test_capture_result_serialization(self, service, sample_image_data, capture_options):
        """Test that CaptureResult can be properly serialized."""
        with patch('pytesseract.image_to_string') as mock_ocr:
            mock_ocr.return_value = "Serialization test"
            
            result = await service.capture_screenshot_with_ocr(sample_image_data, capture_options)
            
            # Test that result can be converted to dict
            result_dict = {
                "success": result.success,
                "note_id": result.note_id,
                "title": result.title,
                "extracted_text": result.extracted_text,
                "tags": result.tags,
                "summary": result.summary,
                "error": result.error,
                "warnings": result.warnings
            }
            
            # Should be JSON serializable
            json_str = json.dumps(result_dict)
            assert isinstance(json_str, str)
            
            # Should be deserializable
            deserialized = json.loads(json_str)
            assert deserialized["success"] == result.success