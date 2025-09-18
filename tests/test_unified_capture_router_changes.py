# ──────────────────────────────────────────────────────────────────────────────
# File: tests/test_unified_capture_router_changes.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Test suite for recent unified capture router changes.

Tests the enhanced quick-note endpoint with flexible request handling
for JSON and form-encoded submissions.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi import Request
from fastapi.responses import JSONResponse
from services.unified_capture_router import router, QuickNoteRequest
from services.unified_capture_service import UnifiedCaptureResponse, CaptureSourceType, CaptureContentType


class TestQuickNoteRequest:
    """Test the QuickNoteRequest model for flexible content handling."""
    
    def test_content_field_only(self):
        """Test request with content field."""
        request = QuickNoteRequest(content="Test note content")
        assert request.content == "Test note content"
        assert request.text is None
        assert request.source == "api"
        assert request.note_type == "thought"
    
    def test_text_field_only(self):
        """Test request with text field."""
        request = QuickNoteRequest(text="Test note text")
        assert request.text == "Test note text"
        assert request.content is None
        assert request.source == "api"
        assert request.note_type == "thought"
    
    def test_both_content_and_text_fields(self):
        """Test request with both content and text fields."""
        request = QuickNoteRequest(content="Test content", text="Test text")
        assert request.content == "Test content"
        assert request.text == "Test text"
    
    def test_custom_source_and_note_type(self):
        """Test request with custom source and note_type."""
        request = QuickNoteRequest(
            content="Test content",
            source="web_ui",
            note_type="idea"
        )
        assert request.source == "web_ui"
        assert request.note_type == "idea"
    
    def test_empty_request(self):
        """Test request with no content."""
        request = QuickNoteRequest()
        assert request.content is None
        assert request.text is None
        assert request.source == "api"
        assert request.note_type == "thought"


class TestQuickNoteEndpoint:
    """Test the enhanced quick-note endpoint with flexible request handling."""
    
    @pytest.fixture
    def mock_service(self):
        """Mock unified capture service."""
        with patch('services.unified_capture_router._service') as mock:
            service = Mock()
            service.unified_capture = AsyncMock()
            service.unified_capture.return_value = UnifiedCaptureResponse(
                success=True,
                note_id=123,
                title="Generated Title",
                content_preview="Test note content...",
                tags=["test"],
                summary="Test summary",
                source_service="unified_capture"
            )
            mock.return_value = service
            yield service
    
    @pytest.fixture
    def mock_request_json(self):
        """Create a mock request with JSON content-type."""
        request = Mock(spec=Request)
        request.headers = {"content-type": "application/json"}
        request.query_params = {}
        return request
    
    @pytest.fixture
    def mock_request_form(self):
        """Create a mock request with form content-type."""
        request = Mock(spec=Request)
        request.headers = {"content-type": "application/x-www-form-urlencoded"}
        request.query_params = {}
        return request
    
    @pytest.fixture
    def mock_request_query(self):
        """Create a mock request with query parameters."""
        request = Mock(spec=Request)
        request.headers = {}
        request.query_params = {"content": "Query parameter content"}
        return request
    
    @pytest.mark.asyncio
    async def test_json_content_field(self, mock_service, mock_request_json):
        """Test JSON request with content field."""
        mock_request_json.json = AsyncMock(return_value={
            "content": "JSON content test",
            "source": "web_ui"
        })
        
        # Import the endpoint function
        from services.unified_capture_router import capture_quick_note
        
        response = await capture_quick_note(mock_request_json)
        
        assert isinstance(response, JSONResponse)
        mock_service.unified_capture.assert_called_once()
        
        # Check the call arguments
        call_args = mock_service.unified_capture.call_args[0][0]
        assert call_args.primary_content == "JSON content test"
        assert call_args.source_type == CaptureSourceType.API
        assert call_args.content_type == CaptureContentType.TEXT
    
    @pytest.mark.asyncio
    async def test_json_text_field(self, mock_service, mock_request_json):
        """Test JSON request with text field."""
        mock_request_json.json = AsyncMock(return_value={
            "text": "JSON text test",
            "note_type": "idea"
        })
        
        from services.unified_capture_router import capture_quick_note
        
        response = await capture_quick_note(mock_request_json)
        
        assert isinstance(response, JSONResponse)
        mock_service.unified_capture.assert_called_once()
        
        call_args = mock_service.unified_capture.call_args[0][0]
        assert call_args.primary_content == "JSON text test"
    
    @pytest.mark.asyncio
    async def test_json_both_fields_content_priority(self, mock_service, mock_request_json):
        """Test JSON request with both content and text fields (content takes priority)."""
        mock_request_json.json = AsyncMock(return_value={
            "content": "Priority content",
            "text": "Secondary text"
        })
        
        from services.unified_capture_router import capture_quick_note
        
        response = await capture_quick_note(mock_request_json)
        
        call_args = mock_service.unified_capture.call_args[0][0]
        assert call_args.primary_content == "Priority content"
    
    @pytest.mark.asyncio
    async def test_form_encoded_content(self, mock_service, mock_request_form):
        """Test form-encoded request."""
        mock_form_data = {
            "content": "Form encoded content",
            "source": "form_client"
        }
        mock_request_form.form = AsyncMock(return_value=mock_form_data)
        
        from services.unified_capture_router import capture_quick_note
        
        response = await capture_quick_note(mock_request_form)
        
        mock_service.unified_capture.assert_called_once()
        call_args = mock_service.unified_capture.call_args[0][0]
        assert call_args.primary_content == "Form encoded content"
    
    @pytest.mark.asyncio
    async def test_query_parameters_rejected(self, mock_service, mock_request_query):
        """Ensure queries with only URL parameters are rejected."""
        mock_request_query.json = AsyncMock(side_effect=Exception("No JSON"))
        mock_request_query.form = AsyncMock(side_effect=Exception("No form"))

        from services.unified_capture_router import capture_quick_note

        response = await capture_quick_note(mock_request_query)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 422
        mock_service.unified_capture.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_empty_content_validation(self, mock_service, mock_request_json):
        """Test validation when no content is provided."""
        mock_request_json.json = AsyncMock(return_value={})
        mock_request_json.query_params = {}
        
        from services.unified_capture_router import capture_quick_note
        
        response = await capture_quick_note(mock_request_json)
        
        # Should return 422 error for no content
        assert response.status_code == 422
        mock_service.unified_capture.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_whitespace_only_content_validation(self, mock_service, mock_request_json):
        """Test validation when only whitespace is provided."""
        mock_request_json.json = AsyncMock(return_value={"content": "   \\n\\t   "})
        mock_request_json.query_params = {}
        
        from services.unified_capture_router import capture_quick_note
        
        response = await capture_quick_note(mock_request_json)
        
        # Should return 422 error for whitespace-only content
        assert response.status_code == 422
        mock_service.unified_capture.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_content_trimming(self, mock_service, mock_request_json):
        """Test that content is properly trimmed."""
        mock_request_json.json = AsyncMock(return_value={
            "content": "  \\n  Trimmed content  \\n  "
        })
        
        from services.unified_capture_router import capture_quick_note
        
        response = await capture_quick_note(mock_request_json)
        
        call_args = mock_service.unified_capture.call_args[0][0]
        assert call_args.primary_content == "Trimmed content"
    
    @pytest.mark.asyncio
    async def test_multipart_form_data(self, mock_service):
        """Test multipart/form-data content type."""
        request = Mock(spec=Request)
        request.headers = {"content-type": "multipart/form-data; boundary=something"}
        request.query_params = {}
        request.json = AsyncMock(side_effect=Exception("No JSON"))
        request.form = AsyncMock(return_value={"text": "Multipart content"})
        
        from services.unified_capture_router import capture_quick_note
        
        response = await capture_quick_note(request)
        
        call_args = mock_service.unified_capture.call_args[0][0]
        assert call_args.primary_content == "Multipart content"
    
    @pytest.mark.asyncio
    async def test_query_params_override_behavior(self, mock_service, mock_request_json):
        """Test that query parameters are merged but don't override payload."""
        mock_request_json.json = AsyncMock(return_value={"content": "JSON content"})
        mock_request_json.query_params = {"source": "query_source", "extra_param": "value"}
        
        from services.unified_capture_router import capture_quick_note
        
        response = await capture_quick_note(mock_request_json)
        
        call_args = mock_service.unified_capture.call_args[0][0]
        assert call_args.primary_content == "JSON content"
        # Query params should be merged but not override existing keys
    
    @pytest.mark.asyncio
    async def test_service_not_initialized_error(self):
        """Test error handling when service is not initialized."""
        with patch('services.unified_capture_router._service', None):
            from services.unified_capture_router import capture_quick_note
            
            request = Mock(spec=Request)
            
            with pytest.raises(Exception):  # Should raise HTTPException with 503
                await capture_quick_note(request)
    
    @pytest.mark.asyncio
    async def test_json_parsing_error_fallback(self, mock_service):
        """Test fallback behavior when JSON parsing fails."""
        request = Mock(spec=Request)
        request.headers = {"content-type": "application/json"}
        request.json = AsyncMock(side_effect=ValueError("Invalid JSON"))
        request.form = AsyncMock(side_effect=Exception("No form"))
        request.query_params = {"text": "Fallback content"}
        
        from services.unified_capture_router import capture_quick_note
        
        response = await capture_quick_note(request)
        
        call_args = mock_service.unified_capture.call_args[0][0]
        assert call_args.primary_content == "Fallback content"
    
    @pytest.mark.asyncio
    async def test_form_parsing_error_fallback(self, mock_service):
        """Test fallback behavior when form parsing fails."""
        request = Mock(spec=Request)
        request.headers = {"content-type": "application/x-www-form-urlencoded"}
        request.json = AsyncMock(side_effect=Exception("No JSON"))
        request.form = AsyncMock(side_effect=ValueError("Invalid form"))
        request.query_params = {"content": "Query fallback"}
        
        from services.unified_capture_router import capture_quick_note
        
        response = await capture_quick_note(request)
        
        call_args = mock_service.unified_capture.call_args[0][0]
        assert call_args.primary_content == "Query fallback"
    
    @pytest.mark.asyncio
    async def test_successful_response_format(self, mock_service, mock_request_json):
        """Test the format of successful responses."""
        mock_request_json.json = AsyncMock(return_value={"content": "Success test"})
        
        from services.unified_capture_router import capture_quick_note
        
        response = await capture_quick_note(mock_request_json)
        
        assert response.status_code == 200
        # Response should contain success indicators
        
        # Verify the service was called with correct parameters
        mock_service.unified_capture.assert_called_once()
        call_args = mock_service.unified_capture.call_args[0][0]
        assert call_args.primary_content == "Success test"
        assert call_args.content_type == CaptureContentType.TEXT
        assert call_args.source_type == CaptureSourceType.API
    
    @pytest.mark.asyncio
    async def test_service_processing_error_handling(self, mock_service, mock_request_json):
        """Test error handling when unified capture service fails."""
        mock_request_json.json = AsyncMock(return_value={"content": "Error test"})
        mock_service.unified_capture.side_effect = Exception("Service processing error")
        
        from services.unified_capture_router import capture_quick_note
        
        with pytest.raises(Exception):  # Should propagate the error
            await capture_quick_note(mock_request_json)


class TestRouterIntegration:
    """Integration tests for the router with actual FastAPI test client."""
    
    @pytest.mark.asyncio
    async def test_router_mounting(self):
        """Test that the router is properly configured."""
        assert router.prefix == "/api/unified-capture"
        assert "unified-capture" in router.tags
    
    def test_quick_note_request_model_validation(self):
        """Test Pydantic model validation for QuickNoteRequest."""
        # Valid requests
        valid_data = [
            {"content": "Test"},
            {"text": "Test"},
            {"content": "Test", "source": "custom"},
            {"text": "Test", "note_type": "idea"},
            {}  # Empty should work with defaults
        ]
        
        for data in valid_data:
            request = QuickNoteRequest(**data)
            assert isinstance(request, QuickNoteRequest)
        
        # Test default values
        empty_request = QuickNoteRequest()
        assert empty_request.source == "api"
        assert empty_request.note_type == "thought"
        assert empty_request.content is None
        assert empty_request.text is None


# Additional edge case tests
class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_long_content(self):
        """Test handling of very long content."""
        long_content = "A" * 10000  # 10K characters
        request = QuickNoteRequest(content=long_content)
        assert len(request.content) == 10000
    
    def test_unicode_content_handling(self):
        """Test proper handling of Unicode content."""
        unicode_content = "Test with emojis 🚀 and special chars: áéíóú ñ 中文"
        request = QuickNoteRequest(content=unicode_content)
        assert request.content == unicode_content
    
    def test_special_characters_in_source(self):
        """Test handling of special characters in source field."""
        request = QuickNoteRequest(
            content="Test",
            source="my-custom_source.123"
        )
        assert request.source == "my-custom_source.123"
    
    def test_newlines_and_formatting_preservation(self):
        """Test that newlines and formatting are preserved."""
        formatted_content = """Line 1
        
        Line 2 with indent
        
        - Bullet point
        - Another point"""
        
        request = QuickNoteRequest(content=formatted_content)
        assert request.content == formatted_content
        # Test for actual newlines (not escaped), should find them
        assert "\n" in request.content or "\r" in request.content or len(request.content.split("\n")) > 1


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
