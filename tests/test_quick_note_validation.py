# ──────────────────────────────────────────────────────────────────────────────
# File: tests/test_quick_note_validation.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Simplified validation tests for the quick-note endpoint functionality.

Tests the core logic without complex async mocking.
"""

import pytest
from services.unified_capture_router import QuickNoteRequest


class TestQuickNoteRequestValidation:
    """Test QuickNoteRequest model validation and behavior."""
    
    def test_content_field_processing(self):
        """Test processing with content field."""
        request = QuickNoteRequest(content="Test content")
        # Test content field extraction logic
        note_content = (request.content or request.text or "").strip()
        assert note_content == "Test content"
    
    def test_text_field_processing(self):
        """Test processing with text field."""
        request = QuickNoteRequest(text="Test text")
        # Test text field extraction logic
        note_content = (request.content or request.text or "").strip()
        assert note_content == "Test text"
    
    def test_content_priority_over_text(self):
        """Test that content field takes priority over text field."""
        request = QuickNoteRequest(content="Priority content", text="Secondary text")
        # Mimic the logic from the actual endpoint
        note_content = (request.content or request.text or "").strip()
        assert note_content == "Priority content"
    
    def test_text_fallback_when_no_content(self):
        """Test that text field is used when content is None."""
        request = QuickNoteRequest(content=None, text="Fallback text")
        note_content = (request.content or request.text or "").strip()
        assert note_content == "Fallback text"
    
    def test_empty_content_detection(self):
        """Test empty content detection logic."""
        # Empty string
        request1 = QuickNoteRequest(content="")
        note_content1 = (request1.content or request1.text or "").strip()
        assert not note_content1  # Should be empty/falsy
        
        # None values
        request2 = QuickNoteRequest(content=None, text=None)
        note_content2 = (request2.content or request2.text or "").strip()
        assert not note_content2  # Should be empty/falsy
        
        # Only whitespace
        request3 = QuickNoteRequest(content="   \n\t   ")
        note_content3 = (request3.content or request3.text or "").strip()
        assert not note_content3  # Should be empty after strip
    
    def test_whitespace_trimming(self):
        """Test whitespace trimming logic."""
        request = QuickNoteRequest(content="  \n  Trimmed content  \n  ")
        note_content = (request.content or request.text or "").strip()
        assert note_content == "Trimmed content"
    
    def test_default_values(self):
        """Test default field values."""
        request = QuickNoteRequest()
        assert request.source == "api"
        assert request.note_type == "thought"
        assert request.content is None
        assert request.text is None
    
    def test_custom_source_and_note_type(self):
        """Test custom source and note_type values."""
        request = QuickNoteRequest(
            content="Test",
            source="web_ui",
            note_type="idea"
        )
        assert request.source == "web_ui"
        assert request.note_type == "idea"
    
    def test_long_content_handling(self):
        """Test handling of long content."""
        long_content = "A" * 1000
        request = QuickNoteRequest(content=long_content)
        note_content = (request.content or request.text or "").strip()
        assert len(note_content) == 1000
        assert note_content == long_content
    
    def test_unicode_content(self):
        """Test Unicode content handling."""
        unicode_content = "Test with emojis 🚀 and special chars: áéíóú ñ 中文"
        request = QuickNoteRequest(content=unicode_content)
        note_content = (request.content or request.text or "").strip()
        assert note_content == unicode_content
    
    def test_multiline_content_preservation(self):
        """Test that multiline content is preserved."""
        multiline_content = """Line 1
Line 2

Line 3 after blank line"""
        request = QuickNoteRequest(content=multiline_content)
        note_content = (request.content or request.text or "").strip()
        assert "\n" in note_content
        assert "Line 1" in note_content
        assert "Line 3 after blank line" in note_content


class TestPayloadProcessingLogic:
    """Test the payload processing logic from the endpoint."""
    
    def test_payload_key_filtering(self):
        """Test the key filtering logic used in the endpoint."""
        # Simulate the filtering logic from the actual endpoint
        payload = {
            "content": "Test content",
            "source": "test_source", 
            "note_type": "idea",
            "extra_key": "should_be_filtered",
            "another_extra": "also_filtered"
        }
        
        # Mimic the filtering logic: only keep known keys
        filtered = {k: v for k, v in payload.items() if k in {"content", "text", "source", "note_type"}}
        
        assert "content" in filtered
        assert "source" in filtered
        assert "note_type" in filtered
        assert "extra_key" not in filtered
        assert "another_extra" not in filtered
        assert len(filtered) == 3  # content, source, note_type
    
    def test_payload_with_text_field(self):
        """Test payload processing with text field."""
        payload = {"text": "Test text", "source": "api"}
        filtered = {k: v for k, v in payload.items() if k in {"content", "text", "source", "note_type"}}
        
        request = QuickNoteRequest(**filtered)
        note_content = (request.content or request.text or "").strip()
        assert note_content == "Test text"
    
    def test_query_param_merging_logic(self):
        """Test query parameter merging logic."""
        # Simulate payload from request body
        payload = {"content": "Body content"}
        
        # Simulate query parameters
        query_params = {"source": "query_source", "extra_param": "value"}
        
        # Mimic the merging logic: query params don't override existing keys
        for k, v in query_params.items():
            payload.setdefault(k, v)
        
        assert payload["content"] == "Body content"  # Not overridden
        assert payload["source"] == "query_source"   # Added from query
        assert payload["extra_param"] == "value"     # Added from query


class TestErrorConditions:
    """Test error conditions and edge cases."""
    
    def test_validation_with_empty_strings(self):
        """Test validation behavior with empty strings."""
        test_cases = [
            {"content": ""},
            {"text": ""},
            {"content": "", "text": ""},
            {"content": "   "},
            {"text": "   "},
            {"content": None, "text": None}
        ]
        
        for case in test_cases:
            request = QuickNoteRequest(**case)
            note_content = (request.content or request.text or "").strip()
            # All should result in empty content after processing
            assert not note_content, f"Case {case} should result in empty content"
    
    def test_valid_content_cases(self):
        """Test cases that should have valid content."""
        valid_cases = [
            {"content": "Valid content"},
            {"text": "Valid text"},
            {"content": "  Valid with whitespace  "},
            {"content": "\nValid with newlines\n"},
            {"content": "A"},  # Single character
            {"text": "🚀"},  # Single emoji
        ]
        
        for case in valid_cases:
            request = QuickNoteRequest(**case)
            note_content = (request.content or request.text or "").strip()
            # All should result in non-empty content after processing
            assert note_content, f"Case {case} should result in valid content"
    
    def test_special_character_sources(self):
        """Test that source field accepts various formats."""
        special_sources = [
            "web-ui",
            "my_app",
            "client.v2",
            "source123",
            "CamelCaseSource"
        ]
        
        for source in special_sources:
            request = QuickNoteRequest(content="Test", source=source)
            assert request.source == source
    
    def test_note_type_variations(self):
        """Test various note_type values."""
        note_types = [
            "thought",
            "idea", 
            "todo",
            "meeting-notes",
            "quick_note"
        ]
        
        for note_type in note_types:
            request = QuickNoteRequest(content="Test", note_type=note_type)
            assert request.note_type == note_type


if __name__ == "__main__":
    pytest.main([__file__, "-v"])