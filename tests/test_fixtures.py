"""
Comprehensive test fixtures and mock data for enhanced capture system testing.

Provides reusable fixtures for all test suites including database setup,
sample data, and mock services.
"""

import pytest
import tempfile
import sqlite3
from pathlib import Path
import base64
import json
from datetime import datetime
from unittest.mock import Mock, AsyncMock
from PIL import Image
import io

# Import services for mocking
from services.unified_capture_service import CaptureSourceType, CaptureContentType
from services.enhanced_discord_service import DiscordContext, ThreadCapture


class TestDatabaseFixtures:
    """Database-related test fixtures."""
    
    @pytest.fixture
    def temp_database(self):
        """Create temporary SQLite database for testing."""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        # Initialize with full schema
        conn = sqlite3.connect(temp_db.name)
        
        # Core tables
        conn.execute('''
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                body TEXT,
                content TEXT,
                summary TEXT,
                tags TEXT,
                actions TEXT,
                type TEXT DEFAULT 'note',
                timestamp TEXT,
                created_at TEXT,
                updated_at TEXT,
                metadata TEXT,
                user_id INTEGER DEFAULT 1,
                status TEXT DEFAULT 'complete',
                audio_filename TEXT,
                file_metadata TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                email TEXT,
                created_at TEXT,
                is_active INTEGER DEFAULT 1
            )
        ''')
        
        # Vector embeddings table (optional)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS note_vecs (
                note_id INTEGER PRIMARY KEY,
                embedding TEXT,
                FOREIGN KEY(note_id) REFERENCES notes(id)
            )
        ''')
        
        # FTS5 table for search
        conn.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts 
            USING fts5(title, body, content, tags, content=notes, content_rowid=id)
        ''')
        
        # Insert test user
        conn.execute("""
            INSERT INTO users (username, hashed_password, email, created_at) 
            VALUES (?, ?, ?, ?)
        """, (
            "testuser",
            "$2b$12$test_hashed_password_here",
            "test@example.com",
            datetime.now().isoformat()
        ))
        
        # Insert sample notes for testing
        sample_notes = [
            ("Test Note 1", "This is a test note", "test, sample", "note"),
            ("Discord Message", "Message from Discord bot", "discord, message", "discord_note"),
            ("OCR Result", "Text extracted from image", "ocr, image", "screenshot"),
            ("PDF Document", "Content from PDF file", "pdf, document", "pdf"),
            ("Voice Memo", "Transcribed voice content", "voice, memo", "voice_memo")
        ]
        
        for i, (title, content, tags, note_type) in enumerate(sample_notes, 1):
            conn.execute("""
                INSERT INTO notes (title, body, content, tags, type, created_at, updated_at, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                title, content, content, tags, note_type,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                1
            ))
        
        conn.commit()
        conn.close()
        
        yield temp_db.name
        
        # Cleanup
        Path(temp_db.name).unlink()
    
    @pytest.fixture
    def mock_db_connection(self, temp_database):
        """Mock database connection function."""
        def get_test_conn():
            return sqlite3.connect(temp_database)
        return get_test_conn


class TestDataFixtures:
    """Sample data fixtures for testing."""
    
    @pytest.fixture
    def sample_text_content(self):
        """Sample text content for testing."""
        return {
            "short": "Quick test note",
            "medium": "This is a medium length test note with some details about testing functionality.",
            "long": "This is a very long test note. " * 50,
            "with_emojis": "Test note with emojis 🎉 and unicode characters ñáéíóú",
            "with_markdown": "# Test Note\n\n**Bold text** and *italic text*\n\n- List item 1\n- List item 2",
            "with_code": "Test note with `inline code` and:\n```python\nprint('hello world')\n```"
        }
    
    @pytest.fixture 
    def sample_location_data(self):
        """Sample location data from iOS."""
        return {
            "san_francisco": {
                "latitude": 37.7749,
                "longitude": -122.4194,
                "address": "San Francisco, CA, USA",
                "accuracy": 10.0,
                "altitude": 15.0
            },
            "new_york": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "address": "New York, NY, USA",
                "accuracy": 5.0,
                "altitude": 10.0
            },
            "london": {
                "latitude": 51.5074,
                "longitude": -0.1278,
                "address": "London, UK",
                "accuracy": 8.0,
                "altitude": 25.0
            }
        }
    
    @pytest.fixture
    def sample_context_data(self):
        """Sample context data from various sources."""
        return {
            "ios": {
                "timestamp": "2024-12-15T10:30:00Z",
                "device": "iPhone 15 Pro",
                "app": "Shortcuts",
                "battery_level": 0.85,
                "network_type": "WiFi",
                "os_version": "iOS 17.2"
            },
            "discord": {
                "guild_id": 123456789,
                "guild_name": "Test Server",
                "channel_id": 987654321,
                "channel_name": "general",
                "user_id": 555666777,
                "username": "testuser",
                "message_id": 111222333
            },
            "web_ui": {
                "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                "ip_address": "192.168.1.100",
                "session_id": "test_session_123"
            }
        }
    
    @pytest.fixture
    def sample_image_data(self):
        """Generate sample base64 image data."""
        # Create test images with different properties
        images = {}
        
        # Small test image
        small_img = Image.new('RGB', (100, 100), color='white')
        buffer = io.BytesIO()
        small_img.save(buffer, format='PNG')
        images['small'] = base64.b64encode(buffer.getvalue()).decode()
        
        # Medium test image
        medium_img = Image.new('RGB', (500, 300), color='lightblue')
        buffer = io.BytesIO()
        medium_img.save(buffer, format='PNG')
        images['medium'] = base64.b64encode(buffer.getvalue()).decode()
        
        # Large test image
        large_img = Image.new('RGB', (1920, 1080), color='lightgray')
        buffer = io.BytesIO()
        large_img.save(buffer, format='JPEG', quality=85)
        images['large'] = base64.b64encode(buffer.getvalue()).decode()
        
        return images
    
    @pytest.fixture
    def sample_pdf_data(self):
        """Generate sample PDF data."""
        return {
            'small': base64.b64encode(b"fake_small_pdf_content").decode(),
            'medium': base64.b64encode(b"fake_medium_pdf_content" * 100).decode(),
            'large': base64.b64encode(b"fake_large_pdf_content" * 1000).decode()
        }
    
    @pytest.fixture
    def sample_audio_data(self):
        """Generate sample audio data."""
        return {
            'short': base64.b64encode(b"fake_short_audio_content").decode(),
            'medium': base64.b64encode(b"fake_medium_audio_content" * 10).decode(),
            'long': base64.b64encode(b"fake_long_audio_content" * 100).decode()
        }
    
    @pytest.fixture
    def sample_discord_thread(self):
        """Sample Discord thread conversation."""
        messages = [
            {
                "id": 1001,
                "author": "alice",
                "content": "Hey team, let's discuss the new feature proposal",
                "timestamp": "2024-12-15T10:00:00Z",
                "attachments": []
            },
            {
                "id": 1002,
                "author": "bob",
                "content": "Great idea! I think we should focus on user experience first",
                "timestamp": "2024-12-15T10:02:00Z",
                "attachments": []
            },
            {
                "id": 1003,
                "author": "charlie",
                "content": "Agreed. Here's the mockup I created",
                "timestamp": "2024-12-15T10:05:00Z",
                "attachments": ["mockup.png"]
            },
            {
                "id": 1004,
                "author": "alice",
                "content": "Perfect! Let's schedule a review meeting for tomorrow",
                "timestamp": "2024-12-15T10:08:00Z",
                "attachments": []
            }
        ]
        
        return ThreadCapture(
            thread_id=888999000,
            thread_name="Feature Discussion",
            messages=messages,
            participants=["alice", "bob", "charlie"],
            start_time="2024-12-15T10:00:00Z",
            end_time="2024-12-15T10:08:00Z",
            message_count=4
        )
    
    @pytest.fixture
    def sample_urls(self):
        """Sample URLs for testing."""
        return {
            'article': 'https://example.com/test-article',
            'blog_post': 'https://blog.example.com/how-to-test',
            'documentation': 'https://docs.example.com/api-reference',
            'youtube': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
            'github': 'https://github.com/user/repo',
            'pdf': 'https://example.com/document.pdf',
            'invalid': 'not-a-valid-url',
            'unreachable': 'https://unreachable.example.com'
        }


class TestMockFixtures:
    """Mock services and dependencies for testing."""
    
    @pytest.fixture
    def mock_ollama_services(self):
        """Mock Ollama AI services."""
        mocks = {}
        
        def create_title_mock(content=""):
            if "error" in content.lower():
                return "Error Note"
            elif "meeting" in content.lower():
                return "Meeting Notes"
            elif "voice" in content.lower():
                return "Voice Memo"
            elif "discord" in content.lower():
                return "Discord Message"
            else:
                return "Generated Title"
        
        def create_summary_mock(content="", prompt=""):
            return {
                "summary": f"AI summary of: {content[:50]}...",
                "tags": ["ai-generated", "test"],
                "actions": ["Review content", "Follow up"]
            }
        
        mocks['title'] = Mock(side_effect=create_title_mock)
        mocks['summarize'] = Mock(side_effect=create_summary_mock)
        
        return mocks
    
    @pytest.fixture
    def mock_ocr_service(self):
        """Mock OCR service."""
        def mock_ocr(image):
            # Simulate different OCR results based on image properties
            return "Extracted text from test image"
        
        return Mock(side_effect=mock_ocr)
    
    @pytest.fixture
    def mock_pdf_service(self):
        """Mock PDF processing service."""
        def mock_pdf_extract(pdf_data):
            return "Extracted text from test PDF document"
        
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "Test PDF content"
        mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
        mock_doc.close = Mock()
        
        return mock_doc
    
    @pytest.fixture
    def mock_web_service(self):
        """Mock web ingestion service."""
        mock_service = Mock()
        
        def mock_ingest_url(url, user_id=1):
            if "unreachable" in url:
                return {"success": False, "error": "URL not reachable"}
            else:
                return {
                    "success": True,
                    "note_id": 123,
                    "title": f"Web Content: {url.split('/')[-1]}",
                    "content": f"Content from {url}",
                    "tags": ["web", "scraped"]
                }
        
        mock_service.ingest_url = AsyncMock(side_effect=mock_ingest_url)
        return mock_service
    
    @pytest.fixture
    def mock_embeddings_service(self):
        """Mock embeddings service."""
        def mock_embed(text):
            # Generate deterministic embedding based on text
            import hashlib
            hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            # Generate 384-dimensional embedding (typical sentence-transformer size)
            embedding = [(hash_val + i) % 100 / 100.0 for i in range(384)]
            return embedding
        
        mock_embedder = Mock()
        mock_embedder.embed = Mock(side_effect=mock_embed)
        return mock_embedder
    
    @pytest.fixture
    def mock_discord_bot(self):
        """Mock Discord bot."""
        mock_bot = Mock()
        mock_bot.command_prefix = '!sb '
        
        # Mock commands
        mock_bot.hybrid_command = Mock(return_value=lambda func: func)
        mock_bot.event = Mock(return_value=lambda func: func)
        
        return mock_bot


class TestScenarioFixtures:
    """Complex test scenarios combining multiple fixtures."""
    
    @pytest.fixture
    def bulk_capture_scenario(self, sample_text_content, sample_urls):
        """Scenario for bulk capture testing."""
        return {
            'text_notes': [
                {
                    'content': sample_text_content['short'],
                    'title': 'Bulk Note 1',
                    'tags': ['bulk', 'test', 'short']
                },
                {
                    'content': sample_text_content['medium'],
                    'title': 'Bulk Note 2', 
                    'tags': ['bulk', 'test', 'medium']
                }
            ],
            'urls': [
                sample_urls['article'],
                sample_urls['blog_post']
            ],
            'mixed_requests': [
                {
                    'content_type': 'text',
                    'content': 'Mixed batch item 1',
                    'source': 'api'
                },
                {
                    'content_type': 'url',
                    'url': sample_urls['article'],
                    'source': 'web_ui'
                }
            ]
        }
    
    @pytest.fixture
    def error_testing_scenario(self):
        """Scenario for error handling testing."""
        return {
            'invalid_image': 'invalid_base64_data',
            'empty_content': '',
            'oversized_content': 'x' * 100000,  # 100KB of text
            'malformed_json': '{"invalid": json}',
            'sql_injection_attempt': "'; DROP TABLE notes; --",
            'xss_attempt': '<script>alert("xss")</script>',
            'unicode_edge_cases': '🏳️‍🌈🧙‍♂️👨‍👩‍👧‍👦',
            'control_characters': '\x00\x01\x02\x03'
        }
    
    @pytest.fixture
    def performance_testing_scenario(self):
        """Scenario for performance testing."""
        return {
            'concurrent_requests': 10,
            'large_batch_size': 50,
            'stress_test_duration': 30,  # seconds
            'memory_limit_mb': 100,
            'timeout_seconds': 5
        }


class TestUtilities:
    """Utility functions for testing."""
    
    @pytest.fixture
    def test_helpers(self):
        """Helper functions for tests."""
        class TestHelpers:
            
            @staticmethod
            def create_test_note(title="Test Note", content="Test content", tags="test"):
                """Create a standardized test note."""
                return {
                    'title': title,
                    'body': content,
                    'content': content,
                    'tags': tags,
                    'type': 'test_note',
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'user_id': 1
                }
            
            @staticmethod
            def assert_capture_result(result, expected_success=True):
                """Assert standard capture result structure."""
                assert 'success' in result
                assert result['success'] == expected_success
                
                if expected_success:
                    assert 'note_id' in result
                    assert 'title' in result
                    assert result['note_id'] is not None
                else:
                    assert 'error' in result
                    assert result['error'] is not None
            
            @staticmethod
            def create_capture_request(content_type, source_type, **kwargs):
                """Create standardized capture request."""
                from services.unified_capture_service import UnifiedCaptureRequest
                
                return UnifiedCaptureRequest(
                    content_type=content_type,
                    source_type=source_type,
                    primary_content=kwargs.get('content', 'Test content'),
                    metadata=kwargs.get('metadata', {}),
                    **{k: v for k, v in kwargs.items() if k not in ['content', 'metadata']}
                )
            
            @staticmethod
            def measure_performance(func, *args, **kwargs):
                """Measure function execution time and memory usage."""
                import time
                import psutil
                import os
                
                process = psutil.Process(os.getpid())
                start_memory = process.memory_info().rss
                start_time = time.time()
                
                result = func(*args, **kwargs)
                
                end_time = time.time()
                end_memory = process.memory_info().rss
                
                return {
                    'result': result,
                    'execution_time': end_time - start_time,
                    'memory_delta': end_memory - start_memory,
                    'peak_memory': end_memory
                }
        
        return TestHelpers()


# Export all fixtures for easy importing
__all__ = [
    'TestDatabaseFixtures',
    'TestDataFixtures', 
    'TestMockFixtures',
    'TestScenarioFixtures',
    'TestUtilities'
]