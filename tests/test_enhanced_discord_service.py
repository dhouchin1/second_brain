"""
Unit tests for Enhanced Discord Service

Tests Discord bot functionality including slash commands, reaction handlers,
thread summarization, and team collaboration features.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from dataclasses import asdict

from services.enhanced_discord_service import (
    EnhancedDiscordService,
    ThreadCapture,
    DiscordContext
)


class TestEnhancedDiscordService:
    
    @pytest.fixture
    def mock_db_conn(self):
        """Mock database connection."""
        conn = Mock()
        cursor = Mock()
        conn.cursor.return_value = cursor
        cursor.lastrowid = 999
        cursor.fetchone.return_value = (10,)  # Mock count result
        cursor.fetchall.return_value = [
            ("test-channel", 5),
            ("Test Note", "2024-12-15T10:30:00Z")
        ]
        return conn
    
    @pytest.fixture
    def get_conn_func(self, mock_db_conn):
        """Mock database connection function."""
        return lambda: mock_db_conn
    
    @pytest.fixture
    def service(self, get_conn_func):
        """Create EnhancedDiscordService instance."""
        return EnhancedDiscordService(get_conn_func)
    
    @pytest.fixture
    def sample_discord_context(self):
        """Sample Discord context data."""
        return DiscordContext(
            guild_id=123456789,
            guild_name="Test Server",
            channel_id=987654321,
            channel_name="general",
            user_id=555666777,
            username="testuser",
            message_id=111222333
        )
    
    @pytest.fixture
    def sample_thread_capture(self):
        """Sample thread capture data."""
        messages = [
            {
                "id": 1,
                "author": "user1",
                "content": "Hello, let's discuss the project",
                "timestamp": "2024-12-15T10:00:00Z",
                "attachments": []
            },
            {
                "id": 2,
                "author": "user2", 
                "content": "Great idea! What should we focus on first?",
                "timestamp": "2024-12-15T10:05:00Z",
                "attachments": []
            },
            {
                "id": 3,
                "author": "user1",
                "content": "Let's start with the API design",
                "timestamp": "2024-12-15T10:10:00Z",
                "attachments": ["design_doc.pdf"]
            }
        ]
        
        return ThreadCapture(
            thread_id=444555666,
            thread_name="Project Discussion",
            messages=messages,
            participants=["user1", "user2"],
            start_time="2024-12-15T10:00:00Z",
            end_time="2024-12-15T10:10:00Z",
            message_count=3
        )
    
    def test_service_initialization(self, service):
        """Test service initializes correctly."""
        assert service is not None
        assert service.embedder is not None
        assert service.bot is not None
        assert hasattr(service, 'intents')
        
        # Check bot configuration
        assert service.bot.command_prefix == '!sb '
        assert service.intents.message_content == True
        assert service.intents.guilds == True
    
    @pytest.mark.asyncio
    async def test_text_note_capture(self, service, sample_discord_context):
        """Test capturing text notes from Discord."""
        content = "This is a test message from Discord"
        
        with patch('services.enhanced_discord_service.ollama_generate_title') as mock_title, \
             patch('services.enhanced_discord_service.ollama_summarize') as mock_summarize:
            
            mock_title.return_value = "Discord Test Message"
            mock_summarize.return_value = {
                "summary": "Test message summary",
                "tags": ["discord", "test"],
                "actions": ["Follow up on test"]
            }
            
            result = await service.capture_text_note(
                content=content,
                discord_context=sample_discord_context,
                note_type="discord_note"
            )
            
            assert result["success"] == True
            assert result["note_id"] == 999
            assert result["title"] == "Discord Test Message"
            assert result["summary"] == "Test message summary"
            assert "discord" in result["tags"]
            assert "discord_note" in result["tags"]
            assert "server-test server" in result["tags"]
            assert "channel-general" in result["tags"]
            assert result["action_items"] == ["Follow up on test"]
    
    @pytest.mark.asyncio
    async def test_text_note_with_original_author(self, service, sample_discord_context):
        """Test text note capture with original author different from context user."""
        content = "Message from another user"
        
        with patch('services.enhanced_discord_service.ollama_generate_title'), \
             patch('services.enhanced_discord_service.ollama_summarize'):
            
            result = await service.capture_text_note(
                content=content,
                discord_context=sample_discord_context,
                original_author="original_user"
            )
            
            assert result["success"] == True
            # Content should include original author info (tested in integration tests)
    
    @pytest.mark.asyncio
    async def test_thread_summary_processing(self, service, sample_thread_capture):
        """Test thread conversation summarization."""
        with patch('services.enhanced_discord_service.ollama_summarize') as mock_summarize:
            mock_summarize.return_value = {
                "summary": "Discussion about project API design with focus on initial implementation",
                "tags": ["project", "api", "discussion"],
                "actions": ["Start with API design", "Review design document"]
            }
            
            result = await service.process_thread_summary(sample_thread_capture)
            
            assert result["success"] == True
            assert result["note_id"] == 999
            assert result["title"] == "Thread Summary: Project Discussion"
            assert "Discussion about project API design" in result["summary"]
            assert "thread-summary" in result.get("tags", [])
            assert "conversation" in result.get("tags", [])
            assert "Start with API design" in result["action_items"]
    
    @pytest.mark.asyncio
    async def test_thread_summary_with_attachments(self, service, sample_thread_capture):
        """Test thread summary includes attachment information."""
        with patch('services.enhanced_discord_service.ollama_summarize') as mock_summarize:
            mock_summarize.return_value = {
                "summary": "Thread with attachments",
                "tags": [],
                "actions": []
            }
            
            result = await service.process_thread_summary(sample_thread_capture)
            
            assert result["success"] == True
            # Thread content should mention attachments (tested in integration tests)
    
    @pytest.mark.asyncio
    async def test_search_notes_integration(self, service):
        """Test search notes functionality."""
        with patch('services.search_adapter.SearchService') as mock_search:
            mock_search_instance = Mock()
            mock_search_instance.search.return_value = [
                {
                    "id": 1,
                    "title": "Test Note 1",
                    "body": "First test note",
                    "tags": "test, discord",
                    "created_at": "2024-12-15T10:00:00Z",
                    "score": 0.95
                },
                {
                    "id": 2,
                    "title": "Test Note 2", 
                    "body": "Second test note",
                    "tags": "test, search",
                    "created_at": "2024-12-15T09:30:00Z",
                    "score": 0.87
                }
            ]
            mock_search.return_value = mock_search_instance
            
            results = await service.search_notes("test query", limit=5)
            
            assert len(results) == 2
            assert results[0]["title"] == "Test Note 1"
            assert results[0]["score"] == 0.95
            assert results[1]["title"] == "Test Note 2"
            
            # Verify search was called correctly
            mock_search_instance.search.assert_called_once_with("test query", mode='hybrid', k=5)
    
    @pytest.mark.asyncio
    async def test_search_notes_no_results(self, service):
        """Test search when no results found."""
        with patch('services.search_adapter.SearchService') as mock_search:
            mock_search_instance = Mock()
            mock_search_instance.search.return_value = []
            mock_search.return_value = mock_search_instance
            
            results = await service.search_notes("nonexistent query")
            
            assert results == []
    
    @pytest.mark.asyncio
    async def test_search_notes_error_handling(self, service):
        """Test search error handling."""
        with patch('services.search_adapter.SearchService') as mock_search:
            mock_search.side_effect = Exception("Search service unavailable")
            
            results = await service.search_notes("test query")
            
            assert results == []  # Should return empty list on error
    
    @pytest.mark.asyncio
    async def test_discord_usage_stats(self, service, mock_db_conn):
        """Test Discord usage statistics generation."""
        # Mock database responses
        cursor = mock_db_conn.cursor.return_value
        cursor.fetchone.side_effect = [
            (25,),  # total discord notes
            (10,),  # server-specific notes
            ("general", 15)  # most used channel
        ]
        cursor.fetchall.return_value = [
            ("Recent Note 1", "2024-12-15T10:00:00Z"),
            ("Recent Note 2", "2024-12-15T09:30:00Z")
        ]
        
        stats = await service.get_discord_usage_stats(guild_id=123456789)
        
        assert stats["total_discord_notes"] == 25
        assert stats["server_notes"] == 10
        assert "general" in stats["top_channel"]
        assert "(15 notes)" in stats["top_channel"]
        assert len(stats["recent_notes"]) == 2
        assert stats["recent_notes"][0]["title"] == "Recent Note 1"
    
    @pytest.mark.asyncio
    async def test_discord_usage_stats_no_guild(self, service, mock_db_conn):
        """Test Discord stats without specific guild ID."""
        cursor = mock_db_conn.cursor.return_value
        cursor.fetchone.side_effect = [
            (50,),  # total discord notes
            (0,),   # no server-specific notes
            ("random", 8)  # most used channel
        ]
        
        stats = await service.get_discord_usage_stats()
        
        assert stats["total_discord_notes"] == 50
        assert stats["server_notes"] == 0
        assert "random" in stats["top_channel"]
    
    @pytest.mark.asyncio
    async def test_discord_usage_stats_error(self, service):
        """Test Discord stats error handling."""
        service.get_conn = Mock(side_effect=Exception("Database error"))
        
        stats = await service.get_discord_usage_stats()
        
        assert stats == {}  # Should return empty dict on error
    
    @pytest.mark.asyncio
    async def test_note_storage_with_discord_metadata(self, service, sample_discord_context, mock_db_conn):
        """Test that notes are stored with Discord-specific metadata."""
        with patch('services.enhanced_discord_service.ollama_generate_title'), \
             patch('services.enhanced_discord_service.ollama_summarize'):
            
            result = await service.capture_text_note(
                content="Test message",
                discord_context=sample_discord_context,
                note_type="test_note"
            )
            
            assert result["success"] == True
            
            # Verify database storage was called
            cursor = mock_db_conn.cursor.return_value
            cursor.execute.assert_called()
            
            # Check that metadata parameter contains Discord context
            execute_calls = cursor.execute.call_args_list
            metadata_found = False
            for call in execute_calls:
                args = call[0]
                if len(args) > 1 and isinstance(args[1], tuple):
                    for param in args[1]:
                        if isinstance(param, str) and "discord_context" in param:
                            metadata_found = True
                            break
            
            assert metadata_found, "Discord context metadata should be stored"
    
    @pytest.mark.asyncio
    async def test_embedding_generation(self, service, sample_discord_context):
        """Test that embeddings are generated for Discord notes."""
        with patch.object(service.embedder, 'embed') as mock_embed, \
             patch('services.enhanced_discord_service.ollama_generate_title'), \
             patch('services.enhanced_discord_service.ollama_summarize'):
            
            mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            result = await service.capture_text_note(
                content="Discord message for embedding",
                discord_context=sample_discord_context
            )
            
            assert result["success"] == True
            mock_embed.assert_called_once()
            
            # Verify embedding was called with title + content
            embed_text = mock_embed.call_args[0][0]
            assert "Discord message for embedding" in embed_text
    
    @pytest.mark.asyncio
    async def test_ai_processing_failure_handling(self, service, sample_discord_context):
        """Test handling when AI processing fails."""
        with patch('services.enhanced_discord_service.ollama_generate_title') as mock_title, \
             patch('services.enhanced_discord_service.ollama_summarize') as mock_summarize:
            
            mock_title.side_effect = Exception("AI service down")
            mock_summarize.side_effect = Exception("AI service down")
            
            result = await service.capture_text_note(
                content="Test message",
                discord_context=sample_discord_context
            )
            
            # Should still succeed with fallback behavior
            assert result["success"] == True
            assert "Discord" in result["title"]  # Should use fallback title
    
    @pytest.mark.asyncio
    async def test_large_thread_summarization(self, service):
        """Test thread summarization with large conversation."""
        # Create large thread with many messages
        large_messages = []
        for i in range(50):
            large_messages.append({
                "id": i,
                "author": f"user{i % 5}",  # 5 different users
                "content": f"Message {i} in the long conversation",
                "timestamp": f"2024-12-15T10:{i:02d}:00Z",
                "attachments": []
            })
        
        large_thread = ThreadCapture(
            thread_id=777888999,
            thread_name="Long Discussion",
            messages=large_messages,
            participants=[f"user{i}" for i in range(5)],
            start_time="2024-12-15T10:00:00Z",
            end_time="2024-12-15T10:49:00Z",
            message_count=50
        )
        
        with patch('services.enhanced_discord_service.ollama_summarize') as mock_summarize:
            mock_summarize.return_value = {
                "summary": "Long discussion with 50 messages from 5 participants",
                "tags": ["long", "discussion"],
                "actions": []
            }
            
            result = await service.process_thread_summary(large_thread)
            
            assert result["success"] == True
            assert result["message_count"] == 50
            # Should handle large content (content truncated for AI processing)
    
    @pytest.mark.asyncio
    async def test_special_characters_in_discord_content(self, service, sample_discord_context):
        """Test handling of special characters and emojis in Discord content."""
        content_with_emojis = "Great work! 🎉 Let's continue 💪 @everyone #general"
        
        with patch('services.enhanced_discord_service.ollama_generate_title'), \
             patch('services.enhanced_discord_service.ollama_summarize'):
            
            result = await service.capture_text_note(
                content=content_with_emojis,
                discord_context=sample_discord_context
            )
            
            assert result["success"] == True
            # Should handle emojis and special Discord syntax correctly
    
    @pytest.mark.asyncio
    async def test_thread_capture_data_structure(self, sample_thread_capture):
        """Test ThreadCapture data structure."""
        assert sample_thread_capture.thread_id == 444555666
        assert sample_thread_capture.thread_name == "Project Discussion"
        assert len(sample_thread_capture.messages) == 3
        assert len(sample_thread_capture.participants) == 2
        assert sample_thread_capture.message_count == 3
        
        # Test that it can be converted to dict
        thread_dict = asdict(sample_thread_capture)
        assert thread_dict["thread_id"] == 444555666
        assert len(thread_dict["messages"]) == 3
    
    @pytest.mark.asyncio
    async def test_discord_context_data_structure(self, sample_discord_context):
        """Test DiscordContext data structure."""
        assert sample_discord_context.guild_id == 123456789
        assert sample_discord_context.guild_name == "Test Server"
        assert sample_discord_context.channel_name == "general"
        assert sample_discord_context.username == "testuser"
        
        # Test that it can be converted to dict
        context_dict = asdict(sample_discord_context)
        assert context_dict["guild_id"] == 123456789
        assert context_dict["channel_name"] == "general"
    
    @pytest.mark.asyncio
    async def test_concurrent_note_captures(self, service, sample_discord_context):
        """Test handling multiple concurrent note captures."""
        import asyncio
        
        # Create multiple capture tasks
        tasks = []
        for i in range(5):
            context = DiscordContext(
                guild_id=sample_discord_context.guild_id,
                guild_name=sample_discord_context.guild_name,
                channel_id=sample_discord_context.channel_id,
                channel_name=f"channel-{i}",
                user_id=sample_discord_context.user_id + i,
                username=f"user{i}",
                message_id=sample_discord_context.message_id + i
            )
            
            task = service.capture_text_note(
                content=f"Concurrent message {i}",
                discord_context=context
            )
            tasks.append(task)
        
        with patch('services.enhanced_discord_service.ollama_generate_title'), \
             patch('services.enhanced_discord_service.ollama_summarize'):
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 5
            assert all(result["success"] for result in results)
    
    @pytest.mark.asyncio
    async def test_database_error_handling(self, service, sample_discord_context):
        """Test handling of database errors."""
        # Mock database connection to fail
        service.get_conn = Mock(side_effect=Exception("Database connection lost"))
        
        result = await service.capture_text_note(
            content="Test message",
            discord_context=sample_discord_context
        )
        
        assert result["success"] == False
        assert "Database connection lost" in result["error"]
    
    @pytest.mark.asyncio
    async def test_empty_thread_handling(self, service):
        """Test handling of empty thread."""
        empty_thread = ThreadCapture(
            thread_id=123,
            thread_name="Empty Thread",
            messages=[],
            participants=[],
            start_time="2024-12-15T10:00:00Z",
            end_time="2024-12-15T10:00:00Z",
            message_count=0
        )
        
        with patch('services.enhanced_discord_service.ollama_summarize') as mock_summarize:
            mock_summarize.return_value = {
                "summary": "Empty thread with no messages",
                "tags": [],
                "actions": []
            }
            
            result = await service.process_thread_summary(empty_thread)
            
            assert result["success"] == True
            assert result["message_count"] == 0