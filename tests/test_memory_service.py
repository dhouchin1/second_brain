import pytest
import sqlite3
import tempfile
import os
from services.memory_service import MemoryService

@pytest.fixture
def temp_db():
    """Create temporary database for testing"""
    fd, path = tempfile.mkstemp()
    conn = sqlite3.connect(path)

    # Create tables
    with open('db/migrations/015_memory.sql', 'r') as f:
        conn.executescript(f.read())

    yield conn

    conn.close()
    os.close(fd)
    os.unlink(path)

def test_add_episodic_memory(temp_db):
    """Test adding episodic memory"""
    memory = MemoryService(temp_db, embeddings_service=None)

    episode_id = memory.add_episodic_memory(
        user_id=1,
        content="User asked about Python",
        summary="Python discussion",
        importance=0.8,
        context="Programming"
    )

    assert episode_id.startswith("ep_")

    # Verify it was stored
    cursor = temp_db.cursor()
    cursor.execute("SELECT COUNT(*) FROM episodic_memories WHERE episode_id = ?", (episode_id,))
    count = cursor.fetchone()[0]
    assert count == 1

def test_search_episodic(temp_db):
    """Test searching episodic memories"""
    memory = MemoryService(temp_db, embeddings_service=None)

    # Add some memories
    memory.add_episodic_memory(1, "Discussion about Python", "Python talk", 0.8)
    memory.add_episodic_memory(1, "Discussion about JavaScript", "JS talk", 0.7)
    memory.add_episodic_memory(1, "Discussion about databases", "DB talk", 0.6)

    # Search
    results = memory.search_episodic(1, "Python", limit=5)

    assert len(results) > 0
    assert "Python" in results[0]['content'] or "Python" in results[0]['summary']

def test_add_semantic_memory(temp_db):
    """Test adding semantic memory"""
    memory = MemoryService(temp_db, embeddings_service=None)

    fact_id = memory.add_semantic_memory(
        user_id=1,
        fact="User prefers Python",
        category="preference",
        confidence=0.9
    )

    assert fact_id.startswith("fact_")

    # Verify it was stored
    cursor = temp_db.cursor()
    cursor.execute("SELECT COUNT(*) FROM semantic_memories WHERE fact_id = ?", (fact_id,))
    count = cursor.fetchone()[0]
    assert count == 1

def test_search_semantic(temp_db):
    """Test searching semantic memories"""
    memory = MemoryService(temp_db, embeddings_service=None)

    # Add some facts
    memory.add_semantic_memory(1, "User prefers Python", "preference", 0.9)
    memory.add_semantic_memory(1, "User works as engineer", "context", 1.0)
    memory.add_semantic_memory(1, "User likes coffee", "preference", 0.8)

    # Search
    results = memory.search_semantic(1, "Python", limit=5)

    assert len(results) > 0
    assert "Python" in results[0]['fact']

def test_conversation_session(temp_db):
    """Test conversation session management"""
    memory = MemoryService(temp_db, embeddings_service=None)

    # Start session
    session_id = memory.start_conversation(1)
    assert session_id.startswith("session_")

    # Add messages
    memory.add_message(session_id, "user", "Hello")
    memory.add_message(session_id, "assistant", "Hi there!")
    memory.add_message(session_id, "user", "How are you?")

    # Get conversation
    messages = memory.get_conversation(session_id)

    assert len(messages) == 3
    assert messages[0]['role'] == 'user'
    assert messages[0]['content'] == 'Hello'
    assert messages[1]['role'] == 'assistant'
    assert messages[2]['role'] == 'user'

def test_get_all_user_facts(temp_db):
    """Test getting all facts for a user"""
    memory = MemoryService(temp_db, embeddings_service=None)

    # Add facts for user 1
    memory.add_semantic_memory(1, "Fact 1", "preference", 0.9)
    memory.add_semantic_memory(1, "Fact 2", "knowledge", 0.8)
    memory.add_semantic_memory(1, "Fact 3", "preference", 0.7)

    # Add facts for user 2
    memory.add_semantic_memory(2, "Other fact", "preference", 0.9)

    # Get all facts for user 1
    facts = memory.get_all_user_facts(1)

    assert len(facts) == 3
    assert all(f['fact'].startswith('Fact') for f in facts)

    # Get facts by category
    prefs = memory.get_all_user_facts(1, category='preference')
    assert len(prefs) == 2

def test_update_semantic_memory(temp_db):
    """Test updating semantic memory"""
    memory = MemoryService(temp_db, embeddings_service=None)

    fact_id = memory.add_semantic_memory(1, "User likes Python", "preference", 0.8)

    # Update fact
    memory.update_semantic_memory(fact_id, fact="User loves Python", confidence=0.95)

    # Verify update
    cursor = temp_db.cursor()
    cursor.execute("SELECT fact, confidence FROM semantic_memories WHERE fact_id = ?", (fact_id,))
    row = cursor.fetchone()
    assert row[0] == "User loves Python"
    assert row[1] == 0.95

def test_delete_semantic_memory(temp_db):
    """Test deleting semantic memory"""
    memory = MemoryService(temp_db, embeddings_service=None)

    fact_id = memory.add_semantic_memory(1, "Temporary fact", "general", 0.5)

    # Verify it exists
    cursor = temp_db.cursor()
    cursor.execute("SELECT COUNT(*) FROM semantic_memories WHERE fact_id = ?", (fact_id,))
    assert cursor.fetchone()[0] == 1

    # Delete it
    memory.delete_semantic_memory(fact_id)

    # Verify it's gone
    cursor.execute("SELECT COUNT(*) FROM semantic_memories WHERE fact_id = ?", (fact_id,))
    assert cursor.fetchone()[0] == 0

def test_recent_episodes(temp_db):
    """Test getting recent episodes"""
    memory = MemoryService(temp_db, embeddings_service=None)

    # Add multiple episodes
    for i in range(5):
        memory.add_episodic_memory(1, f"Content {i}", f"Summary {i}", 0.5 + i*0.1)

    # Get recent episodes
    recent = memory.get_recent_episodes(1, limit=3)

    assert len(recent) == 3
    # Should be in reverse chronological order (most recent first)
    assert "Summary 4" in recent[0]['summary']

def test_end_conversation(temp_db):
    """Test ending a conversation session"""
    memory = MemoryService(temp_db, embeddings_service=None)

    session_id = memory.start_conversation(1)
    memory.add_message(session_id, "user", "Test")

    # End conversation
    memory.end_conversation(session_id)

    # Verify ended_at is set
    cursor = temp_db.cursor()
    cursor.execute("SELECT ended_at FROM conversation_sessions WHERE session_id = ?", (session_id,))
    ended_at = cursor.fetchone()[0]
    assert ended_at is not None
