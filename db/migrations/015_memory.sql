-- Episodic Memory (conversation history)
CREATE TABLE IF NOT EXISTS episodic_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    episode_id TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    importance REAL DEFAULT 0.5,
    context TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_episodic_user_date ON episodic_memories(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_episodic_importance ON episodic_memories(importance DESC);

-- Virtual table for FTS on episodic memories
CREATE VIRTUAL TABLE IF NOT EXISTS episodic_fts USING fts5(
    episode_id UNINDEXED,
    content,
    summary,
    content=episodic_memories,
    content_rowid=id
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS episodic_after_insert AFTER INSERT ON episodic_memories BEGIN
    INSERT INTO episodic_fts(rowid, episode_id, content, summary)
    VALUES (new.id, new.episode_id, new.content, new.summary);
END;

CREATE TRIGGER IF NOT EXISTS episodic_after_update AFTER UPDATE ON episodic_memories BEGIN
    UPDATE episodic_fts SET content = new.content, summary = new.summary
    WHERE rowid = new.id;
END;

CREATE TRIGGER IF NOT EXISTS episodic_after_delete AFTER DELETE ON episodic_memories BEGIN
    DELETE FROM episodic_fts WHERE rowid = old.id;
END;

-- Semantic Memory (user facts and preferences)
CREATE TABLE IF NOT EXISTS semantic_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    fact_id TEXT UNIQUE NOT NULL,
    fact TEXT NOT NULL,
    category TEXT DEFAULT 'general',  -- 'preference', 'knowledge', 'context', 'general'
    confidence REAL DEFAULT 1.0,
    source TEXT,  -- Where this fact came from
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_semantic_user_category ON semantic_memories(user_id, category);
CREATE INDEX IF NOT EXISTS idx_semantic_confidence ON semantic_memories(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_semantic_updated ON semantic_memories(updated_at DESC);

-- Virtual table for FTS on semantic memories
CREATE VIRTUAL TABLE IF NOT EXISTS semantic_fts USING fts5(
    fact_id UNINDEXED,
    fact,
    category,
    content=semantic_memories,
    content_rowid=id
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS semantic_after_insert AFTER INSERT ON semantic_memories BEGIN
    INSERT INTO semantic_fts(rowid, fact_id, fact, category)
    VALUES (new.id, new.fact_id, new.fact, new.category);
END;

CREATE TRIGGER IF NOT EXISTS semantic_after_update AFTER UPDATE ON semantic_memories BEGIN
    UPDATE semantic_fts SET fact = new.fact, category = new.category
    WHERE rowid = new.id;
END;

CREATE TRIGGER IF NOT EXISTS semantic_after_delete AFTER DELETE ON semantic_memories BEGIN
    DELETE FROM semantic_fts WHERE rowid = old.id;
END;

-- Conversation sessions
CREATE TABLE IF NOT EXISTS conversation_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    session_id TEXT UNIQUE NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    message_count INTEGER DEFAULT 0,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON conversation_sessions(user_id, last_activity DESC);

-- Individual messages in conversations
CREATE TABLE IF NOT EXISTS conversation_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,  -- 'user' or 'assistant'
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES conversation_sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON conversation_messages(session_id, created_at ASC);
