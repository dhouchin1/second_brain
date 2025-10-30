-- Vector storage for episodic memories (requires sqlite-vec extension)
CREATE TABLE IF NOT EXISTS episodic_vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT NOT NULL UNIQUE,
    embedding BLOB NOT NULL,
    FOREIGN KEY (episode_id) REFERENCES episodic_memories(episode_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_episodic_vectors_episode ON episodic_vectors(episode_id);

-- Vector storage for semantic memories (requires sqlite-vec extension)
CREATE TABLE IF NOT EXISTS semantic_vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id TEXT NOT NULL UNIQUE,
    embedding BLOB NOT NULL,
    FOREIGN KEY (fact_id) REFERENCES semantic_memories(fact_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_semantic_vectors_fact ON semantic_vectors(fact_id);
