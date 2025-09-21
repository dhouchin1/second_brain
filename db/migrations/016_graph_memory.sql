-- Graph memory foundational tables
-- Provides lightweight metadata, fact storage, and embeddings to support
-- ArchiveBox/Obsidian ingestion without disrupting the existing notes schema.

CREATE TABLE IF NOT EXISTS gm_sources (
  id INTEGER PRIMARY KEY,
  source_key TEXT UNIQUE,
  source_type TEXT NOT NULL DEFAULT 'unknown',
  uri TEXT,
  checksum TEXT,
  metadata TEXT DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_gm_sources_key ON gm_sources(source_key);
CREATE INDEX IF NOT EXISTS idx_gm_sources_checksum ON gm_sources(checksum);

CREATE TABLE IF NOT EXISTS gm_facts (
  id INTEGER PRIMARY KEY,
  subject TEXT NOT NULL,
  predicate TEXT NOT NULL,
  object TEXT NOT NULL,
  object_type TEXT NOT NULL DEFAULT 'string',
  confidence REAL NOT NULL DEFAULT 0.7,
  source_id INTEGER,
  valid_at TEXT NOT NULL DEFAULT (datetime('now')),
  invalid_at TEXT,
  last_seen_at TEXT NOT NULL DEFAULT (datetime('now')),
  extra TEXT DEFAULT '{}',
  FOREIGN KEY(source_id) REFERENCES gm_sources(id) ON DELETE SET NULL,
  UNIQUE(subject, predicate, object, valid_at, source_id)
);

CREATE INDEX IF NOT EXISTS idx_gm_facts_subject ON gm_facts(subject);
CREATE INDEX IF NOT EXISTS idx_gm_facts_predicate ON gm_facts(predicate);
CREATE INDEX IF NOT EXISTS idx_gm_facts_subject_predicate ON gm_facts(subject, predicate);
CREATE INDEX IF NOT EXISTS idx_gm_facts_current ON gm_facts(subject, predicate, object)
  WHERE invalid_at IS NULL;

CREATE TABLE IF NOT EXISTS gm_fact_embeddings (
  id INTEGER PRIMARY KEY,
  fact_id INTEGER NOT NULL,
  model TEXT,
  dimension INTEGER,
  vector BLOB NOT NULL,
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY(fact_id) REFERENCES gm_facts(id) ON DELETE CASCADE,
  UNIQUE(fact_id, model)
);

CREATE TABLE IF NOT EXISTS gm_documents (
  id INTEGER PRIMARY KEY,
  storage_table TEXT NOT NULL,
  storage_rowid INTEGER NOT NULL,
  source_id INTEGER,
  checksum TEXT,
  path TEXT,
  mime TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY(source_id) REFERENCES gm_sources(id) ON DELETE SET NULL,
  UNIQUE(storage_table, storage_rowid)
);

CREATE INDEX IF NOT EXISTS idx_gm_documents_checksum ON gm_documents(checksum);
