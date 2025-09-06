# Initialize master brain.db and migrate compatible data from notes

-- Required for creating triggers that modify FTS5 virtual tables
PRAGMA trusted_schema=ON;

-- Attach/create the new master database
ATTACH DATABASE 'brain.db' AS brain;

-- Core content tables
CREATE TABLE IF NOT EXISTS brain.files (
  id INTEGER PRIMARY KEY,
  path TEXT,
  name TEXT,
  ext TEXT,
  mime_type TEXT,
  size_bytes INTEGER,
  checksum TEXT,
  title TEXT,
  url TEXT,
  source TEXT,           -- e.g., 'notes', 'web', 'audio', 'pdf'
  external_id TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS brain.idx_files_source ON files(source);
CREATE UNIQUE INDEX IF NOT EXISTS brain.idx_files_source_external ON files(source, external_id);

CREATE TABLE IF NOT EXISTS brain.frontmatter (
  id INTEGER PRIMARY KEY,
  file_id INTEGER NOT NULL,
  key TEXT NOT NULL,
  value TEXT,
  FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS brain.idx_frontmatter_file ON frontmatter(file_id);
CREATE INDEX IF NOT EXISTS brain.idx_frontmatter_key ON frontmatter(key);

CREATE TABLE IF NOT EXISTS brain.tags (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS brain.file_tags (
  file_id INTEGER NOT NULL,
  tag_id INTEGER NOT NULL,
  PRIMARY KEY(file_id, tag_id),
  FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE,
  FOREIGN KEY(tag_id) REFERENCES tags(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS brain.idx_file_tags_tag ON file_tags(tag_id);

CREATE TABLE IF NOT EXISTS brain.links (
  id INTEGER PRIMARY KEY,
  from_file_id INTEGER,
  to_file_id INTEGER,
  from_chunk_id INTEGER,
  to_chunk_id INTEGER,
  link_type TEXT,       -- e.g., 'ref', 'cite', 'embed'
  context TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY(from_file_id) REFERENCES files(id) ON DELETE CASCADE,
  FOREIGN KEY(to_file_id) REFERENCES files(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS brain.idx_links_from_to ON links(from_file_id, to_file_id);

CREATE TABLE IF NOT EXISTS brain.chunks (
  id INTEGER PRIMARY KEY,
  file_id INTEGER NOT NULL,
  idx INTEGER NOT NULL DEFAULT 0,  -- chunk order within a file
  content TEXT NOT NULL,
  token_count INTEGER,
  start_offset INTEGER,
  end_offset INTEGER,
  section TEXT,          -- logical section name if available
  heading_level INTEGER,  -- optional heading level
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS brain.idx_chunks_file ON chunks(file_id, idx);

-- FTS5 over chunks with triggers (contentless, synced from chunks)
CREATE VIRTUAL TABLE IF NOT EXISTS brain.chunk_fts USING fts5(
  content, section,
  content='chunks', content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS brain.chunk_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunk_fts(rowid, content, section)
  VALUES (new.id, new.content, new.section);
END;

CREATE TRIGGER IF NOT EXISTS brain.chunk_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunk_fts(chunk_fts, rowid, content, section)
  VALUES('delete', old.id, old.content, old.section);
  INSERT INTO chunk_fts(rowid, content, section)
  VALUES (new.id, new.content, new.section);
END;

CREATE TRIGGER IF NOT EXISTS brain.chunk_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunk_fts(chunk_fts, rowid, content, section)
  VALUES('delete', old.id, old.content, old.section);
END;

-- Embeddings storage (model-agnostic)
CREATE TABLE IF NOT EXISTS brain.embeddings (
  id INTEGER PRIMARY KEY,
  item_type TEXT NOT NULL,           -- 'file' | 'chunk' | 'note'
  item_id INTEGER NOT NULL,
  model TEXT NOT NULL DEFAULT 'all-MiniLM-L6-v2',
  vector BLOB NOT NULL,              -- serialized float32 array
  dim INTEGER NOT NULL DEFAULT 384,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS brain.idx_embeddings_item ON embeddings(item_type, item_id);
CREATE INDEX IF NOT EXISTS brain.idx_embeddings_model ON embeddings(model);
CREATE INDEX IF NOT EXISTS brain.idx_embeddings_updated ON embeddings(updated_at);

-- Mapping table for ANN indices or external vec tables
CREATE TABLE IF NOT EXISTS brain.vec_map (
  id INTEGER PRIMARY KEY,
  embedding_id INTEGER NOT NULL,
  item_type TEXT NOT NULL,
  item_id INTEGER NOT NULL,
  FOREIGN KEY(embedding_id) REFERENCES embeddings(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS brain.idx_vec_map_item ON vec_map(item_type, item_id);

-- Activity log
CREATE TABLE IF NOT EXISTS brain.activity (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL DEFAULT (datetime('now')),
  actor TEXT,                   -- user/system identifier
  type TEXT NOT NULL,           -- event type
  ref_type TEXT,                -- e.g., 'file','chunk','note'
  ref_id INTEGER,
  message TEXT,
  metadata TEXT                 -- JSON
);
CREATE INDEX IF NOT EXISTS brain.idx_activity_ts ON activity(ts);
CREATE INDEX IF NOT EXISTS brain.idx_activity_type ON activity(type);

-- Settings KV store
CREATE TABLE IF NOT EXISTS brain.settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Optional housekeeping triggers
CREATE TRIGGER IF NOT EXISTS brain.embeddings_touch AFTER UPDATE ON embeddings BEGIN
  UPDATE embeddings SET updated_at = datetime('now') WHERE id = new.id;
END;

-- Initial data migration from main.notes -> brain.files/chunks/frontmatter
-- Map existing notes as files with a single chunk per note body/content
INSERT OR IGNORE INTO brain.files(id, title, source, external_id, url, created_at, updated_at)
SELECT id, COALESCE(title, ''), 'notes', CAST(id AS TEXT), COALESCE(external_url, NULL), created_at, updated_at
FROM main.notes;

-- Prefer 'content' if present; otherwise fall back to 'body'
INSERT OR IGNORE INTO brain.chunks(id, file_id, idx, content, section, created_at, updated_at)
SELECT n.id, n.id, 0,
       COALESCE(NULLIF(TRIM(n.content), ''), n.body),
       'note',
       n.created_at, n.updated_at
FROM main.notes AS n;

-- Persist original tags as a frontmatter field for later normalization
INSERT OR IGNORE INTO brain.frontmatter(file_id, key, value)
SELECT id, 'tags', tags FROM main.notes WHERE COALESCE(TRIM(tags), '') <> '';

-- Build FTS for migrated chunks
INSERT INTO brain.chunk_fts(chunk_fts) VALUES('rebuild');

DETACH DATABASE brain;
