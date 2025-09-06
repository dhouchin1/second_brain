# Backfill chunks in brain.db from notes (idempotent)

PRAGMA trusted_schema=ON;

ATTACH DATABASE 'brain.db' AS brain;

INSERT INTO brain.chunks(file_id, idx, content, section, created_at, updated_at)
SELECT n.id, 0,
       COALESCE(NULLIF(TRIM(n.content), ''), n.body),
       'note',
       n.created_at, n.updated_at
FROM main.notes AS n
WHERE NOT EXISTS (
  SELECT 1 FROM brain.chunks c WHERE c.file_id = n.id AND c.idx = 0
);

-- Rebuild FTS to ensure coverage
INSERT INTO brain.chunk_fts(chunk_fts) VALUES('rebuild');

DETACH DATABASE brain;

