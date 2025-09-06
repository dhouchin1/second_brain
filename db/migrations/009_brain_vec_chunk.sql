# Optional vector index for chunks using sqlite-vec (will be skipped if extension unavailable)

ATTACH DATABASE 'brain.db' AS brain;

-- This requires the vec0 virtual table provider. Migration runner will skip on failure.
CREATE VIRTUAL TABLE IF NOT EXISTS brain.vec_chunk USING vec0(
  embedding float[384],
  chunk_id INTEGER PRIMARY KEY
);

DETACH DATABASE brain;

