# ──────────────────────────────────────────────────────────────────────────────
# File: db/migrations/002_vec.sql
# ──────────────────────────────────────────────────────────────────────────────
-- Requires sqlite-vec to be loaded before running
-- If this migration fails, it's okay; the application will operate in keyword-only mode
-- until the extension is available.
-- Updated for all-MiniLM-L6-v2 model (384 dimensions)
CREATE VIRTUAL TABLE IF NOT EXISTS note_vecs USING vec0(
  embedding float[384],
  note_id INTEGER PRIMARY KEY
);

-- Convenience index for lookups
-- (vec0 manages its own internal index structures for ANN search)