#!/usr/bin/env python3
"""
Backfill vectors into sqlite-vec table `note_vecs` for existing notes.

Requirements:
- sqlite-vec extension loaded (run `python scripts/sqlite_vec_check.py`)
- `note_vecs` table present (apply migrations)
- Embedding provider configured via env (OLLAMA_URL/EMBEDDINGS_PROVIDER)

Usage:
  python scripts/backfill_note_vectors.py --db notes.db --limit 500
  EMBEDDINGS_PROVIDER=none python scripts/backfill_note_vectors.py
"""
import argparse
import json
import os
import sqlite3
from typing import Optional

from services.embeddings import Embeddings

def main():
    ap = argparse.ArgumentParser(description='Backfill note vectors into note_vecs')
    ap.add_argument('--db', default=os.getenv('SQLITE_DB', 'notes.db'), help='Path to SQLite DB')
    ap.add_argument('--limit', type=int, default=0, help='Max notes to process (0 = all)')
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Sanity checks
    row = cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='note_vecs'").fetchone()
    if not row:
        print('ERROR: note_vecs table not found. Ensure sqlite-vec is loaded and run migrations (002_vec).')
        return 2

    # Find notes missing vectors
    sql = (
        "SELECT n.id, n.title, COALESCE(n.body, n.content, '') AS body "
        "FROM notes n LEFT JOIN note_vecs v ON v.note_id = n.id "
        "WHERE v.note_id IS NULL ORDER BY n.id DESC"
    )
    if args.limit and args.limit > 0:
        sql += f" LIMIT {int(args.limit)}"
    notes = cur.execute(sql).fetchall()
    if not notes:
        print('No notes need vectors. Done.')
        return 0

    embedder = Embeddings()
    ok = 0
    for row in notes:
        note_id = row['id']
        text = f"{row['title'] or ''}\n\n{row['body'] or ''}".strip()
        if not text:
            continue
        try:
            vec = embedder.embed(text)
            cur.execute(
                "INSERT OR REPLACE INTO note_vecs(note_id, embedding) VALUES (?, ?)",
                (note_id, json.dumps(vec))
            )
            ok += 1
        except Exception as e:
            print(f"WARN: embedding failed for note {note_id}: {e}")
            conn.rollback()
        else:
            conn.commit()

    print(f"Backfill complete: {ok} vectors stored out of {len(notes)} candidates")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

