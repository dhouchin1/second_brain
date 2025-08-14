# ──────────────────────────────────────────────────────────────────────────────
# File: db/migrations/001_core.sql
# ──────────────────────────────────────────────────────────────────────────────
-- Core schema for local-first search + jobs (FTS5 required)
PRAGMA foreign_keys=ON;

-- Notes (simplified)
CREATE TABLE IF NOT EXISTS notes (
  id INTEGER PRIMARY KEY,
  title TEXT NOT NULL DEFAULT '',
  body  TEXT NOT NULL DEFAULT '',
  tags  TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_notes_updated_at ON notes(updated_at);
CREATE INDEX IF NOT EXISTS idx_notes_created_at ON notes(created_at);

-- FTS5 virtual table referencing notes
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
  title, body, tags,
  content='notes', content_rowid='id',
  tokenize='unicode61 remove_diacritics 2 stemmer porter'
);

-- Triggers to keep FTS5 in sync with notes
CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
  INSERT INTO notes_fts(rowid, title, body, tags)
  VALUES (new.id, new.title, new.body, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
  INSERT INTO notes_fts(notes_fts, rowid, title, body, tags)
  VALUES('delete', old.id, old.title, old.body, old.tags);
  INSERT INTO notes_fts(rowid, title, body, tags)
  VALUES (new.id, new.title, new.body, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
  INSERT INTO notes_fts(notes_fts, rowid, title, body, tags)
  VALUES('delete', old.id, old.title, old.body, old.tags);
END;

-- Jobs & Rules (embedded automation)
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY,
  type TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending', -- pending|running|done|failed
  not_before TEXT,                         -- schedule time (UTC)
  attempts INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL DEFAULT 3,
  payload TEXT NOT NULL DEFAULT '{}',      -- JSON
  last_error TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  taken_by TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status_time ON jobs(status, not_before);

CREATE TABLE IF NOT EXISTS job_logs (
  id INTEGER PRIMARY KEY,
  job_id INTEGER NOT NULL,
  ts TEXT NOT NULL DEFAULT (datetime('now')),
  level TEXT NOT NULL DEFAULT 'info',
  message TEXT NOT NULL,
  FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
);

-- Optional: simple rules registry for future expansion
CREATE TABLE IF NOT EXISTS rules (
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  enabled INTEGER NOT NULL DEFAULT 1,
  definition TEXT NOT NULL DEFAULT '{}' -- JSON (conditions, actions)
);

-- housekeeping trigger
CREATE TRIGGER IF NOT EXISTS jobs_touch AFTER UPDATE ON jobs BEGIN
  UPDATE jobs SET updated_at = datetime('now') WHERE id = new.id;
END;


# ──────────────────────────────────────────────────────────────────────────────
# File: db/migrations/002_vec.sql
# ──────────────────────────────────────────────────────────────────────────────
-- Requires sqlite-vec to be loaded before running
-- If this migration fails, it's okay; the application will operate in keyword-only mode
-- until the extension is available.
CREATE VIRTUAL TABLE IF NOT EXISTS note_vecs USING vec0(
  embedding float[768],
  note_id INTEGER PRIMARY KEY
);

-- Convenience index for lookups
-- (vec0 manages its own internal index structures for ANN search)


# ──────────────────────────────────────────────────────────────────────────────
# File: services/embeddings.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Local embedding helpers.
- Default provider: Ollama embeddings API (http://localhost:11434)
- Fallback: deterministic pseudo-embedding (for dev without Ollama)
Configure via env:
  EMBEDDINGS_PROVIDER=ollama|none
  EMBEDDINGS_MODEL=nomic-embed-text (or your preferred local model)
"""
from __future__ import annotations
import hashlib
import json
import os
import random
import struct
import urllib.request

DEFAULT_DIM = 768

class Embeddings:
    def __init__(self, provider: str | None = None, model: str | None = None, dim: int = DEFAULT_DIM):
        self.provider = provider or os.getenv('EMBEDDINGS_PROVIDER', 'ollama')
        self.model = model or os.getenv('EMBEDDINGS_MODEL', 'nomic-embed-text')
        self.dim = int(os.getenv('EMBEDDINGS_DIM', str(dim)))

    def embed(self, text: str) -> list[float]:
        if self.provider == 'ollama':
            return self._ollama_embed(text)
        return self._pseudo_embed(text)

    def _ollama_embed(self, text: str) -> list[float]:
        data = json.dumps({"model": self.model, "input": text}).encode('utf-8')
        req = urllib.request.Request(
            os.getenv('OLLAMA_URL', 'http://localhost:11434/api/embeddings'),
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode('utf-8'))
        vec = payload.get('embedding') or payload.get('data', [{}])[0].get('embedding')
        if not vec:
            raise RuntimeError('No embedding returned from Ollama')
        return vec

    def _pseudo_embed(self, text: str) -> list[float]:
        # Stable pseudo-embedding using a hash; useful for offline dev
        h = hashlib.sha256(text.encode('utf-8')).digest()
        rng = random.Random(h)
        return [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]

    @staticmethod
    def pack_f32(array: list[float]) -> bytes:
        return struct.pack('<%sf' % len(array), *array)


# ──────────────────────────────────────────────────────────────────────────────
# File: services/search_adapter.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Search adapter over SQLite FTS5 + (optional) sqlite-vec.
- Loads sqlite-vec extension if SQLITE_VEC_PATH is set.
- Runs migrations from db/migrations/*
- Provides keyword, semantic, and hybrid search.
"""
from __future__ import annotations
import os
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from services.embeddings import Embeddings

MIGRATIONS = [
    Path('db/migrations/001_core.sql'),
    Path('db/migrations/002_vec.sql'),
]

class SearchService:
    def __init__(self, db_path: str = 'notes.db', vec_ext_path: Optional[str] = None):
        self.db_path = db_path
        self.vec_ext_path = vec_ext_path or os.getenv('SQLITE_VEC_PATH')
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._enable_extensions()
        self._run_migrations()
        self.embedder = Embeddings()

    def _enable_extensions(self):
        self.conn.execute('PRAGMA foreign_keys=ON;')
        try:
            self.conn.enable_load_extension(True)
            if self.vec_ext_path:
                self.conn.load_extension(self.vec_ext_path)
        except Exception as e:
            # Extension loading is optional; log and continue
            print(f"[search] sqlite-vec not loaded: {e}")

    def _run_migrations(self):
        cur = self.conn.cursor()
        for path in MIGRATIONS:
            if not path.exists():
                continue
            sql = path.read_text(encoding='utf-8')
            try:
                cur.executescript(sql)
                self.conn.commit()
            except sqlite3.OperationalError as e:
                # vec migration may fail if extension not loaded; ignore
                print(f"[search] migration {path.name} skipped/error: {e}")
                self.conn.rollback()

    # ─── Indexing ────────────────────────────────────────────────────────────
    def upsert_note(self, note_id: Optional[int], title: str, body: str, tags: str = '') -> int:
        cur = self.conn.cursor()
        if note_id is None:
            cur.execute("INSERT INTO notes(title, body, tags) VALUES (?,?,?)", (title, body, tags))
            note_id = cur.lastrowid
        else:
            cur.execute("UPDATE notes SET title=?, body=?, tags=?, updated_at=datetime('now') WHERE id=?",
                        (title, body, tags, note_id))
        self.conn.commit()
        # FTS5 is updated by triggers. Now (optionally) update vectors.
        self._upsert_vector(note_id, f"{title}\n\n{body}")
        return note_id

    def _vec_table_exists(self) -> bool:
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='note_vecs'")
        return cur.fetchone() is not None

    def _upsert_vector(self, note_id: int, text: str):
        if not self._vec_table_exists():
            return
        vec = self.embedder.embed(text)
        # Try JSON text insert first (supported by sqlite-vec), then fall back to BLOB
        cur = self.conn.cursor()
        try:
            cur.execute("INSERT OR REPLACE INTO note_vecs(note_id, embedding) VALUES (?, ?)", (note_id, json.dumps(vec)))
            self.conn.commit()
            return
        except Exception:
            pass
        try:
            from services.embeddings import Embeddings as _E
            blob = _E.pack_f32(vec)
            cur.execute("INSERT OR REPLACE INTO note_vecs(note_id, embedding) VALUES (?, ?)", (note_id, sqlite3.Binary(blob)))
            self.conn.commit()
        except Exception as e:
            print(f"[search] vector upsert failed (note {note_id}): {e}")
            self.conn.rollback()

    # ─── Search ─────────────────────────────────────────────────────────────
    def search(self, q: str, mode: str = 'hybrid', k: int = 20) -> list[sqlite3.Row]:
        if mode not in {'hybrid','keyword','semantic'}:
            mode = 'hybrid'
        if mode == 'keyword' or not self._vec_table_exists():
            return self._keyword(q, k)
        if mode == 'semantic':
            return self._semantic(q, k)
        return self._hybrid(q, k)

    def _keyword(self, q: str, k: int) -> list[sqlite3.Row]:
        cur = self.conn.cursor()
        rows = cur.execute(
            """
            SELECT n.*,
                   bm25(notes_fts) AS kw_rank,
                   snippet(notes_fts, 1, '<b>', '</b>', '…', 12) AS snippet
            FROM notes_fts JOIN notes n ON notes_fts.rowid = n.id
            WHERE notes_fts MATCH ?
            ORDER BY kw_rank
            LIMIT ?
            """, (q, k)).fetchall()
        return rows

    def _semantic(self, q: str, k: int) -> list[sqlite3.Row]:
        if not self._vec_table_exists():
            return []
        qvec = self.embedder.embed(q)
        cur = self.conn.cursor()
        rows = cur.execute(
            """
            WITH vs AS (
              SELECT note_id AS id,
                     1.0 - vec_cosine_distance(embedding, ?) AS vs_rank
              FROM note_vecs
              ORDER BY vs_rank DESC
              LIMIT ?
            )
            SELECT n.*, vs.vs_rank AS score FROM vs JOIN notes n ON n.id = vs.id
            ORDER BY score DESC
            """, (json.dumps(qvec), k)).fetchall()
        return rows

    def _hybrid(self, q: str, k: int) -> list[sqlite3.Row]:
        if not self._vec_table_exists():
            return self._keyword(q, k)
        qvec = self.embedder.embed(q)
        cur = self.conn.cursor()
        rows = cur.execute(
            """
            WITH kw AS (
              SELECT rowid AS id, bm25(notes_fts) AS kw_rank
              FROM notes_fts
              WHERE notes_fts MATCH ?
              ORDER BY kw_rank
              LIMIT 50
            ),
            vs AS (
              SELECT note_id AS id, 1.0 - vec_cosine_distance(embedding, ?) AS vs_rank
              FROM note_vecs
              ORDER BY vs_rank DESC
              LIMIT 50
            ),
            unioned AS (
              SELECT id, (1.0/(1.0+kw_rank)) AS kw_s, 0.0 AS vs_s FROM kw
              UNION ALL
              SELECT id, 0.0, vs_rank FROM vs
            )
            SELECT n.*,
                   COALESCE(SUM(kw_s),0)*0.6 + COALESCE(SUM(vs_s),0)*0.4 AS score
            FROM unioned u JOIN notes n ON n.id = u.id
            GROUP BY n.id
            ORDER BY score DESC
            LIMIT ?
            """, (q, json.dumps(qvec), k)).fetchall()
        return rows


# ──────────────────────────────────────────────────────────────────────────────
# File: services/jobs.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Embedded job runner (SQLite-only) for digests, reindexing, etc.
Usage:
  from services.jobs import JobRunner
  runner = JobRunner(db_path)
  runner.start(app)  # FastAPI lifespan or on_startup event
"""
from __future__ import annotations
import asyncio
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Optional

UTC = timezone.utc

class JobRunner:
    def __init__(self, db_path: str = 'notes.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.handlers: Dict[str, Callable[[sqlite3.Row, sqlite3.Connection], None]] = {
            'digest': self._handle_digest,
            'reindex': self._handle_reindex,
        }
        self._stop = asyncio.Event()

    def start(self, app=None):
        loop = asyncio.get_event_loop()
        loop.create_task(self._worker())

    async def _worker(self):
        while not self._stop.is_set():
            job = self._take_job()
            if not job:
                await asyncio.sleep(1.0)
                continue
            try:
                self._update_job_status(job['id'], 'running')
                handler = self.handlers.get(job['type'])
                if not handler:
                    raise RuntimeError(f"No handler for job type {job['type']}")
                handler(job, self.conn)
                self._update_job_status(job['id'], 'done')
            except Exception as e:
                self._fail_job(job, e)

    def stop(self):
        self._stop.set()

    def enqueue(self, type_: str, payload: Optional[dict] = None, when: Optional[datetime] = None):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO jobs(type, not_before, payload) VALUES (?,?,?)",
            (type_, (when or datetime.now(tz=UTC)).isoformat(), json.dumps(payload or {}))
        )
        self.conn.commit()
        return cur.lastrowid

    def _take_job(self) -> Optional[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE jobs
            SET status='running', taken_by='local', updated_at=datetime('now')
            WHERE id = (
              SELECT id FROM jobs
              WHERE status='pending' AND (not_before IS NULL OR not_before <= datetime('now'))
              ORDER BY created_at
              LIMIT 1
            )
            RETURNING *;
            """
        )
        return cur.fetchone()

    def _update_job_status(self, job_id: int, status: str):
        self.conn.execute("UPDATE jobs SET status=? WHERE id=?", (status, job_id))
        self.conn.commit()

    def _fail_job(self, job: sqlite3.Row, e: Exception):
        attempts = job['attempts'] + 1
        if attempts < job['max_attempts']:
            # backoff 2^attempts minutes
            delay = 2 ** attempts
            self.conn.execute(
                "UPDATE jobs SET status='pending', attempts=?, last_error=?, not_before=datetime('now', ?) WHERE id=?",
                (attempts, str(e), f'+{delay} minutes', job['id'])
            )
        else:
            self.conn.execute(
                "UPDATE jobs SET status='failed', attempts=?, last_error=? WHERE id=?",
                (attempts, str(e), job['id'])
            )
        self.conn.commit()

    # ─── Handlers ────────────────────────────────────────────────────────────
    def _handle_digest(self, job: sqlite3.Row, conn: sqlite3.Connection):
        # Naive digest: gather recent notes and write a new summary note stub
        payload = json.loads(job['payload'] or '{}')
        cur = conn.cursor()
        cur.execute("SELECT id, title FROM notes WHERE created_at >= datetime('now','-1 day') ORDER BY created_at DESC")
        items = cur.fetchall()
        titles = '\n'.join(f"- {r['title']}" for r in items)
        title = payload.get('title') or f"Daily Digest — {datetime.now(tz=UTC).date()}"
        body = f"Auto-generated digest stub (local). Items:\n\n{titles}\n"
        cur.execute("INSERT INTO notes(title, body, tags) VALUES (?,?,?)", (title, body, '#digest'))
        conn.commit()

    def _handle_reindex(self, job: sqlite3.Row, conn: sqlite3.Connection):
        # Rebuild FTS from content
        cur = conn.cursor()
        cur.execute("INSERT INTO notes_fts(notes_fts) VALUES('rebuild')")
        conn.commit()


# ──────────────────────────────────────────────────────────────────────────────
# File: api/routes_search.py
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import os, json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.search_adapter import SearchService

router = APIRouter(prefix="/search", tags=["search"])
service = SearchService(db_path=os.getenv('SQLITE_DB','notes.db'), vec_ext_path=os.getenv('SQLITE_VEC_PATH'))

class IndexNoteIn(BaseModel):
    id: int | None = None
    title: str
    body: str
    tags: str = ''

class SearchIn(BaseModel):
    q: str
    mode: str = 'hybrid'
    k: int = 20

@router.post("/index")
def index_note(payload: IndexNoteIn):
    note_id = service.upsert_note(payload.id, payload.title, payload.body, payload.tags)
    return {"ok": True, "id": note_id}

@router.post("")
def search(payload: SearchIn):
    rows = service.search(payload.q, payload.mode, payload.k)
    return {"results": [dict(r) for r in rows]}

# ──────────────────────────────────────────────────────────────────────────────
# File: api/routes_capture.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Apple Shortcuts-friendly capture endpoints:
- POST /capture          (JSON)   → create a text note
- POST /capture/audio    (multipart/form-data) → save audio, convert to WAV, optional Whisper transcript → note
"""
from __future__ import annotations
import os, io, tempfile, subprocess, time
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from services.search_adapter import SearchService
import shutil

router = APIRouter(prefix="", tags=["capture"])
svc = SearchService(db_path=os.getenv('SQLITE_DB','notes.db'), vec_ext_path=os.getenv('SQLITE_VEC_PATH'))

AUDIO_DIR = Path(os.getenv('AUDIO_DIR', 'audio'))
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

class CaptureIn(BaseModel):
    type: str = 'text'
    text: str
    tags: str | None = ''
    title: str | None = None

@router.post('/capture')
def capture_text(payload: CaptureIn):
    title = payload.title or (payload.text[:80] + ('…' if len(payload.text) > 80 else ''))
    note_id = svc.upsert_note(None, title, payload.text, payload.tags or '')
    return {"ok": True, "id": note_id}


def _run(cmd: list[str], timeout: int = 120) -> tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
    return p.returncode, out, err


def _to_wav(src_path: Path, dst_path: Path) -> None:
    ffmpeg = os.getenv('FFMPEG_BIN', 'ffmpeg')
    code, out, err = _run([ffmpeg, '-y', '-i', str(src_path), '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(dst_path)])
    if code != 0:
        raise RuntimeError(f"ffmpeg failed: {err[:300]}")


def _transcribe(wav_path: Path) -> str | None:
    bin_ = os.getenv('WHISPER_BIN', 'whisper-cli')
    model = os.getenv('WHISPER_MODEL', 'ggml-base.en.bin')
    if not shutil.which(bin_):
        return None
    with tempfile.TemporaryDirectory() as td:
        outbase = Path(td) / 'out'
        code, out, err = _run([bin_, '-m', model, '-f', str(wav_path), '-otxt', '-of', str(outbase)])
        if code != 0:
            return None
        txt = (outbase.with_suffix('.txt'))
        return txt.read_text(encoding='utf-8') if txt.exists() else None

@router.post('/capture/audio')
def capture_audio(file: UploadFile = File(...), title: str | None = Form(None), tags: str | None = Form('')):
    # Save upload
    ts = int(time.time())
    raw_path = AUDIO_DIR / f"rec_{ts}_{file.filename or 'audio'}"
    with raw_path.open('wb') as f:
        f.write(file.file.read())
    # Convert to wav
    wav_path = raw_path.with_suffix('.wav')
    try:
        _to_wav(raw_path, wav_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio convert failed: {e}")
    # Try to transcribe (optional)
    transcript = None
    try:
        transcript = _transcribe(wav_path)
    except Exception:
        transcript = None
    # Build note body
    body_lines = [f"[Audio] {raw_path.name}", f"WAV: {wav_path.name}"]
    if transcript:
        body_lines += ["", "Transcript:", transcript]
    body = "
".join(body_lines)
    note_title = title or (transcript[:80] + '…' if transcript else f"Voice Capture {ts}")
    note_id = svc.upsert_note(None, note_title, body, tags or '')
    return {"ok": True, "id": note_id, "transcribed": bool(transcript)}

# ──────────────────────────────────────────────────────────────────────────────
# File: tests/test_search_smoke.py
# ──────────────────────────────────────────────────────────────────────────────
import os
import sqlite3
import pytest
from services.search_adapter import SearchService

@pytest.fixture()
def svc(tmp_path):
    db = tmp_path / 'test.db'
    # No sqlite-vec in CI: run keyword-only
    os.environ.pop('SQLITE_VEC_PATH', None)
    s = SearchService(db_path=str(db))
    s.upsert_note(None, 'Hello World', 'This is a hello note about FastAPI and search.', '#hello')
    s.upsert_note(None, 'Grocery List', 'eggs\nmilk\nbananas', '#list')
    return s

@pytest.mark.parametrize('mode',["keyword","hybrid"]) 
def test_search_basic(svc, mode):
    res = svc.search('hello', mode=mode, k=5)
    assert any('Hello World' in r['title'] for r in res)


# ──────────────────────────────────────────────────────────────────────────────
# File: scripts/dev_seed.py
# ──────────────────────────────────────────────────────────────────────────────
"""Seed a few notes for manual testing."""
import os
from services.search_adapter import SearchService

if __name__ == '__main__':
    svc = SearchService(db_path=os.getenv('SQLITE_DB','notes.db'), vec_ext_path=os.getenv('SQLITE_VEC_PATH'))
    svc.upsert_note(None, 'Second Brain setup', 'Finish Shortcuts + Bookmarklet today.', '#todo #setup')
    svc.upsert_note(None, 'ArchiveBox idea', 'Queue links nightly and snapshot with WARC.', '#archive #idea')
    svc.upsert_note(None, 'Daily Digest sketch', 'Outline jobs and rules for local runner.', '#digest #idea')
    print('Seeded!')


# ──────────────────────────────────────────────────────────────────────────────
# File: README_quickstart.md
# ──────────────────────────────────────────────────────────────────────────────
# Quickstart — FTS5 + sqlite-vec search & embedded jobs

## 0) Prepare env
- Python 3.11+
- Ensure SQLite has FTS5 (most modern builds do). On macOS: the system sqlite includes FTS5.
- (Optional) Build or download **sqlite-vec** and note the `.dylib/.so` path.
- (Optional) Ollama running locally with an embeddings model (e.g., `nomic-embed-text`).

```bash
export SQLITE_DB=notes.db
# If you have sqlite-vec, set the path to the loadable extension file
export SQLITE_VEC_PATH=/absolute/path/to/sqlite-vec0.dylib   # or .so on Linux
export EMBEDDINGS_PROVIDER=ollama
export EMBEDDINGS_MODEL=nomic-embed-text
```

## 1) Run migrations + seed
```bash
python scripts/dev_seed.py
```

If the vec migration prints a warning, you’re in keyword-only mode; hybrid kicks in once sqlite-vec loads successfully.

## 2) Wire routes into FastAPI
In your `app.py` or `app/main.py`:
```python
from fastapi import FastAPI
from api.routes_search import router as search_router
from api.routes_capture import router as capture_router
from services.jobs import JobRunner

app = FastAPI()
app.include_router(search_router)
app.include_router(capture_router)

# start embedded worker
runner = JobRunner(db_path=os.getenv('SQLITE_DB','notes.db'))
@app.on_event('startup')
def _start_worker():
    runner.start(app)
```

## 3) Run the API
```bash
uvicorn app:app --reload --port 8082
```

## 4) Test it quickly
```bash
# Index a note (JSON)
curl -s localhost:8082/search/index -X POST -H 'content-type: application/json' \
  -d '{"title":"Hello World","body":"FTS5 loves SQLite","tags":"#demo"}' | jq

# Search (hybrid)
curl -s localhost:8082/search -X POST -H 'content-type: application/json' \
  -d '{"q":"sqlite","mode":"hybrid","k":10}' | jq

# Capture TEXT (Apple Shortcut compatible)
curl -s localhost:8082/capture -X POST -H 'content-type: application/json' \
  -d '{"type":"text","text":"Captured from iOS Share Sheet","tags":"#ios #capture"}' | jq

# Capture AUDIO (multipart, e.g., an .m4a recording)
curl -s -X POST 'http://localhost:8082/capture/audio' \
  -F 'file=@/path/to/recording.m4a' \
  -F 'title=Voice memo test' \
  -F 'tags=#ios #audio' | jq
```

## 5) Run tests
```bash
pytest -q tests/test_search_smoke.py
```

## 6) Try a digest
```bash
python -c "from services.jobs import JobRunner; r=JobRunner(os.getenv('SQLITE_DB','notes.db')); r.enqueue('digest',{}); print('queued')"
# The background worker will pick it up; check notes list for a new digest stub.
```

## 7) Apple Shortcuts — iOS Setup (Share Sheet)
**Text/Web Capture Shortcut**
1. Create a new Shortcut → **Add to Share Sheet** → accepts **Text**.
2. Actions:
   - **If** input is not text → use **Get Details of Safari Web Page** → **Get Article**/**Get Name** to form text.
   - **Get Contents of URL**
     - URL: `http://YOUR-LAN-IP:8082/capture`
     - Method: **POST**
     - Request Body: **JSON**
       - `type`: `text`
       - `text`: **Provided Input** (Magic Variable)
       - `tags`: e.g. `#ios #share`
     - Headers: `Content-Type: application/json`
3. Optional: Show result.

**Voice Capture Shortcut**
1. Create a new Shortcut → **Record Audio**.
2. **Get Contents of URL**
   - URL: `http://YOUR-LAN-IP:8082/capture/audio`
   - Method: **POST**
   - Request Body: **Form**
     - Add File field **file** → value = recorded audio (Magic Var from *Record Audio*)
     - Text fields: `title` and `tags` as desired
3. Run once to allow local network permissions.

> On macOS/iOS, find your LAN IP in **Settings → Wi‑Fi → (i)**. Replace `YOUR-LAN-IP` above (e.g., `http://10.0.0.87:8084`).

**Troubleshooting**
- If audio upload errors with ffmpeg: ensure `brew install ffmpeg` on the server Mac. We force 16 kHz mono WAV (pcm_s16le) which Whisper likes.
- If transcription is missing, set env vars: `WHISPER_BIN` (path to `whisper-cli`) and `WHISPER_MODEL` (e.g., `ggml-base.en.bin`). If absent, capture still works without transcript.
- If `sqlite-vec` isn’t installed, searches still work via FTS5; set `SQLITE_VEC_PATH` later to enable hybrid.
