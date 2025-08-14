#!/usr/bin/env bash
# scripts/scaffolds/scaffold_031.sh
# SQLite-backed LLM job queue, worker with retries/backoff, and app integration
set -euo pipefail

STAMP="$(date +%Y%m%d-%H%M%S)"
bk(){ [[ -f "$1" ]] && { mkdir -p .bak; cp -p "$1" ".bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }

mkdir -p scripts/workers

# 1) job_queue.py
cat > job_queue.py <<'PY'
import sqlite3, time, datetime
from pathlib import Path
from typing import Optional, Tuple
from config import settings

DB_PATH = settings.db_path

def _conn():
    return sqlite3.connect(str(DB_PATH))

def init_queue():
    conn = _conn(); c = conn.cursor()
    c.execute("""
      CREATE TABLE IF NOT EXISTS llm_jobs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_type TEXT NOT NULL,
        note_id INTEGER,
        status TEXT NOT NULL DEFAULT 'pending', -- pending|in_progress|failed|done
        attempts INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        priority INTEGER NOT NULL DEFAULT 100,
        next_run_at TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT
      )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_llm_jobs_status_next ON llm_jobs(status, next_run_at)")
    conn.commit(); conn.close()

def enqueue_note_processing(note_id: int, priority: int = 100):
    conn = _conn(); c = conn.cursor()
    c.execute("INSERT INTO llm_jobs(job_type, note_id, priority, next_run_at) VALUES (?,?,?,datetime('now'))",
              ("note_process", note_id, priority))
    conn.commit(); conn.close()

def fetch_next() -> Optional[Tuple]:
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    conn = _conn(); c = conn.cursor()
    row = c.execute("""
      SELECT id, job_type, note_id, attempts FROM llm_jobs
      WHERE status='pending' AND (next_run_at IS NULL OR next_run_at <= ?)
      ORDER BY priority ASC, id ASC
      LIMIT 1
    """, (now,)).fetchone()
    if not row:
      conn.close(); return None
    job_id = row[0]
    c.execute("UPDATE llm_jobs SET status='in_progress', updated_at=datetime('now') WHERE id=?", (job_id,))
    conn.commit(); conn.close()
    return row

def mark_done(job_id: int):
    conn = _conn(); c = conn.cursor()
    c.execute("UPDATE llm_jobs SET status='done', updated_at=datetime('now') WHERE id=?", (job_id,))
    conn.commit(); conn.close()

def mark_failed(job_id: int, attempts: int, err: str):
    # exponential backoff: 2^attempts minutes (capped)
    delay_min = min(60, 2 ** max(0, attempts))
    conn = _conn(); c = conn.cursor()
    c.execute("""
      UPDATE llm_jobs
      SET status='pending', attempts=?, last_error=?, next_run_at=datetime('now', ? || ' minutes'), updated_at=datetime('now')
      WHERE id=?
    """, (attempts, err[:1000], str(delay_min), job_id))
    conn.commit(); conn.close()
PY

# 2) worker
cat > scripts/workers/llm_worker.py <<'PY'
#!/usr/bin/env python3
import time, traceback
from job_queue import init_queue, fetch_next, mark_done, mark_failed
from tasks import process_note  # reuse your existing processor

def main():
    print("[llm_worker] starting…")
    init_queue()
    while True:
        job = fetch_next()
        if not job:
            time.sleep(1.5); continue
        job_id, job_type, note_id, attempts = job
        try:
            if job_type == "note_process":
                process_note(note_id)  # existing function that enriches note
            else:
                raise RuntimeError(f"unknown job_type={job_type}")
            mark_done(job_id)
            print(f"[llm_worker] done job {job_id} note={note_id}")
        except Exception as e:
            attempts += 1
            err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            print(f"[llm_worker] fail job {job_id} attempt={attempts}: {e}")
            mark_failed(job_id, attempts, err)

if __name__ == "__main__":
    main()
PY
chmod +x scripts/workers/llm_worker.py

# 3) integrate in app.py: init_queue on startup + swap capture to enqueue
if ! grep -q "scaffold_031 additions (job queue)" app.py 2>/dev/null; then
  bk app.py
  cat >> app.py <<'PY'

# ==== scaffold_031 additions (job queue) ====
from job_queue import init_queue as _init_job_queue, enqueue_note_processing

@app.on_event("startup")
def _sb_init_job_queue():
    try:
        _init_job_queue()
    except Exception as e:
        print("job_queue init error:", e)
PY

  # replace background task call in /capture with enqueue
  # (safe sed; if pattern not found, we just append a note)
  if grep -n "background_tasks\.add_task\(process_note, note_id\)" -n app.py >/dev/null 2>&1; then
    sed -i '' "s/background_tasks\.add_task(process_note, note_id)/enqueue_note_processing(note_id)/" app.py 2>/dev/null || \
    sed -i "s/background_tasks\.add_task(process_note, note_id)/enqueue_note_processing(note_id)/" app.py
    echo "• replaced background_tasks.add_task with enqueue_note_processing"
  else
    echo "• NOTE: could not find background_tasks.add_task(process_note, note_id); capture may already enqueue."
  fi
fi

# 4) Makefile target
if [[ -f Makefile ]]; then bk Makefile; fi
cat >> Makefile <<'MK'

# === Workers ===
worker:
	@. .venv/bin/activate && python scripts/workers/llm_worker.py
MK

echo "Done 031.

Run the worker:
  make worker
Capture will enqueue a job; the worker processes with retries/backoff.
"
