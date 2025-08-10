import pathlib
import sqlite3
from fastapi.responses import HTMLResponse
from datetime import datetime
from fastapi import FastAPI, Request, Form, UploadFile, File, Body, Query, HTTPException
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from collections import defaultdict
import subprocess
import os
from llm_utils import ollama_summarize, ollama_generate_title
from markupsafe import Markup, escape
import re
from typing import Optional

import hashlib

# ---- Config ----
BASE_DIR = pathlib.Path(__file__).parent.resolve()
DB_PATH = BASE_DIR / "notes.db"
AUDIO_DIR = BASE_DIR / "audio"
WHISPER_CPP_PATH = BASE_DIR / "whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL_PATH = BASE_DIR / "whisper.cpp/models/ggml-base.en.bin"

# ---- FastAPI Setup ----
app = FastAPI()
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

def highlight(text, term):
    if not text or not term:
        return text
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    return Markup(pattern.sub(lambda m: f"<mark>{escape(m.group(0))}</mark>", text))
templates.env.filters['highlight'] = highlight

def get_conn():
    return sqlite3.connect(str(DB_PATH))

def init_db():
    conn = get_conn()
    c = conn.cursor()
    # Base table
    c.execute('''
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            summary TEXT,
            tags TEXT,
            type TEXT,
            timestamp TEXT,
            audio_filename TEXT,
            content TEXT,
            pinned INTEGER DEFAULT 0,
            note_color TEXT,
            audio_hash TEXT,
            deleted_at TEXT
        )
    ''')
    # FTS5 index referencing notes
    c.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
            title, summary, tags, content, content='notes', content_rowid='id'
        )
    ''')
    # Helpful indexes / constraints
    c.execute("CREATE INDEX IF NOT EXISTS idx_notes_timestamp ON notes(timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_notes_type ON notes(type)")
    c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_notes_audio_hash ON notes(audio_hash) WHERE audio_hash IS NOT NULL")
    conn.commit()
    conn.close()
init_db()  # Ensure tables are ready

def run_cmd_with_retries(cmd: list[str], retries: int = 2):
    """
    Run a subprocess command with simple retries. Returns (returncode, stdout, stderr).
    """
    last_rc, out, err = 1, "", ""
    for attempt in range(retries + 1):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        last_rc, out, err = proc.returncode, proc.stdout, proc.stderr
        if last_rc == 0:
            break
    return last_rc, out, err

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def transcribe_audio(audio_path):
    import time
    wav_path = audio_path.with_suffix('.converted.wav')
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", str(audio_path),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav_path)
    ]
    rc, _, err = run_cmd_with_retries(ffmpeg_cmd, retries=2)
    if rc != 0:
        print("ffmpeg failed to convert audio:", err)
        return "", None
    try:
        size = os.path.getsize(wav_path)
        print(f"Converted audio: {wav_path} (size: {size} bytes)")
    except OSError:
        print(f"Converted audio: {wav_path}")
    out_txt_path = wav_path.with_suffix(wav_path.suffix + '.txt')
    whisper_cmd = [
        str(WHISPER_CPP_PATH),
        "-m", str(WHISPER_MODEL_PATH),
        "-f", str(wav_path),
        "-otxt"
    ]
    rc, _, err = run_cmd_with_retries(whisper_cmd, retries=2)
    print(f"Looking for transcript at: {out_txt_path}")
    for _ in range(20):
        if out_txt_path.exists() and out_txt_path.stat().st_size > 0:
            break
        time.sleep(0.1)
    if out_txt_path.exists() and out_txt_path.stat().st_size > 0:
        content = out_txt_path.read_text().strip()
        print(f"Transcript content: '{content}'")
        return content, wav_path.name
    else:
        print("Whisper.cpp failed or output file missing/empty:", err)
        return "", wav_path.name

    
def find_related_notes(note_id, tags, conn):
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    if not tag_list:
        return []
    q = " OR ".join(["tags LIKE ?"] * len(tag_list))
    params = [f"%{tag}%" for tag in tag_list]
    sql = f"SELECT id, title FROM notes WHERE id != ? AND ({q}) LIMIT 3"
    params = [note_id] + params
    rows = conn.execute(sql, params).fetchall()
    return [{"id": row[0], "title": row[1]} for row in rows]

@app.get("/")
def dashboard(request: Request, q: str = "", tag: str = ""):
    conn = get_conn()
    c = conn.cursor()
    if q:
        rows = c.execute("""
            SELECT n.*
            FROM notes_fts fts
            JOIN notes n ON n.id = fts.rowid
            WHERE notes_fts MATCH ?
            ORDER BY n.pinned DESC, n.timestamp DESC LIMIT 100
        """, (q,)).fetchall()
    elif tag:
        rows = c.execute("SELECT * FROM notes WHERE tags LIKE ? ORDER BY timestamp DESC", (f"%{tag}%",)).fetchall()
    else:
        rows = c.execute("SELECT * FROM notes ORDER BY pinned DESC, timestamp DESC LIMIT 100").fetchall()
    notes = [dict(zip([col[0] for col in c.description], row)) for row in rows]
    notes_by_day = defaultdict(list)
    for note in notes:
        day = note["timestamp"][:10] if note.get("timestamp") else "Unknown"
        notes_by_day[day].append(note)
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "notes_by_day": dict(notes_by_day),
            "q": q,
            "tag": tag,
        },
    )

@app.get("/detail/{note_id}")
def detail(request: Request, note_id: int):
    conn = get_conn()
    c = conn.cursor()
    row = c.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
    if not row:
        return RedirectResponse("/", status_code=302)
    note = dict(zip([col[0] for col in c.description], row))
    related = find_related_notes(note_id, note.get("tags", ""), conn)
    return templates.TemplateResponse(
        "detail.html",
        {"request": request, "note": note, "related": related}
    )

@app.get("/audio/{filename}")
def get_audio(filename: str):
    # Prevent path traversal by resolving inside AUDIO_DIR
    target = (AUDIO_DIR / filename).resolve()
    base = AUDIO_DIR.resolve()
    if not str(target).startswith(str(base) + os.sep) and target != base:
        raise HTTPException(status_code=400, detail="Invalid path")
    if target.exists():
        return FileResponse(str(target))
    return {"error": "Audio not found"}

@app.post("/capture")
async def capture(
    request: Request,
    note: str = Form(""),
    tags: str = Form(""),
    file: UploadFile = File(None)
):
    content = note.strip()
    note_type = "note"
    summary = ""
    title = ""
    audio_filename = None

    if file:
        AUDIO_DIR.mkdir(exist_ok=True)
        raw_bytes = await file.read()
        audio_sha = sha256_bytes(raw_bytes)
        # If we've already processed this exact audio, redirect to the existing note
        conn_check = get_conn()
        cur_check = conn_check.cursor()
        existing = cur_check.execute("SELECT id FROM notes WHERE audio_hash=?", (audio_sha,)).fetchone()
        conn_check.close()
        if existing:
            return RedirectResponse(f"/detail/{existing[0]}", status_code=303)

        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
        safe_name = f"{timestamp}-{file.filename.replace(' ', '_')}"
        audio_path = AUDIO_DIR / safe_name
        with open(audio_path, "wb") as out_f:
            out_f.write(raw_bytes)

        transcript, converted_name = transcribe_audio(audio_path)
        audio_filename = converted_name
        if transcript:
            content = transcript
            print(f"[capture] Transcript to summarize: {repr(content[:200])}")
            title = ollama_generate_title(transcript)
            if not title or title.lower().startswith("untitled"):
                title = transcript.splitlines()[0][:60] if transcript.strip() else "[Audio Note]"
            summary = ollama_summarize(transcript)
        else:
            title = f"[Transcription failed for {safe_name}]"
            content = ""
            summary = title
        note_type = "audio"
    else:
        print(f"[capture] Note content to summarize: {repr(content[:200])}")
        # AI TITLE!
        title = ollama_generate_title(content) if content else "[No Title]"
        if not title or title.lower().startswith("untitled"):
            title = content[:60] if content else "[No Title]"
        summary = ollama_summarize(content)
        audio_filename = None

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"About to insert note. Title: '{title}', summary: '{summary}', audio_filename: '{audio_filename}'")
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO notes (title, content, summary, tags, type, timestamp, audio_filename) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (title, content, summary, tags, note_type, now, audio_filename)
    )
    conn.commit()
    note_id = c.lastrowid
    # Persist FTS row
    c.execute(
        "INSERT INTO notes_fts(rowid, title, summary, tags, content) VALUES (?, ?, ?, ?, ?)",
        (note_id, title, summary, tags, content)
    )
    # If this was an audio upload, persist the hash now
    if audio_filename and 'audio' in note_type:
        c.execute("UPDATE notes SET audio_hash=? WHERE id=?", (audio_sha, note_id))

    conn.commit()
    conn.close()
    return RedirectResponse("/", status_code=302)

@app.post("/webhook/apple")
async def webhook_apple(data: dict = Body(...)):
    print("APPLE WEBHOOK RECEIVED:", data)
    note = data.get("note", "")
    tags = data.get("tags", "")
    note_type = data.get("type", "apple")
    summary = ollama_summarize(note)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO notes (title, content, summary, tags, type, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
        (note[:60] + "..." if len(note) > 60 else note, note, summary, tags, note_type, now)
    )
    conn.commit()
    note_id = c.lastrowid
    c.execute(
        "INSERT INTO notes_fts(rowid, title, summary, tags, content) VALUES (?, ?, ?, ?, ?)",
        (note_id, note[:60] + "..." if len(note) > 60 else note, summary, tags, note)
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.put("/api/notes/{note_id}/title")
async def api_update_title(note_id: int, payload: dict = Body(...)):
    title = (payload.get("title") or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title required")
    conn = get_conn()
    cur = conn.cursor()
    # Update base table
    cur.execute("UPDATE notes SET title=? WHERE id=?", (title, note_id))
    if conn.total_changes == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Note not found")
    # Update FTS mirror
    cur.execute("UPDATE notes_fts SET title=? WHERE rowid=?", (title, note_id))
    conn.commit(); conn.close()
    return {"ok": True}

@app.get("/api/notes/recent")
def api_notes_recent(limit: int = 50):
    conn = get_conn(); conn.row_factory = sqlite3.Row; c = conn.cursor()
    rows = c.execute("SELECT id, title, summary, tags, timestamp FROM notes ORDER BY timestamp DESC LIMIT ?", (max(1,min(500,limit)),)).fetchall()
    items = [dict(r) for r in rows]
    conn.close()
    return {"items": items}


@app.put("/api/notes/{note_id}/color")
async def api_note_color(note_id: int, payload: dict = Body(...)):
    color = (payload.get("color") or "").strip()[:20]
    conn = get_conn(); c = conn.cursor()
    c.execute("UPDATE notes SET note_color=? WHERE id=?", (color or None, note_id))
    if conn.total_changes == 0:
        conn.close(); raise HTTPException(status_code=404, detail="Note not found")
    conn.commit(); conn.close()
    return {"ok": True, "color": color}




@app.get("/export/note/{note_id}/print", response_class=HTMLResponse)
def export_note_print(request: Request, note_id: int):
    conn = get_conn(); conn.row_factory = sqlite3.Row; c = conn.cursor()
    r = c.execute("SELECT * FROM notes WHERE id=?", (note_id,)).fetchone()
    if not r:
        conn.close(); raise HTTPException(status_code=404, detail="Note not found")
    conn.close()
    return templates.TemplateResponse("print_note.html", {"request": request, "note": dict(r)})


@app.get("/agenda", response_class=HTMLResponse)
def agenda(request: Request):
    """
    Scans all notes for due markers:
      - @due(YYYY-MM-DD) or @due(YYYY-MM-DD HH:MM)
      - [due:YYYY-MM-DD] or [due:YYYY-MM-DD HH:MM]
    Shows upcoming and overdue items.
    """
    conn = get_conn(); conn.row_factory = sqlite3.Row; c = conn.cursor()
    rows = c.execute("SELECT id, title, content, timestamp FROM notes WHERE (deleted_at IS NULL OR deleted_at='') ORDER BY timestamp DESC").fetchall() if "deleted_at" in "".join([r[1] for r in c.execute("PRAGMA table_info(notes)").fetchall()]) else c.execute("SELECT id, title, content, timestamp FROM notes ORDER BY timestamp DESC").fetchall()
    items = []
    now = datetime.now()
    RX1 = re.compile(r"@due\((\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}))?\)")
    RX2 = re.compile(r"\[due:(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}))?\]", re.I)
    for r in rows:
        text = (r["content"] or "")
        for m in RX1.finditer(text):
            d, t = m.group(1), m.group(2) or "00:00"
            dt = datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M")
            items.append({"note_id": r["id"], "title": r["title"] or f"Note {r['id']}", "due": dt, "source": m.group(0)})
        for m in RX2.finditer(text):
            d, t = m.group(1), m.group(2) or "00:00"
            dt = datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M")
            items.append({"note_id": r["id"], "title": r["title"] or f"Note {r['id']}", "due": dt, "source": m.group(0)})
    items.sort(key=lambda x: x["due"])
    upcoming = [i for i in items if i["due"] >= now]
    overdue  = [i for i in items if i["due"] <  now]
    conn.close()
    return templates.TemplateResponse("agenda.html", {"request": request, "upcoming": upcoming, "overdue": overdue, "now": now})


@app.post("/pin/{note_id}")
def toggle_pin(note_id: int, request: Request):
    conn = get_conn(); c = conn.cursor()
    row = c.execute("SELECT pinned FROM notes WHERE id=?", (note_id,)).fetchone()
    if not row:
        conn.close(); raise HTTPException(status_code=404, detail="Note not found")
    newv = 0 if (row[0] or 0) else 1
    c.execute("UPDATE notes SET pinned=? WHERE id=?", (newv, note_id))
    conn.commit(); conn.close()
    # Redirect back to referrer if possible, else dashboard
    ref = request.headers.get("referer") or "/"
    return RedirectResponse(ref, status_code=303)

# ---- Activity Timeline ----
@app.get("/activity")
def activity_timeline(
    request: Request,
    activity_type: str = Query("all", alias="type"),
    start: str = "",
    end: str = "",
):
    conn = get_conn()
    c = conn.cursor()

    base_query = "SELECT id, summary, type, timestamp FROM notes"
    conditions = []
    params: list[str] = []
    if activity_type and activity_type != "all":
        conditions.append("type = ?")
        params.append(activity_type)
    if start:
        conditions.append("date(timestamp) >= date(?)")
        params.append(start)
    if end:
        conditions.append("date(timestamp) <= date(?)")
        params.append(end)
    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)
    base_query += " ORDER BY timestamp DESC LIMIT 100"

    rows = c.execute(base_query, params).fetchall()
    activities = [
        dict(zip([col[0] for col in c.description], row)) for row in rows
    ]
    conn.close()
    return templates.TemplateResponse(
        "activity_timeline.html",
        {
            "request": request,
            "activities": activities,
            "activity_type": activity_type,
            "start": start,
            "end": end,
        },
    )

# ---- Health Check ----
@app.get("/health")
def health():
    conn = get_conn()
    c = conn.cursor()
    tables = c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    conn.close()
    return {"tables": [t[0] for t in tables]}


# --- API: Update tags, content, and search ---

@app.put("/api/notes/{note_id}/tags")
async def api_update_tags(note_id: int, payload: dict = Body(...)):
    tags = (payload.get("tags") or "").strip()
    conn = get_conn(); cur = conn.cursor()
    cur.execute("UPDATE notes SET tags=? WHERE id=?", (tags, note_id))
    if conn.total_changes == 0:
        conn.close(); raise HTTPException(status_code=404, detail="Note not found")
    cur.execute("UPDATE notes_fts SET tags=? WHERE rowid=?", (tags, note_id))
    conn.commit(); conn.close()
    return {"ok": True, "tags": tags}

@app.put("/api/notes/{note_id}/content")
async def api_update_content(note_id: int, payload: dict = Body(...)):
    content = (payload.get("content") or "").strip()
    summary = (payload.get("summary") or "").strip()
    title = (payload.get("title") or None)
    conn = get_conn(); cur = conn.cursor()
    if title:
        cur.execute("UPDATE notes SET content=?, summary=?, title=? WHERE id=?", (content, summary, title, note_id))
    else:
        cur.execute("UPDATE notes SET content=?, summary=? WHERE id=?", (content, summary, note_id))
    if conn.total_changes == 0:
        conn.close(); raise HTTPException(status_code=404, detail="Note not found")
    # Update FTS mirror
    if title:
        cur.execute("UPDATE notes_fts SET content=?, summary=?, title=? WHERE rowid=?", (content, summary, title, note_id))
    else:
        cur.execute("UPDATE notes_fts SET content=?, summary=? WHERE rowid=?", (content, summary, note_id))
    conn.commit(); conn.close()
    return {"ok": True}

@app.get("/api/search")
def api_search(q: str = Query(..., min_length=1), limit: int = 50):
    limit = max(1, min(200, limit))
    conn = get_conn(); conn.row_factory = sqlite3.Row; c = conn.cursor()
    rows = c.execute("""
        SELECT n.id, n.title, n.summary, n.tags, n.type, n.timestamp
        FROM notes_fts fts
        JOIN notes n ON n.id = fts.rowid
        WHERE notes_fts MATCH ?
        ORDER BY n.pinned DESC, n.timestamp DESC
        LIMIT ?
    """, (q, limit)).fetchall()
    items = [dict(r) for r in rows]
    conn.close()
    return {"items": items}