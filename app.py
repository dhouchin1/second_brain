from datetime import datetime
from fastapi import FastAPI, Request, Form, Body
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pathlib, sqlite3, os, re, json, asyncio
from collections import Counter
import httpx

app = FastAPI(title="Second Brain Premium")

BASE_DIR = pathlib.Path(__file__).parent.resolve()
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

def get_conn():
    db = BASE_DIR / "notes.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            body TEXT NOT NULL,
            tags TEXT,
            created_at TEXT
        )
    """)
    return conn

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    c = get_conn().cursor()
    total_notes = c.execute("SELECT COUNT(*) AS c FROM notes").fetchone()["c"]
    recent = c.execute(
        "SELECT id, body, tags, created_at FROM notes ORDER BY datetime(created_at) DESC LIMIT 5"
    ).fetchall()
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "stats": {"notes": total_notes}, "recent": recent},
    )

@app.get("/notes", response_class=HTMLResponse)
async def notes_page(request: Request, tag: str = ""):
    c = get_conn().cursor()
    if tag:
        rows = c.execute(
            "SELECT id, body, tags, created_at "
            "FROM notes WHERE (','||IFNULL(tags,'')||',') LIKE ? "
            "ORDER BY datetime(created_at) DESC",
            (f"%,{tag},%",),
        ).fetchall()
    else:
        rows = c.execute(
            "SELECT id, body, tags, created_at FROM notes "
            "ORDER BY datetime(created_at) DESC"
        ).fetchall()
    return templates.TemplateResponse("notes.html", {"request": request, "notes": rows, "active_tag": tag})

@app.get("/notes/{note_id}", response_class=HTMLResponse)
async def note_detail(request: Request, note_id: int):
    c = get_conn().cursor()
    row = c.execute(
        "SELECT id, body, tags, created_at FROM notes WHERE id = ?", (note_id,)
    ).fetchone()
    if not row:
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse("note_detail.html", {"request": request, "note": row})

@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request, q: str = ""):
    c = get_conn().cursor()
    rows = []
    if q:
        rows = c.execute(
            "SELECT id, body, tags, created_at FROM notes "
            "WHERE body LIKE ? OR IFNULL(tags,'') LIKE ? "
            "ORDER BY datetime(created_at) DESC",
            (f"%{q}%", f"%{q}%"),
        ).fetchall()
    return templates.TemplateResponse("search.html", {"request": request, "q": q, "notes": rows})

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse("settings.html", {"request": request})

@app.post("/notes", response_class=HTMLResponse)
async def create_note(request: Request, body: str = Form(...), tags: str = Form("")):
    conn = get_conn()
    c = conn.cursor()
    now = datetime.utcnow().isoformat(timespec="seconds")
    # normalize tags: strip/uniq/lower
    tag_list = [t.strip().lower().replace("#","") for t in tags.split(",") if t.strip()]
    tag_list = list(dict.fromkeys(tag_list))
    tags_norm = ",".join(tag_list) if tag_list else None

    c.execute("INSERT INTO notes (body, tags, created_at) VALUES (?, ?, ?)", (body, tags_norm, now))
    conn.commit()
    note_id = c.lastrowid
    row = c.execute(
        "SELECT id, body, tags, created_at FROM notes WHERE id = ?", (note_id,)
    ).fetchone()

    if request.headers.get("HX-Request"):
        return templates.TemplateResponse("partials/note_item.html", {"request": request, "n": row})
    return RedirectResponse(f"/notes/{note_id}", status_code=303)

# ---------- Automated Tag Suggestions (Ollama) ----------
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

TAG_PROMPT = """You are a tagging assistant. From the user note below, extract 3-7 SHORT tags.
Rules:
- lowercase
- no punctuation except hyphens
- no spaces; use hyphen for multiword (e.g., time-management)
- no leading '#'
- return ONLY a comma-separated list of tags, nothing else.

Note:
"""

STOPWORDS = set("""
a an and the i you he she it we they to of for in on with from this that these those be is are was were am will would can could should as at by not or if into over under about after before during up down out very just
""".split())

def naive_tags(text: str, k: int = 6):
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{3,}", text.lower())
    words = [w for w in words if w not in STOPWORDS]
    counts = Counter(words)
    return [w for w, _ in counts.most_common(k)]

@app.post("/tags/suggest")
async def suggest_tags(payload: dict = Body(...)):
    text = (payload.get("text") or "").strip()
    if len(text) < 8:
        return JSONResponse({"tags": []})

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            # Prefer the /api/generate endpoint for simple prompts
            req = {
                "model": OLLAMA_MODEL,
                "prompt": TAG_PROMPT + text + "\nTags:",
                "stream": False,
            }
            r = await client.post(f"{OLLAMA_BASE}/api/generate", json=req)
            r.raise_for_status()
            data = r.json()
            raw = data.get("response", "")
            tags = [t.strip().lower().replace("#","").replace(" ", "-") for t in raw.split(",") if t.strip()]
            tags = [re.sub(r"[^a-z0-9\-]", "", t) for t in tags]
            tags = [t for t in tags if t]
            if not tags:
                tags = naive_tags(text)
    except Exception:
        tags = naive_tags(text)

    # uniq + cap length
    tags = list(dict.fromkeys(tags))[:7]
    return JSONResponse({"tags": tags})
