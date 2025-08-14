#!/usr/bin/env bash
# scripts/scaffold_011.sh
set -euo pipefail
STAMP="$(date +%Y%m%d-%H%M%S)"
b(){ [[ -f "$1" ]] && { mkdir -p .bak; cp -p "$1" ".bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }

mkdir -p templates/partials static scripts

# =========================== app.py (consolidated) ===========================
b app.py
cat > app.py <<'PY'
from datetime import datetime
from fastapi import FastAPI, Request, Form, Body
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pathlib, sqlite3, os, re, httpx, json, math, io, zipfile, typing
from collections import Counter
try:
    import numpy as np
except Exception:
    np = None  # optional; fallback code paths work without it

app = FastAPI(title="Second Brain Premium")
BASE_DIR = pathlib.Path(__file__).parent.resolve()
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

PAGE_SIZE = 20

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS notes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  body TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tags (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS note_tags (
  note_id INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
  tag_id  INTEGER NOT NULL REFERENCES tags(id)  ON DELETE CASCADE,
  PRIMARY KEY (note_id, tag_id)
);

CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT
);

-- Embeddings (semantic related & search)
CREATE TABLE IF NOT EXISTS note_embeddings (
  note_id INTEGER PRIMARY KEY REFERENCES notes(id) ON DELETE CASCADE,
  dim INTEGER NOT NULL,
  vec BLOB NOT NULL
);

-- Full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts
USING fts5(body, content='notes', content_rowid='id');

CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
  INSERT INTO notes_fts(rowid, body) VALUES (new.id, new.body);
END;

CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
  INSERT INTO notes_fts(notes_fts, rowid, body) VALUES('delete', old.id, old.body);
  INSERT INTO notes_fts(rowid, body) VALUES (new.id, new.body);
END;

CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
  INSERT INTO notes_fts(notes_fts, rowid, body) VALUES('delete', old.id, old.body);
END;
"""

def get_conn():
    db = BASE_DIR / "notes.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn

# ---------------- helpers ----------------
def norm_tag(name: str) -> str:
    return re.sub(r"[^a-z0-9\-]", "", (name or "").strip().lower().replace("#","").replace(" ", "-"))

def parse_tags(csv: str):
    return [t for t in {norm_tag(x) for x in (csv or "").split(",")} if t]

def ensure_tag(conn, name: str) -> int:
    cur = conn.execute("INSERT OR IGNORE INTO tags(name) VALUES (?)", (name,))
    if cur.lastrowid:
        return cur.lastrowid
    return conn.execute("SELECT id FROM tags WHERE name=?", (name,)).fetchone()["id"]

def set_note_tags(conn, note_id: int, names):
    conn.execute("DELETE FROM note_tags WHERE note_id=?", (note_id,))
    for nm in names:
        tid = ensure_tag(conn, nm)
        conn.execute("INSERT OR IGNORE INTO note_tags(note_id, tag_id) VALUES (?,?)", (note_id, tid))

def map_note_tags(conn, rows):
    ids = [r["id"] for r in rows] if rows else []
    if not ids: return {}
    q = f"""
    SELECT n.id, GROUP_CONCAT(t.name) AS tags
    FROM notes n
    LEFT JOIN note_tags nt ON nt.note_id=n.id
    LEFT JOIN tags t ON t.id=nt.tag_id
    WHERE n.id IN ({",".join("?"*len(ids))})
    GROUP BY n.id
    """
    return {r["id"]: (r["tags"] or "") for r in conn.execute(q, ids).fetchall()}

def hx_trigger_dict(event: str, payload: dict) -> dict:
    return {event: payload}

# -------- settings helpers --------
def get_setting(conn, key: str, default: str | None = None):
    r = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return (r["value"] if r else None) or default

def put_setting(conn, key: str, value: str):
    conn.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
    conn.commit()

# -------- embeddings --------
EMBED_BASE_ENV  = os.getenv("OLLAMA_EMBED_BASE_URL")
EMBED_MODEL_ENV = os.getenv("OLLAMA_EMBED_MODEL")

def _vec_to_blob(vec):
    if np is None:
        return (",".join(f"{float(x):.7f}" for x in vec)).encode("utf-8")
    arr = np.asarray(vec, dtype=np.float32)
    return arr.tobytes()

def _blob_to_vec(blob, dim_hint=None):
    if np is None:
        s = blob.decode("utf-8")
        return [float(x) for x in s.split(",") if x]
    arr = np.frombuffer(blob, dtype=np.float32)
    return arr.astype(np.float32)

def embed_text(text: str, base: str, model: str) -> typing.List[float]:
    payload = {"model": model, "prompt": text}
    r = httpx.post(f"{base}/api/embeddings", json=payload, timeout=30.0)
    r.raise_for_status()
    data = r.json()
    vec = data.get("embedding") or (data.get("data",[{}])[0].get("embedding"))
    if not vec:
        raise RuntimeError("No embedding returned")
    return vec

def ensure_embedding_for_note(conn, note_id: int, body: str):
    base = EMBED_BASE_ENV or get_setting(conn, "OLLAMA_EMBED_BASE_URL", "http://localhost:11434")
    model = EMBED_MODEL_ENV or get_setting(conn, "OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
    text = body.strip()
    if not text: return
    try:
        vec = embed_text(text, base, model)
        conn.execute(
            "INSERT INTO note_embeddings(note_id, dim, vec) VALUES (?,?,?) "
            "ON CONFLICT(note_id) DO UPDATE SET dim=excluded.dim, vec=excluded.vec",
            (note_id, len(vec), _vec_to_blob(vec))
        )
        conn.commit()
    except Exception as e:
        print("Embedding error:", e)

def cosine(a, b):
    if np is not None:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0: return 0.0
        return float(np.dot(a, b) / denom)
    dot = sum(x*y for x,y in zip(a,b))
    na  = math.sqrt(sum(x*x for x in a))
    nb  = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0: return 0.0
    return dot / (na * nb)

def related_notes_semantic(conn, note_id: int, limit: int = 6):
    row = conn.execute("SELECT e.dim, e.vec FROM note_embeddings e WHERE e.note_id=?", (note_id,)).fetchone()
    if not row: return []
    dim = row["dim"]; qvec = _blob_to_vec(row["vec"], dim_hint=dim)
    others = conn.execute("SELECT note_id, dim, vec FROM note_embeddings WHERE note_id != ?", (note_id,)).fetchall()
    if not others: return []
    scored = []
    for r in others:
        if r["dim"] != dim: continue
        vec = _blob_to_vec(r["vec"], dim_hint=dim)
        s = cosine(qvec, vec)
        scored.append((r["note_id"], s))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_ids = [i for (i, _) in scored[:limit]]
    if not top_ids: return []
    q = f"SELECT id, body, created_at FROM notes WHERE id IN ({','.join('?'*len(top_ids))})"
    rows = conn.execute(q, top_ids).fetchall()
    order = {nid:i for i,nid in enumerate(top_ids)}
    rows.sort(key=lambda r: order[r["id"]])
    return rows

# ---------------- pages ----------------
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) c FROM notes").fetchone()["c"]
    recent = conn.execute("SELECT id, body, created_at FROM notes ORDER BY datetime(created_at) DESC LIMIT 8").fetchall()
    tagmap = map_note_tags(conn, recent)
    recent_list = [dict(r) | {"tags": tagmap.get(r["id"], "")} for r in recent]
    # top tags
    trows = conn.execute("""
      SELECT t.name, COUNT(nt.note_id) usage
      FROM tags t LEFT JOIN note_tags nt ON nt.tag_id=t.id
      GROUP BY t.id ORDER BY usage DESC, t.name ASC LIMIT 20
    """).fetchall()
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "stats": {"notes": total},
        "recent": recent_list,
        "top_tags": trows
    })

@app.get("/notes", response_class=HTMLResponse)
def notes_page(request: Request, tag: str = "", page: int = 1):
    conn = get_conn()
    base_sql = "FROM notes n"
    params = []
    if tag:
        base_sql += " WHERE EXISTS (SELECT 1 FROM note_tags nt JOIN tags t ON t.id=nt.tag_id WHERE nt.note_id=n.id AND t.name=?)"
        params.append(tag)
    count = conn.execute(f"SELECT COUNT(*) c {base_sql}", params).fetchone()["c"]
    offset = (page - 1) * PAGE_SIZE
    rows = conn.execute(f"SELECT n.id, n.body, n.created_at {base_sql} ORDER BY datetime(n.created_at) DESC LIMIT ? OFFSET ?", (*params, PAGE_SIZE, offset)).fetchall()
    tagmap = map_note_tags(conn, rows)
    notes = [dict(r) | {"tags": tagmap.get(r["id"], "")} for r in rows]
    has_more = (offset + PAGE_SIZE) < count
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse("partials/note_page.html", {"request": request, "notes": notes, "tag": tag, "next_page": (page+1) if has_more else None})
    return templates.TemplateResponse("notes.html", {"request": request, "notes": notes, "active_tag": tag, "next_page": (2 if has_more else None)})

@app.get("/notes/{note_id}", response_class=HTMLResponse)
def note_detail(request: Request, note_id: int):
    conn = get_conn()
    r = conn.execute("SELECT id, body, created_at FROM notes WHERE id=?", (note_id,)).fetchone()
    if not r: return RedirectResponse("/", status_code=302)
    note = dict(r) | {"tags": map_note_tags(conn, [r]).get(r["id"], "")}
    related_rows = related_notes_semantic(conn, note_id, limit=6)
    if not related_rows:
        related_rows = conn.execute("""
          WITH note_tags_set AS (
            SELECT t.id AS tag_id FROM note_tags nt JOIN tags t ON t.id=nt.tag_id WHERE nt.note_id=?
          )
          SELECT n.id, n.body, n.created_at, COUNT(*) AS overlap
          FROM notes n
          JOIN note_tags nt ON nt.note_id = n.id
          WHERE n.id != ? AND nt.tag_id IN (SELECT tag_id FROM note_tags_set)
          GROUP BY n.id
          ORDER BY overlap DESC, datetime(n.created_at) DESC
          LIMIT 6
        """, (note_id, note_id)).fetchall()
    rel_tagmap = map_note_tags(conn, related_rows)
    related = [dict(x) | {"tags": rel_tagmap.get(x["id"], "")} for x in related_rows]
    return templates.TemplateResponse("note_detail.html", {"request": request, "note": note, "related": related})

@app.get("/notes/{note_id}/edit", response_class=HTMLResponse)
def note_edit_partial(request: Request, note_id: int):
    conn = get_conn()
    r = conn.execute("SELECT id, body, created_at FROM notes WHERE id=?", (note_id,)).fetchone()
    if not r: return HTMLResponse("")
    note = dict(r) | {"tags": map_note_tags(conn, [r]).get(r["id"], "")}
    return templates.TemplateResponse("partials/note_edit_form.html", {"request": request, "note": note})

@app.post("/notes/{note_id}/update", response_class=HTMLResponse)
def note_update(request: Request, note_id: int, body: str = Form(...), tags: str = Form("")):
    conn = get_conn()
    conn.execute("UPDATE notes SET body=? WHERE id=?", (body, note_id))
    set_note_tags(conn, note_id, parse_tags(tags))
    conn.commit()
    ensure_embedding_for_note(conn, note_id, body)  # best-effort
    r = conn.execute("SELECT id, body, created_at FROM notes WHERE id=?", (note_id,)).fetchone()
    note = dict(r) | {"tags": map_note_tags(conn, [r]).get(r["id"], "")}
    headers = {"HX-Trigger": json.dumps(hx_trigger_dict("toast", {"type":"success","message":"Note updated"}))}
    return templates.TemplateResponse("partials/note_view.html", {"request": request, "note": note}, headers=headers)

@app.post("/notes/{note_id}/delete", response_class=RedirectResponse)
def note_delete(note_id: int):
    conn = get_conn()
    conn.execute("DELETE FROM notes WHERE id=?", (note_id,))
    conn.commit()
    return RedirectResponse("/notes", status_code=303)

@app.post("/notes", response_class=HTMLResponse)
def create_note(request: Request, body: str = Form(...), tags: str = Form("")):
    conn = get_conn()
    now = datetime.utcnow().isoformat(timespec="seconds")
    cur = conn.execute("INSERT INTO notes(body, created_at) VALUES (?,?)", (body, now))
    nid = cur.lastrowid
    set_note_tags(conn, nid, parse_tags(tags))
    conn.commit()
    ensure_embedding_for_note(conn, nid, body)  # best-effort
    if request.headers.get("HX-Request"):
        row = conn.execute("SELECT id, body, created_at FROM notes WHERE id=?", (nid,)).fetchone()
        note = dict(row) | {"tags": map_note_tags(conn, [row]).get(nid, "")}
        headers = {"HX-Trigger": json.dumps(hx_trigger_dict("toast", {"type":"success","message":"Note saved"}))}
        return templates.TemplateResponse("partials/note_item.html", {"request": request, "n": note}, headers=headers)
    return RedirectResponse(f"/notes/{nid}", status_code=303)

@app.post("/notes/batch", response_class=HTMLResponse)
def notes_batch(request: Request, action: str = Form(...), ids: str = Form(...), tag: str = Form("")):
    conn = get_conn()
    id_list = [int(x) for x in ids.split(",") if x.strip().isdigit()]
    if not id_list:
        return Response(status_code=204, headers={"HX-Trigger": json.dumps(hx_trigger_dict("toast", {"type":"info","message":"No notes selected"}))})
    if action == "delete":
        conn.executemany("DELETE FROM notes WHERE id=?", [(i,) for i in id_list])
    elif action == "add_tag":
        nm = norm_tag(tag); 
        if nm:
            tid = ensure_tag(conn, nm)
            conn.executemany("INSERT OR IGNORE INTO note_tags(note_id, tag_id) VALUES (?,?)", [(i, tid) for i in id_list])
    elif action == "remove_tag":
        nm = norm_tag(tag)
        if nm:
            row = conn.execute("SELECT id FROM tags WHERE name=?", (nm,)).fetchone()
            if row:
                conn.executemany("DELETE FROM note_tags WHERE note_id=? AND tag_id=?", [(i, row["id"]) for i in id_list])
    conn.commit()
    headers = {"HX-Trigger": json.dumps(hx_trigger_dict("toast", {"type":"success","message":"Batch action completed"}))}
    return Response(status_code=204, headers=headers)

# ---------------- Search (FTS + Semantic) ----------------
def semantic_search(conn, query: str, page: int = 1, page_size: int = PAGE_SIZE):
    base = os.getenv("OLLAMA_EMBED_BASE_URL", get_setting(conn, "OLLAMA_EMBED_BASE_URL", "http://localhost:11434"))
    model = os.getenv("OLLAMA_EMBED_MODEL", get_setting(conn, "OLLAMA_EMBED_MODEL", "nomic-embed-text:latest"))
    qvec = embed_text(query, base, model)
    rows = conn.execute("SELECT note_id, dim, vec FROM note_embeddings").fetchall()
    scored = []
    for r in rows:
        if r["dim"] != len(qvec): continue
        v = _blob_to_vec(r["vec"], dim_hint=r["dim"])
        s = cosine(v, qvec)
        scored.append((r["note_id"], s))
    scored.sort(key=lambda x: x[1], reverse=True)
    total = len(scored)
    start = (page - 1) * page_size
    end   = start + page_size
    page_ids = [i for (i, _) in scored[start:end]]
    if not page_ids:
        return [], 0
    q = f"SELECT id, body, created_at FROM notes WHERE id IN ({','.join('?'*len(page_ids))})"
    items = conn.execute(q, page_ids).fetchall()
    order = {nid:i for i,nid in enumerate(page_ids)}
    items.sort(key=lambda r: order[r["id"]])
    return items, total

@app.get("/search", response_class=HTMLResponse)
def search_page(request: Request, q: str = "", page: int = 1, embed: int = 0):
    conn = get_conn()
    items = []; total = 0
    using_embed = bool(embed)
    if q:
        if using_embed:
            try:
                items, total = semantic_search(conn, q, page=page, page_size=PAGE_SIZE)
            except Exception:
                using_embed = False
        if not using_embed:
            total = conn.execute("SELECT COUNT(*) c FROM notes_fts WHERE notes_fts MATCH ?", (q,)).fetchone()["c"]
            offset = (page - 1) * PAGE_SIZE
            items = conn.execute("""
              SELECT n.id, n.body, n.created_at
              FROM notes_fts f JOIN notes n ON n.id=f.rowid
              WHERE notes_fts MATCH ?
              ORDER BY bm25(notes_fts) LIMIT ? OFFSET ?
            """, (q, PAGE_SIZE, offset)).fetchall()
    tagmap = map_note_tags(conn, items)
    notes = [dict(r) | {"tags": tagmap.get(r["id"], "")} for r in items]
    has_more = (page * PAGE_SIZE) < total
    ctx = {"request": request, "q": q, "notes": notes, "next_page": (page+1) if has_more else None, "embed": 1 if using_embed else 0}
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse("partials/note_page.html", ctx)
    return templates.TemplateResponse("search.html", ctx)

@app.get("/api/q", response_class=HTMLResponse)
def quick_search_partial(request: Request, q: str):
    conn = get_conn()
    q = (q or "").strip()
    if not q: return HTMLResponse("")
    rows = conn.execute("""
      SELECT n.id, n.body, n.created_at
      FROM notes_fts f JOIN notes n ON n.id=f.rowid
      WHERE notes_fts MATCH ?
      ORDER BY bm25(notes_fts) LIMIT 20
    """, (q,)).fetchall()
    tagmap = map_note_tags(conn, rows)
    items = [dict(r) | {"tags": tagmap.get(r["id"], "")} for r in rows]
    return templates.TemplateResponse("partials/search_results.html", {"request": request, "items": items})

# ---------------- Tags page (missing before) ----------------
@app.get("/tags", response_class=HTMLResponse)
def tags_page(request: Request):
    conn = get_conn()
    rows = conn.execute("""
      SELECT t.name, COUNT(nt.note_id) usage
      FROM tags t LEFT JOIN note_tags nt ON nt.tag_id=t.id
      GROUP BY t.id
      ORDER BY usage DESC, t.name ASC
    """).fetchall()
    return templates.TemplateResponse("tags.html", {"request": request, "tags": rows})

# ---------------- Tag APIs (autocomplete + LLM suggest) ----------------
STOPWORDS = set("a an and the i you he she it we they to of for in on with from this that these those be is are was were am will would can could should as at by not or if into over under about after before during up down out very just".split())
def naive_tags(text: str, k: int = 6):
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{3,}", text.lower())
    words = [w for w in words if w not in STOPWORDS]
    cnt = Counter(words)
    return [w for w,_ in cnt.most_common(k)]

@app.post("/tags/suggest")
def suggest_tags(payload: dict = Body(...)):
    text = (payload.get("text") or "").strip()
    if len(text) < 8: return JSONResponse({"tags": []})
    conn = get_conn()
    base = get_setting(conn, "OLLAMA_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    model = get_setting(conn, "OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
    prompt = ("You are a tagging assistant. From the note, extract 3-7 short tags.\n"
              "- lowercase\n- hyphen for multiword\n- no '#'\n- comma-separated only\n\nNote:\n" + text + "\nTags:")
    try:
        r = httpx.post(f"{base}/api/generate", json={"model": model, "prompt": prompt, "stream": False}, timeout=10.0)
        r.raise_for_status()
        resp = r.json().get("response","")
        tags = [norm_tag(t) for t in resp.split(",")]
        tags = [t for t in tags if t] or naive_tags(text)
    except Exception:
        tags = naive_tags(text)
    return {"tags": list(dict.fromkeys(tags))[:7]}

@app.get("/api/tags")
def tags_autocomplete(q: str = ""):
    conn = get_conn()
    qn = norm_tag(q)
    if qn:
        rows = conn.execute("""
          SELECT t.name, COUNT(nt.note_id) usage
          FROM tags t LEFT JOIN note_tags nt ON nt.tag_id=t.id
          WHERE t.name LIKE ? || '%'
          GROUP BY t.id
          ORDER BY usage DESC, t.name ASC
          LIMIT 10
        """, (qn,)).fetchall()
    else:
        rows = conn.execute("""
          SELECT t.name, COUNT(nt.note_id) usage
          FROM tags t LEFT JOIN note_tags nt ON nt.tag_id=t.id
          GROUP BY t.id ORDER BY usage DESC, t.name ASC LIMIT 10
        """).fetchall()
    return {"tags": [r["name"] for r in rows]}

# ---------------- Settings + Embeddings Admin ----------------
@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    conn = get_conn()
    ctx = {
        "base": get_setting(conn, "OLLAMA_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")),
        "model": get_setting(conn, "OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "llama3.1:8b")),
        "emb_base": get_setting(conn, "OLLAMA_EMBED_BASE_URL", EMBED_BASE_ENV or "http://localhost:11434"),
        "emb_model": get_setting(conn, "OLLAMA_EMBED_MODEL", EMBED_MODEL_ENV or "nomic-embed-text:latest"),
    }
    return templates.TemplateResponse("settings.html", {"request": request, **ctx})

@app.post("/settings/ollama", response_class=RedirectResponse)
def settings_ollama(base: str = Form(""), model: str = Form(""), emb_base: str = Form(""), emb_model: str = Form("")):
    conn = get_conn()
    put_setting(conn, "OLLAMA_BASE_URL", base.strip() or "http://localhost:11434")
    put_setting(conn, "OLLAMA_MODEL", model.strip() or "llama3.1:8b")
    put_setting(conn, "OLLAMA_EMBED_BASE_URL", emb_base.strip() or "http://localhost:11434")
    put_setting(conn, "OLLAMA_EMBED_MODEL", emb_model.strip() or "nomic-embed-text:latest")
    return RedirectResponse("/settings?ok=1", status_code=303)

@app.post("/settings/ollama/test")
def settings_ollama_test(base: str = Form(""), model: str = Form("")):
    base = base.strip() or "http://localhost:11434"
    model = model.strip() or "llama3.1:8b"
    try:
        r = httpx.get(f"{base}/api/tags", timeout=3.0)  # quick reachability check
        ok = True; lat = int(r.elapsed.total_seconds()*1000)
    except Exception:
        ok = False; lat = None
    return JSONResponse({"ok": ok, "latency_ms": lat})

@app.get("/embeddings", response_class=HTMLResponse)
def embeddings_admin(request: Request):
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) c FROM notes").fetchone()["c"]
    embedded = conn.execute("SELECT COUNT(*) c FROM note_embeddings").fetchone()["c"]
    missing = total - embedded
    return templates.TemplateResponse("embeddings.html", {"request": request, "total": total, "embedded": embedded, "missing": missing})

@app.post("/embeddings/rebuild", response_class=HTMLResponse)
def embeddings_rebuild(request: Request, limit: int = Form(100), force: int = Form(0)):
    conn = get_conn()
    base = get_setting(conn, "OLLAMA_EMBED_BASE_URL", EMBED_BASE_ENV or "http://localhost:11434")
    model = get_setting(conn, "OLLAMA_EMBED_MODEL", EMBED_MODEL_ENV or "nomic-embed-text:latest")
    if force:
        rows = conn.execute("SELECT id, body FROM notes ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    else:
        rows = conn.execute("""
          SELECT n.id, n.body
          FROM notes n
          LEFT JOIN note_embeddings e ON e.note_id = n.id
          WHERE e.note_id IS NULL
          ORDER BY n.id DESC LIMIT ?
        """, (limit,)).fetchall()
    done = 0; errors = 0
    for r in rows:
        try:
            vec = embed_text(r["body"], base, model)
            conn.execute("INSERT INTO note_embeddings(note_id, dim, vec) VALUES (?,?,?) ON CONFLICT(note_id) DO UPDATE SET dim=excluded.dim, vec=excluded.vec",
                         (r["id"], len(vec), _vec_to_blob(vec)))
            done += 1
        except Exception:
            errors += 1
    conn.commit()
    total = conn.execute("SELECT COUNT(*) c FROM notes").fetchone()["c"]
    embedded = conn.execute("SELECT COUNT(*) c FROM note_embeddings").fetchone()["c"]
    missing = total - embedded
    return templates.TemplateResponse("partials/emb_progress.html",
        {"request": request, "total": total, "embedded": embedded, "missing": missing, "just_done": done, "errors": errors})

# ---------------- Export ----------------
def _all_notes(conn):
    rows = conn.execute("SELECT id, body, created_at FROM notes ORDER BY id ASC").fetchall()
    tagmap = map_note_tags(conn, rows)
    out = []
    for r in rows:
        out.append({"id": r["id"], "body": r["body"], "created_at": r["created_at"], "tags": [t for t in (tagmap.get(r["id"], "") or "").split(",") if t]})
    return out

@app.get("/export", response_class=HTMLResponse)
def export_page(request: Request):
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) c FROM notes").fetchone()["c"]
    tags  = conn.execute("SELECT COUNT(*) c FROM tags").fetchone()["c"]
    return templates.TemplateResponse("export.html", {"request": request, "total": total, "tags": tags})

@app.get("/export/json")
def export_json():
    conn = get_conn()
    data = {"exported_at": datetime.utcnow().isoformat(timespec="seconds")+"Z", "notes": _all_notes(conn)}
    payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    headers = {"Content-Disposition": 'attachment; filename="second_brain_export.json"'}
    return Response(content=payload, media_type="application/json", headers=headers)

def _slugify(title: str) -> str:
    s = (title or "").strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s]+", "-", s)
    return s[:60] or "note"

def _title_from_body(body: str) -> str:
    for line in (body or "").splitlines():
        t = line.strip()
        if t: return t
    return "Untitled"

@app.get("/export/markdown.zip")
def export_markdown_zip():
    conn = get_conn()
    notes = _all_notes(conn)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for n in notes:
            title = _title_from_body(n["body"])
            name = f'{n["id"]}-{_slugify(title)}.md'
            fm_tags = ", ".join(n["tags"])
            content = f"""--- 
id: {n["id"]}
created_at: {n["created_at"]}
tags: [{fm_tags}]
---
{n["body"]}
"""
            zf.writestr(name, content)
    buf.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="second_brain_markdown.zip"'}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)
PY

# =========================== templates: dashboard, tags, missing partial ===========================
b templates/dashboard.html
cat > templates/dashboard.html <<'HTML'
{% extends "base.html" %}
{% block title %}Dashboard — Second Brain Premium{% endblock %}
{% block content %}
<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
  <section class="lg:col-span-2 rounded-2xl bg-white/95 dark:bg-slate-950/90 shadow-xl ring-1 ring-black/5 p-6">
    <h2 class="text-lg font-semibold text-slate-900 dark:text-white">Quick Capture</h2>
    <form class="mt-3 grid gap-3" hx-post="/notes" hx-target="#recent-list" hx-swap="afterbegin">
      <textarea id="body" name="body" rows="5" placeholder="Jot something down…"
        class="w-full rounded-xl border-slate-300 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"></textarea>
      <input type="hidden" id="tagsInput" name="tags" value="">
      <div>
        <div class="text-xs text-slate-500 mb-1">Tags</div>
        <div id="tag-editor" class="min-h-[44px] rounded-xl px-2 py-2 bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 flex flex-wrap gap-2 relative">
          <input id="tag-entry" type="text" placeholder="type, Tab/Enter to accept"
                 class="bg-transparent outline-none flex-1 min-w-[140px] text-slate-800 dark:text-slate-200 placeholder:text-slate-400" />
          <div id="tag-auto" class="absolute left-2 right-2 top-full mt-1 hidden rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 overflow-hidden"></div>
        </div>
        <div id="tag-suggestions" class="mt-2 flex flex-wrap gap-2"></div>
      </div>
      <div class="flex gap-2">
        <button class="px-4 py-2 rounded-xl bg-brand-600 text-white">Save</button>
        <a href="/notes" class="px-4 py-2 rounded-xl bg-slate-200 dark:bg-slate-800">All Notes</a>
      </div>
    </form>
  </section>

  <aside class="rounded-2xl bg-white/95 dark:bg-slate-950/90 shadow-xl ring-1 ring-black/5 p-6">
    <h2 class="text-lg font-semibold text-slate-900 dark:text-white">Stats</h2>
    <div class="mt-2 text-sm text-slate-600 dark:text-slate-300">Notes: <b>{{ stats.notes }}</b></div>
    <h3 class="mt-4 text-sm font-medium text-slate-900 dark:text-white">Top tags</h3>
    <div class="mt-2 flex flex-wrap gap-2">
      {% for t in top_tags %}
        <a href="/notes?tag={{ t.name }}" class="inline-block px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800">#{{ t.name }} ({{ t.usage }})</a>
      {% else %}
        <div class="text-slate-500 text-sm">No tags yet.</div>
      {% endfor %}
    </div>
    <div class="mt-6">
      <a href="/embeddings" class="text-sm text-slate-600 hover:underline dark:text-slate-300">Embeddings Admin →</a>
    </div>
  </aside>
</div>

<section class="mt-6 rounded-2xl bg-white/95 dark:bg-slate-950/90 shadow-xl ring-1 ring-black/5 p-6">
  <h2 class="text-lg font-semibold text-slate-900 dark:text-white">Recent</h2>
  <ul id="recent-list" class="mt-3 divide-y divide-slate-200/80 dark:divide-white/10">
    {% for n in recent %}
      {% include 'partials/note_item.html' %}
    {% else %}
      <li class="py-6 text-slate-500">No notes yet — try Quick Capture above.</li>
    {% endfor %}
  </ul>
</section>
{% endblock %}
HTML

b templates/tags.html
cat > templates/tags.html <<'HTML'
{% extends "base.html" %}
{% block title %}Tags — Second Brain Premium{% endblock %}
{% block content %}
<div class="rounded-2xl bg-white/95 dark:bg-slate-950/90 shadow-xl ring-1 ring-black/5 p-6">
  <h1 class="text-xl font-semibold text-slate-900 dark:text-white">Tags</h1>
  <div class="mt-4 flex flex-wrap gap-2">
    {% for t in tags %}
      <a href="/notes?tag={{ t.name }}" class="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700">
        <span>#{{ t.name }}</span>
        <span class="text-xs text-slate-500">{{ t.usage }}</span>
      </a>
    {% else %}
      <div class="text-slate-500">No tags yet.</div>
    {% endfor %}
  </div>
</div>
{% endblock %}
HTML

# Missing partial used by /notes POST hx response
b templates/partials/note_item.html
cat > templates/partials/note_item.html <<'HTML'
<li class="py-4">
  <a href="/notes/{{ n.id }}" class="block text-slate-900 dark:text-white font-medium hover:underline">
    {{ (n.body[:140] ~ ('…' if n.body|length > 140 else '')) | e }}
  </a>
  <div class="mt-1 text-xs text-slate-500">
    {{ n.created_at }}
    {% if n.tags %}
      •
      {% for t in n.tags.split(',') if t.strip() %}
        <a href="/notes?tag={{ t.strip() }}"
           class="inline-block ml-1 px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700">#{{ t.strip() }}</a>
      {% endfor %}
    {% endif %}
  </div>
</li>
HTML

# =========================== Fix scroll sentinel (notes vs search) ===========================
b templates/partials/scroll_sentinel.html
cat > templates/partials/scroll_sentinel.html <<'HTML'
{% if q %}
  <div class="mt-4 h-10 flex items-center justify-center"
       hx-get="/search?q={{ q | urlencode }}&page={{ next_page }}{% if embed %}&embed=1{% endif %}"
       hx-trigger="revealed once"
       hx-target="#notes-list"
       hx-swap="beforeend">
    <div class="animate-pulse text-white/70">Loading…</div>
  </div>
{% else %}
  <div class="mt-4 h-10 flex items-center justify-center"
       hx-get="/notes?page={{ next_page }}{% if tag %}&tag={{ tag }}{% endif %}"
       hx-trigger="revealed once"
       hx-target="#notes-list"
       hx-swap="beforeend">
    <div class="animate-pulse text-white/70">Loading…</div>
  </div>
{% endif %}
HTML

# =========================== Keep search_results if missing ===========================
[[ -f templates/partials/search_results.html ]] || cat > templates/partials/search_results.html <<'HTML'
<ul class="divide-y divide-slate-200/80 dark:divide-white/10">
  {% for n in items %}
    <li class="py-3 px-4 hover:bg-slate-50 dark:hover:bg-slate-900">
      <a href="/notes/{{ n.id }}" class="block">
        <div class="font-medium text-slate-900 dark:text-white">{{ (n.body[:100] ~ ('…' if n.body|length > 100 else '')) | e }}</div>
        <div class="text-xs text-slate-500">{{ n.created_at }}
          {% if n.tags %} •
            {% for t in n.tags.split(',') if t.strip() %}
              <span class="inline-block ml-1 px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800">#{{ t.strip() }}</span>
            {% endfor %}
          {% endif %}
        </div>
      </a>
    </li>
  {% else %}
    <li class="py-6 text-center text-slate-500">No results.</li>
  {% endfor %}
</ul>
HTML

# =========================== requirements.txt (ensure) ===========================
if [[ ! -f requirements.txt ]]; then
  cat > requirements.txt <<'REQ'
fastapi==0.115.0
uvicorn[standard]==0.30.6
jinja2==3.1.4
httpx==0.27.2
numpy>=1.24,<3
python-multipart==0.0.9
REQ
else
  grep -q "fastapi" requirements.txt || echo "fastapi==0.115.0" >> requirements.txt
  grep -q "uvicorn" requirements.txt || echo "uvicorn[standard]==0.30.6" >> requirements.txt
  grep -q "jinja2" requirements.txt || echo "jinja2==3.1.4" >> requirements.txt
  grep -q "httpx" requirements.txt || echo "httpx==0.27.2" >> requirements.txt
  grep -q "numpy" requirements.txt || echo "numpy>=1.24,<3" >> requirements.txt
  grep -q "python-multipart" requirements.txt || echo "python-multipart==0.0.9" >> requirements.txt
fi

# =========================== quick verifier script ===========================
b scripts/verify_scaffold.sh
cat > scripts/verify_scaffold.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
ok(){ printf "✅ %s\n" "$*"; }
warn(){ printf "⚠️  %s\n" "$*" >&2; FAIL=1; }
FAIL=0

need_file(){ [[ -f "$1" ]] && ok "exists: $1" || warn "missing: $1"; }

need_file app.py
need_file templates/base.html
need_file templates/dashboard.html
need_file templates/notes.html
need_file templates/search.html
need_file templates/tags.html
need_file templates/note_detail.html
need_file templates/settings.html
need_file templates/export.html
need_file templates/embeddings.html
need_file templates/partials/note_item.html
need_file templates/partials/note_row_selectable.html
need_file templates/partials/note_edit_form.html
need_file templates/partials/note_view.html
need_file templates/partials/scroll_sentinel.html
need_file templates/partials/search_results.html
need_file templates/partials/emb_progress.html
need_file static/app.js

grep -q "/search" app.py && ok "endpoint: /search" || warn "endpoint missing: /search"
grep -q "/notes\"" app.py && ok "endpoint: /notes" || warn "endpoint missing: /notes"
grep -q "/tags\"" app.py && ok "endpoint: /tags" || warn "endpoint missing: /tags"
grep -q "/export" app.py && ok "endpoint: /export" || warn "endpoint missing: /export"
grep -q "/embeddings" app.py && ok "endpoint: /embeddings" || warn "endpoint missing: /embeddings"
grep -q "notes_fts" app.py && ok "sqlite FTS present" || warn "sqlite FTS missing"
grep -q "note_embeddings" app.py && ok "embeddings table present" || warn "embeddings table missing"

if [[ $FAIL -eq 0 ]]; then
  echo "All checks passed."
  exit 0
else
  echo "Some checks failed. See warnings above."
  exit 1
fi
SH
chmod +x scripts/verify_scaffold.sh

echo "Done.

Run:
  bash scripts/scaffold_011.sh
  # then sanity-check
  bash scripts/verify_scaffold.sh

Restart your app:
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  uvicorn app:app --reload --host 0.0.0.0 --port 8084
"
