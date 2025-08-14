#!/usr/bin/env bash
# scripts/scaffold_018.sh
set -euo pipefail

STAMP="$(date +%Y%m%d-%H%M%S)"
b(){ [[ -f "$1" ]] && { mkdir -p .bak; cp -p "$1" ".bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }

mkdir -p templates/partials out/exports scripts

# 1) Append Python block to app.py once
if ! grep -q "scaffold_018 additions" app.py 2>/dev/null; then
  b app.py
  cat >> app.py <<'PY'

# ==== scaffold_018 additions (exports page+CSV, inline tags, similar notes, bulk import) ====
# NOTE: we define Response here to also cover earlier scaffolds that returned Response.
try:
    from fastapi import Response
except Exception:
    from fastapi.responses import Response  # fallback

import csv, io, zipfile, json, os, re, datetime as _dt
from typing import List
from fastapi import UploadFile, File

def _ensure_schema_018(conn):
    c = conn.cursor()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS export_archives (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL,
      kind TEXT NOT NULL,
      file_path TEXT NOT NULL,
      period_start TEXT NOT NULL,
      period_end TEXT NOT NULL,
      created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_export_user ON export_archives(user_id, id DESC);
    """)
    conn.commit()

@app.on_event("startup")
def _s18_startup():
    try:
        _ensure_schema_018(get_conn())
    except Exception as e:
        print("scaffold_018 init failed:", e)

# ---------- Inline Tags (CSV in notes.tags) ----------
def _split_tags(csv_tags: str) -> List[str]:
    return [t.strip().lower() for t in (csv_tags or "").split(",") if t.strip()]

def _join_tags(tags: List[str]) -> str:
    # keep stable order & uniqueness
    seen = set(); out=[]
    for t in tags:
        t = t.strip().lower().replace("#","").replace(" ", "-")
        if t and t not in seen:
            seen.add(t); out.append(t)
    return ",".join(out)

def _update_fts_row(c, note_id: int):
    row = c.execute("SELECT title, summary, tags, actions, content FROM notes WHERE id=?", (note_id,)).fetchone()
    if row:
        title, summary, tags, actions, content = row
        c.execute("DELETE FROM notes_fts WHERE rowid=?", (note_id,))
        c.execute("INSERT INTO notes_fts(rowid, title, summary, tags, actions, content) VALUES (?,?,?,?,?,?)",
                  (note_id, title, summary, tags, actions, content))

@app.get("/notes/{note_id}/tags/partial")
def note_tags_partial(note_id: int, current_user: User = Depends(get_current_user)):
    conn = get_conn(); c = conn.cursor()
    row = c.execute("SELECT tags FROM notes WHERE id=? AND user_id=?", (note_id, current_user.id)).fetchone()
    tags = _split_tags(row[0] if row else "")
    conn.close()
    return templates.TemplateResponse("partials/note_tags.html",
        {"request": Request, "note_id": note_id, "tags": tags})

@app.post("/notes/{note_id}/tag/add")
def note_tag_add(note_id: int, tag: str = Form(...), current_user: User = Depends(get_current_user)):
    tag = (tag or "").strip()
    if not tag: return Response(status_code=204)
    conn = get_conn(); c = conn.cursor()
    row = c.execute("SELECT tags FROM notes WHERE id=? AND user_id=?", (note_id, current_user.id)).fetchone()
    if not row: 
        conn.close(); return Response(status_code=404)
    tags = _split_tags(row[0])
    tags.append(tag)
    new_csv = _join_tags(tags)
    c.execute("UPDATE notes SET tags=? WHERE id=?", (new_csv, note_id))
    _update_fts_row(c, note_id)
    c.execute("INSERT INTO audit_log(action, note_id, meta, created_at) VALUES ('tag_add', ?, ?, datetime('now'))",
              (note_id, f"tag={tag}")) if c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='audit_log'").fetchone() else None
    conn.commit(); conn.close()
    return Response(status_code=204)

@app.post("/notes/{note_id}/tag/remove")
def note_tag_remove(note_id: int, tag: str = Form(...), current_user: User = Depends(get_current_user)):
    tag = (tag or "").strip().lower()
    conn = get_conn(); c = conn.cursor()
    row = c.execute("SELECT tags FROM notes WHERE id=? AND user_id=?", (note_id, current_user.id)).fetchone()
    if not row: 
        conn.close(); return Response(status_code=404)
    tags = [t for t in _split_tags(row[0]) if t != tag]
    new_csv = _join_tags(tags)
    c.execute("UPDATE notes SET tags=? WHERE id=?", (new_csv, note_id))
    _update_fts_row(c, note_id)
    c.execute("INSERT INTO audit_log(action, note_id, meta, created_at) VALUES ('tag_remove', ?, ?, datetime('now'))",
              (note_id, f"tag={tag}")) if c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='audit_log'").fetchone() else None
    conn.commit(); conn.close()
    return Response(status_code=204)

# ---------- Similar Notes (tags overlap or FTS) ----------
@app.get("/detail/{note_id}/similar")
def similar_notes(note_id: int, mode: str = "tags", current_user: User = Depends(get_current_user)):
    conn = get_conn(); c = conn.cursor()
    row = c.execute("SELECT tags, title, summary, content FROM notes WHERE id=? AND user_id=?", (note_id, current_user.id)).fetchone()
    if not row:
        conn.close(); return {"items": []}
    tags, title, summary, content = row
    items=[]
    if mode == "fts":
        q = " ".join([w for w in (title or "").split()[:6]]) or (summary or "")[:60] or (content or "")[:60]
        if q.strip():
            rows = c.execute("""
              SELECT n.id, n.title, n.timestamp, snippet(notes_fts, 4, '<mark>','</mark>',' … ', 8) AS snippet
              FROM notes_fts fts JOIN notes n ON n.id = fts.rowid
              WHERE notes_fts MATCH ? AND n.user_id = ? AND n.id != ?
              ORDER BY bm25(notes_fts) LIMIT 6
            """, (q, current_user.id, note_id)).fetchall()
            items = [{"id": r[0], "title": r[1], "timestamp": r[2], "snippet": r[3]} for r in rows]
    else:
        tag_list = _split_tags(tags)
        if tag_list:
            # compute overlap by tag LIKE; approximate but fast
            wh = " OR ".join(["n.tags LIKE ?"] * len(tag_list))
            params = [f"%{t}%" for t in tag_list]
            rows = c.execute(f"""
              SELECT n.id, n.title, n.timestamp
              FROM notes n
              WHERE n.user_id=? AND n.id != ? AND ({wh})
              ORDER BY n.timestamp DESC LIMIT 12
            """, (current_user.id, note_id, *params)).fetchall()
            items = [{"id": r[0], "title": r[1], "timestamp": r[2]} for r in rows]
    conn.close()
    # Render partial if browser requests HTML; otherwise JSON
    accept = Request.headers.get("accept","") if hasattr(Request, "headers") else ""
    if "text/html" in accept:
        return templates.TemplateResponse("partials/similar_list.html",
                {"request": Request, "items": items})
    return {"items": items}

# ---------- Exports page + CSV generator ----------
def _export_notes_csv_for_user(conn, user_id: int, start_iso: str, end_iso: str, kind: str) -> str:
    (settings.base_dir / "out" / "exports").mkdir(parents=True, exist_ok=True)
    name = f"notes_{user_id}_{kind}_{start_iso}_to_{end_iso}.csv"
    fpath = settings.base_dir / "out" / "exports" / name
    rows = conn.execute("""
      SELECT id, timestamp, replace(replace(content, char(13),' '), char(10), ' ') AS content,
             COALESCE(tags,'') AS tags, COALESCE(title,'') AS title, COALESCE(summary,'') AS summary, COALESCE(actions,'') AS actions, COALESCE(type,'') AS type
      FROM notes WHERE user_id = ? AND date(timestamp) BETWEEN date(?) AND date(?) ORDER BY datetime(timestamp) ASC
    """, (user_id, start_iso, end_iso)).fetchall()
    with open(fpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id","timestamp","title","summary","tags","actions","type","content"])
        for r in rows: w.writerow([r[0], r[1], r[4], r[5], r[3], r[6], r[7], r[2]])
    conn.execute("INSERT INTO export_archives(user_id, kind, file_path, period_start, period_end, created_at) VALUES (?,?,?,?,?,?)",
                 (user_id, kind, str(fpath), start_iso, end_iso, _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    return str(fpath)

def _iso_week_range(today: _dt.date) -> tuple[str,str]:
    start = today - _dt.timedelta(days=today.weekday())
    end = start + _dt.timedelta(days=6)
    return start.isoformat(), end.isoformat()

@app.get("/exports")
def exports_page(request: Request, current_user: User = Depends(get_current_user)):
    conn = get_conn(); c = conn.cursor()
    rows = c.execute("""
      SELECT id, kind, file_path, period_start, period_end, created_at
      FROM export_archives WHERE user_id=? ORDER BY id DESC
    """, (current_user.id,)).fetchall()
    conn.close()
    return templates.TemplateResponse("exports.html",
        {"request": request, "rows": rows})

@app.post("/exports/run")
def exports_run(kind: str = Form("daily"), start: str = Form(""), end: str = Form(""),
                current_user: User = Depends(get_current_user)):
    conn = get_conn()
    today = _dt.date.today()
    if start and end:
        s, e = start, end
    elif (kind or "daily").lower() == "weekly":
        s, e = _iso_week_range(today)
    else:
        s = e = today.isoformat()
    _export_notes_csv_for_user(conn, current_user.id, s, e, (kind or "daily").lower())
    conn.close()
    return RedirectResponse("/exports", status_code=303)

@app.get("/exports/download/{id}")
def exports_download(id: int, current_user: User = Depends(get_current_user)):
    conn = get_conn(); c = conn.cursor()
    row = c.execute("SELECT file_path FROM export_archives WHERE id=? AND user_id=?", (id, current_user.id)).fetchone()
    conn.close()
    if not row: return Response(status_code=404)
    path = row[0]
    if not os.path.exists(path): return Response(status_code=410)
    return FileResponse(path, filename=os.path.basename(path), media_type="text/csv")

# ---------- Bulk Import APIs ----------
def _title_from_content(body: str) -> str:
    for line in (body or "").splitlines():
        t = line.strip()
        if t: return t[:80]
    return "note"

def _parse_md_with_frontmatter(text: str):
    text = text or ""
    if not text.startswith("---\n"):
        return {}, text
    parts = text.split("\n",1)[1].split("\n---",1)
    if len(parts) < 2: return {}, text
    fm_raw, rest = parts[0], parts[1].lstrip("\n")
    meta = {}
    for line in fm_raw.splitlines():
        if ":" not in line: continue
        k,v = line.split(":",1)
        meta[k.strip()] = v.strip().strip("'\"")
    return meta, rest

@app.post("/api/import/json")
async def import_json_api(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    data = json.loads((await file.read()).decode("utf-8", errors="replace"))
    notes = data.get("notes") if isinstance(data, dict) else data
    if not isinstance(notes, list): return {"ok": False, "error": "invalid format"}
    conn = get_conn(); c = conn.cursor()
    added = 0
    for n in notes:
        body = n.get("content","") or n.get("body","") or ""
        title = n.get("title") or _title_from_content(body)
        summary = n.get("summary","")
        tags = _join_tags(_split_tags(",".join(n.get("tags",[]) if isinstance(n.get("tags"), list) else (n.get("tags","") or "").split(","))))
        actions = n.get("actions","")
        ts = n.get("timestamp") or _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        typ = n.get("type","import")
        c.execute("""INSERT INTO notes(title, content, summary, tags, actions, type, timestamp, status, user_id)
                     VALUES (?,?,?,?,?,?,?,?,?)""",
                  (title, body, summary, tags, actions, typ, ts, "complete", current_user.id))
        nid = c.lastrowid
        c.execute("INSERT INTO notes_fts(rowid, title, summary, tags, actions, content) VALUES (?,?,?,?,?,?)",
                  (nid, title, summary, tags, actions, body))
        added += 1
    conn.commit(); conn.close()
    return {"ok": True, "imported": added}

@app.post("/api/import/markdown")
async def import_markdown_api(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    buf = await file.read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(buf), "r")
    except Exception:
        return {"ok": False, "error": "upload a .zip of markdown files"}
    conn = get_conn(); c = conn.cursor()
    added = 0
    for zi in zf.infolist():
        if not zi.filename.lower().endswith(".md"): continue
        text = zf.read(zi).decode("utf-8", errors="replace")
        meta, body = _parse_md_with_frontmatter(text)
        title = meta.get("title") or _title_from_content(body)
        tags = _join_tags(_split_tags(",".join(meta.get("tags","").split(",") if isinstance(meta.get("tags"), str) else meta.get("tags",[]))))
        actions = "\n".join(meta.get("actions",[])) if isinstance(meta.get("actions", list)) else meta.get("actions","")
        ts = meta.get("timestamp") or _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        typ = meta.get("type","md")
        c.execute("""INSERT INTO notes(title, content, summary, tags, actions, type, timestamp, status, user_id)
                     VALUES (?,?,?,?,?,?,?,?,?)""",
                  (title, body, meta.get("summary",""), tags, actions, typ, ts, "complete", current_user.id))
        nid = c.lastrowid
        c.execute("INSERT INTO notes_fts(rowid, title, summary, tags, actions, content) VALUES (?,?,?,?,?,?)",
                  (nid, title, meta.get("summary",""), tags, actions, body))
        added += 1
    conn.commit(); conn.close()
    return {"ok": True, "imported": added}
PY
else
  echo "• app.py already contains scaffold_018 additions — skipping append"
fi

# 2) Templates: Exports page + Partials
b templates/exports.html
cat > templates/exports.html <<'HTML'
{% extends "base.html" %}
{% block title %}Exports — Second Brain{% endblock %}
{% block content %}
<div class="rounded-2xl bg-white/95 dark:bg-slate-950/90 shadow-xl ring-1 ring-black/5 p-6">
  <h1 class="text-xl font-semibold">CSV Exports</h1>

  <form class="mt-4 flex flex-wrap items-end gap-3" action="/exports/run" method="post">
    <div>
      <label class="text-sm block mb-1">Kind</label>
      <select name="kind" class="rounded-xl border-slate-300 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 px-3 py-2">
        <option value="daily">Daily (today)</option>
        <option value="weekly">Weekly (current ISO week)</option>
      </select>
    </div>
    <div>
      <label class="text-sm block mb-1">Start (YYYY-MM-DD)</label>
      <input name="start" placeholder="(optional)" class="rounded-xl border-slate-300 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 px-3 py-2">
    </div>
    <div>
      <label class="text-sm block mb-1">End (YYYY-MM-DD)</label>
      <input name="end" placeholder="(optional)" class="rounded-xl border-slate-300 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 px-3 py-2">
    </div>
    <button class="px-3 py-2 rounded-xl bg-brand text-white">Generate</button>
  </form>

  <h2 class="mt-6 text-sm font-medium">Archives</h2>
  <table class="mt-2 w-full text-sm">
    <thead><tr class="text-left border-b dark:border-slate-800">
      <th class="py-2">Created</th><th>Kind</th><th>Range</th><th>File</th>
    </tr></thead>
    <tbody>
      {% for r in rows %}
        <tr class="border-b dark:border-slate-800">
          <td class="py-2">{{ r.created_at }}</td>
          <td>{{ r.kind }}</td>
          <td>{{ r.period_start }} → {{ r.period_end }}</td>
          <td><a class="hover:underline" href="/exports/download/{{ r.id }}">{{ r.file_path.split('/')[-1] }}</a></td>
        </tr>
      {% else %}
        <tr><td colspan="4" class="py-6 text-slate-500">No exports yet.</td></tr>
      {% endfor %}
    </tbody>
  </table>
</div>
{% endblock %}
HTML

b templates/partials/note_tags.html
cat > templates/partials/note_tags.html <<'HTML'
<div id="note-tags" class="mt-3 flex flex-wrap gap-2">
  {% for t in tags %}
    <form method="post" action="/notes/{{ note_id }}/tag/remove" class="inline">
      <input type="hidden" name="tag" value="{{ t }}">
      <button class="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-sm" title="Remove #{{ t }}">
        <span>#{{ t }}</span><span aria-hidden="true">×</span>
      </button>
    </form>
  {% else %}
    <span class="text-sm text-slate-500">No tags yet.</span>
  {% endfor %}
  <form method="post" action="/notes/{{ note_id }}/tag/add" class="inline-flex items-center gap-2">
    <input name="tag" placeholder="add-tag" class="ml-2 px-2 py-1 rounded border border-slate-300 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 text-sm">
    <button class="px-2 py-1 rounded bg-slate-200 dark:bg-slate-800 text-sm">Add</button>
  </form>
</div>
HTML

b templates/partials/similar_list.html
cat > templates/partials/similar_list.html <<'HTML'
<ul class="mt-2 space-y-2">
  {% for it in items %}
    <li class="p-3 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
      <a class="font-medium hover:underline text-slate-900 dark:text-white" href="/detail/{{ it.id }}">
        {{ it.title or ('#'+it.id|string) }}
      </a>
      {% if it.snippet %}
        <div class="text-xs text-slate-500 mt-1">{{ it.snippet|safe }}</div>
      {% endif %}
      <div class="text-xs text-slate-500 mt-1">{{ it.timestamp }}</div>
    </li>
  {% else %}
    <li class="text-slate-500">Nothing similar yet.</li>
  {% endfor %}
</ul>
HTML

# 3) (Optional) Add nav link to /exports if your base.html has a nav list
if [[ -f templates/base.html ]] && ! grep -q 'href="/exports"' templates/base.html; then
  b templates/base.html
  awk '1; /<\/ul>/ && !x {print "        <li><a class=\"hover:underline\" href=\"/exports\">Exports</a></li>"; x=1}' templates/base.html > templates/base.html.new && mv templates/base.html.new templates/base.html
fi

# 4) Tiny verifier
cat > scripts/verify_scaffold_018.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
grep -q "scaffold_018 additions" app.py && echo "ok marker" || { echo "missing marker"; exit 1; }
for f in templates/exports.html templates/partials/note_tags.html templates/partials/similar_list.html; do
  [[ -f "$f" ]] && echo "ok $f" || { echo "missing $f"; exit 1; }
done
echo "Looks good."
SH
chmod +x scripts/verify_scaffold_018.sh

echo "Done.

New routes:
  • /exports, /exports/run, /exports/download/{id}   (per-user CSV archives)
  • /notes/{id}/tags/partial  (partial UI for inline chips)
  • /notes/{id}/tag/add  and  /notes/{id}/tag/remove
  • /detail/{id}/similar?mode=tags|fts   (JSON or partial)
  • /api/import/json   and   /api/import/markdown    (upload JSON/ZIP)

Next:
  bash scripts/scaffold_018.sh
  bash scripts/verify_scaffold_018.sh
  Restart your server.

Wire-up tips:
  • In detail.html, where you show a note, drop:
      {% include 'partials/note_tags.html' %}
    (or fetch /notes/{{ note.id }}/tags/partial into a container)
  • For “Similar notes”, insert a container that loads
      /detail/{{ note.id }}/similar?mode=tags
    and add a simple toggle to hit mode=fts.
"
