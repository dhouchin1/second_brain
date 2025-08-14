#!/usr/bin/env bash
# scripts/scaffold_015.sh  (fixed)
# Adds: metrics panel + CSV, audit_log + triggers, tag add/remove endpoints,
# toolbar partial + small CSS tweak. Safe to re-run.
set -euo pipefail

STAMP="$(date +%Y%m%d-%H%M%S)"
backup() {
  [[ -f "$1" ]] && { mkdir -p .bak; cp -p "$1" ".bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true
}

mkdir -p templates/partials static

# 1) Append scaffold_015 Python to app.py (only once)
if ! grep -q "scaffold_015 additions" app.py 2>/dev/null; then
  backup app.py
  cat >> app.py <<'PY'

# ==== scaffold_015 additions (metrics, audit, note toolbar helpers) ====
from fastapi import Request, Form, HTMLResponse, RedirectResponse, JSONResponse, Response
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
import csv, io as _io

def _apply_scaffold_015(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS audit_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      action TEXT NOT NULL,
      note_id INTEGER,
      meta TEXT,
      created_at TEXT NOT NULL
    );

    CREATE TRIGGER IF NOT EXISTS s15_notes_ins AFTER INSERT ON notes BEGIN
      INSERT INTO audit_log(action,note_id,meta,created_at)
      VALUES('note_insert', NEW.id, 'len='||length(NEW.body), datetime('now'));
    END;

    CREATE TRIGGER IF NOT EXISTS s15_notes_upd AFTER UPDATE ON notes BEGIN
      INSERT INTO audit_log(action,note_id,meta,created_at)
      VALUES('note_update', NEW.id, 'len='||length(NEW.body), datetime('now'));
    END;

    CREATE TRIGGER IF NOT EXISTS s15_notes_del AFTER DELETE ON notes BEGIN
      INSERT INTO audit_log(action,note_id,meta,created_at)
      VALUES('note_delete', OLD.id, 'len='||length(OLD.body), datetime('now'));
    END;

    CREATE TRIGGER IF NOT EXISTS s15_tags_add AFTER INSERT ON note_tags BEGIN
      INSERT INTO audit_log(action,note_id,meta,created_at)
      SELECT 'tag_add', NEW.note_id, 'tag='||(SELECT name FROM tags WHERE id=NEW.tag_id), datetime('now'));
    END;

    CREATE TRIGGER IF NOT EXISTS s15_tags_del AFTER DELETE ON note_tags BEGIN
      INSERT INTO audit_log(action,note_id,meta,created_at)
      SELECT 'tag_remove', OLD.note_id, 'tag='||(SELECT name FROM tags WHERE id=OLD.tag_id), datetime('now'));
    END;
    """)
    conn.commit()

@app.on_event("startup")
def _s15_startup():
    try:
        c = get_conn(); _apply_scaffold_015(c)
    except Exception as e:
        print("scaffold_015 schema apply failed:", e)

def _series_last_n_days(conn, n=30):
    import datetime as _dt
    today = _dt.date.today()
    days = [(today - _dt.timedelta(days=i)) for i in range(n-1, -1, -1)]
    rows = conn.execute("SELECT date(created_at) d, COUNT(*) c FROM notes GROUP BY d").fetchall()
    m = {r["d"]: r["c"] for r in rows}
    return [(d.isoformat(), int(m.get(d.isoformat(), 0))) for d in days]

@app.get('/metrics/panel', response_class=HTMLResponse)
def metrics_panel(request: Request, days: int = 30):
    conn = get_conn()
    days = max(7, min(120, int(days or 30)))
    data = _series_last_n_days(conn, days)
    values = [c for _, c in data]
    mx = max(values) if values else 1
    w, h = 360, 64
    pts = []
    for i, v in enumerate(values):
        x = int(i * (w-8) / max(1, len(values)-1)) + 4
        y = int(h-6 - (0 if mx==0 else (v/mx)*(h-12)))
        pts.append(f"{x},{y}")
    total = sum(values); today = values[-1] if values else 0
    return templates.TemplateResponse('partials/metrics_panel.html', {'request':request, 'data':data, 'points':' '.join(pts), 'total': total, 'today': today, 'maxv': mx})

@app.get('/metrics/notes.csv')
def metrics_csv(days: int = 90):
    conn = get_conn()
    days = max(7, min(365, int(days or 90)))
    data = _series_last_n_days(conn, days)
    buf = _io.StringIO(); w = csv.writer(buf); w.writerow(['date','count'])
    for d,c in data: w.writerow([d,c])
    payload = buf.getvalue().encode('utf-8')
    return Response(content=payload, media_type='text/csv',
                    headers={'Content-Disposition': 'attachment; filename="notes_metrics.csv"'})

@app.get('/audit', response_class=HTMLResponse)
def audit_page(request: Request, limit: int = 200):
    conn = get_conn()
    rows = conn.execute("SELECT id, action, note_id, meta, created_at FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return templates.TemplateResponse('audit.html', {'request': request, 'rows': rows})

@app.post('/notes/{note_id}/tag/add', response_class=HTMLResponse)
def note_add_tag(request: Request, note_id: int, tag: str = Form(...)):
    conn = get_conn()
    nm = norm_tag(tag)
    if not nm: return Response(status_code=204)
    tid = ensure_tag(conn, nm)
    conn.execute('INSERT OR IGNORE INTO note_tags(note_id, tag_id) VALUES (?,?)', (note_id, tid))
    conn.commit()
    hdr = {'HX-Trigger': json.dumps({'toast': {'message': 'Added #' + nm}})}
    return Response(status_code=204, headers=hdr)

@app.post('/notes/{note_id}/tag/remove', response_class=HTMLResponse)
def note_remove_tag(request: Request, note_id: int, tag: str = Form(...)):
    conn = get_conn()
    nm = norm_tag(tag)
    row = conn.execute('SELECT id FROM tags WHERE name=?', (nm,)).fetchone()
    if row:
        conn.execute('DELETE FROM note_tags WHERE note_id=? AND tag_id=?', (note_id, row['id']))
        conn.commit()
        hdr = {'HX-Trigger': json.dumps({'toast': {'message': 'Removed #' + nm}})}
        return Response(status_code=204, headers=hdr)
    return Response(status_code=204)
PY
else
  echo "• app.py already contains scaffold_015 additions — skipping append"
fi

# 2) Templates
backup templates/partials/metrics_panel.html
cat > templates/partials/metrics_panel.html <<'HTML'
<div class="rounded-2xl bg-white/95 dark:bg-slate-950/90 shadow ring-1 ring-black/5 p-4">
  <div class="flex items-center justify-between">
    <h3 class="text-sm font-medium text-slate-900 dark:text-white">Last {{ data|length }} days</h3>
    <a class="text-xs hover:underline text-slate-600 dark:text-slate-300" href="/metrics/notes.csv">CSV</a>
  </div>
  <div class="mt-2 flex items-end gap-4">
    <svg width="360" height="64" viewBox="0 0 360 64" class="rounded bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
      <polyline fill="none" stroke="currentColor" stroke-width="2" points="{{ points }}"></polyline>
    </svg>
    <div class="text-sm text-slate-600 dark:text-slate-300">
      <div>Total: <b>{{ total }}</b></div>
      <div>Today: <b>{{ today }}</b></div>
      <div>Max/day: <b>{{ maxv }}</b></div>
    </div>
  </div>
</div>
HTML

backup templates/audit.html
cat > templates/audit.html <<'HTML'
{% extends "base.html" %}
{% block title %}Audit — Second Brain Premium{% endblock %}
{% block content %}
<div class="rounded-2xl bg-white/95 dark:bg-slate-950/90 shadow-xl ring-1 ring-black/5 p-6">
  <h1 class="text-xl font-semibold">Audit log</h1>
  <table class="mt-4 w-full text-sm">
    <thead><tr class="text-left border-b dark:border-slate-800">
      <th class="py-2">Time (UTC)</th><th>Action</th><th>Note</th><th>Meta</th>
    </tr></thead>
    <tbody>
      {% for r in rows %}
        <tr class="border-b dark:border-slate-800">
          <td class="py-2">{{ r.created_at }}</td>
          <td><span class="badge">{{ r.action }}</span></td>
          <td>{% if r.note_id %}<a class="hover:underline" href="/notes/{{ r.note_id }}">#{{ r.note_id }}</a>{% else %}—{% endif %}</td>
          <td><code class="text-xs">{{ r.meta }}</code></td>
        </tr>
      {% else %}
        <tr><td colspan="4" class="py-6 text-slate-500">No activity yet.</td></tr>
      {% endfor %}
    </tbody>
  </table>
</div>
{% endblock %}
HTML

backup templates/partials/note_toolbar.html
cat > templates/partials/note_toolbar.html <<'HTML'
<div class="sticky top-0 z-30 backdrop-blur bg-white/70 dark:bg-slate-950/60 border-b border-slate-200/70 dark:border-slate-800 px-3 py-2 flex items-center gap-2">
  <a href="/notes" class="px-2 py-1 rounded bg-slate-100 dark:bg-slate-900">← Notes</a>
  <button class="px-3 py-1.5 rounded bg-brand-600 text-white" 
          hx-get="/notes/{{ note.id }}/edit" 
          hx-target="#note-main" hx-swap="outerHTML">Edit</button>
  <button class="px-3 py-1.5 rounded bg-slate-200 dark:bg-slate-800" onclick="navigator.clipboard.writeText(window.location.href).then(()=>document.body.dispatchEvent(new CustomEvent('toast',{detail:{message:'Link copied'}})))">Copy link</button>
  <button class="px-3 py-1.5 rounded bg-slate-200 dark:bg-slate-800" onclick="(async()=>{const t=prompt('Add tag (e.g., research-notes)'); if(!t)return; await fetch('/notes/{{ note.id }}/tag/add',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'tag='+encodeURIComponent(t)}); location.reload();})()">Add tag</button>
  <form method="post" action="/notes/{{ note.id }}/delete" onsubmit="return confirm('Delete this note?')" class="inline">
    <button class="px-3 py-1.5 rounded bg-rose-600 text-white">Delete</button>
  </form>
  <div class="ml-auto text-xs text-slate-500">{{ note.created_at }}</div>
</div>
HTML

# 3) Small CSS tweak
if [[ -f static/brand.css ]] && ! grep -q "#note-main" static/brand.css; then
  backup static/brand.css
  cat >> static/brand.css <<'CSS'

/* scaffold_015 tweaks */
#note-main{ scroll-margin-top: 72px; }
CSS
fi

# 4) Optionally inject metrics panel into dashboard if a "Recent" section exists (best-effort, safe)
if [[ -f templates/dashboard.html ]] && ! grep -q "/metrics/panel" templates/dashboard.html; then
  backup templates/dashboard.html
  # Append a metrics panel loader near the end of the file to avoid brittle in-place rewriting
  cat >> templates/dashboard.html <<'HTML'

<!-- scaffold_015: metrics panel -->
<div id="metrics-panel" class="mt-4" hx-get="/metrics/panel" hx-trigger="load" hx-swap="outerHTML">
  <div class="rounded-2xl bg-white/95 dark:bg-slate-950/90 ring-1 ring-black/5 p-4 text-slate-500">Loading metrics…</div>
</div>
HTML
fi

echo "scaffold_015.sh applied successfully.

Notes:
- If you want the toolbar visible on note pages, include this in your note detail template inside the main content block:
    {% include 'partials/note_toolbar.html' %}
- New routes: /metrics/panel, /metrics/notes.csv, /audit, and POST /notes/{id}/tag/add|remove

Tip: run with bash ->  bash scripts/scaffold_015.sh
"