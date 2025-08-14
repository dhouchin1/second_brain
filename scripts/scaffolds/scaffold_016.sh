#!/usr/bin/env bash
# scripts/scaffold_016.sh
set -euo pipefail
STAMP="$(date +%Y%m%d-%H%M%S)"
b(){ [[ -f "$1" ]] && { mkdir -p .bak; cp -p "$1" ".bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }

mkdir -p templates/partials out/exports

# 0) Ensure FileResponse import exists
if ! grep -q "from fastapi.responses import .*FileResponse" app.py; then
  b app.py
  awk 'BEGIN{done=0}
       {print}
       /from fastapi.responses import/ && !done {
         if($0 !~ /FileResponse/){ sub(/\)$/, ", FileResponse)"); }
         done=1
       }' app.py > app.py.new || true
  if [[ -s app.py.new ]]; then mv app.py.new app.py; fi
  # If pattern didn’t match (older app.py), just append a safe import
  if ! grep -q "FileResponse" app.py; then
    echo "from fastapi.responses import FileResponse" >> app.py
  fi
fi

# 1) App: schema (export_archives), exports endpoints, audit filters, note tags partial, similar notes endpoint
b app.py
awk '1; END{
print "\n# ==== scaffold_016 additions (exports, audit filters, inline tags, similar sidebar) ===="
print "import datetime as _dt, csv, os, pathlib"
print ""
print "def _apply_scaffold_016(conn):"
print "    conn.executescript(\"\"\""
print "    CREATE TABLE IF NOT EXISTS export_archives ("
print "      id INTEGER PRIMARY KEY AUTOINCREMENT,"
print "      kind TEXT NOT NULL," 
print "      file_path TEXT NOT NULL,"
print "      period_start TEXT NOT NULL,"
print "      period_end TEXT NOT NULL,"
print "      created_at TEXT NOT NULL"
print "    );"
print "    \"\"\"); conn.commit()"
print ""
print "@app.on_event('startup')"
print "def _s16_startup():"
print "    try:"
print "        c = get_conn(); _apply_scaffold_016(c)"
print "    except Exception as e:"
print "        print('scaffold_016 schema apply failed:', e)"
print ""
print "def _export_notes_csv(conn, start_iso: str, end_iso: str, kind: str) -> str:"
print "    p = BASE_DIR / 'out' / 'exports'; p.mkdir(parents=True, exist_ok=True)"
print "    name = f\"notes_{kind}_{start_iso}_to_{end_iso}.csv\""
print "    fpath = p / name"
print "    rows = conn.execute(\"\"\""
print "      SELECT n.id, n.created_at, replace(replace(n.body, char(13), ' '), char(10), ' ') AS body,"
print "             COALESCE((SELECT GROUP_CONCAT(t.name) FROM note_tags nt JOIN tags t ON t.id=nt.tag_id WHERE nt.note_id=n.id),'') AS tags"
print "      FROM notes n WHERE date(n.created_at) BETWEEN date(?) AND date(?)"
print "      ORDER BY datetime(n.created_at) ASC"
print "    \"\"\", (start_iso, end_iso)).fetchall()"
print "    with open(fpath, 'w', newline='', encoding='utf-8') as f:"
print "        w = csv.writer(f); w.writerow(['id','created_at','tags','body']);"
print "        for r in rows: w.writerow([r['id'], r['created_at'], r['tags'], r['body']])"
print "    conn.execute(\"INSERT INTO export_archives(kind, file_path, period_start, period_end, created_at) VALUES (?,?,?,?,?)\","
print "                 (kind, str(fpath), start_iso, end_iso, _dt.datetime.utcnow().isoformat(timespec='seconds')))"
print "    conn.commit()"
print "    return str(fpath)"
print ""
print "def _week_range(today: _dt.date) -> tuple[str,str]:"
print "    # Monday..Sunday ISO week"
print "    start = today - _dt.timedelta(days=today.weekday())"
print "    end = start + _dt.timedelta(days=6)"
print "    return start.isoformat(), end.isoformat()"
print ""
print "@app.get('/exports', response_class=HTMLResponse)"
print "def exports_page(request: Request):"
print "    conn = get_conn()"
print "    rows = conn.execute(\"SELECT id, kind, file_path, period_start, period_end, created_at FROM export_archives ORDER BY id DESC\").fetchall()"
print "    return templates.TemplateResponse('exports.html', {'request': request, 'rows': rows})"
print ""
print "@app.post('/exports/run', response_class=RedirectResponse)"
print "def exports_run(kind: str = Form('daily'), start: str = Form(''), end: str = Form('')):"
print "    kind = (kind or 'daily').lower()"
print "    conn = get_conn()"
print "    today = _dt.date.today()"
print "    if start and end:"
print "        s, e = start, end"
print "    elif kind == 'weekly':"
print "        s, e = _week_range(today)"
print "    else:"
print "        # daily: today"
print "        s = e = today.isoformat()"
print "    _export_notes_csv(conn, s, e, kind)"
print "    return RedirectResponse('/exports', status_code=303)"
print ""
print "@app.get('/exports/download/{id}')"
print "def exports_download(id: int):"
print "    conn = get_conn()"
print "    row = conn.execute(\"SELECT file_path FROM export_archives WHERE id=?\", (id,)).fetchone()"
print "    if not row: return Response(status_code=404)"
print "    path = row['file_path']"
print "    if not os.path.exists(path): return Response(status_code=410)"
print "    fn = os.path.basename(path)"
print "    return FileResponse(path, filename=fn, media_type='text/csv')"
print ""
print "# --- Audit filters ---"
print "@app.get('/audit', response_class=HTMLResponse)"
print "def audit_page(request: Request, action: str = '', tag: str = '', date_from: str = '', date_to: str = '', limit: int = 200):"
print "    conn = get_conn()"
print "    where = []; params = []"
print "    if action: where.append('action=?'); params.append(action)"
print "    if date_from: where.append(\"date(created_at) >= date(?)\"); params.append(date_from)"
print "    if date_to:   where.append(\"date(created_at) <= date(?)\"); params.append(date_to)"
print "    # tag filter: match tag in meta (for tag_add/remove) OR current note tags"
print "    if tag:"
print "        t = tag.lower();"
print "        where.append(\"( (action IN ('tag_add','tag_remove') AND lower(meta) LIKE ?) OR EXISTS (SELECT 1 FROM note_tags nt JOIN tags t ON t.id=nt.tag_id WHERE nt.note_id=a.note_id AND t.name=?))\")"
print "        params.extend([f'%tag=' + t + '%', t])"
print "    sql = \"SELECT a.id, a.action, a.note_id, a.meta, a.created_at FROM audit_log a\""
print "    if where: sql += \" WHERE \" + \" AND \".join(where)"
print "    sql += \" ORDER BY a.id DESC LIMIT ?\"; params.append(limit)"
print "    rows = conn.execute(sql, params).fetchall()"
print "    # Distinct actions for filter UI"
print "    acts = [r['action'] for r in conn.execute(\"SELECT DISTINCT action FROM audit_log ORDER BY action\").fetchall()]"
print "    return templates.TemplateResponse('audit.html', {'request': request, 'rows': rows, 'filters': {'action': action, 'tag': tag, 'date_from': date_from, 'date_to': date_to}, 'actions': acts})"
print ""
print "# --- Inline tag chips (partial) ---"
print "def _note_tags(conn, note_id: int):"
print "    return [r['name'] for r in conn.execute(\"\"\""
print "        SELECT t.name FROM tags t JOIN note_tags nt ON nt.tag_id=t.id WHERE nt.note_id=? ORDER BY t.name"
print "    \"\"\", (note_id,)).fetchall()]"
print ""
print "@app.get('/notes/{note_id}/tags/partial', response_class=HTMLResponse)"
print "def note_tags_partial(request: Request, note_id: int):"
print "    conn = get_conn(); tags = _note_tags(conn, note_id)"
print "    return templates.TemplateResponse('partials/note_tags.html', {'request': request, 'note_id': note_id, 'tags': tags})"
print ""
print "@app.post('/notes/{note_id}/tag/add', response_class=HTMLResponse)"
print "def note_add_tag_htmx(request: Request, note_id: int, tag: str = Form(...)):"
print "    conn = get_conn()"
print "    nm = norm_tag(tag)"
print "    if nm:"
print "        tid = ensure_tag(conn, nm)"
print "        conn.execute('INSERT OR IGNORE INTO note_tags(note_id, tag_id) VALUES (?,?)', (note_id, tid)); conn.commit()"
print "    # return updated tag chips if HTMX"
print "    if request.headers.get('HX-Request'):"
print "        return note_tags_partial(request, note_id)"
print "    return RedirectResponse(f'/notes/{note_id}', status_code=303)"
print ""
print "@app.post('/notes/{note_id}/tag/remove', response_class=HTMLResponse)"
print "def note_remove_tag_htmx(request: Request, note_id: int, tag: str = Form(...)):"
print "    conn = get_conn()"
print "    nm = norm_tag(tag)"
print "    row = conn.execute('SELECT id FROM tags WHERE name=?', (nm,)).fetchone()"
print "    if row:"
print "        conn.execute('DELETE FROM note_tags WHERE note_id=? AND tag_id=?', (note_id, row['id'])); conn.commit()"
print "    if request.headers.get('HX-Request'):"
print "        return note_tags_partial(request, note_id)"
print "    return RedirectResponse(f'/notes/{note_id}', status_code=303)"
print ""
print "# --- Similar notes partial (semantic vs tags overlap) ---"
print "@app.get('/notes/{note_id}/similar', response_class=HTMLResponse)"
print "def note_similar(request: Request, note_id: int, mode: str = 'embed'):"
print "    conn = get_conn()"
print "    items = []"
print "    if mode == 'embed':"
print "        items = related_notes_semantic(conn, note_id, limit=6)"
print "    else:"
print "        items = conn.execute(\"\"\""
print "          WITH note_tags_set AS ("
print "            SELECT t.id AS tag_id FROM note_tags nt JOIN tags t ON t.id=nt.tag_id WHERE nt.note_id=?"
print "          )"
print "          SELECT n.id, n.body, n.created_at, COUNT(*) AS overlap"
print "          FROM notes n"
print "          JOIN note_tags nt ON nt.note_id = n.id"
print "          WHERE n.id != ? AND nt.tag_id IN (SELECT tag_id FROM note_tags_set)"
print "          GROUP BY n.id"
print "          ORDER BY overlap DESC, datetime(n.created_at) DESC"
print "          LIMIT 6"
print "        \"\"\", (note_id, note_id)).fetchall()"
print "    tagmap = map_note_tags(conn, items)"
print "    notes = [dict(r) | {'tags': tagmap.get(r['id'], '')} for r in items]"
print "    return templates.TemplateResponse('partials/similar_list.html', {'request': request, 'notes': notes})"
}' app.py > app.py.new && mv app.py.new app.py

# 2) Templates: exports page
b templates/exports.html
cat > templates/exports.html <<'HTML'
{% extends "base.html" %}
{% block title %}Exports — Second Brain Premium{% endblock %}
{% block content %}
<div class="rounded-2xl bg-white/95 dark:bg-slate-950/90 shadow-xl ring-1 ring-black/5 p-6">
  <h1 class="text-xl font-semibold">CSV Exports</h1>

  <form class="mt-4 flex flex-wrap items-end gap-3" action="/exports/run" method="post">
    <div>
      <label class="text-sm block mb-1">Kind</label>
      <select name="kind" class="rounded-xl border-slate-300 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 px-3 py-2">
        <option value="daily">Daily (today)</option>
        <option value="weekly">Weekly (ISO week)</option>
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
    <button class="px-3 py-2 rounded-xl bg-brand-600 text-white">Generate</button>
    <div class="text-xs text-slate-500">Tip: cron <code>curl -XPOST http://…/exports/run -d kind=daily</code></div>
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

# 3) Audit page with filters (replace existing file with filter UI)
b templates/audit.html
cat > templates/audit.html <<'HTML'
{% extends "base.html" %}
{% block title %}Audit — Second Brain Premium{% endblock %}
{% block content %}
<div class="rounded-2xl bg-white/95 dark:bg-slate-950/90 shadow-xl ring-1 ring-black/5 p-6">
  <h1 class="text-xl font-semibold">Audit log</h1>

  <form class="mt-4 grid grid-cols-1 md:grid-cols-5 gap-3" method="get" action="/audit">
    <div>
      <label class="block text-sm mb-1">Action</label>
      <select name="action" class="w-full rounded-xl border-slate-300 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 px-3 py-2">
        <option value="">(any)</option>
        {% for a in actions %}
          <option value="{{ a }}" {% if filters.action==a %}selected{% endif %}>{{ a }}</option>
        {% endfor %}
      </select>
    </div>
    <div>
      <label class="block text-sm mb-1">Tag</label>
      <input name="tag" value="{{ filters.tag or '' }}" placeholder="e.g., research-notes" class="w-full rounded-xl border-slate-300 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 px-3 py-2">
    </div>
    <div>
      <label class="block text-sm mb-1">From</label>
      <input name="date_from" type="date" value="{{ filters.date_from or '' }}" class="w-full rounded-xl border-slate-300 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 px-3 py-2">
    </div>
    <div>
      <label class="block text-sm mb-1">To</label>
      <input name="date_to" type="date" value="{{ filters.date_to or '' }}" class="w-full rounded-xl border-slate-300 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 px-3 py-2">
    </div>
    <div class="flex items-end">
      <button class="px-3 py-2 rounded-xl bg-brand-600 text-white">Filter</button>
    </div>
  </form>

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

# 4) Note tag chips partial (HTMX, with remove)
b templates/partials/note_tags.html
cat > templates/partials/note_tags.html <<'HTML'
<div id="note-tags" class="mt-3 flex flex-wrap gap-2">
  {% for t in tags %}
    <form hx-post="/notes/{{ note_id }}/tag/remove" hx-target="#note-tags" hx-swap="outerHTML" class="inline">
      <input type="hidden" name="tag" value="{{ t }}">
      <button class="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-sm" title="Remove #{{ t }}">
        <span>#{{ t }}</span><span aria-hidden="true">×</span>
      </button>
    </form>
  {% else %}
    <span class="text-sm text-slate-500">No tags yet.</span>
  {% endfor %}
</div>
HTML

# 5) Similar list partial
b templates/partials/similar_list.html
cat > templates/partials/similar_list.html <<'HTML'
<ul class="mt-2 space-y-2">
  {% for n in notes %}
    <li class="p-3 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
      <a class="font-medium hover:underline text-slate-900 dark:text-white" href="/notes/{{ n.id }}">
        {{ (n.body[:100] ~ ('…' if n.body|length > 100 else '')) | e }}
      </a>
      {% if n.tags %}
        <div class="mt-1 text-xs text-slate-500">
          {% for t in n.tags.split(',') if t.strip() %}<span class="badge">#{{ t.strip() }}</span>{% endfor %}
        </div>
      {% endif %}
    </li>
  {% else %}
    <li class="text-slate-500">Nothing similar yet.</li>
  {% endfor %}
</ul>
HTML

# 6) Patch note_detail.html to load tag chips + similar sidebar (if not present)
if [[ -f templates/note_detail.html ]]; then
  if ! grep -q "id=\"note-tags\"" templates/note_detail.html; then
    b templates/note_detail.html
    awk '
      BEGIN{doneTags=0; doneSimilar=0}
      {print}
      /<\/h1>|<h1/ && !doneTags {
        print "<!-- Inline tag chips (HTMX) -->"
        print "<div hx-get=\"/notes/{{ note.id }}/tags/partial\" hx-trigger=\"load\" hx-target=\"this\" hx-swap=\"outerHTML\">"
        print "  <div class=\"text-sm text-slate-500\">Loading tags…</div>"
        print "</div>"
        doneTags=1
      }
      /<\/main>|<\/section>/ && !doneSimilar {
        print "<aside class=\"mt-8\">"
        print "  <div class=\"flex items-center gap-2\">"
        print "    <div class=\"text-sm font-medium\">Similar notes</div>"
        print "    <button class=\"text-xs px-2 py-1 rounded bg-slate-200 dark:bg-slate-800\""
        print "            hx-get=\"/notes/{{ note.id }}/similar?mode=embed\" hx-target=\"#similar-list\" hx-swap=\"innerHTML\">Semantic</button>"
        print "    <button class=\"text-xs px-2 py-1 rounded bg-slate-200 dark:bg-slate-800\""
        print "            hx-get=\"/notes/{{ note.id }}/similar?mode=tags\" hx-target=\"#similar-list\" hx-swap=\"innerHTML\">Tags overlap</button>"
        print "  </div>"
        print "  <div id=\"similar-list\" class=\"mt-2\" hx-get=\"/notes/{{ note.id }}/similar?mode=embed\" hx-trigger=\"load\" hx-swap=\"innerHTML\"></div>"
        print "</aside>"
        doneSimilar=1
      }
    ' templates/note_detail.html > templates/note_detail.html.new && mv templates/note_detail.html.new templates/note_detail.html
  fi
fi

# 7) Add nav link to /exports and /audit if missing
if [[ -f templates/base.html ]]; then
  if ! grep -q 'href="/exports"' templates/base.html; then
    b templates/base.html
    awk '1; /href="\/export"/ && !x {print "        <li><a class=\"hover:text-slate-900 dark:hover:text-white\" href=\"/exports\">Exports</a></li>"; x=1}' templates/base.html > templates/base.html.new && mv templates/base.html.new templates/base.html
  fi
  if ! grep -q 'href="/audit"' templates/base.html; then
    b templates/base.html
    awk '1; /<\/ul>/ && !x {print "        <li><a class=\"hover:text-slate-900 dark:hover:text-white\" href=\"/audit\">Audit</a></li>"; x=1}' templates/base.html > templates/base.html.new && mv templates/base.html.new templates/base.html
  fi
fi

# 8) requirements (no new deps)
[[ -f requirements.txt ]] || cat > requirements.txt <<'REQ'
fastapi==0.115.0
uvicorn[standard]==0.30.6
jinja2==3.1.4
httpx==0.27.2
numpy>=1.24,<3
python-multipart==0.0.9
REQ

echo "Done.

New routes:
  • /exports                — list & generate CSV archives
  • /exports/run            — POST (kind=daily|weekly[, start, end])
  • /exports/download/{id}  — download CSV
  • /audit (filters)        — action / tag / date range
  • /notes/{id}/tags/partial — HTMX partial for inline tag chips
  • /notes/{id}/similar?mode=embed|tags — Similar notes sidebar

UI:
  • Note pages now show removable tag chips
  • Sidebar toggle to switch Similar mode (Semantic vs Tag overlap)

Automation:
  • Use cron or a GitHub Action to hit /exports/run (e.g., daily). Example:
      */15 * * * * curl -fsS -XPOST http://localhost:8084/exports/run -d kind=daily >/dev/null

Restart:
  source .venv/bin/activate || python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  uvicorn app:app --reload --host 0.0.0.0 --port 8084
"