#!/usr/bin/env bash
# scripts/scaffolds/scaffold_038_ui.sh
# New UI (Tailwind glass), extra routes (/dashboard2, /notes, /tags, /settings),
# Quick Capture -> /capture, toast + shortcuts. Non-destructive to existing UI.
set -euo pipefail

STAMP="$(date +%Y%m%d-%H%M%S)"
bk(){ [[ -f "$1" ]] && { mkdir -p .bak; cp -p "$1" ".bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }

mkdir -p templates templates/partials static/js

# 0) Ensure HTMLResponse import is present
if grep -q "from fastapi.responses import" app.py; then
  if ! grep -q "HTMLResponse" app.py; then
    bk app.py
    perl -0777 -pe "s/from fastapi\.responses import ([^\n]+)/from fastapi.responses import \1, HTMLResponse/g" -i app.py
  fi
else
  bk app.py
  printf "\nfrom fastapi.responses import HTMLResponse\n" >> app.py
fi

# 1) Base template (non-destructive): create base_glass.html; only create base.html if missing
if [[ ! -f templates/base_glass.html ]]; then
cat > templates/base_glass.html <<'HTML'
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{% if title %}{{ title }} — {% endif %}{{ settings.brand_name if settings and settings.brand_name else "LocalKeep" }}</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          colors: {
            sb: { 50:'#f4f6ff',100:'#e8ecff',200:'#cfd7ff',300:'#a9b8ff',400:'#7a8aff',500:'#5665ff',600:'#3b47e6',700:'#2e37b4',800:'#262f8a',900:'#232a6d' }
          },
          boxShadow: { soft:'0 10px 25px -10px rgba(0,0,0,0.25)' }
        }
      }
    }
  </script>
  <link rel="manifest" href="/static/manifest.webmanifest">
  <link rel="icon" href="/static/icons/icon-192.svg">
  <meta property="og:title" content="LocalKeep">
  <meta property="og:description" content="Local-first notes that think with you.">
  <meta property="og:image" content="/static/opengraph.svg">
  <meta name="theme-color" content="#6D28D9">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    html, body { height: 100%; }
    body { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  </style>
</head>
<body class="min-h-screen bg-gradient-to-br from-indigo-600 via-purple-600 to-fuchsia-600 text-slate-100">
  <div class="min-h-screen bg-black/10 backdrop-blur-sm">
    <header class="sticky top-0 z-40 bg-slate-900/60 backdrop-blur border-b border-white/10">
      <div class="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 flex items-center justify-between h-14">
        <a href="/dashboard2" class="flex items-center gap-2 font-semibold">
          <span class="text-2xl">🧠</span>
          <span class="hidden sm:block">{{ settings.brand_name if settings and settings.brand_name else "LocalKeep" }}</span>
        </a>
        <div class="flex items-center gap-2">
          <form action="/notes" method="get" class="hidden md:block">
            <input name="q" placeholder="Search notes, audio, tags…" class="w-96 rounded-xl bg-white/10 border border-white/10 px-4 py-2 text-sm placeholder:text-slate-300 focus:outline-none focus:ring-2 focus:ring-white/20" />
          </form>
          <a href="/notes#quick" class="rounded-xl bg-white/10 hover:bg-white/20 border border-white/10 px-3 py-2 text-sm">+ Quick Capture</a>
          <a href="/settings" class="rounded-xl bg-white/10 hover:bg-white/20 border border-white/10 px-3 py-2 text-sm">Settings</a>
        </div>
      </div>
    </header>

    <div class="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 grid grid-cols-1 lg:grid-cols-12 gap-6 py-6">
      <aside class="lg:col-span-3 xl:col-span-2">
        <nav class="rounded-2xl bg-slate-900/60 border border-white/10 shadow-soft p-2">
          <a href="/dashboard2" class="flex items-center gap-3 px-3 py-2 rounded-xl hover:bg-white/10 {{ 'bg-white/10' if active == 'dashboard' else '' }}">🏠 <span>Dashboard</span></a>
          <a href="/notes" class="flex items-center gap-3 px-3 py-2 rounded-xl hover:bg-white/10 {{ 'bg-white/10' if active == 'notes' else '' }}">📝 <span>Notes</span></a>
          <a href="/tags" class="flex items-center gap-3 px-3 py-2 rounded-xl hover:bg-white/10 {{ 'bg-white/10' if active == 'tags' else '' }}">🏷️ <span>Tags</span></a>
          <a href="/audio" class="flex items-center gap-3 px-3 py-2 rounded-xl hover:bg-white/10 {{ 'bg-white/10' if active == 'audio' else '' }}">🎙️ <span>Audio</span></a>
          <a href="/exports" class="flex items-center gap-3 px-3 py-2 rounded-xl hover:bg-white/10 {{ 'bg-white/10' if active == 'exports' else '' }}">⬇️ <span>Export</span></a>
        </nav>
      </aside>

      <main class="lg:col-span-9 xl:col-span-10">
        {% block content %}{% endblock %}
      </main>
    </div>
  </div>

  <script defer src="/static/js/ui.js"></script>
</body>
</html>
HTML
fi

# Create base.html if missing at all (for thanks/landing if needed)
if [[ ! -f templates/base.html ]]; then
  cp templates/base_glass.html templates/base.html
fi

# 2) Dashboard, Notes, Tags, Settings templates
cat > templates/dashboard_glass.html <<'HTML'
{% extends 'base_glass.html' %}
{% set active = 'dashboard' %}
{% set title = 'Dashboard' %}
{% block content %}
  <div class="grid md:grid-cols-2 xl:grid-cols-4 gap-4 mb-6">
    <div class="rounded-2xl bg-slate-900/60 border border-white/10 shadow-soft p-4">
      <div class="text-slate-300 text-sm">Notes</div>
      <div class="text-3xl font-semibold">{{ stats.notes }}</div>
    </div>
    <div class="rounded-2xl bg-slate-900/60 border border-white/10 shadow-soft p-4">
      <div class="text-slate-300 text-sm">Tags</div>
      <div class="text-3xl font-semibold">{{ stats.tags }}</div>
    </div>
    <div class="rounded-2xl bg-slate-900/60 border border-white/10 shadow-soft p-4">
      <div class="text-slate-300 text-sm">Audio Memos</div>
      <div class="text-3xl font-semibold">{{ stats.audio }}</div>
    </div>
    <div class="rounded-2xl bg-slate-900/60 border border-white/10 shadow-soft p-4">
      <div class="text-slate-300 text-sm">Storage (MB)</div>
      <div class="text-3xl font-semibold">{{ stats.storage_mb }}</div>
    </div>
  </div>

  <div class="grid lg:grid-cols-3 gap-6">
    <section class="lg:col-span-2 rounded-2xl bg-slate-900/60 border border-white/10 shadow-soft">
      <div class="p-4 border-b border-white/10 flex items-center justify-between">
        <h2 class="font-semibold">Quick Capture</h2>
        <span class="text-xs text-slate-300">Cmd/Ctrl+Enter to save</span>
      </div>
      <form id="quick-capture" class="p-4 space-y-3" method="post" action="/capture" enctype="multipart/form-data" data-toast="Capturing…">
        <textarea name="note" rows="4" placeholder="What's on your mind?" class="w-full rounded-xl bg-white/10 border border-white/10 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-white/20"></textarea>
        <input name="tags" placeholder="#tags (comma separated)" class="w-full rounded-xl bg-white/10 border border-white/10 px-4 py-2" />
        <div class="flex items-center gap-3">
          <input type="file" name="file" accept="audio/*" class="text-sm">
          <button class="rounded-xl bg-sb-500 hover:bg-sb-400 px-4 py-2 font-medium">Save Note</button>
        </div>
      </form>
    </section>

    <section class="rounded-2xl bg-slate-900/60 border border-white/10 shadow-soft">
      <div class="p-4 border-b border-white/10">
        <h2 class="font-semibold">Recent Notes</h2>
      </div>
      <ul class="divide-y divide-white/10">
        {% if recent and recent|length > 0 %}
          {% for n in recent %}
          <li>
            <a href="/detail/{{ n.id }}" class="block p-4 hover:bg-white/5">
              <div class="text-sm text-slate-300 line-clamp-2">{{ n.preview }}</div>
              <div class="mt-2 flex flex-wrap gap-2">
                {% for t in (n.tags or '').split(',') if t %}
                  <span class="text-xs bg-white/10 border border-white/10 rounded-full px-2 py-0.5">{{ t.strip() }}</span>
                {% endfor %}
              </div>
            </a>
          </li>
          {% endfor %}
        {% else %}
          <li class="p-4 text-sm text-slate-400">No notes yet. Start capturing your thoughts!</li>
        {% endif %}
      </ul>
    </section>
  </div>
{% endblock %}
HTML

cat > templates/notes.html <<'HTML'
{% extends 'base_glass.html' %}
{% set active = 'notes' %}
{% set title = 'Notes' %}
{% block content %}
  <div class="rounded-2xl bg-slate-900/60 border border-white/10 shadow-soft">
    <div class="p-4 border-b border-white/10 flex items-center gap-3">
      <form action="/notes" method="get" class="flex-1">
        <input name="q" value="{{ q or '' }}" placeholder="Search notes…" class="w-full rounded-xl bg-white/10 border border-white/10 px-4 py-2" />
      </form>
      <form action="/notes" method="get" class="flex items-center gap-2">
        <select name="tag" class="rounded-xl bg-white/10 border border-white/10 px-3 py-2">
          <option value="">All tags</option>
          {% for t in tags %}
          <option value="{{ t.name }}" {% if active_tag==t.name %}selected{% endif %}>#{{ t.name }} ({{ t.count }})</option>
          {% endfor %}
        </select>
        <button class="rounded-xl bg-white/10 border border-white/10 px-3 py-2">Filter</button>
      </form>
    </div>
    <div id="quick" class="p-4 border-b border-white/10">
      <form id="quick-capture2" method="post" action="/capture" enctype="multipart/form-data" data-toast="Capturing…">
        <textarea name="note" rows="3" placeholder="Quick note…" class="w-full rounded-xl bg-white/10 border border-white/10 px-4 py-2"></textarea>
        <div class="mt-2 flex items-center gap-2">
          <input name="tags" placeholder="#tags" class="rounded-xl bg-white/10 border border-white/10 px-3 py-2" />
          <button class="rounded-xl bg-sb-500 hover:bg-sb-400 px-4 py-2 font-medium">Add</button>
        </div>
      </form>
    </div>
    <ul class="divide-y divide-white/10">
      {% for n in notes %}
      <li class="p-4">
        <a href="/detail/{{ n.id }}" class="block">
          <div class="text-slate-100">{{ n.title or n.preview }}</div>
          <div class="mt-1 text-xs text-slate-400">{{ n.ts }}</div>
          <div class="mt-2 flex flex-wrap gap-2">
            {% for t in (n.tags or '').split(',') if t %}
              <span class="text-xs bg-white/10 border border-white/10 rounded-full px-2 py-0.5">{{ t.strip() }}</span>
            {% endfor %}
          </div>
        </a>
      </li>
      {% else %}
      <li class="p-4 text-sm text-slate-400">No notes found.</li>
      {% endfor %}
    </ul>
  </div>
{% endblock %}
HTML

cat > templates/tags.html <<'HTML'
{% extends 'base_glass.html' %}
{% set active = 'tags' %}
{% set title = 'Tags' %}
{% block content %}
  <div class="grid lg:grid-cols-3 gap-6">
    <section class="lg:col-span-2 rounded-2xl bg-slate-900/60 border border-white/10 shadow-soft">
      <div class="p-4 border-b border-white/10 flex items-center justify-between">
        <h2 class="font-semibold">All Tags</h2>
        <form method="post" action="/tags/rename" onsubmit="sbToast('Renaming…')" class="flex items-center gap-2">
          <input name="old" placeholder="old" class="rounded-xl bg-white/10 border border-white/10 px-3 py-2 text-sm" />
          <span>→</span>
          <input name="new" placeholder="new" class="rounded-xl bg-white/10 border border-white/10 px-3 py-2 text-sm" />
          <button class="rounded-xl bg-white/10 border border-white/10 px-3 py-2 text-sm">Rename</button>
        </form>
      </div>
      <div class="p-4 flex flex-wrap gap-2">
        {% for t in tags %}
          <a href="/notes?tag={{ t.name }}" class="px-3 py-1 rounded-full bg-white/10 border border-white/10 hover:bg-white/20 text-sm">#{{ t.name }} <span class="opacity-70">({{ t.count }})</span></a>
        {% endfor %}
      </div>
    </section>

    <section class="rounded-2xl bg-slate-900/60 border border-white/10 shadow-soft p-4">
      <h2 class="font-semibold mb-3">Tag Suggestions</h2>
      <ul class="space-y-2 text-sm text-slate-300">
        {% for s in suggestions %}
          <li class="flex items-center justify-between bg-white/5 rounded-xl px-3 py-2">
            <span>#{{ s }}</span>
            <form method="post" action="/tags/create">
              <input type="hidden" name="name" value="{{ s }}" />
              <button class="text-sb-300 hover:underline">Add</button>
            </form>
          </li>
        {% endfor %}
      </ul>
    </section>
  </div>
{% endblock %}
HTML

cat > templates/settings.html <<'HTML'
{% extends 'base_glass.html' %}
{% set active = 'settings' %}
{% set title = 'Settings' %}
{% block content %}
  <div class="rounded-2xl bg-slate-900/60 border border-white/10 shadow-soft p-6 space-y-6">
    <h2 class="text-xl font-semibold">Settings</h2>
    <form method="post" action="/settings" onsubmit="sbToast('Saving…')">
      <div class="grid md:grid-cols-2 gap-6">
        <label class="block">
          <span class="text-sm text-slate-300">AI Summarizer (on capture)</span>
          <select name="summarizer" class="mt-1 w-full rounded-xl bg-white/10 border border-white/10 px-3 py-2">
            <option value="on" {% if cfg.summarizer=='on' %}selected{% endif %}>On</option>
            <option value="off" {% if cfg.summarizer=='off' %}selected{% endif %}>Off</option>
          </select>
        </label>
        <label class="block">
          <span class="text-sm text-slate-300">Default Export Format</span>
          <select name="export_format" class="mt-1 w-full rounded-xl bg-white/10 border border-white/10 px-3 py-2">
            <option value="markdown" {% if cfg.export_format=='markdown' %}selected{% endif %}>Markdown</option>
            <option value="json" {% if cfg.export_format=='json' %}selected{% endif %}>JSON</option>
          </select>
        </label>
      </div>
      <div class="pt-4">
        <button class="rounded-xl bg-sb-500 hover:bg-sb-400 px-4 py-2 font-medium">Save</button>
      </div>
    </form>
  </div>
{% endblock %}
HTML

# 3) ui.js (toast + ctrl/cmd+enter)
cat > static/js/ui.js <<'JS'
(function () {
  // Toast
  window.sbToast = function (msg) {
    const t = document.createElement('div');
    t.textContent = msg || 'Working…';
    t.style.position='fixed'; t.style.right='12px'; t.style.bottom='12px';
    t.style.background='rgba(15,23,42,.92)'; t.style.border='1px solid rgba(255,255,255,.12)';
    t.style.color='#fff'; t.style.padding='10px 12px'; t.style.borderRadius='12px'; t.style.zIndex='9999';
    document.body.appendChild(t);
    setTimeout(()=>{ t.style.opacity='0'; t.style.transition='opacity .3s'; setTimeout(()=>t.remove(), 300); }, 1400);
  };
  // Submit on Ctrl/Cmd+Enter
  document.addEventListener('keydown', (e)=>{
    const el = document.activeElement;
    if (!el || el.tagName!=='TEXTAREA') return;
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      const form = el.closest('form');
      if (form) { form.submit(); }
    }
  });
  // data-toast on forms
  document.addEventListener('submit', (e)=>{
    const f=e.target; if(f.matches('form[data-toast]')){ sbToast(f.getAttribute('data-toast')||'Working…'); }
  });
})();
JS

# 4) Append new routes + helpers into app.py (idempotent)
if ! grep -q "scaffold_038_ui" app.py; then
  bk app.py
  cat >> app.py <<'PY'

# ==== scaffold_038_ui: helpers & routes (non-destructive) ====
from fastapi import Response

def _stats_and_recent(current_user_id:int):
    conn = get_conn(); c = conn.cursor()
    notes = c.execute("SELECT COUNT(*) FROM notes WHERE user_id=?", (current_user_id,)).fetchone()[0]
    audio = c.execute("SELECT COUNT(*) FROM notes WHERE user_id=? AND audio_filename IS NOT NULL", (current_user_id,)).fetchone()[0]
    rows = c.execute("SELECT tags FROM notes WHERE user_id=? AND tags IS NOT NULL", (current_user_id,)).fetchall()
    tag_set = {}
    for (csv,) in rows:
        for t in (csv or "").split(","):
            t=t.strip()
            if t: tag_set[t]=tag_set.get(t,0)+1
    tags_count = len(tag_set)
    storage_mb = 0.0
    try:
        total=0
        for (fn,) in c.execute("SELECT audio_filename FROM notes WHERE user_id=? AND audio_filename IS NOT NULL", (current_user_id,)).fetchall():
            p = (settings.audio_dir / fn)
            if p.exists(): total += p.stat().st_size
        storage_mb = round(total/1024/1024, 2)
    except Exception:
        pass
    recent_rows = c.execute("""
      SELECT id, title, tags, timestamp, substr(COALESCE(content,''),1,140)
      FROM notes
      WHERE user_id=?
      ORDER BY datetime(COALESCE(timestamp, datetime('now'))) DESC
      LIMIT 10
    """, (current_user_id,)).fetchall()
    conn.close()
    rec=[{"id":r[0], "title":r[1], "tags":r[2] or "", "ts":r[3], "preview":r[4]} for r in recent_rows]
    return {"notes": notes, "audio": audio, "tags": tags_count, "storage_mb": storage_mb}, rec

def _search_notes(current_user_id:int, q:str="", tag:str=""):
    conn = get_conn(); c = conn.cursor()
    rows=[]
    if q:
        rows = c.execute("""
          SELECT n.id, n.title, n.tags, n.timestamp, substr(COALESCE(n.content,''),1,160)
          FROM notes_fts f JOIN notes n ON n.id=f.rowid
          WHERE notes_fts MATCH ? AND n.user_id=?
          ORDER BY datetime(COALESCE(n.timestamp, datetime('now'))) DESC LIMIT 200
        """, (q, current_user_id)).fetchall()
    elif tag:
        rows = c.execute("""
          SELECT id, title, tags, timestamp, substr(COALESCE(content,''),1,160)
          FROM notes WHERE user_id=? AND tags LIKE ?
          ORDER BY datetime(COALESCE(timestamp, datetime('now'))) DESC LIMIT 200
        """, (current_user_id, f"%{tag}%")).fetchall()
    else:
        rows = c.execute("""
          SELECT id, title, tags, timestamp, substr(COALESCE(content,''),1,160)
          FROM notes WHERE user_id=?
          ORDER BY datetime(COALESCE(timestamp, datetime('now'))) DESC LIMIT 200
        """, (current_user_id,)).fetchall()
    conn.close()
    return [{"id":r[0], "title":r[1], "tags":r[2] or "", "ts":r[3], "preview":r[4]} for r in rows]

def _all_tags_with_counts(current_user_id:int):
    conn = get_conn(); c = conn.cursor()
    rows = c.execute("SELECT tags FROM notes WHERE user_id=? AND tags IS NOT NULL", (current_user_id,)).fetchall()
    m={}
    for (csv,) in rows:
        for t in (csv or "").split(","):
            t=t.strip()
            if t: m[t]=m.get(t,0)+1
    conn.close()
    return [{"name":k, "count":v} for k,v in sorted(m.items(), key=lambda kv: (-kv[1], kv[0]))]

def _suggest_tags(current_user_id:int):
    # simple: top 5 tags; later can call ollama for suggestions
    return [t["name"] for t in _all_tags_with_counts(current_user_id)[:5]]

@app.get("/dashboard2")
def dashboard2(request: Request, current_user: User = Depends(get_current_user)):
    stats, recent = _stats_and_recent(current_user.id)
    return templates.TemplateResponse("dashboard_glass.html", {"request": request, "stats": stats, "recent": recent, "settings": settings})

@app.get("/notes")
def notes_index(request: Request, q:str="", tag:str="", current_user: User = Depends(get_current_user)):
    items = _search_notes(current_user.id, q=q, tag=tag)
    tags = _all_tags_with_counts(current_user.id)
    return templates.TemplateResponse("notes.html", {"request": request, "notes": items, "tags": tags, "q": q, "active_tag": tag, "settings": settings})

@app.get("/tags")
def tags_page(request: Request, current_user: User = Depends(get_current_user)):
    tags = _all_tags_with_counts(current_user.id)
    sugg = _suggest_tags(current_user.id)
    return templates.TemplateResponse("tags.html", {"request": request, "tags": tags, "suggestions": sugg, "settings": settings})

@app.get("/settings")
def settings_get(request: Request, current_user: User = Depends(get_current_user)):
    # in-DB user settings (simple key/value)
    conn = get_conn(); c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS user_settings (user_id INTEGER, key TEXT, value TEXT, PRIMARY KEY(user_id,key))")
    def g(key, default): 
        row=c.execute("SELECT value FROM user_settings WHERE user_id=? AND key=?", (current_user.id, key)).fetchone()
        return row[0] if row else default
    cfg = {"summarizer": g("summarizer","on"), "export_format": g("export_format","markdown")}
    conn.close()
    return templates.TemplateResponse("settings.html", {"request": request, "cfg": cfg, "settings": settings})

@app.post("/settings")
def settings_post(request: Request, export_format:str=Form("markdown"), summarizer:str=Form("on"), current_user: User = Depends(get_current_user)):
    conn = get_conn(); c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO user_settings(user_id,key,value) VALUES (?,?,?)", (current_user.id,"export_format",export_format))
    c.execute("INSERT OR REPLACE INTO user_settings(user_id,key,value) VALUES (?,?,?)", (current_user.id,"summarizer",summarizer))
    conn.commit(); conn.close()
    if "application/json" in request.headers.get("accept",""):
        return {"ok": True}
    return RedirectResponse("/settings", status_code=303)
PY
fi

echo "✓ scaffold_038_ui complete."
echo "Try the new UI at: http://localhost:8084/dashboard2"