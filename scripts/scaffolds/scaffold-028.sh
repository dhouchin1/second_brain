#!/usr/bin/env bash
# scripts/scaffolds/scaffold_028.sh
# Large UI/UX refresh + add commonly expected APIs + exports + FTS rebuild, toasts, palette, tag suggest, dev login page
set -euo pipefail

STAMP="$(date +%Y%m%d-%H%M%S)"
bk(){ [[ -f "$1" ]] && { mkdir -p .bak; cp -p "$1" ".bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }
say(){ printf "\033[1;36m%s\033[0m\n" "$*"; }

mkdir -p static templates templates/partials

# 0) CSS/JS enhancements (extends earlier brand.css/ui.js if present)
touch static/brand.css
bk static/brand.css
cat >> static/brand.css <<'CSS'

/* ==== scaffold_028 UI upgrades ==== */
:root{ --brand:#6d28d9; --ink:#0b1220; --muted:#6b7280; --bg:#f8fafc; --panel:rgba(255,255,255,.96); }
body{ background:var(--bg); }
.container{ max-width:1100px; margin:0 auto; padding:1rem; }
.card{ background:var(--panel); border:1px solid #e5e7eb; border-radius:14px; box-shadow:0 6px 16px rgba(0,0,0,.05); padding:1rem; }
.toolbar{ display:flex; align-items:center; gap:.5rem; margin-bottom:1rem; }
.kbd{ font:inherit; border:1px solid #cbd5e1; border-bottom-width:2px; padding:.1rem .35rem; border-radius:.35rem; background:#f1f5f9; color:#334155; }
.badge{ display:inline-block; padding:.15rem .4rem; border-radius:.4rem; background:#eef2ff; color:#3730a3; border:1px solid #c7d2fe; font-size:.75rem;}
.grid{ display:grid; gap:1rem; }
.grid-2{ grid-template-columns: 1fr 320px; }
.note{ padding:.7rem; border-radius:.7rem; border:1px solid #e5e7eb; background:#fff;}
.note h3{ margin:0 0 .3rem 0; font-size:1rem;}
.tags{ display:flex; gap:.3rem; flex-wrap:wrap;}
.tag{ padding:.2rem .45rem; border:1px solid #e2e8f0; background:#f8fafc; border-radius:.5rem; font-size:.8rem;}
input.input, textarea.input{
  width:100%; padding:.6rem .7rem; border-radius:.6rem; border:1px solid #e2e8f0; background:#fff;
}
.btn{ padding:.48rem .8rem; border-radius:.6rem; border:1px solid #d1d5db; background:#111827; color:#fff; }
.btn.secondary{ background:#eef2ff; color:#3730a3; border-color:#c7d2fe; }
.link{ color:#1f2937; text-decoration:none; }
.link:hover{ text-decoration:underline; }

/* toasts (reuse) */
.sb-toast{ position:fixed; right:1rem; bottom:1rem; background:#111827; color:#fff; padding:.6rem .9rem; border-radius:.6rem; box-shadow:0 10px 18px rgba(0,0,0,.25); opacity:0; transform:translateY(10px); transition:.25s;}
.sb-toast.show{ opacity:1; transform:translateY(0); }

/* command palette */
#cmdk{ display:none; position:fixed; inset:0; background:rgba(0,0,0,.35); padding-top:10vh; }
.cmdk-panel{ margin:0 auto; width:min(760px,92vw); background:#fff; border-radius:12px; border:1px solid #e5e7eb; box-shadow:0 20px 60px rgba(0,0,0,.25); padding:14px;}
.cmdk-panel input{ width:100%; }
.tag-suggest{ display:flex; gap:.35rem; flex-wrap:wrap; margin-top:.28rem;}
.tag-pill{ padding:.25rem .45rem; border-radius:.5rem; border:1px solid #e2e8f0; background:#f8fafc; font-size:.85rem; cursor:pointer;}
CSS

touch static/ui.js
bk static/ui.js
cat >> static/ui.js <<'JS'
// ==== scaffold_028 ui helpers (extends prior) ====
(function(){
  window.sbToast = window.sbToast || function(msg){
    const n=document.createElement('div'); n.className='sb-toast'; n.textContent=msg||'Done';
    document.body.appendChild(n); setTimeout(()=>n.classList.add('show'),10);
    setTimeout(()=>n.classList.remove('show'),3500); setTimeout(()=>n.remove(),4100);
  };

  function openPalette(){
    const dlg=document.getElementById('cmdk'); if(!dlg) return; dlg.style.display='block';
    const inp=dlg.querySelector('input[name="q"]'); if(inp){ inp.value=''; inp.focus(); }
  }
  function closePalette(){ const dlg=document.getElementById('cmdk'); if(dlg) dlg.style.display='none'; }
  document.addEventListener('keydown', (e)=>{
    const k=e.key.toLowerCase();
    if((e.metaKey||e.ctrlKey) && k==='k'){ e.preventDefault(); openPalette(); }
    if(k==='escape'){ closePalette(); }
  });

  async function tagAutocomplete(inp){
    const val=inp.value.trim(); const box=inp.nextElementSibling;
    if(!box || !box.classList.contains('tag-suggest')) return;
    if(!val){ box.innerHTML=''; return; }
    try{
      const r=await fetch('/api/tags?q='+encodeURIComponent(val));
      const d=await r.json();
      box.innerHTML='';
      (d.tags||[]).forEach(t=>{
        const b=document.createElement('button'); b.type='button'; b.className='tag-pill'; b.textContent='#'+t;
        b.onclick=()=>{ inp.value=t; box.innerHTML=''; };
        box.appendChild(b);
      });
    }catch(e){}
  }
  document.addEventListener('input', (e)=>{ if(e.target.matches('input[data-tag-ac]')) tagAutocomplete(e.target); });

  window.sbOpenPalette=openPalette; window.sbClosePalette=closePalette;
})();
JS

# 1) New/overhauled templates (safe overwrite after backup)
bk templates/base.html
cat > templates/base.html <<'HTML'
<!doctype html>
<html>
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{% block title %}Second Brain{% endblock %}</title>
  <link rel="stylesheet" href="/static/brand.css">
</head>
<body>
  <header class="card" style="margin:12px; display:flex; align-items:center; gap:12px;">
    <a class="link" href="/" style="font-weight:700;">🧠 Second Brain</a>
    <form action="/" method="get" style="margin-left:auto; display:flex; gap:6px;">
      <input class="input" name="q" placeholder="Search… (⌘/Ctrl+K)">
      <button class="btn secondary" type="submit">Search</button>
    </form>
    <a class="btn secondary" href="/about">About</a>
    <a class="btn secondary" href="/exports">Exports</a>
  </header>

  {% include 'partials/cmdk.html' %}
  <script src="/static/ui.js"></script>

  <div class="container">
    {% block content %}{% endblock %}
  </div>
</body>
</html>
HTML

cat > templates/partials/cmdk.html <<'HTML'
<div id="cmdk">
  <div class="cmdk-panel">
    <form action="/" method="get">
      <input class="input" type="text" name="q" placeholder="Search notes…" autofocus>
    </form>
    <div class="text-xs" style="color:#6b7280; margin-top:.4rem;">Type query and press Enter. Esc to close.</div>
  </div>
</div>
HTML

# dashboard
bk templates/dashboard.html
cat > templates/dashboard.html <<'HTML'
{% extends "base.html" %}
{% block title %}Dashboard — Second Brain{% endblock %}
{% block content %}
<div class="toolbar">
  <form action="/" method="get" style="display:flex; gap:.5rem;">
    <input class="input" name="q" value="{{ q }}" placeholder="Search title/summary/tags/content…">
    <input class="input" name="tag" value="{{ tag }}" placeholder="Filter by tag (e.g. project-x)">
    <button class="btn" type="submit">Filter</button>
  </form>
</div>

{% if last_sync %}
<div class="card" style="margin-bottom:1rem;">
  <b>Last export sync:</b> {{ last_sync }}
</div>
{% endif %}

<div class="grid grid-2">
  <div>
    {% for day, items in notes_by_day.items() %}
      <div class="card" style="margin-bottom:1rem;">
        <div style="font-weight:600; margin-bottom:.4rem;">{{ day }}</div>
        <div class="grid" style="grid-template-columns: 1fr;">
          {% for n in items %}
            <div class="note">
              <a class="link" href="/detail/{{ n.id }}"><h3>{{ n.title or '(untitled)' }}</h3></a>
              <div class="tags">
                {% for t in (n.tags or '').split(',') if t %}
                  <span class="tag">#{{ t }}</span>
                {% endfor %}
              </div>
              <div style="color:#6b7280; font-size:.85rem; margin-top:.3rem;">{{ n.summary or (n.content[:140] ~ '…') }}</div>
              <div style="color:#94a3b8; font-size:.75rem; margin-top:.2rem;">{{ n.timestamp }}</div>
            </div>
          {% endfor %}
        </div>
      </div>
    {% else %}
      <div class="card">No notes yet.</div>
    {% endfor %}
  </div>

  <aside>
    <div class="card">
      <div style="font-weight:600; margin-bottom:.5rem;">Quick capture</div>
      <form action="/capture" method="post" enctype="multipart/form-data" onsubmit="sbToast('Capturing…')">
        <textarea class="input" name="note" placeholder="Paste or type…" rows="4"></textarea>
        <input class="input" data-tag-ac name="tags" placeholder="tags (comma separated)" style="margin-top:.5rem;">
        <div class="tag-suggest"></div>
        <div style="display:flex; gap:.6rem; margin-top:.6rem;">
          <input type="file" name="file">
          <button class="btn" type="submit">Save</button>
        </div>
      </form>
      <div style="margin-top:.6rem;">
        <form action="/dev/login" method="post" onsubmit="sbToast('Logging in…')">
          <input class="input" name="username" placeholder="dev username">
          <input class="input" name="password" type="password" placeholder="dev password" style="margin-top:.4rem;">
          <button class="btn secondary" type="submit" style="margin-top:.4rem;">Set dev cookie</button>
        </form>
        <div class="text-xs" style="color:#6b7280; margin-top:.3rem;">Dev cookie populates Authorization header via middleware.</div>
      </div>
    </div>
  </aside>
</div>
{% endblock %}
HTML

# detail
bk templates/detail.html
cat > templates/detail.html <<'HTML'
{% extends "base.html" %}
{% block title %}Note — Second Brain{% endblock %}
{% block content %}
<div class="grid grid-2">
  <div>
    <div class="card">
      <div style="display:flex; align-items:center; gap:.6rem; justify-content:space-between;">
        <h2 style="margin:0;">{{ note.title or '(untitled)' }}</h2>
        <form action="/delete/{{ note.id }}" method="post" onsubmit="return confirm('Delete this note?')">
          <button class="btn secondary">Delete</button>
        </form>
      </div>
      <div style="color:#64748b; font-size:.85rem; margin:.3rem 0;">{{ note.timestamp }} • type={{ note.type }}</div>
      {% if note.audio_filename %}
        <audio controls src="/audio/{{ note.audio_filename }}" style="width:100%; margin:.6rem 0;"></audio>
      {% endif %}
      <pre style="white-space:pre-wrap; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:.8rem;">{{ note.content }}</pre>

      <h3 style="margin-top:1rem;">Tags</h3>
      <div class="tags" style="margin-bottom:.4rem;">
        {% for t in (note.tags or '').split(',') if t %}
          <form method="post" action="/notes/{{ note.id }}/tag/remove" style="display:inline;">
            <input type="hidden" name="tag" value="{{ t }}">
            <button class="tag" title="Remove">#{{ t }} ×</button>
          </form>
        {% else %}
          <span class="badge">no tags</span>
        {% endfor %}
      </div>
      <form method="post" action="/notes/{{ note.id }}/tag/add" onsubmit="sbToast('Adding tag…')">
        <input class="input" data-tag-ac name="tag" placeholder="add tag">
        <div class="tag-suggest"></div>
        <button class="btn" style="margin-top:.4rem;">Add</button>
      </form>

      <h3 style="margin-top:1rem;">Summary</h3>
      <div class="card" style="background:#fff;">{{ note.summary or '—' }}</div>
    </div>
  </div>
  <aside>
    <div class="card">
      <div style="display:flex; align-items:center; justify-content:space-between;">
        <h3 style="margin:0;">Similar</h3>
        <div>
          <a class="link" href="/detail/{{ note.id }}/similar?mode=tags">tags</a> ·
          <a class="link" href="/detail/{{ note.id }}/similar?mode=fts">fts</a>
        </div>
      </div>
      <div id="similar">
        {% for it in related %}
          <div class="note" style="margin-top:.5rem;">
            <a class="link" href="/detail/{{ it.id }}"><b>{{ it.title or ('#' ~ it.id) }}</b></a>
          </div>
        {% else %}
          <div class="text-xs" style="color:#6b7280;">No related items.</div>
        {% endfor %}
      </div>
    </div>
    <div class="card" style="margin-top:1rem;">
      <a class="btn secondary" href="/edit/{{ note.id }}">Edit</a>
    </div>
  </aside>
</div>
{% endblock %}
HTML

# edit
bk templates/edit.html
cat > templates/edit.html <<'HTML'
{% extends "base.html" %}
{% block title %}Edit — Second Brain{% endblock %}
{% block content %}
<div class="card">
  <h2>Edit note</h2>
  <form method="post" action="/edit/{{ note.id }}" onsubmit="sbToast('Saving…')">
    <label>Content</label>
    <textarea class="input" name="content" rows="12">{{ note.content }}</textarea>
    <label style="margin-top:.5rem;">Tags (csv)</label>
    <input class="input" name="tags" value="{{ note.tags or '' }}">
    <button class="btn" style="margin-top:.6rem;">Save</button>
  </form>
</div>
{% endblock %}
HTML

# 2) Backfill common API routes (autocomplete, tag suggest, fts rebuild, export JSON/MD)
if ! grep -q "scaffold_028 additions (api backfill)" app.py 2>/dev/null; then
  bk app.py
  cat >> app.py <<'PY'

# ==== scaffold_028 additions (api backfill) ====
from fastapi.responses import JSONResponse
import io, zipfile, json as _json

@app.get("/api/tags")
def tags_autocomplete(q: str = "", current_user: User = Depends(get_current_user)):
    conn = get_conn(); c = conn.cursor()
    q = (q or "").strip().lower()
    rows = c.execute("SELECT tags FROM notes WHERE user_id = ? AND tags IS NOT NULL", (current_user.id,)).fetchall()
    conn.close()
    seen=set()
    for (csv,) in rows:
        for t in (csv or "").split(","):
            t=t.strip().lower()
            if q and q not in t: continue
            if t: seen.add(t)
    # return top 20 alphabetically
    return {"tags": sorted(seen)[:20]}

@app.post("/tags/suggest")
def tags_suggest(body: dict = Body(...), current_user: User = Depends(get_current_user)):
    text = body.get("text","")
    res = ollama_summarize(text)
    return {"tags": res.get("tags", [])}

@app.post("/fts/rebuild")
def fts_rebuild(current_user: User = Depends(get_current_user)):
    conn = get_conn(); c = conn.cursor()
    c.execute("DELETE FROM notes_fts WHERE rowid IN (SELECT id FROM notes WHERE user_id = ?)", (current_user.id,))
    rows = c.execute("SELECT id, title, summary, tags, actions, content FROM notes WHERE user_id = ?", (current_user.id,)).fetchall()
    c.executemany("INSERT INTO notes_fts(rowid, title, summary, tags, actions, content) VALUES (?,?,?,?,?,?)", rows)
    conn.commit(); conn.close()
    return {"ok": True, "count": len(rows)}

@app.get("/export/json")
def export_json(current_user: User = Depends(get_current_user)):
    conn = get_conn(); c = conn.cursor()
    rows = c.execute("SELECT id, title, summary, tags, actions, type, timestamp, audio_filename, content FROM notes WHERE user_id = ? ORDER BY timestamp DESC", (current_user.id,)).fetchall()
    notes=[{"id":r[0],"title":r[1],"summary":r[2],"tags":(r[3] or "").split(",") if r[3] else [],"actions":(r[4] or "").splitlines(),"type":r[5],"timestamp":r[6],"audio_filename":r[7],"content":r[8]} for r in rows]
    conn.close()
    return JSONResponse({"notes": notes})

@app.get("/export/markdown.zip")
def export_markdown_zip(current_user: User = Depends(get_current_user)):
    buf = io.BytesIO()
    zf = zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED)
    conn = get_conn(); c = conn.cursor()
    rows = c.execute("SELECT title, summary, tags, actions, type, timestamp, content FROM notes WHERE user_id = ?", (current_user.id,)).fetchall()
    for (title, summary, tags, actions, typ, ts, content) in rows:
        name = (ts or "note").replace(":","-").replace(" ","_")[:19] + "-" + (safe_filename(title or "note")) + ".md"
        fm = []
        fm.append("---")
        if title: fm.append(f"title: {title}")
        if ts: fm.append(f"timestamp: {ts}")
        if typ: fm.append(f"type: {typ}")
        if tags: fm.append(f"tags: [{tags}]")
        if actions: fm.append("actions: |"); [fm.append(f"  - {a}") for a in (actions or "").splitlines() if a.strip()]
        if summary: fm.append(f"summary: |"); [fm.append(f"  {line}") for line in (summary or "").splitlines()]
        fm.append("---")
        body = "\n".join(fm) + "\n\n" + (content or "")
        zf.writestr(name, body)
    zf.close()
    buf.seek(0)
    return FileResponse(buf, media_type="application/zip", filename="notes_markdown.zip")
PY
fi

# 3) Dev login page (optional convenience)
cat > templates/login.html <<'HTML'
{% extends "base.html" %}
{% block title %}Dev Login — Second Brain{% endblock %}
{% block content %}
<div class="card" style="max-width:520px; margin:3rem auto;">
  <h2>Developer Login Cookie</h2>
  <p class="text-sm" style="color:#6b7280;">Sets an <code>sb_token</code> cookie so you can browse HTML without a header extension (dev only).</p>
  <form action="/dev/login" method="post" onsubmit="sbToast('Logging in…')">
    <input class="input" name="username" placeholder="username">
    <input class="input" type="password" name="password" placeholder="password" style="margin-top:.4rem;">
    <button class="btn" style="margin-top:.6rem;">Set Cookie</button>
  </form>
</div>
{% endblock %}
HTML

echo "Done 028.

UI updated (base/dashboard/detail/edit), JS/CSS enhanced, and API backfills added:
  • /api/tags (autocomplete)   • POST /tags/suggest   • POST /fts/rebuild
  • GET /export/json           • GET /export/markdown.zip

Tip: ship it with PR automation:
  make ship S=scripts/scaffolds/scaffold_028.sh M=\"feat(ui): overhaul + api backfills\"
"
