#!/usr/bin/env bash
# scripts/scaffolds/scaffold_036.sh
# Bundle: stats/recent/search APIs + sidebar/widgets, /audio & /exports pages,
# toasts + tag rename, Discord control commands.
set -euo pipefail

STAMP="$(date +%Y%m%d-%H%M%S)"
bk(){ [[ -f "$1" ]] && { mkdir -p .bak; cp -p "$1" ".bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }

mkdir -p templates templates/partials static scripts

# 0) FastAPI import fixes (HTMLResponse, JSONResponse)
if grep -q "from fastapi.responses import" app.py; then
  if ! grep -q "HTMLResponse" app.py; then
    bk app.py
    perl -0777 -pe "s/from fastapi\.responses import ([^\n]+)/from fastapi.responses import \1, HTMLResponse/g" -i app.py
    echo "• ensured HTMLResponse import"
  fi
  if ! grep -q "JSONResponse" app.py; then
    bk app.py
    perl -0777 -pe "s/from fastapi\.responses import ([^\n]+)/from fastapi.responses import \1, JSONResponse/g" -i app.py
    echo "• ensured JSONResponse import"
  fi
else
  bk app.py
  printf "\nfrom fastapi.responses import HTMLResponse, JSONResponse\n" >> app.py
fi

# 1) APIs: /api/stats, /api/recent, /api/search
if ! grep -q "scaffold_036 additions (stats,recent,search)" app.py 2>/dev/null; then
  bk app.py
  cat >> app.py <<'PY'

# ==== scaffold_036 additions (stats,recent,search) ====
from collections import Counter

@app.get("/api/stats")
def api_stats(current_user: User = Depends(get_current_user)):
    conn = get_conn(); c = conn.cursor()
    total = c.execute("SELECT COUNT(*) FROM notes WHERE user_id=?", (current_user.id,)).fetchone()[0]
    by_type_rows = c.execute("SELECT COALESCE(type,'note') as t, COUNT(*) FROM notes WHERE user_id=? GROUP BY t", (current_user.id,)).fetchall()
    by_type = {k:v for k,v in by_type_rows}
    days = c.execute("""
      SELECT date(COALESCE(timestamp, datetime('now'))) d, COUNT(*)
      FROM notes WHERE user_id=?
      GROUP BY d ORDER BY d DESC LIMIT 7
    """, (current_user.id,)).fetchall()
    rows = c.execute("SELECT tags FROM notes WHERE user_id=? AND tags IS NOT NULL", (current_user.id,)).fetchall()
    tag_counter = Counter()
    for (csv,) in rows:
        for t in (csv or "").split(","):
            t=t.strip().lower()
            if t: tag_counter[t]+=1
    top_tags = [{"tag": t, "count": n} for t,n in tag_counter.most_common(10)]
    conn.close()
    return {"total": total, "by_type": by_type, "last_days": [{"day": d, "count": n} for d,n in days], "top_tags": top_tags}

@app.get("/api/recent")
def api_recent(limit: int = 10, current_user: User = Depends(get_current_user)):
    limit = max(1, min(50, limit))
    conn = get_conn(); c = conn.cursor()
    rows = c.execute("""
      SELECT id, title, COALESCE(type,'note'), timestamp, substr(content,1,120)
      FROM notes WHERE user_id=? ORDER BY datetime(COALESCE(timestamp,datetime('now'))) DESC LIMIT ?
    """, (current_user.id, limit)).fetchall()
    conn.close()
    return {"items": [{"id": r[0], "title": r[1] or "(untitled)", "type": r[2], "ts": r[3], "preview": r[4]} for r in rows]}

@app.get("/api/search")
def api_search(q: str = "", current_user: User = Depends(get_current_user)):
    q = (q or "").strip()
    if not q:
        return {"items": []}
    conn = get_conn(); c = conn.cursor()
    rows = c.execute("""
      SELECT n.id, n.title, n.timestamp
      FROM notes_fts fts JOIN notes n ON n.id = fts.rowid
      WHERE notes_fts MATCH ? AND n.user_id = ?
      ORDER BY n.timestamp DESC LIMIT 25
    """, (q, current_user.id)).fetchall()
    conn.close()
    return {"items": [{"id": r[0], "title": r[1] or "(untitled)", "ts": r[2]} for r in rows]}
PY
fi

# 2) Ensure exports APIs exist (fallback, in case they weren't scaffolded earlier)
if ! grep -q "def export_json" app.py; then
  bk app.py
  cat >> app.py <<'PY'

# ==== scaffold_036 additions (exports fallback) ====
import io, zipfile

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
        if summary: fm.append("summary: |"); [fm.append(f"  {line}") for line in (summary or "").splitlines()]
        fm.append("---")
        body = "\n".join(fm) + "\n\n" + (content or "")
        zf.writestr(name, body)
    zf.close()
    buf.seek(0)
    return FileResponse(buf, media_type="application/zip", filename="notes_markdown.zip")
PY
fi

# 3) Tag rename endpoint + toasts support
if ! grep -q "scaffold_036 additions (tag rename)" app.py 2>/dev/null; then
  bk app.py
  cat >> app.py <<'PY'

# ==== scaffold_036 additions (tag rename) ====
@app.post("/tags/rename")
def tags_rename(
    request: Request,
    old: str = Form(...),
    new: str = Form(...),
    current_user: User = Depends(get_current_user),
):
    old = (old or "").strip().lower()
    new = (new or "").strip().lower()
    if not old or not new or old == new:
        raise HTTPException(status_code=400, detail="Invalid rename")
    conn = get_conn(); c = conn.cursor()
    rows = c.execute("SELECT id, tags FROM notes WHERE user_id=? AND tags LIKE ?", (current_user.id, f"%{old}%")).fetchall()
    changed = 0
    for nid, csv in rows:
        tags = [t.strip().lower() for t in (csv or "").split(",") if t.strip()]
        if old in tags:
            tags = [new if t==old else t for t in tags]
            seen=set(); dedup=[]
            for t in tags:
                if t not in seen:
                    seen.add(t); dedup.append(t)
            new_csv = ",".join(dedup)
            c.execute("UPDATE notes SET tags=? WHERE id=? AND user_id=?", (new_csv, nid, current_user.id))
            r = c.execute("SELECT title, summary, actions, content FROM notes WHERE id=?", (nid,)).fetchone()
            if r:
                title, summary, actions, content = r
                c.execute("DELETE FROM notes_fts WHERE rowid=?", (nid,))
                c.execute("INSERT INTO notes_fts(rowid, title, summary, tags, actions, content) VALUES (?,?,?,?,?,?)",
                          (nid, title, summary, new_csv, actions, content))
            changed += 1
    conn.commit(); conn.close()
    if "application/json" in request.headers.get("accept",""):
        return {"ok": True, "changed": changed}
    return RedirectResponse("/", status_code=302)
PY
fi

# 4) Pages: /audio, /exports
if ! grep -q "scaffold_036 additions (pages audio,exports)" app.py 2>/dev/null; then
  bk app.py
  cat >> app.py <<'PY'

# ==== scaffold_036 additions (pages audio,exports) ====
@app.get("/audio")
def audio_page(request: Request, current_user: User = Depends(get_current_user)):
    conn = get_conn(); c = conn.cursor()
    rows = c.execute("""
      SELECT id, title, timestamp, audio_filename, summary
      FROM notes WHERE user_id=? AND audio_filename IS NOT NULL
      ORDER BY datetime(COALESCE(timestamp,datetime('now'))) DESC LIMIT 200
    """, (current_user.id,)).fetchall()
    conn.close()
    items = [{"id":r[0], "title":r[1] or "(audio)", "ts":r[2], "file":r[3], "summary":r[4]} for r in rows]
    return templates.TemplateResponse("audio.html", {"request": request, "items": items})

@app.get("/exports")
def exports_page(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse("exports.html", {"request": request})
PY
fi

# 5) Sidebar + widgets + templates
cat > templates/partials/sidebar.html <<'HTML'
<nav class="card" style="position:sticky; top:12px;">
  <div style="font-weight:600; margin-bottom:.4rem;">Navigation</div>
  <div style="display:flex; flex-direction:column; gap:.35rem;">
    <a class="link" href="/">🏠 Dashboard</a>
    <a class="link" href="/audio">🎙 Audio</a>
    <a class="link" href="/exports">📦 Exports</a>
    <a class="link" href="/about">ℹ️ About</a>
  </div>
</nav>
HTML

cat > templates/partials/stats_widget.html <<'HTML'
<div id="stats-widget" class="card">
  <div style="font-weight:600; margin-bottom:.5rem;">Stats</div>
  <div class="grid" style="grid-template-columns: repeat(3, minmax(0,1fr));">
    <div><div class="text-sm" style="color:#64748b;">Total</div><div id="stat-total" style="font-weight:700;">—</div></div>
    <div><div class="text-sm" style="color:#64748b;">Notes</div><div id="stat-note">—</div></div>
    <div><div class="text-sm" style="color:#64748b;">Audio</div><div id="stat-audio">—</div></div>
  </div>
  <div style="margin-top:.5rem;">
    <div class="text-sm" style="color:#64748b;">Top tags</div>
    <div id="stat-tags" class="tags" style="margin-top:.2rem;"></div>
  </div>
</div>
<script>
(async ()=>{
  try{
    const r = await fetch('/api/stats'); const d = await r.json();
    document.getElementById('stat-total').textContent = d.total ?? '0';
    document.getElementById('stat-note').textContent = (d.by_type && (d.by_type.note || d.by_type['note'])) ?? 0;
    document.getElementById('stat-audio').textContent = (d.by_type && (d.by_type.audio || d.by_type['audio'])) ?? 0;
    const t = document.getElementById('stat-tags'); t.innerHTML='';
    (d.top_tags||[]).forEach(x=>{
      const s=document.createElement('span'); s.className='tag'; s.textContent='#'+x.tag+' '+x.count; t.appendChild(s);
    });
  }catch(e){}
})();
</script>
HTML

cat > templates/partials/recent_widget.html <<'HTML'
<div id="recent-widget" class="card">
  <div style="font-weight:600; margin-bottom:.5rem;">Recent</div>
  <div id="recent-list" class="grid" style="grid-template-columns: 1fr;"></div>
</div>
<script>
(async ()=>{
  try{
    const r = await fetch('/api/recent?limit=8'); const d = await r.json();
    const box=document.getElementById('recent-list'); box.innerHTML='';
    (d.items||[]).forEach(it=>{
      const row=document.createElement('div'); row.className='note';
      row.innerHTML = `<a class="link" href="/detail/${it.id}"><b>${it.title}</b></a><div class="text-xs" style="color:#94a3b8;">${it.ts||''}</div>`;
      box.appendChild(row);
    });
  }catch(e){}
})();
</script>
HTML

# Create pages
cat > templates/audio.html <<'HTML'
{% extends "base.html" %}
{% block title %}Audio — Second Brain{% endblock %}
{% block content %}
<div class="grid grid-2">
  <div>
    <div class="card"><h2>Audio Notes</h2></div>
    {% for it in items %}
      <div class="note" style="margin-top:.6rem;">
        <a class="link" href="/detail/{{ it.id }}"><b>{{ it.title }}</b></a>
        <div class="text-xs" style="color:#94a3b8;">{{ it.ts }}</div>
        <audio controls src="/audio/{{ it.file }}" style="width:100%; margin-top:.3rem;"></audio>
        {% if it.summary %}<div class="text-sm" style="color:#64748b; margin-top:.2rem;">{{ it.summary }}</div>{% endif %}
      </div>
    {% else %}
      <div class="card">No audio notes yet.</div>
    {% endfor %}
  </div>
  <aside>
    {% include 'partials/sidebar.html' %}
    {% include 'partials/recent_widget.html' %}
  </aside>
</div>
{% endblock %}
HTML

cat > templates/exports.html <<'HTML'
{% extends "base.html" %}
{% block title %}Exports — Second Brain{% endblock %}
{% block content %}
<div class="grid grid-2">
  <div>
    <div class="card">
      <h2>Exports</h2>
      <p class="text-sm" style="color:#64748b;">Download your dataset in convenient formats.</p>
      <div style="display:flex; gap:.6rem; flex-wrap:wrap;">
        <a class="btn" href="/export/json" onclick="sbToast('Generating JSON…')">Download JSON</a>
        <a class="btn" href="/export/markdown.zip" onclick="sbToast('Preparing Markdown ZIP…')">Download Markdown ZIP</a>
      </div>
    </div>
  </div>
  <aside>
    {% include 'partials/sidebar.html' %}
    {% include 'partials/stats_widget.html' %}
  </aside>
</div>
{% endblock %}
HTML

# 6) Inject sidebar/widgets into existing pages (if present)
for f in templates/dashboard.html templates/detail.html templates/edit.html; do
  [[ -f "$f" ]] || continue
  bk "$f"
done

if [[ -f templates/dashboard.html ]]; then
  if grep -q "<aside>" templates/dashboard.html; then
    awk '1;/<aside>/{print "    {% include '\''partials/sidebar.html'\'' %}\n    {% include '\''partials/stats_widget.html'\'' %}\n    {% include '\''partials/recent_widget.html'\'' %}";}' templates/dashboard.html > templates/dashboard.new && mv templates/dashboard.new templates/dashboard.html
  else
    printf "{%% include 'partials/sidebar.html' %%}\n{%% include 'partials/stats_widget.html' %%}\n{%% include 'partials/recent_widget.html' %%}\n\n" | cat - templates/dashboard.html > templates/dashboard.new && mv templates/dashboard.new templates/dashboard.html
  fi
  # ensure quick capture form shows toast
  perl -0777 -pe "s/<form action=\"\/capture\" method=\"post\" enctype=\"multipart\/form-data\"[^>]*>/<form action=\"\/capture\" method=\"post\" enctype=\"multipart\/form-data\" data-toast=\"Capturing…\">/g" -i templates/dashboard.html || true
fi

if [[ -f templates/detail.html ]]; then
  if grep -q "<aside>" templates/detail.html; then
    awk '1;/<aside>/{print "    {% include '\''partials/sidebar.html'\'' %}\n    {% include '\''partials/recent_widget.html'\'' %}";}' templates/detail.html > templates/detail.new && mv templates/detail.new templates/detail.html
  else
    printf "{%% include 'partials/sidebar.html' %%}\n{%% include 'partials/recent_widget.html' %%}\n\n" | cat - templates/detail.html > templates/detail.new && mv templates/detail.new templates/detail.html
  fi
  # add rename form if not present
  if ! grep -q "Rename tag" templates/detail.html; then
    awk '1;/<h3 style="margin-top:1rem;">Tags<\/h3>/{print "      <div class=\"text-xs\" style=\"color:#94a3b8; margin-bottom:.35rem;\">Tip: rename replaces across all your notes.</div>"}' templates/detail.html > templates/detail.tmp && mv templates/detail.tmp templates/detail.html
    awk '1;/<\/form>/{ if(!x){print "      <form method=\"post\" action=\"/tags/rename\" onsubmit=\"sbToast('\''Renaming tag…'\'')\" style=\"margin-top:.5rem; display:flex; gap:.4rem; flex-wrap:wrap;\">\
        <input class=\"input\" name=\"old\" placeholder=\"old tag\">\
        <input class=\"input\" name=\"new\" placeholder=\"new tag\">\
        <button class=\"btn secondary\" type=\"submit\">Rename tag</button>\
      </form>"; x=1 }}' templates/detail.html > templates/detail.tmp && mv templates/detail.tmp templates/detail.html
  fi
  # toast on delete
  perl -0777 -pe "s/<form action=\"\/delete\/\{\{ note.id \}\}\" method=\"post\"/<form action=\"\/delete\/{{ note.id }}\" method=\"post\" onsubmit=\"sbToast('Deleting…')\"/g" -i templates/detail.html || true
fi

if [[ -f templates/edit.html ]]; then
  perl -0777 -pe "s/<form method=\"post\" action=\"\/edit\/\{\{ note.id \}\}\"/<form method=\"post\" action=\"\/edit\/{{ note.id }}\" data-toast=\"Saving…\"/g" -i templates/edit.html || true
fi

# 7) UI JS: generic data-toast support (append if missing)
if [[ -f static/ui.js ]] && ! grep -q "data-toast" static/ui.js; then
  echo >> static/ui.js
  cat >> static/ui.js <<'JS'
// Generic toast-on-submit for forms with data-toast
document.addEventListener('submit', (e)=>{
  const f=e.target; if(f.matches('form[data-toast]')){ try{ sbToast(f.getAttribute('data-toast')||'Working…'); }catch(_){} }
});
JS
fi

# 8) Discord control bot
if [[ -f scripts/discord_bot.py ]]; then bk scripts/discord_bot.py; fi
cat > scripts/discord_bot.py <<'PY'
import os, asyncio, httpx, discord, re
from dotenv import load_dotenv
load_dotenv()

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
BASE  = os.getenv("DISCORD_FORWARD_URL", "http://localhost:8084")
CAPTURE_URL = BASE.rstrip("/") + "/capture"
API_STATS   = BASE.rstrip("/") + "/api/stats"
API_RECENT  = BASE.rstrip("/") + "/api/recent"
API_SEARCH  = BASE.rstrip("/") + "/api/search"
TAG_RENAME  = BASE.rstrip("/") + "/tags/rename"

BEARER = os.getenv("DISCORD_FORWARD_BEARER")
PREFIX = os.getenv("DISCORD_BOT_PREFIX", "!")
ALLOWED = set(filter(None, (os.getenv("DISCORD_ALLOWED_GUILDS","").split(","))))

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)

def auth_headers():
    return {"Authorization": f"Bearer {BEARER}"} if BEARER else {}

async def http_json(url, method="GET", **kw):
    kw.setdefault("timeout", 30)
    headers = kw.pop("headers", {})
    headers.update(auth_headers())
    async with httpx.AsyncClient(timeout=kw.pop("timeout")) as cli:
        r = await cli.request(method, url, headers=headers, **kw)
        r.raise_for_status()
        if "application/json" in r.headers.get("content-type",""):
            return r.json()
        return r.text

async def forward_text(note:str, tags:str="discord"):
    data = {"note": note, "tags": tags}
    async with httpx.AsyncClient(timeout=30) as cli:
        r = await cli.post(CAPTURE_URL, data=data, headers=auth_headers())
        r.raise_for_status()

async def handle_command(msg: discord.Message):
    if msg.author.bot: return
    if ALLOWED and str(msg.guild.id) not in ALLOWED: return
    content = msg.content.strip()

    # !ping
    if content.startswith(PREFIX+"ping"):
        await msg.reply("pong"); return

    # !stats
    if content.startswith(PREFIX+"stats"):
        try:
            d = await http_json(API_STATS)
            total = d.get("total", 0); by = d.get("by_type", {})
            top = ", ".join([f"#{t['tag']}({t['count']})" for t in d.get("top_tags", [])[:5]]) or "—"
            out = f"**Stats** — total: {total}, notes: {by.get('note',0)}, audio: {by.get('audio',0)}\nTop tags: {top}"
            await msg.reply(out)
        except Exception as e:
            await msg.reply(f"stats error: {e}")
        return

    # !recent [n]
    if content.startswith(PREFIX+"recent"):
        m = re.match(rf"^{re.escape(PREFIX)}recent\s+(\d+)$", content)
        n = int(m.group(1)) if m else 5
        try:
            d = await http_json(API_RECENT+f"?limit={max(1,min(20,n))}")
            lines=[]
            for it in d.get("items", []):
                lines.append(f"- #{it['id']} {it['title']} · {it.get('ts','')}")
            await msg.reply("**Recent**\n" + ("\n".join(lines) if lines else "No recent items."))
        except Exception as e:
            await msg.reply(f"recent error: {e}")
        return

    # !search <q...>
    if content.startswith(PREFIX+"search"):
        q = content[len(PREFIX)+len("search"):].strip()
        if not q: await msg.reply("Usage: !search <query>"); return
        try:
            d = await http_json(API_SEARCH+f"?q={httpx.QueryParams({'q':q})['q']}")
            items = d.get("items", [])[:10]
            if not items: await msg.reply("No matches."); return
            lines = [f"- #{it['id']} {it['title']} · {it.get('ts','')}" for it in items]
            await msg.reply("**Search**\n" + "\n".join(lines))
        except Exception as e:
            await msg.reply(f"search error: {e}")
        return

    # !renameTag <old> <new>
    if content.startswith(PREFIX+"renameTag"):
        parts = content.split()
        if len(parts) != 3:
            await msg.reply("Usage: !renameTag <old> <new>")
            return
        old, new = parts[1].lower(), parts[2].lower()
        try:
            async with httpx.AsyncClient(timeout=60) as cli:
                r = await cli.post(TAG_RENAME, data={"old": old, "new": new}, headers=auth_headers())
                if r.status_code == 200 or r.status_code == 302:
                    await msg.reply(f"Renamed #{old} → #{new}")
                else:
                    await msg.reply(f"rename failed: {r.status_code} {r.text}")
        except Exception as e:
            await msg.reply(f"rename error: {e}")
        return

    # !capture <text...>
    if content.startswith(PREFIX+"capture"):
        note = content[len(PREFIX)+len("capture"):].strip()
        if not note:
            await msg.reply("Usage: !capture <text>")
            return
        try:
            await forward_text(f"[{msg.author.display_name}] {note}", tags="discord")
            await msg.add_reaction("🧠")
        except Exception as e:
            await msg.reply(f"capture error: {e}")
        return

    # Fallback: forward plain text
    if content:
        try:
            await forward_text(f"[{msg.author.display_name}] {content}", tags="discord")
            await msg.add_reaction("🧠")
        except Exception:
            pass

@bot.event
async def on_ready():
    print(f"Bot ready: {bot.user} (latency {bot.latency*1000:.0f}ms)")

@bot.event
async def on_message(message: discord.Message):
    await handle_command(message)

if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit("Set DISCORD_BOT_TOKEN in .env")
    bot.run(TOKEN)
PY

# 9) Makefile helper for Discord
if [[ -f Makefile ]] && ! grep -q "^discord:" Makefile; then
  cat >> Makefile <<'MK'

# === Discord ===
discord:
	@. .venv/bin/activate && python scripts/discord_bot.py
MK
fi

echo "scaffold_036 complete.

Next:
  # restart app
  uvicorn app:app --reload --host 0.0.0.0 --port 8084

  # optional: run Discord bot
  source .venv/bin/activate
  make discord
"
