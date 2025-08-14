#!/usr/bin/env bash
# scripts/scaffolds/scaffold_032.sh
# Add /api/stats, /api/recent, /api/search; Sidebar include; Stats & Recent widgets
set -euo pipefail

STAMP="$(date +%Y%m%d-%H%M%S)"
bk(){ [[ -f "$1" ]] && { mkdir -p .bak; cp -p "$1" ".bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }

mkdir -p templates/partials static

# 0) Ensure JSONResponse import
if ! grep -q "from fastapi.responses import .*JSONResponse" app.py; then
  bk app.py
  perl -0777 -pe "s/from fastapi\.responses import ([^\n]+)/from fastapi.responses import \1, JSONResponse/g" -i app.py || true
fi

# 1) Append APIs if not present
if ! grep -q "scaffold_032 additions (stats,recent,search)" app.py 2>/dev/null; then
  bk app.py
  cat >> app.py <<'PY'

# ==== scaffold_032 additions (stats,recent,search) ====
from collections import Counter

@app.get("/api/stats")
def api_stats(current_user: User = Depends(get_current_user)):
    conn = get_conn(); c = conn.cursor()
    total = c.execute("SELECT COUNT(*) FROM notes WHERE user_id=?", (current_user.id,)).fetchone()[0]
    by_type = dict(c.execute("SELECT COALESCE(type,'note') as t, COUNT(*) FROM notes WHERE user_id=? GROUP BY t", (current_user.id,)).fetchall())
    # last 7 days counts
    days = c.execute("""
      SELECT date(COALESCE(timestamp, datetime('now'))) d, COUNT(*)
      FROM notes WHERE user_id=?
      GROUP BY d ORDER BY d DESC LIMIT 7
    """, (current_user.id,)).fetchall()
    # tags histogram
    rows = c.execute("SELECT tags FROM notes WHERE user_id=? AND tags IS NOT NULL", (current_user.id,)).fetchall()
    tag_counter = Counter()
    for (csv,) in rows:
        for t in (csv or "").split(","):
            t = t.strip().lower()
            if t: tag_counter[t]+=1
    top_tags = [{"tag": t, "count": n} for t,n in tag_counter.most_common(10)]
    conn.close()
    return {"total": total, "by_type": by_type, "last_days": [{"day": d, "count": n} for d,n in days], "top_tags": top_tags}

@app.get("/api/recent")
def api_recent(limit: int = 10, current_user: User = Depends(get_current_user)):
    limit = max(1, min(50, limit))
    conn = get_conn(); c = conn.cursor()
    rows = c.execute("""
      SELECT id, title, type, timestamp, substr(content,1,120)
      FROM notes WHERE user_id=? ORDER BY datetime(COALESCE(timestamp,datetime('now'))) DESC LIMIT ?
    """, (current_user.id, limit)).fetchall()
    conn.close()
    return {"items": [{"id": r[0], "title": r[1] or "(untitled)", "type": r[2] or "note", "ts": r[3], "preview": r[4]} for r in rows]}

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

# 2) Sidebar include
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

# 3) Stats & Recent widgets (partials with fetch)
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

# 4) Inject sidebar + widgets into dashboard/detail/edit if files exist
if [[ -f templates/dashboard.html ]]; then
  bk templates/dashboard.html
  # Insert sidebar in the aside if present, else prepend navigation block at top
  if grep -q "<aside>" templates/dashboard.html; then
    awk '1;/<aside>/{print "    {% include '\''partials/sidebar.html'\'' %}\n    {% include '\''partials/stats_widget.html'\'' %}\n    {% include '\''partials/recent_widget.html'\'' %}";}' templates/dashboard.html > templates/dashboard.new && mv templates/dashboard.new templates/dashboard.html
  else
    printf "{%% include 'partials/sidebar.html' %%}\n{%% include 'partials/stats_widget.html' %%}\n{%% include 'partials/recent_widget.html' %%}\n\n" | cat - templates/dashboard.html > templates/dashboard.new && mv templates/dashboard.new templates/dashboard.html
  fi
fi

if [[ -f templates/detail.html ]]; then
  bk templates/detail.html
  if grep -q "<aside>" templates/detail.html; then
    awk '1;/<aside>/{print "    {% include '\''partials/sidebar.html'\'' %}\n    {% include '\''partials/recent_widget.html'\'' %}";}' templates/detail.html > templates/detail.new && mv templates/detail.new templates/detail.html
  else
    printf "{%% include 'partials/sidebar.html' %%}\n{%% include 'partials/recent_widget.html' %%}\n\n" | cat - templates/detail.html > templates/detail.new && mv templates/detail.new templates/detail.html
  fi
fi

if [[ -f templates/edit.html ]]; then
  bk templates/edit.html
  printf "{%% include 'partials/sidebar.html' %%}\n\n" | cat - templates/edit.html > templates/edit.new && mv templates/edit.new templates/edit.html
fi

echo "Done 032: stats/recent/search APIs, sidebar, widgets wired."
