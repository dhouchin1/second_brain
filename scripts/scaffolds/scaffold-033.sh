#!/usr/bin/env bash
# scripts/scaffolds/scaffold_033.sh
# Add /audio page (list audio notes) and /exports page (download center)
set -euo pipefail

STAMP="$(date +%Y%m%d-%H%M%S)"; bk(){ [[ -f "$1" ]] && { mkdir -p .bak; cp -p "$1" ".bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }
mkdir -p templates

# 1) Python routes
if ! grep -q "scaffold_033 additions (pages audio,exports)" app.py 2>/dev/null; then
  bk app.py
  cat >> app.py <<'PY'

# ==== scaffold_033 additions (pages audio,exports) ====
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
    # Simple download center; advanced CSV archives can be added via future migrations
    return templates.TemplateResponse("exports.html", {"request": request})
PY
fi

# 2) Templates
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
      <p class="text-sm" style="color:#64748b;">Download full dataset in convenient formats.</p>
      <div style="display:flex; gap:.6rem; flex-wrap:wrap;">
        <a class="btn" href="/export/json" onclick="sbToast('Generating JSON…')">Download JSON</a>
        <a class="btn" href="/export/markdown.zip" onclick="sbToast('Preparing Markdown ZIP…')">Download Markdown ZIP</a>
      </div>
      <div class="text-xs" style="color:#94a3b8; margin-top:.5rem;">CSV archives page can be added via advanced exports scaffold.</div>
    </div>
  </div>
  <aside>
    {% include 'partials/sidebar.html' %}
    {% include 'partials/stats_widget.html' %}
  </aside>
</div>
{% endblock %}
HTML

echo "Done 033: /audio and /exports pages created."
