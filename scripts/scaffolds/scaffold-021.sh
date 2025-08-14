#!/usr/bin/env bash
# scripts/scaffolds/scaffold_021.sh
# Healthz, diagnostics JSON, pretty About page (idempotent)
set -euo pipefail

STAMP="$(date +%Y%m%d-%H%M%S)"
bk(){ [[ -f "$1" ]] && { mkdir -p .bak; cp -p "$1" ".bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }

mkdir -p templates scripts

# 1) Append diagnostics to app.py once
if ! grep -q "scaffold_021 additions (healthz, about, diagnostics)" app.py 2>/dev/null; then
  bk app.py
  cat >> app.py <<'PY'

# ==== scaffold_021 additions (healthz, about, diagnostics) ====
import os, subprocess, json

def _sb_read_version():
    try:
        vp = (settings.base_dir / "VERSION")
        return vp.read_text(encoding="utf-8").strip()
    except Exception:
        return "0.0.0"

def _probe_sqlite(conn):
    c = conn.cursor()
    tabs = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")]
    trgs = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='trigger'")]
    write_ok = True
    try:
        c.execute("CREATE TABLE IF NOT EXISTS __write_test (id INTEGER)")
        c.execute("DROP TABLE __write_test")
        conn.commit()
    except Exception:
        write_ok = False
    def has(name): return name in tabs
    return {
        "tables": tabs,
        "triggers": trgs,
        "has": {
            "users": has("users"),
            "notes": has("notes"),
            "notes_fts": has("notes_fts"),
            "audit_log": has("audit_log"),
            "export_archives": has("export_archives"),
            "sync_status": has("sync_status"),
        },
        "write_ok": write_ok,
    }

def _git_info():
    def run(cmd):
        try:
            return subprocess.run(cmd, shell=True, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()
        except Exception as e:
            return f"(error: {e})"
    return {
        "branch": run("git rev-parse --abbrev-ref HEAD"),
        "commit": run("git rev-parse --short HEAD"),
        "remote": run("git config --get remote.origin.url"),
        "status": run("git status --porcelain -b"),
    }

@app.get("/healthz")
def healthz():
    try:
        conn = get_conn()
        conn.execute("SELECT 1").fetchone()
        conn.close()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/api/diag")
def api_diag(current_user: User = Depends(get_current_user)):
    conn = get_conn()
    db = _probe_sqlite(conn)
    counts = {}
    try:
        counts["notes"] = conn.execute("SELECT COUNT(*) FROM notes WHERE user_id=?", (current_user.id,)).fetchone()[0]
        counts["users"] = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    except Exception:
        counts = {}
    conn.close()
    return {
        "version": _sb_read_version(),
        "commit": os.getenv("APP_BUILD_SHA","")[:12],
        "db": db,
        "counts": counts,
        "git": _git_info(),
        "features": {
            "fts_present": db["has"]["notes_fts"],
            "audit_present": db["has"]["audit_log"],
            "exports_present": db["has"]["export_archives"],
        },
        "user": {"id": current_user.id, "username": current_user.username},
    }

@app.get("/about", response_class=HTMLResponse)
def about(request: Request, current_user: User = Depends(get_current_user)):
    info = api_diag(current_user)  # reuse JSON generator
    info_json = json.dumps(info, indent=2)
    return templates.TemplateResponse("about.html", {"request": request, "info": info, "info_json": info_json})
PY
fi

# 2) About template
bk templates/about.html
cat > templates/about.html <<'HTML'
{% extends "base.html" %}
{% block title %}About — Second Brain{% endblock %}
{% block content %}
<div class="rounded-2xl bg-white/95 dark:bg-slate-950/90 shadow-xl ring-1 ring-black/5 p-6 space-y-4">
  <h1 class="text-xl font-semibold">About this instance</h1>

  <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
    <div class="p-4 border rounded-xl dark:border-slate-800">
      <div class="text-sm text-slate-500">Version</div>
      <div class="text-2xl font-semibold">{{ info.version }}</div>
      {% if info.commit %}<div class="text-xs text-slate-500">commit {{ info.commit }}</div>{% endif %}
    </div>
    <div class="p-4 border rounded-xl dark:border-slate-800">
      <div class="text-sm text-slate-500">Notes (you)</div>
      <div class="text-2xl font-semibold">{{ info.counts.notes or 0 }}</div>
    </div>
    <div class="p-4 border rounded-xl dark:border-slate-800">
      <div class="text-sm text-slate-500">Users</div>
      <div class="text-2xl font-semibold">{{ info.counts.users or 0 }}</div>
    </div>
  </div>

  <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
    <div class="p-4 border rounded-xl dark:border-slate-800">
      <h2 class="font-medium">Database</h2>
      <div class="mt-1 text-sm">Write OK:
        {% if info.db.write_ok %}<span class="badge" style="background:#DCFCE7;color:#065F46;">yes</span>
        {% else %}<span class="badge" style="background:#FEE2E2;color:#991B1B;">no</span>{% endif %}
      </div>
      <ul class="mt-2 text-sm space-y-1">
        {% for k,v in info.db.has.items() %}
          <li>{{ k }}:
            {% if v %}<span class="badge" style="background:#DBEAFE;color:#1E3A8A;">present</span>
            {% else %}<span class="badge" style="background:#FEE2E2;color:#991B1B;">missing</span>{% endif %}
          </li>
        {% endfor %}
      </ul>
    </div>
    <div class="p-4 border rounded-xl dark:border-slate-800">
      <h2 class="font-medium">Git</h2>
      <div class="text-sm"><b>Branch:</b> {{ info.git.branch }}</div>
      <div class="text-sm"><b>Commit:</b> {{ info.git.commit }}</div>
      <div class="text-sm"><b>Remote:</b> <code>{{ info.git.remote }}</code></div>
      <details class="mt-2">
        <summary class="text-sm">Status</summary>
        <pre class="text-xs whitespace-pre-wrap">{{ info.git.status }}</pre>
      </details>
    </div>
  </div>

  <details class="p-4 border rounded-xl dark:border-slate-800">
    <summary class="font-medium">Raw diagnostics JSON</summary>
    <pre class="text-xs whitespace-pre-wrap">{{ info_json }}</pre>
  </details>
</div>
{% endblock %}
HTML

# 3) Optional nav link to /about
if [[ -f templates/base.html ]] && ! grep -q 'href="/about"' templates/base.html; then
  bk templates/base.html
  awk '1; /<\/ul>/ && !x {print "        <li><a class=\"hover:underline\" href=\"/about\">About</a></li>"; x=1}' templates/base.html > templates/base.html.new && mv templates/base.html.new templates/base.html
fi

# 4) Local diagnostics runner (writes BuildLogs/DiagReport_*.md)
cat > scripts/diag_local.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
BASE="${1:-http://localhost:8084}"
now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
outdir="BuildLogs"; mkdir -p "$outdir"
file="$outdir/DiagReport_${now//[:]/-}.md"
git_cmd(){ git "$@" 2>/dev/null || true; }
pyver="$(python3 -c 'import sys;print(".".join(map(str,sys.version_info[:3])))' 2>/dev/null || echo none)"
curl_s(){ curl -sS "$@" || true; }
{
  echo "---"; echo "type: diag"; echo "timestamp: \"$now\""; echo "base: \"$BASE\""; echo "---"; echo
  echo "# Diagnostics Report"
  echo "## System"; echo "- Python: \`$pyver\`"; echo "- Host: \`$(hostname -s 2>/dev/null || echo unknown)\`"; echo
  echo "## Git"; echo "```"; echo "branch: $(git_cmd rev-parse --abbrev-ref HEAD)"; echo "commit: $(git_cmd rev-parse --short HEAD)"; echo "remote: $(git_cmd config --get remote.origin.url)"; echo "status:"; git_cmd status --porcelain -b; echo "```"; echo
  echo "## HTTP"
  echo "### /healthz"; echo "```json"; curl_s "$BASE/healthz"; echo "```"
  echo "### /version"; echo "```json"; curl_s "$BASE/version"; echo "```"
  echo "### /api/diag (auth-protected; may 401)"; echo "```"; curl_s -i "$BASE/api/diag"; echo "```"
} > "$file"
echo "✓ Wrote $file"
SH
chmod +x scripts/diag_local.sh

echo "scaffold_021 complete.

Verify:
  curl -s http://localhost:8084/healthz
  curl -s http://localhost:8084/version
  # visit /about (in browser, JWT auth required)
  bash scripts/diag_local.sh
"
