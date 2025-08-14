#!/usr/bin/env bash
# scripts/scaffolds/scaffold_025.sh
# Modern header, toasts, command palette, tag autocomplete, dev cookie login
set -euo pipefail
STAMP="$(date +%Y%m%d-%H%M%S)"
bk(){ [[ -f "$1" ]] && { mkdir -p .bak; cp -p "$1" ".bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }

mkdir -p static templates/partials

# 1) Static JS/CSS
# JS
cat > static/ui.js <<'JS'
// Minimal helpers: toast, cmd-k palette, tag autocomplete
(function(){
  // Toast
  window.sbToast = function(msg){
    const n=document.createElement('div'); n.className='sb-toast'; n.textContent=msg||'Done';
    document.body.appendChild(n); setTimeout(()=>n.classList.add('show'),10);
    setTimeout(()=>n.classList.remove('show'),3500); setTimeout(()=>n.remove(),4100);
  };

  // Cmd/Ctrl+K Command Palette -> focuses search input or opens /api/search live preview (simplified)
  function openPalette(){
    const dlg=document.getElementById('cmdk'); if(!dlg) return;
    dlg.style.display='block'; const inp=dlg.querySelector('input[name="q"]'); inp.value=''; inp.focus();
  }
  function closePalette(){ const dlg=document.getElementById('cmdk'); if(dlg) dlg.style.display='none'; }

  document.addEventListener('keydown', (e)=>{
    const k=e.key.toLowerCase();
    if((e.metaKey||e.ctrlKey) && k==='k'){ e.preventDefault(); openPalette(); }
    if(k==='escape'){ closePalette(); }
  });

  // Tag autocomplete: fetch /api/tags?q=...
  async function autocomplete(inp){
    const val=inp.value.trim(); const box=inp.nextElementSibling;
    if(!box || !box.classList.contains('tag-suggest')) return;
    if(!val){ box.innerHTML=''; return; }
    const r=await fetch('/api/tags?q='+encodeURIComponent(val)); const d=await r.json();
    box.innerHTML=''; (d.tags||[]).forEach(t=>{
      const b=document.createElement('button'); b.type='button'; b.className='tag-pill'; b.textContent='#'+t;
      b.onclick=()=>{ inp.value=t; box.innerHTML=''; };
      box.appendChild(b);
    });
  }
  document.addEventListener('input', (e)=>{
    const t=e.target; if(t.matches('input[data-tag-ac]')) autocomplete(t);
  });

  // Expose helpers
  window.sbOpenPalette=openPalette; window.sbClosePalette=closePalette;
})();
JS

# CSS
if [[ -f static/brand.css ]]; then bk static/brand.css; fi
cat >> static/brand.css <<'CSS'

/* === UI Overhaul (scaffold_025) === */
:root{ --brand:#4f46e5; --ink:#0f172a; --muted:#64748b; --panel:rgba(255,255,255,.92); --panel-dark:rgba(2,6,23,.92);}
.header{ position:sticky; top:0; z-index:50; backdrop-filter:blur(6px); background:var(--panel); border-bottom:1px solid rgba(0,0,0,.06); display:flex; gap:.75rem; align-items:center; padding:.6rem .9rem;}
.header a{ text-decoration:none; color:var(--ink); font-weight:600;}
.header .right{ margin-left:auto; display:flex; gap:.5rem; }
.btn{ padding:.5rem .8rem; border-radius:.6rem; background:#eef2ff; color:#3730a3; border:1px solid #c7d2fe;}
.input{ padding:.5rem .7rem; border-radius:.6rem; border:1px solid #e2e8f0; min-width:240px;}
.sb-toast{ position:fixed; right:1rem; bottom:1rem; background:#111827; color:#fff; padding:.6rem .9rem; border-radius:.6rem; box-shadow:0 10px 18px rgba(0,0,0,.25); opacity:0; transform:translateY(10px); transition:.25s;}
.sb-toast.show{ opacity:1; transform:translateY(0); }
#cmdk{ display:none; position:fixed; inset:0; background:rgba(0,0,0,.3); align-items:flex-start; justify-content:center; padding-top:10vh;}
.cmdk-panel{ width:min(720px,92vw); background:#fff; border-radius:12px; box-shadow:0 10px 30px rgba(0,0,0,.25); border:1px solid #e5e7eb; padding:12px;}
.cmdk-panel input{ width:100%; }
.tag-suggest{ display:flex; gap:.35rem; flex-wrap:wrap; margin-top:.25rem;}
.tag-pill{ padding:.25rem .45rem; border-radius:.5rem; border:1px solid #e2e8f0; background:#f8fafc; font-size:.85rem; }
CSS

# 2) Command palette partial & header include
cat > templates/partials/cmdk.html <<'HTML'
<div id="cmdk">
  <div class="cmdk-panel">
    <form action="/" method="get">
      <input class="input" type="text" name="q" placeholder="Search…" autofocus>
    </form>
    <div class="text-xs text-slate-500 mt-2">Tip: type keywords and press Enter.</div>
  </div>
</div>
HTML

# 3) Base header injection (create base.html if missing)
if [[ ! -f templates/base.html ]]; then
cat > templates/base.html <<'HTML'
<!doctype html><html><head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="stylesheet" href="/static/brand.css">
  <title>{% block title %}Second Brain{% endblock %}</title>
</head><body class="bg-slate-50 text-slate-900">
  <div class="header">
    <a href="/">Second Brain</a>
    <form action="/" method="get"><input class="input" name="q" placeholder="Search…"></form>
    <div class="right">
      <button class="btn" onclick="sbOpenPalette()">⌘K</button>
      <a class="btn" href="/about">About</a>
      <a class="btn" href="/exports">Exports</a>
    </div>
  </div>
  {% include 'partials/cmdk.html' %}
  <script src="/static/ui.js"></script>
  <main class="p-4">{% block content %}{% endblock %}</main>
</body></html>
HTML
else
  if ! grep -q "partials/cmdk.html" templates/base.html; then
    bk templates/base.html
    awk '1; END{print "{% include '\''partials/cmdk.html'\'' %>"}' templates/base.html >/dev/null 2>&1 || true
    # Safer: append includes & script at end
    printf "\n{% include 'partials/cmdk.html' %}\n<script src=\"/static/ui.js\"></script>\n" >> templates/base.html
    echo "• appended cmdk include + ui.js to base.html"
  fi
fi

# 4) Dev cookie login shim (middleware + /dev/login)
if ! grep -q "scaffold_025 additions (dev cookie login)" app.py 2>/dev/null; then
  bk app.py
  cat >> app.py <<'PY'

# ==== scaffold_025 additions (dev cookie login) ====
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Response

class _AuthCookieToHeader(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        try:
            if getattr(settings, "dev_login_enabled", 0):
                if "authorization" not in request.headers and "sb_token" in request.cookies:
                    token = request.cookies.get("sb_token")
                    if token:
                        # inject header for oauth2_scheme
                        request.scope["headers"] = list(request.scope["headers"]) + [(b"authorization", f"Bearer {token}".encode())]
        except Exception:
            pass
        return await call_next(request)

app.add_middleware(_AuthCookieToHeader)

@app.post("/dev/login")
def dev_login(username: str = Form(...), password: str = Form(...), response: Response = None):
    if not getattr(settings, "dev_login_enabled", 0):
        raise HTTPException(status_code=403, detail="Dev login disabled")
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(status_code=400, detail="Bad credentials")
    token = create_access_token({"sub": user.username}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    if response is None:  # FastAPI will give us one
        response = Response()
    response.set_cookie("sb_token", token, httponly=False, samesite="Lax")
    return {"ok": True, "token": token}
PY
fi

echo "Done 025. Restart the app; press ⌘/Ctrl+K for palette. Use /dev/login to set a cookie for browsing."
