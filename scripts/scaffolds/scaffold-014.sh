#!/usr/bin/env bash
# scripts/scaffold_014.sh
set -euo pipefail
STAMP="$(date +%Y%m%d-%H%M%S)"
b(){ [[ -f "$1" ]] && { mkdir -p .bak; cp -p "$1" ".bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }

mkdir -p templates/partials static scripts

# ============ 1) brand.css (theme tokens) ============
b static/brand.css
cat > static/brand.css <<'CSS'
:root{
  --brand: var(--brand-raw, #4f46e5);
  --brand-700: color-mix(in srgb, var(--brand) 90%, #000);
  --brand-600: var(--brand);
  --brand-500: color-mix(in srgb, var(--brand) 85%, #fff);
}
:root[data-theme="dark"]{ color-scheme: dark; }
.bg-brand-600{ background-color: var(--brand-600); }
.text-brand-600{ color: var(--brand-600); }
.border-brand-600{ border-color: var(--brand-600); }
.badge{ display:inline-block; padding:.15rem .45rem; border-radius:.5rem; font-size:.75rem; background:rgba(0,0,0,.05);}
.toast{ position:fixed; right:1rem; bottom:1rem; z-index:50; background:#111827; color:#fff; padding:.75rem 1rem; border-radius:.75rem; box-shadow:0 10px 20px rgba(0,0,0,.2); margin-top:.5rem;}
.kbd{font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size:.75rem; padding:.1rem .35rem; border:1px solid rgba(0,0,0,.2); border-bottom-width:2px; border-radius:.375rem; background:#fff; color:#111827;}
/* Command palette */
#cmdk{ position:fixed; inset:0; display:none; align-items:flex-start; justify-content:center; padding-top:10vh; background:rgba(0,0,0,.35); z-index:60;}
#cmdk .panel{ width:min(680px, 92vw); background:#0b1020; color:#eef2ff; border:1px solid rgba(255,255,255,.08); border-radius:12px; overflow:hidden; }
#cmdk input{ width:100%; padding:12px 14px; background:#0f172a; color:#e2e8f0; border:none; outline:none; }
#cmdk ul{ max-height:50vh; overflow:auto; }
#cmdk li{ padding:10px 12px; border-top:1px solid rgba(255,255,255,.06); cursor:pointer; }
#cmdk li:hover{ background:#111827; }
CSS

# ============ 2) app.js (toasts, shortcuts, tag editor, command palette) ============
b static/app.js
cat > static/app.js <<'JS'
(function(){
  // --- theme init from localStorage ---
  const theme = localStorage.getItem('sb.theme') || 'auto';
  if(theme==='dark' || (theme==='auto' && window.matchMedia('(prefers-color-scheme: dark)').matches)){
    document.documentElement.setAttribute('data-theme','dark');
  }
  const brand = localStorage.getItem('sb.brand');
  if(brand){ document.documentElement.style.setProperty('--brand-raw', brand); }

  // --- toasts: listen for custom HX-Trigger event "toast" ---
  document.body.addEventListener('toast', (e)=>{
    const d = e.detail || {};
    const msg = d.message || 'Done';
    const el = document.createElement('div');
    el.className = 'toast';
    el.textContent = msg;
    document.body.appendChild(el);
    setTimeout(()=>{ el.remove(); }, Math.min(6000, (msg.length*60)));
  });

  // --- shortcuts ---
  const go = (p)=>{ window.location.href = p; };
  document.addEventListener('keydown', (ev)=>{
    if(ev.target && ['INPUT','TEXTAREA'].includes(ev.target.tagName)) return;
    if ((ev.metaKey||ev.ctrlKey) && ev.key.toLowerCase() === 'k') { ev.preventDefault(); openCmdK(); return; }
    const k = ev.key.toLowerCase();
    if(k==='/'){ ev.preventDefault(); const q = document.querySelector('input[name="q"]'); if(q){ q.focus(); q.select(); } return; }
    if(k==='n'){ const b = document.getElementById('body'); if(b){ b.focus(); } return; }
    // g + key chords
    window.__sbChord = window.__sbChord || {g:false};
    if(k==='g'){ window.__sbChord.g = true; setTimeout(()=>window.__sbChord.g=false, 900); return; }
    if(window.__sbChord.g){
      if(k==='t'){ go('/tags'); }
      if(k==='e'){ go('/embeddings'); }
      if(k==='s'){ go('/settings'); }
      window.__sbChord.g = false;
      return;
    }
    if(k==='?'){ openHelp(); return; }
  });

  // --- tag editor ---
  const tagInput = document.getElementById('tag-entry');
  const tagHidden = document.getElementById('tagsInput');
  const tagAuto = document.getElementById('tag-auto');
  const tagWrap = document.getElementById('tag-editor');
  const tagsSet = new Set();
  function renderChips(){
    // remove old chips
    [...tagWrap.querySelectorAll('.chip')].forEach(e=>e.remove());
    [...tagsSet].forEach(t=>{
      const chip = document.createElement('span');
      chip.className = 'chip inline-flex items-center gap-1 px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-sm';
      chip.textContent = '#'+t;
      const x = document.createElement('button'); x.type='button'; x.textContent='×'; x.className='ml-1 text-slate-500';
      x.onclick = ()=>{ tagsSet.delete(t); renderChips(); };
      chip.appendChild(x);
      tagWrap.insertBefore(chip, tagInput);
    });
    tagHidden.value = [...tagsSet].join(',');
  }
  async function suggestLLM(){
    const body = document.getElementById('body'); if(!body||body.value.trim().length<12) return;
    try{
      const r = await fetch('/tags/suggest', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text: body.value})});
      const j = await r.json();
      const box = document.getElementById('tag-suggestions'); if(!box) return;
      box.innerHTML = '';
      (j.tags||[]).forEach(t=>{
        const b = document.createElement('button'); b.type='button'; b.className='badge'; b.textContent='#'+t;
        b.onclick = ()=>{ tagsSet.add(t); renderChips(); };
        box.appendChild(b);
      });
    }catch(_){}
  }
  async function autocomplete(q){
    if(!q){ tagAuto.classList.add('hidden'); tagAuto.innerHTML=''; return; }
    try{
      const r = await fetch('/api/tags?q='+encodeURIComponent(q));
      const j = await r.json();
      tagAuto.innerHTML = (j.tags||[]).map(t=>`<button type="button" data-t="${t}" class="w-full text-left px-2 py-1 hover:bg-slate-100 dark:hover:bg-slate-800">${'#'+t}</button>`).join('');
      tagAuto.classList.remove('hidden');
      [...tagAuto.querySelectorAll('button')].forEach(b=>{
        b.onclick = ()=>{ tagsSet.add(b.dataset.t); tagAuto.classList.add('hidden'); tagInput.value=''; renderChips(); };
      });
    }catch(_){}
  }
  if(tagInput){
    tagInput.addEventListener('keydown',(e)=>{
      if(['Enter','Tab',','].includes(e.key)){ e.preventDefault(); const v = (tagInput.value||'').trim().toLowerCase().replace(/^#/,'').replace(/\s+/g,'-'); if(v){ tagsSet.add(v); tagInput.value=''; renderChips(); } tagAuto.classList.add('hidden'); }
      if(e.key==='Backspace' && !tagInput.value && tagsSet.size){ const last=[...tagsSet].pop(); tagsSet.delete(last); renderChips(); }
    });
    tagInput.addEventListener('input', ()=>autocomplete(tagInput.value.trim().toLowerCase()));
    const body = document.getElementById('body'); if(body){ body.addEventListener('blur', suggestLLM); }
  }

  // --- toasts on HTMX "HX-Trigger" payloads ---
  document.body.addEventListener('htmx:afterOnLoad', (evt)=>{
    const hdr = evt.detail.xhr.getResponseHeader('HX-Trigger');
    if(!hdr) return;
    try{
      const obj = JSON.parse(hdr);
      if(obj.toast){ document.body.dispatchEvent(new CustomEvent('toast',{detail:obj.toast})); }
    }catch(_){}
  });

  // --- command palette ---
  const cmdk = document.createElement('div');
  cmdk.id='cmdk'; cmdk.innerHTML = `
    <div class="panel">
      <input id="cmdk-input" placeholder="Search notes, or type a command… (e.g. >tags)" />
      <ul id="cmdk-list"></ul>
    </div>`;
  document.body.appendChild(cmdk);

  function openCmdK(){ cmdk.style.display='flex'; const i=document.getElementById('cmdk-input'); i.value=''; i.focus(); renderCmd(''); }
  function closeCmdK(){ cmdk.style.display='none'; }
  cmdk.addEventListener('click',(e)=>{ if(e.target===cmdk) closeCmdK(); });
  document.addEventListener('keydown',(e)=>{ if(e.key==='Escape' && cmdk.style.display==='flex') closeCmdK(); });

  async function renderCmd(q){
    const list = document.getElementById('cmdk-list');
    const items = [];
    const nav = [
      {label:'Go to Dashboard', href:'/'},
      {label:'Go to Notes', href:'/notes'},
      {label:'Go to Tags', href:'/tags'},
      {label:'Go to Search', href:'/search'},
      {label:'Go to Embeddings', href:'/embeddings'},
      {label:'Go to Export', href:'/export'},
      {label:'Go to Settings', href:'/settings'},
      {label:'Go to Compare', href:'/compare'}
    ];
    if(q.startsWith('>')){
      const qq = q.slice(1).trim();
      const cmds = [
        {label:'New note (focus)', action:()=>{ closeCmdK(); const b=document.getElementById('body'); if(b){ b.focus(); }}},
        {label:'Theme: Light', action:()=>applyTheme('light')},
        {label:'Theme: Dark', action:()=>applyTheme('dark')},
        {label:'Theme: Auto', action:()=>applyTheme('auto')}
      ];
      list.innerHTML = cmds.filter(c=>c.label.toLowerCase().includes(qq)).map((c,i)=>`<li data-i="${i}" data-action="1">${c.label}</li>`).join('');
      [...list.querySelectorAll('li')].forEach(li=>li.onclick=()=>cmds[+li.dataset.i].action());
      return;
    }
    // search notes quickly via /api/q
    let hits=[];
    if(q.trim().length){
      try{
        const r = await fetch('/api/q?q='+encodeURIComponent(q.trim()));
        const html = await r.text();
        hits = Array.from((new DOMParser().parseFromString(html,'text/html')).querySelectorAll('a[href^="/notes/"]'))
          .slice(0,6).map(a=>({label:a.textContent.trim(), href:a.getAttribute('href')}));
      }catch(_){}
    }
    const all = [...(q?hits:[]), ...nav].slice(0,12);
    list.innerHTML = all.map(i=>`<li data-href="${i.href||''}">${i.label}</li>`).join('');
    [...list.querySelectorAll('li')].forEach(li=>li.onclick=()=>{ const h=li.dataset.href; if(h){ window.location.href=h; }});
  }
  document.getElementById('cmdk-input')?.addEventListener('input', (e)=>renderCmd(e.target.value));
  window.openCmdK = openCmdK;

  // theme apply
  function applyTheme(mode){
    localStorage.setItem('sb.theme', mode);
    if(mode==='dark' || (mode==='auto' && window.matchMedia('(prefers-color-scheme: dark)').matches)){
      document.documentElement.setAttribute('data-theme','dark');
    }else{
      document.documentElement.removeAttribute('data-theme');
    }
  }
  window.applyBrand = function(hex){
    localStorage.setItem('sb.brand', hex);
    document.documentElement.style.setProperty('--brand-raw', hex);
    fetch('/theme/apply', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'}, body:'brand='+encodeURIComponent(hex)});
  };
})();
JS

# ============ 3) Add Compare, About, Scaffolds, Theme endpoints ============
b app.py
awk '1; END{
print "\n# ==== scaffold_014 additions (compare/about/theme) ===="
print "import hashlib, glob"
print ""
print "def _db_has(conn, name, kind=\"table\"):"
print "    q = \"SELECT name FROM sqlite_master WHERE type=? AND name=?\""
print "    return bool(conn.execute(q, (kind, name)).fetchone())"
print ""
print "@app.get(\"/about\", response_class=HTMLResponse)"
print "def about(request: Request):"
print "    conn = get_conn()"
print "    stats = {"
print "      \"notes\": conn.execute(\"SELECT COUNT(*) c FROM notes\").fetchone()[\"c\"],"
print "      \"tags\": conn.execute(\"SELECT COUNT(*) c FROM tags\").fetchone()[\"c\"],"
print "      \"embedded\": conn.execute(\"SELECT COUNT(*) c FROM note_embeddings\").fetchone()[\"c\"],"
print "    }"
print "    info = {"
print "      \"auth_mode\": os.getenv(\"AUTH_MODE\",\"none\"),"
print "      \"ollama_base\": os.getenv(\"OLLAMA_BASE_URL\", get_setting(conn, \"OLLAMA_BASE_URL\",\"http://localhost:11434\")),"
print "      \"embed_model\": os.getenv(\"OLLAMA_EMBED_MODEL\", get_setting(conn, \"OLLAMA_EMBED_MODEL\",\"nomic-embed-text:latest\")),"
print "    }"
print "    return templates.TemplateResponse(\"about.html\", {\"request\":request, \"stats\":stats, \"info\":info})"
print ""
print "def _probe_features():"
print "    conn = get_conn()"
print "    tables = {"
print "      \"notes\": _db_has(conn, \"notes\"),"
print "      \"tags\": _db_has(conn, \"tags\"),"
print "      \"note_embeddings\": _db_has(conn, \"note_embeddings\"),"
print "      \"notes_fts\": _db_has(conn, \"notes_fts\", kind=\"table\") or _db_has(conn, \"notes_fts\", kind=\"view\"),"
print "      \"discord_bindings\": _db_has(conn, \"discord_bindings\"),"
print "      \"discord_captures\": _db_has(conn, \"discord_captures\"),"
print "      \"discord_guild_prefs\": _db_has(conn, \"discord_guild_prefs\"),"
print "      \"users\": _db_has(conn, \"users\"),"
print "      \"api_tokens\": _db_has(conn, \"api_tokens\"),"
print "    }"
print "    files = {"
print "      \"discord_bot\": os.path.exists(BASE_DIR/\"discord_bot.py\"),"
print "      \"cli\": os.path.exists(BASE_DIR/\"sb.py\"),"
print "      \"makefile\": os.path.exists(BASE_DIR/\"Makefile\"),"
print "      \"static_app_js\": os.path.exists(BASE_DIR/\"static/app.js\"),"
print "      \"brand_css\": os.path.exists(BASE_DIR/\"static/brand.css\"),"
print "    }"
print "    # endpoints (rough check via source text)"
print "    src = (BASE_DIR/\"app.py\").read_text(encoding=\"utf-8\", errors=\"ignore\")"
print "    endpoints = {"
print "      \"/search\": \"/search\" in src,"
print "      \"/export\": \"/export\" in src,"
print "      \"/embeddings\": \"/embeddings\" in src,"
print "      \"/tags\": \"/tags\" in src,"
print "      \"/tokens\": \"/tokens\" in src,"
print "      \"/login\": \"/login\" in src,"
print "      \"/about\": \"/about\" in src,"
print "      \"/compare\": \"/compare\" in src,"
print "    }"
print "    scaffolds = []"
print "    for p in sorted(glob.glob(str(BASE_DIR/\"scripts\"/\"scaffold_*.sh\"))):"
print "        try:"
print "            data = open(p,\"rb\").read()"
print "            h = hashlib.sha256(data).hexdigest()[:12]"
print "            scaffolds.append({\"name\": os.path.basename(p), \"size\": len(data), \"hash\": h})"
print "        except Exception:"
print "            pass"
print "    return {\"tables\": tables, \"files\": files, \"endpoints\": endpoints, \"scaffolds\": scaffolds}"
print ""
print "@app.get(\"/compare\", response_class=HTMLResponse)"
print "def compare(request: Request):"
print "    feat = _probe_features()"
print "    return templates.TemplateResponse(\"compare.html\", {\"request\": request, **feat})"
print ""
print "@app.get(\"/scaffolds\", response_class=HTMLResponse)"
print "def scaffolds_page(request: Request):"
print "    feat = _probe_features()"
print "    return templates.TemplateResponse(\"scaffolds.html\", {\"request\": request, \"scaffolds\": feat[\"scaffolds\"]})"
print ""
print "@app.post(\"/theme/apply\")"
print "def theme_apply(brand: str = Form(\"\"), mode: str = Form(\"\")):"
print "    conn = get_conn()"
print "    if brand: put_setting(conn, \"THEME_BRAND\", brand)"
print "    if mode:  put_setting(conn, \"THEME_MODE\", mode)"
print "    return JSONResponse({\"ok\": True})"
}' app.py > app.py.new && mv app.py.new app.py

# ============ 4) Templates: about, compare, scaffolds ============
b templates/about.html
cat > templates/about.html <<'HTML'
{% extends "base.html" %}
{% block title %}About — Second Brain Premium{% endblock %}
{% block content %}
<div class="rounded-2xl bg-white/95 dark:bg-slate-950/90 shadow-xl ring-1 ring-black/5 p-6 space-y-4">
  <h1 class="text-xl font-semibold">About this instance</h1>
  <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
    <div class="p-4 border rounded-xl dark:border-slate-800">
      <div class="text-sm text-slate-500">Notes</div>
      <div class="text-2xl font-semibold">{{ stats.notes }}</div>
    </div>
    <div class="p-4 border rounded-xl dark:border-slate-800">
      <div class="text-sm text-slate-500">Tags</div>
      <div class="text-2xl font-semibold">{{ stats.tags }}</div>
    </div>
    <div class="p-4 border rounded-xl dark:border-slate-800">
      <div class="text-sm text-slate-500">Embedded</div>
      <div class="text-2xl font-semibold">{{ stats.embedded }}</div>
    </div>
  </div>
  <div class="mt-2 text-sm text-slate-600 dark:text-slate-300">
    Auth mode: <b>{{ info.auth_mode }}</b> • Ollama: <code>{{ info.ollama_base }}</code> • Embed model: <code>{{ info.embed_model }}</code>
  </div>
  <form class="mt-6 flex items-end gap-3" method="post" action="/theme/apply" onsubmit="localStorage.setItem('sb.brand', brand.value)">
    <div>
      <label class="block text-sm mb-1">Brand color (hex)</label>
      <input name="brand" id="brand" placeholder="#4f46e5" class="rounded-xl border-slate-300 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 px-3 py-2">
    </div>
    <div>
      <label class="block text-sm mb-1">Theme</label>
      <select name="mode" class="rounded-xl border-slate-300 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 px-3 py-2">
        <option value="">(leave)</option>
        <option value="light">Light</option>
        <option value="dark">Dark</option>
        <option value="auto">Auto</option>
      </select>
    </div>
    <button class="px-3 py-2 rounded-xl bg-brand-600 text-white">Apply</button>
  </form>
</div>
{% endblock %}
HTML

b templates/compare.html
cat > templates/compare.html <<'HTML'
{% extends "base.html" %}
{% block title %}Compare — Second Brain Premium{% endblock %}
{% block content %}
<div class="rounded-2xl bg-white/95 dark:bg-slate-950/90 shadow-xl ring-1 ring-black/5 p-6">
  <h1 class="text-xl font-semibold">Feature Matrix</h1>
  <div class="mt-4 grid grid-cols-1 md:grid-cols-3 gap-6">
    <section class="border rounded-xl p-4 dark:border-slate-800">
      <h2 class="font-medium">Database</h2>
      <ul class="mt-2 space-y-1 text-sm">
        {% for k,v in tables.items() %}
          <li> {{ k }}:
            {% if v %}<span class="badge" style="background: #DCFCE7; color:#065F46;">present</span>{% else %}
            <span class="badge" style="background: #FEE2E2; color:#991B1B;">missing</span>{% endif %}
          </li>
        {% endfor %}
      </ul>
    </section>
    <section class="border rounded-xl p-4 dark:border-slate-800">
      <h2 class="font-medium">Files</h2>
      <ul class="mt-2 space-y-1 text-sm">
        <li>discord_bot.py: {% if files.discord_bot %}<span class="badge" style="background:#E0E7FF;color:#3730A3;">ok</span>{% else %}<span class="badge" style="background:#FEE2E2;color:#991B1B;">no</span>{% endif %}</li>
        <li>sb.py (CLI): {% if files.cli %}<span class="badge" style="background:#E0E7FF;color:#3730A3;">ok</span>{% else %}<span class="badge" style="background:#FEE2E2;color:#991B1B;">no</span>{% endif %}</li>
        <li>Makefile: {% if files.makefile %}<span class="badge" style="background:#E0E7FF;color:#3730A3;">ok</span>{% else %}<span class="badge" style="background:#FEE2E2;color:#991B1B;">no</span>{% endif %}</li>
        <li>static/app.js: {% if files.static_app_js %}<span class="badge" style="background:#E0E7FF;color:#3730A3;">ok</span>{% else %}<span class="badge" style="background:#FEE2E2;color:#991B1B;">no</span>{% endif %}</li>
        <li>static/brand.css: {% if files.brand_css %}<span class="badge" style="background:#E0E7FF;color:#3730A3;">ok</span>{% else %}<span class="badge" style="background:#FEE2E2;color:#991B1B;">no</span>{% endif %}</li>
      </ul>
    </section>
    <section class="border rounded-xl p-4 dark:border-slate-800">
      <h2 class="font-medium">Endpoints</h2>
      <ul class="mt-2 space-y-1 text-sm">
        {% for k,v in endpoints.items() %}
          <li>{{ k }}:
            {% if v %}<span class="badge" style="background:#DBEAFE;color:#1E3A8A;">exists</span>
            {% else %}<span class="badge" style="background:#FEE2E2;color:#991B1B;">missing</span>{% endif %}
          </li>
        {% endfor %}
      </ul>
    </section>
  </div>

  <h2 class="mt-8 text-lg font-semibold">Applied Scaffolds</h2>
  <table class="mt-2 w-full text-sm">
    <thead><tr class="text-left border-b dark:border-slate-800">
      <th class="py-2">File</th><th>Size</th><th>SHA-256 (12)</th>
    </tr></thead>
    <tbody>
      {% for s in scaffolds %}
        <tr class="border-b dark:border-slate-800">
          <td class="py-2"><a class="hover:underline" href="/scaffolds">{{ s.name }}</a></td>
          <td>{{ s.size }}</td>
          <td><code>{{ s.hash }}</code></td>
        </tr>
      {% else %}
        <tr><td colspan="3" class="py-4 text-slate-500">No scaffold_*.sh files found.</td></tr>
      {% endfor %}
    </tbody>
  </table>
</div>
{% endblock %}
HTML

b templates/scaffolds.html
cat > templates/scaffolds.html <<'HTML'
{% extends "base.html" %}
{% block title %}Scaffolds — Second Brain Premium{% endblock %}
{% block content %}
<div class="rounded-2xl bg-white/95 dark:bg-slate-950/90 shadow-xl ring-1 ring-black/5 p-6">
  <h1 class="text-xl font-semibold">Scaffold scripts</h1>
  <ul class="mt-4 divide-y divide-slate-200/80 dark:divide-white/10">
    {% for s in scaffolds %}
      <li class="py-3 flex items-center justify-between">
        <div>
          <div class="font-medium">{{ s.name }}</div>
          <div class="text-xs text-slate-500">size {{ s.size }} • sha {{ s.hash }}</div>
        </div>
      </li>
    {% else %}
      <li class="py-6 text-slate-500">No scaffold scripts found in /scripts.</li>
    {% endfor %}
  </ul>
</div>
{% endblock %}
HTML

# ============ 5) Patch base.html to load brand.css & app.js and add nav links ============
if [[ -f templates/base.html ]]; then
  if ! grep -q "static/brand.css" templates/base.html; then
    b templates/base.html
    awk '1; /<\/head>/ && !x {print "  <link rel=\"stylesheet\" href=\"/static/brand.css\">"; print "  <script defer src=\"/static/app.js\"></script>"; x=1}' templates/base.html > templates/base.html.new && mv templates/base.html.new templates/base.html
  fi
  # Add Compare link after Export if missing
  if grep -q "href=\"/export\"" templates/base.html && ! grep -q "href=\"/compare\"" templates/base.html; then
    awk '1; /href="\/export"/ && !x {print "        <li><a class=\"hover:text-slate-900 dark:hover:text-white\" href=\"/compare\">Compare</a></li>"; x=1}' templates/base.html > templates/base.html.new && mv templates/base.html.new templates/base.html
  fi
  # Add About link at end if missing
  if ! grep -q "href=\"/about\"" templates/base.html; then
    awk '1; /<\/ul>/ && !x {print "        <li><a class=\"hover:text-slate-900 dark:hover:text-white\" href=\"/about\">About</a></li>"; x=1}' templates/base.html > templates/base.html.new && mv templates/base.html.new templates/base.html
  fi
fi

# ============ 6) requirements ensure ============
if [[ ! -f requirements.txt ]]; then
  cat > requirements.txt <<'REQ'
fastapi==0.115.0
uvicorn[standard]==0.30.6
jinja2==3.1.4
httpx==0.27.2
numpy>=1.24,<3
python-multipart==0.0.9
REQ
else
  grep -q "httpx" requirements.txt || echo "httpx==0.27.2" >> requirements.txt
  grep -q "numpy" requirements.txt || echo "numpy>=1.24,<3" >> requirements.txt
fi

echo "Done.

New routes:
  /about   — system & theme controls
  /compare — feature matrix (DB tables, files, endpoints) + list of scaffolds
  /scaffolds — scaffold inventory
Front-end:
  • real brand color classes via static/brand.css (tunable in /about)
  • command palette (Ctrl/⌘+K), shortcuts, toasts
  • tag editor with autocomplete & LLM suggestions

Restart the app:
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  uvicorn app:app --reload --host 0.0.0.0 --port 8084
"
