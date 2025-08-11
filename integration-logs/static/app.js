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
