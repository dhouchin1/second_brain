// Dark mode toggle
const btn = document.getElementById('themeToggle');
if (btn) {
  btn.addEventListener('click', () => {
    const el = document.documentElement;
    const isDark = el.classList.toggle('dark');
    localStorage.theme = isDark ? 'dark' : 'light';
  });
}

// --- Tiny Tag Editor + Suggestions ---
(function () {
  const editor = document.getElementById('tag-editor');
  const entry  = document.getElementById('tag-entry');
  const hidden = document.getElementById('tagsInput');
  const suggestWrap = document.getElementById('tag-suggestions');
  const noteBody = document.getElementById('body');
  if (!editor || !entry || !hidden) return;

  let tags = new Set();

  function syncHidden() { hidden.value = Array.from(tags).join(','); }
  function renderTokens() {
    // remove all chips except the input
    Array.from(editor.querySelectorAll('.tag-chip')).forEach(el => el.remove());
    tags.forEach(t => {
      const chip = document.createElement('span');
      chip.className = 'tag-chip inline-flex items-center gap-1 px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200';
      chip.innerHTML = `#${t} <button type="button" class="remove text-xs opacity-70 hover:opacity-100">×</button>`;
      chip.querySelector('button.remove').addEventListener('click', () => { tags.delete(t); renderTokens(); syncHidden(); });
      editor.insertBefore(chip, entry);
    });
  }
  function addTag(raw) {
    const t = raw.trim().toLowerCase().replace(/^#/, '').replace(/\s+/g, '-').replace(/[^a-z0-9\-]/g, '');
    if (!t) return;
    tags.add(t);
    renderTokens(); syncHidden();
  }
  function parseEntryAndAdd() {
    const parts = entry.value.split(/[,\s]+/).filter(Boolean);
    parts.forEach(addTag);
    entry.value = '';
  }

  // expose for HTMX form reset hook
  window.TagEditor = { reset() { tags = new Set(); renderTokens(); syncHidden(); } };

  entry.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      parseEntryAndAdd();
    } else if (e.key === 'Backspace' && entry.value === '') {
      // quick remove last tag
      const last = Array.from(tags).pop();
      if (last) { tags.delete(last); renderTokens(); syncHidden(); }
    }
  });

  // LLM suggestions with debounce
  let timer = null;
  const debounce = (fn, ms=600) => (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };

  async function fetchSuggestions(text) {
    try {
      const res = await fetch('/tags/suggest', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ text })
      });
      if (!res.ok) throw new Error('bad response');
      const data = await res.json();
      renderSuggestions(Array.isArray(data.tags) ? data.tags : []);
    } catch (e) {
      renderSuggestions([]);
    }
  }

  function renderSuggestions(list) {
    suggestWrap.innerHTML = '';
    if (!list.length) return;
    list.forEach(tag => {
      if (tags.has(tag)) return;
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'px-2 py-1 rounded-lg bg-brand-600 text-white text-xs hover:bg-brand-700';
      btn.textContent = `#${tag}`;
      btn.addEventListener('click', () => { addTag(tag); });
      suggestWrap.appendChild(btn);
    });
  }

  if (noteBody) {
    noteBody.addEventListener('input', debounce(() => {
      const text = noteBody.value.trim();
      if (text.length < 8) { renderSuggestions([]); return; }
      fetchSuggestions(text);
    }, 650));
  }
})();
