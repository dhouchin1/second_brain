#!/usr/bin/env bash
# scripts/scaffolds/scaffold_034.sh
# Toasts on save; global tag rename endpoint + detail UI
set -euo pipefail

STAMP="$(date +%Y%m%d-%H%M%S)"; bk(){ [[ -f "$1" ]] && { mkdir -p .bak; cp -p "$1" ".bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }

# 1) Endpoint: POST /tags/rename
if ! grep -q "scaffold_034 additions (tags rename)" app.py 2>/dev/null; then
  bk app.py
  cat >> app.py <<'PY'

# ==== scaffold_034 additions (tags rename) ====
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
    # Load notes with tag match
    rows = c.execute("SELECT id, tags FROM notes WHERE user_id=? AND tags LIKE ?", (current_user.id, f"%{old}%")).fetchall()
    changed = 0
    for nid, csv in rows:
        tags = [t.strip().lower() for t in (csv or "").split(",") if t.strip()]
        if old in tags:
            tags = [new if t==old else t for t in tags]
            # de-dup
            seen=[]; dedup=[]
            for t in tags:
                if t not in seen:
                    seen.append(t); dedup.append(t)
            new_csv = ",".join(dedup)
            c.execute("UPDATE notes SET tags=? WHERE id=? AND user_id=?", (new_csv, nid, current_user.id))
            # keep FTS in sync
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

# 2) Detail page UI additions (rename form + toasts on forms)
if [[ -f templates/detail.html ]]; then
  bk templates/detail.html
  # ensure add/rename forms toast; insert rename form under tags section
  if ! grep -q "Rename tag" templates/detail.html; then
    awk '1;/<h3 style="margin-top:1rem;">Tags<\/h3>/{print "      <div class=\"text-xs\" style=\"color:#94a3b8; margin-bottom:.35rem;\">Tip: rename replaces across all your notes.</div>"}' templates/detail.html > templates/detail.tmp && mv templates/detail.tmp templates/detail.html
    awk '1;/<\/form>/{ if(!x){print "      <form method=\"post\" action=\"/tags/rename\" onsubmit=\"sbToast('\''Renaming tag…'\'')\" style=\"margin-top:.5rem; display:flex; gap:.4rem; flex-wrap:wrap;\">\
        <input class=\"input\" name=\"old\" placeholder=\"old tag\">\
        <input class=\"input\" name=\"new\" placeholder=\"new tag\">\
        <button class=\"btn secondary\" type=\"submit\">Rename tag</button>\
      </form>"; x=1 }}' templates/detail.html > templates/detail.tmp && mv templates/detail.tmp templates/detail.html
  fi
  # add toast on delete/edit forms if not present
  perl -0777 -pe "s/<form action=\"\/delete\/\{\{ note.id \}\}\" method=\"post\"/<form action=\"\/delete\/{{ note.id }}\" method=\"post\" onsubmit=\"sbToast('Deleting…')\"/g" -i templates/detail.html || true
fi

# 3) Add generic data-toast support (forms) in ui.js if not already
if [[ -f static/ui.js ]] && ! grep -q "data-toast" static/ui.js; then
  echo >> static/ui.js
  cat >> static/ui.js <<'JS'
// Generic toast-on-submit for forms with data-toast
document.addEventListener('submit', (e)=>{
  const f=e.target; if(f.matches('form[data-toast]')){ try{ sbToast(f.getAttribute('data-toast')||'Working…'); }catch(_){} }
});
JS
fi

# 4) Ensure edit/quick capture forms have toasts/data-toast
if [[ -f templates/edit.html ]]; then
  bk templates/edit.html
  perl -0777 -pe "s/<form method=\"post\" action=\"\/edit\/\{\{ note.id \}\}\"/<form method=\"post\" action=\"\/edit\/{{ note.id }}\" data-toast=\"Saving…\"/g" -i templates/edit.html || true
fi
if [[ -f templates/dashboard.html ]]; then
  bk templates/dashboard.html
  perl -0777 -pe "s/<form action=\"\/capture\" method=\"post\" enctype=\"multipart\/form-data\"[^>]*>/<form action=\"\/capture\" method=\"post\" enctype=\"multipart\/form-data\" data-toast=\"Capturing…\">/g" -i templates/dashboard.html || true
fi

echo "Done 034: tag rename endpoint + UI, toasts hardened."
