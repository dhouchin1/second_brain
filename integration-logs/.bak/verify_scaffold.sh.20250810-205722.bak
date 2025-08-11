#!/usr/bin/env bash
set -euo pipefail
ok(){ printf "✅ %s\n" "$*"; }
warn(){ printf "⚠️  %s\n" "$*" >&2; FAIL=1; }
FAIL=0

need_file(){ [[ -f "$1" ]] && ok "exists: $1" || warn "missing: $1"; }

need_file app.py
need_file templates/base.html
need_file templates/dashboard.html
need_file templates/notes.html
need_file templates/search.html
need_file templates/tags.html
need_file templates/note_detail.html
need_file templates/settings.html
need_file templates/export.html
need_file templates/embeddings.html
need_file templates/partials/note_item.html
need_file templates/partials/note_row_selectable.html
need_file templates/partials/note_edit_form.html
need_file templates/partials/note_view.html
need_file templates/partials/scroll_sentinel.html
need_file templates/partials/search_results.html
need_file templates/partials/emb_progress.html
need_file static/app.js

grep -q "/search" app.py && ok "endpoint: /search" || warn "endpoint missing: /search"
grep -q "/notes\"" app.py && ok "endpoint: /notes" || warn "endpoint missing: /notes"
grep -q "/tags\"" app.py && ok "endpoint: /tags" || warn "endpoint missing: /tags"
grep -q "/export" app.py && ok "endpoint: /export" || warn "endpoint missing: /export"
grep -q "/embeddings" app.py && ok "endpoint: /embeddings" || warn "endpoint missing: /embeddings"
grep -q "notes_fts" app.py && ok "sqlite FTS present" || warn "sqlite FTS missing"
grep -q "note_embeddings" app.py && ok "embeddings table present" || warn "embeddings table missing"

if [[ $FAIL -eq 0 ]]; then
  echo "All checks passed."
  exit 0
else
  echo "Some checks failed. See warnings above."
  exit 1
fi
