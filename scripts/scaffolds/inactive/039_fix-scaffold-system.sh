#!/usr/bin/env bash
# scripts/scaffolds/inactive/039_fix-scaffold-system.sh
# Normalize the scaffolding system:
# - ensure dirs (inactive/, active/, state, logs, hooks)
# - install new.sh / promote.sh if missing
# - migrate legacy scaffold_*.sh and scaffold-*.sh into inactive/
# - set git hooksPath and exec-bit pre-commit
# - chmod +x all scaffold scripts
# - idempotent
set -euo pipefail

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
SCAFF_LOG_FILE="${SCAFF_LOG_FILE:-/dev/null}"

SCAFF_DIR="$ROOT/scripts/scaffolds"
INACTIVE_DIR="$SCAFF_DIR/inactive"
ACTIVE_DIR="$SCAFF_DIR/active"
STATE_DIR="$ROOT/.scaffolds"
APPLIED_DIR="$STATE_DIR/applied"
LOG_DIR="$ROOT/.logs/scaffolds"
HOOKS_DIR="$ROOT/.githooks"

log(){ echo "$(date +"%Y-%m-%d %H:%M:%S") [SCAF-039] $*" | tee -a "$SCAFF_LOG_FILE"; }
bk(){ [[ -f "$1" ]] && { mkdir -p "$ROOT/.bak"; cp -p "$1" "$ROOT/.bak/$(basename "$1").$STAMP.bak"; log "backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }
ensure_dir(){ mkdir -p "$1"; log "mkdir -p $1"; }

ensure_line_in_file() {
  local line="$1" file="$2"
  touch "$file"
  if ! grep -qxF "$line" "$file"; then
    echo "$line" >> "$file"
    log "appended to $(basename "$file"): $line"
  else
    log "line already present in $(basename "$file"): $line"
  fi
}

git_mv_or_mv() {
  local src="$1" dest="$2"
  if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git mv "$src" "$dest"
  else
    mv "$src" "$dest"
  fi
}

get_next_num() {
  local max
  max=$(
    { find "$SCAFF_DIR" -maxdepth 1 -name '[0-9][0-9][0-9]_*.sh' -print 2>/dev/null
      find "$INACTIVE_DIR" -maxdepth 1 -name '[0-9][0-9][0-9]_*.sh' -print 2>/dev/null
      find "$ACTIVE_DIR" -maxdepth 1 -name '[0-9][0-9][0-9]_*.sh' -print 2>/dev/null; } \
    | awk -F'/' '{print $NF}' | awk -F'_' '{print $1}' | sed 's/^0*//' | sort -n | tail -1
  )
  [[ -z "${max:-}" ]] && echo 1 || echo $((max+1))
}

sanitize_slug() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/\.sh$//' | sed -E 's/[^a-z0-9]+/-/g' | sed -E 's/^-+|-+$//g'
}

migrate_one() {
  local src="$1"
  [[ -f "$src" ]] || return 0
  local base num slug dest
  base="$(basename "$src")"

  if [[ "$base" =~ ([0-9]{3}) ]]; then
    num="${BASH_REMATCH[1]}"
  else
    printf -v num "%03d" "$(get_next_num)"
  fi

  slug="$base"; slug="${slug%.*}"
  slug="$(echo "$slug" | sed -E 's/^[sS]caffold[-_]*([0-9]{3})?[-_]*//')"
  slug="$(sanitize_slug "$slug")"
  [[ -z "$slug" ]] && slug="migrated"

  ensure_dir "$INACTIVE_DIR"
  dest="$INACTIVE_DIR/${num}_${slug}.sh"

  if [[ "$src" == "$dest" ]]; then
    chmod +x "$dest"; log "ensured exec: $dest"; return 0
  fi

  if [[ -e "$dest" && "$src" != "$dest" ]]; then
    printf -v num "%03d" "$(get_next_num)"
    dest="$INACTIVE_DIR/${num}_${slug}.sh"
  fi

  git_mv_or_mv "$src" "$dest"
  chmod +x "$dest"
  log "migrated -> $dest"
}

# --- 1) ensure directories
ensure_dir "$SCAFF_DIR"
ensure_dir "$INACTIVE_DIR"
ensure_dir "$ACTIVE_DIR"
ensure_dir "$STATE_DIR"
ensure_dir "$APPLIED_DIR"
ensure_dir "$LOG_DIR"
ensure_dir "$HOOKS_DIR"

# --- 2) .gitignore hygiene (keep sources tracked)
GI="$ROOT/.gitignore"
touch "$GI"
ensure_line_in_file "/.logs/" "$GI"
ensure_line_in_file "/.scaffolds/" "$GI"
ensure_line_in_file "/.bak/" "$GI"
ensure_line_in_file "*.bak" "$GI"

# --- 3) pre-commit hook to enforce exec bits for scaffolds
PC="$HOOKS_DIR/pre-commit"
if [[ ! -f "$PC" ]]; then
  cat > "$PC" <<'HEOF'
#!/usr/bin/env bash
set -euo pipefail
changed=$(git diff --cached --name-only --diff-filter=ACM | grep '^scripts/scaffolds/.*\.sh$' || true)
for f in $changed; do
  git update-index --chmod=+x "$f"
  echo "• set exec bit: $f"
done
HEOF
  chmod +x "$PC"
  log "installed .githooks/pre-commit"
else
  if ! grep -q 'update-index --chmod=+x' "$PC"; then
    cat >> "$PC" <<'HAPPEND'

# ensure exec bit for scaffold scripts
changed=$(git diff --cached --name-only --diff-filter=ACM | grep '^scripts/scaffolds/.*\.sh$' || true)
for f in $changed; do
  git update-index --chmod=+x "$f"
  echo "• set exec bit: $f"
done
HAPPEND
    chmod +x "$PC"
    log "augmented existing pre-commit hook"
  else
    log "pre-commit already contains exec-bit rule"
  fi
fi

# set hooksPath (local repo)
if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$ROOT" config core.hooksPath .githooks || true
  log "git core.hooksPath set to .githooks"
else
  log "not a git repo; skipped hooksPath"
fi

# --- 4) install helpers if missing: new.sh + promote.sh (drafts -> pending)
NEW="$SCAFF_DIR/new.sh"
PROM="$SCAFF_DIR/promote.sh"

if [[ ! -f "$NEW" ]]; then
  cat > "$NEW" <<'NEOF'
#!/usr/bin/env bash
# scripts/scaffolds/new.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INACTIVE_DIR="$DIR/inactive"
mkdir -p "$INACTIVE_DIR"

name="${*:-}"
[[ -z "$name" ]] && { echo "Usage: $0 <short-name-like-brand-env>"; exit 1; }

next=$(
  { find "$DIR" -maxdepth 1 -name '[0-9][0-9][0-9]_*.sh' -print 2>/dev/null
    find "$INACTIVE_DIR" -maxdepth 1 -name '[0-9][0-9][0-9]_*.sh' -print 2>/dev/null
    find "$DIR/active" -maxdepth 1 -name '[0-9][0-9][0-9]_*.sh' -print 2>/dev/null; } \
  | awk -F'/' '{print $NF}' | awk -F'_' '{print $1}' | sed 's/^0*//' | sort -n | tail -1
)
[[ -z "${next:-}" ]] && next=1 || next=$((next+1))
printf -v num "%03d" "$next"

file="$INACTIVE_DIR/${num}_${name}.sh"
cat > "$file" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
SCAFF_LOG_FILE="${SCAFF_LOG_FILE:-/dev/null}"

bk(){ [[ -f "$1" ]] && { mkdir -p "$ROOT/.bak"; cp -p "$1" "$ROOT/.bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }
log(){ echo "$(date +"%Y-%m-%d %H:%M:%S") [SCAF] $*" | tee -a "$SCAFF_LOG_FILE"; }

# TODO: your changes here

log "Done."
EOF

chmod +x "$file"
echo "Created draft scaffold: $file"
NEOF
  chmod +x "$NEW"; log "installed scripts/scaffolds/new.sh"
else
  log "new.sh already present"
fi

if [[ ! -f "$PROM" ]]; then
  cat > "$PROM" <<'PEOF'
#!/usr/bin/env bash
# scripts/scaffolds/promote.sh [--renumber] <name-or-file>
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INACTIVE_DIR="$DIR/inactive"
REN=0
if [[ "${1:-}" == "--renumber" ]]; then REN=1; shift; fi
key="${1:-}"

[[ -z "$key" ]] && { echo "Usage: $0 [--renumber] <name-or-file>"; exit 1; }
[[ -d "$INACTIVE_DIR" ]] || { echo "No inactive dir: $INACTIVE_DIR"; exit 1; }

if [[ -f "$key" ]]; then
  src="$key"
else
  src="$(find "$INACTIVE_DIR" -maxdepth 1 -type f -name "*${key}*.sh" | sort | head -n1 || true)"
fi
[[ -z "${src:-}" ]] && { echo "No matching draft in $INACTIVE_DIR for: $key"; exit 1; }

base="$(basename "$src")"
num="${base%%_*}"
slug="${base#*_}"

dest="$DIR/${num}_${slug}"   # pending (top-level)
if [[ -e "$dest" ]]; then
  if [[ $REN -eq 0 ]]; then
    echo "Active file exists: $dest. Use --renumber to assign a new number."
    exit 1
  fi
  next=$(
    { find "$DIR" -maxdepth 1 -name '[0-9][0-9][0-9]_*.sh' -print 2>/dev/null
      find "$INACTIVE_DIR" -maxdepth 1 -name '[0-9][0-9][0-9]_*.sh' -print 2>/dev/null
      find "$DIR/active" -maxdepth 1 -name '[0-9][0-9][0-9]_*.sh' -print 2>/dev/null; } \
  | awk -F'/' '{print $NF}' | awk -F'_' '{print $1}' | sed 's/^0*//' | sort -n | tail -1
  )
  [[ -z "${next:-}" ]] && next=1 || next=$((next+1))
  printf -v num "%03d" "$next"
  dest="$DIR/${num}_${slug}"
fi

if git -C "$DIR/../.." rev-parse --show-toplevel >/dev/null 2>&1; then
  git mv "$src" "$dest"
else
  mv "$src" "$dest"
fi
chmod +x "$dest"
echo "Promoted (pending): $(basename "$dest")"
PEOF
  chmod +x "$PROM"; log "installed scripts/scaffolds/promote.sh"
else
  log "promote.sh already present"
fi

# --- 5) migrate legacy files into inactive/
for p in \
  "$ROOT"/scaffold_*.sh "$ROOT"/scaffold-*.sh \
  "$SCAFF_DIR"/scaffold_*.sh "$SCAFF_DIR"/scaffold-*.sh
do
  [[ -e "$p" ]] || continue
  migrate_one "$p"
done

# --- 6) chmod +x every scaffold (pending, inactive, active)
find "$SCAFF_DIR" -type f -name '*.sh' -print0 2>/dev/null | xargs -0 -I{} chmod +x "{}" || true
log "ensured +x on all scaffolds"

log "039_fix-scaffold-system complete."
