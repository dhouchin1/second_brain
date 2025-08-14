#!/usr/bin/env bash
# scripts/scaffolds/new.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INACTIVE_DIR="$DIR/inactive"
mkdir -p "$INACTIVE_DIR"

name="${*:-}"
[[ -z "$name" ]] && { echo "Usage: $0 <short-name-like-brand-env>"; exit 1; }

# next number across active + inactive
next=$(
  { find "$DIR" -maxdepth 1 -name '[0-9][0-9][0-9]_*.sh' -print; \
    find "$INACTIVE_DIR" -maxdepth 1 -name '[0-9][0-9][0-9]_*.sh' -print; } 2>/dev/null \
  | awk -F'/' '{print $NF}' | awk -F'_' '{print $1}' | sed 's/^0*//' | sort -n | tail -1
)
[[ -z "${next:-}" ]] && next=1 || next=$((next+1))
printf -v num "%03d" "$next"

file="$INACTIVE_DIR/${num}_${name}.sh"
cat > "$file" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
# Describe what this scaffold does (idempotent!)
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
