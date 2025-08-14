#!/usr/bin/env bash
set -euo pipefail
# Describe what this scaffold does (idempotent!)
ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
SCAFF_LOG_FILE="${SCAFF_LOG_FILE:-/dev/null}"

bk(){ [[ -f "$1" ]] && { mkdir -p "$ROOT/.bak"; cp -p "$1" "$ROOT/.bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }
log(){ echo "$(date +"%Y-%m-%d %H:%M:%S") [SCAF] $*" | tee -a "$SCAFF_LOG_FILE"; }

# --- demo: write a harmless 'probe' file once ---
SELF="$(basename "$0")"
NAME="${SELF%.sh}"
TARGET="$ROOT/static/${NAME}.probe"

mkdir -p "$ROOT/static"

if [[ ! -f "$TARGET" ]]; then
  echo "applied $SELF at $STAMP" > "$TARGET"
  log "Wrote $TARGET"
else
  log "Probe already present: $TARGET"
fi

# example of a guarded env append
ENV_FILE="$ROOT/.env"
grep -q '^DEMO_SCAFFOLD=' "$ENV_FILE" 2>/dev/null || {
  bk "$ENV_FILE"
  printf "\n# Added by %s on %s\nDEMO_SCAFFOLD=%s\n" "$SELF" "$STAMP" "$NAME" >> "$ENV_FILE"
  log "Appended DEMO_SCAFFOLD to .env"
}

log "Done."
