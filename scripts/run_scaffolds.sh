#!/usr/bin/env bash
# scripts/run_scaffolds.sh
set -euo pipefail

# --- repo root ---
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -z "${ROOT}" ]] && ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SCAFF_DIR="$ROOT/scripts/scaffolds"
INACTIVE_DIR="$SCAFF_DIR/inactive"
ACTIVE_DIR="$SCAFF_DIR/active"          # <- new: where applied scaffolds live
STATE_DIR="$ROOT/.scaffolds"
APPLIED_DIR="$STATE_DIR/applied"
LOG_DIR="$ROOT/.logs/scaffolds"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/run-$STAMP.log"

mkdir -p "$STATE_DIR" "$APPLIED_DIR" "$LOG_DIR" "$SCAFF_DIR" "$INACTIVE_DIR" "$ACTIVE_DIR"

# --- logging ---
is_tty() { [[ -t 1 ]]; }
ts() { date +"%Y-%m-%d %H:%M:%S"; }
if is_tty; then
  c_info=$'\e[36m'; c_ok=$'\e[32m'; c_warn=$'\e[33m'; c_err=$'\e[31m'; c_off=$'\e[0m'
else
  c_info=""; c_ok=""; c_warn=""; c_err=""; c_off=""
fi
log(){ echo "$(ts) $1" | tee -a "$LOG_FILE"; }
info(){ log "${c_info}[INFO]${c_off} $*"; }
ok(){ log "${c_ok}[ OK ]${c_off} $*"; }
warn(){ log "${c_warn}[WARN]${c_off} $*"; }
err(){ log "${c_err}[ERR ]${c_off} $*"; }

trap 'err "Scaffold run failed (see $LOG_FILE)"; exit 1' ERR

# --- single-run lock (mac-safe fallback if no flock) ---
if command -v flock >/dev/null 2>&1; then
  exec 9>"$STATE_DIR/.lock"
  flock -n 9 || { err "Another scaffold run is in progress."; exit 1; }
else
  LOCK_DIR="$STATE_DIR/lock"
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    info "Using lock directory fallback (no flock present)"
    trap 'rm -rf "$LOCK_DIR" >/dev/null 2>&1 || true' EXIT
  else
    err "Another scaffold run is in progress."
    exit 1
  fi
fi

# --- helpers ---
hash_file() {
  local f="$1"
  if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$f" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then sha256sum "$f" | awk '{print $1}'
  elif command -v md5 >/dev/null 2>&1; then md5 -q "$f"
  else awk '{sum+=gsub(/./,"&")} END{print sum}' "$f"
  fi
}

bk(){ [[ -f "$1" ]] && { mkdir -p "$ROOT/.bak"; cp -p "$1" "$ROOT/.bak/$(basename "$1").$STAMP.bak"; info "backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }

git_mv_or_mv() {
  local src="$1" dest="$2"
  if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git mv "$src" "$dest"
  else
    mv "$src" "$dest"
  fi
}

move_to_active() {
  local src="$1"
  local base="$(basename "$src")"
  local dest="$ACTIVE_DIR/$base"
  if [[ -e "$dest" && "$dest" != "$src" ]]; then
    dest="$ACTIVE_DIR/${base%.sh}-$STAMP.sh"
  fi
  git_mv_or_mv "$src" "$dest"
  chmod +x "$dest" || true
  info "moved to active: $(basename "$dest")"
}

# --- discover PENDING scripts (top-level only; ignore inactive/ and active/) ---
mapfile -t scripts < <(find "$SCAFF_DIR" -maxdepth 1 -type f -name '[0-9][0-9][0-9]_*.sh' -print | sort)

if [[ ${#scripts[@]} -eq 0 ]]; then
  warn "No pending scaffolds in $SCAFF_DIR (drafts: $INACTIVE_DIR, applied: $ACTIVE_DIR)"
  exit 0
fi

info "Log: $LOG_FILE"
info "Found ${#scripts[@]} pending scaffold(s) in $SCAFF_DIR"

# ensure executability
for s in "${scripts[@]}"; do
  if [[ ! -x "$s" ]]; then
    chmod +x "$s"
    info "chmod +x $(basename "$s")"
  fi
done

# --- dry-run support ---
DRY_RUN="${1:-}"
if [[ "$DRY_RUN" == "--dry-run" ]]; then
  info "Dry run: will list what would run (no execution/move)."
fi

# --- run new or changed scripts ---
ran_any=0
for s in "${scripts[@]}"; do
  base="$(basename "$s")"
  sum="$(hash_file "$s")"
  marker="$APPLIED_DIR/$base.$sum.ok"

  if [[ -f "$marker" ]]; then
    info "skip $base (already applied; hash $sum)"
    # even if previously applied, ensure it lives in active/
    if [[ "$DRY_RUN" != "--dry-run" ]]; then
      [[ "$(dirname "$s")" == "$SCAFF_DIR" ]] && move_to_active "$s"
    fi
    continue
  fi

  info "apply $base (hash $sum)"
  if [[ "$DRY_RUN" == "--dry-run" ]]; then
    continue
  fi

  export ROOT STAMP
  export SCAFF_LOG_FILE="$LOG_FILE"

  if bash "$s"; then
    touch "$marker"
    ok "applied $base"
    ran_any=1
    move_to_active "$s"
  else
    err "failed $base"
    exit 1
  fi
done

[[ "$DRY_RUN" == "--dry-run" ]] && { ok "dry run complete"; exit 0; }

if [[ $ran_any -eq 0 ]]; then
  ok "nothing to do; all pending scaffolds already applied"
else
  ok "scaffold run complete"
fi
