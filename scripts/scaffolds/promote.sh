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

# resolve candidate
if [[ -f "$key" ]]; then
  src="$key"
else
  # allow "042_brand-env.sh" or "brand-env"
  src="$(find "$INACTIVE_DIR" -maxdepth 1 -type f -name "*${key}*.sh" | sort | head -n1 || true)"
fi
[[ -z "${src:-}" ]] && { echo "No matching draft in $INACTIVE_DIR for: $key"; exit 1; }

base="$(basename "$src")"
num="${base%%_*}"
slug="${base#*_}"

dest="$DIR/${num}_${slug}"
if [[ -e "$dest" ]]; then
  if [[ $REN -eq 0 ]]; then
    echo "Active file exists: $dest. Use --renumber to assign a new number."
    exit 1
  fi
  next=$(
    { find "$DIR" -maxdepth 1 -name '[0-9][0-9][0-9]_*.sh' -print; \
      find "$INACTIVE_DIR" -maxdepth 1 -name '[0-9][0-9][0-9]_*.sh' -print; } 2>/dev/null \
    | awk -F'/' '{print $NF}' | awk -F'_' '{print $1}' | sed 's/^0*//' | sort -n | tail -1
  )
  [[ -z "${next:-}" ]] && next=1 || next=$((next+1))
  printf -v num "%03d" "$next"
  dest="$DIR/${num}_${slug}"
fi

# prefer git mv when possible
if git -C "$DIR/../.." rev-parse --show-toplevel >/dev/null 2>&1; then
  git mv "$src" "$dest"
else
  mv "$src" "$dest"
fi
chmod +x "$dest"
echo "Promoted: $(basename "$dest")"
