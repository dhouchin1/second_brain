#!/usr/bin/env bash
set -Eeuo pipefail
command -v git >/dev/null || { echo "git required"; exit 1; }
command -v gh  >/dev/null || { echo "gh required";  exit 1; }

TOP="$(git rev-parse --show-toplevel)"
cd "$TOP"

BASE="$(git rev-parse --abbrev-ref HEAD)"
[[ "$BASE" =~ ^integration/ ]] || { echo "Run from an integration/* branch."; exit 1; }

mkdir -p integration-logs scripts/scaffolds/active scripts/scaffolds/inactive

LABEL="scaffold"
gh label create "$LABEL" --color "633974" --description "Scaffold application" 2>/dev/null || true

# Build ordered list of pending scaffolds (top-level only; ignore inactive/ & active/)
PENDING=$(
  find "scripts/scaffolds" -maxdepth 1 -type f -name 'scaffold-*.sh' -print 2>/dev/null \
  | while IFS= read -r f; do
      b="$(basename "$f")"
      n="$(echo "$b" | sed -E 's/^scaffold-([0-9]+).*/\1/')"  # extract number
      printf "%08d|%s\n" "${n:-0}" "$f"
    done \
  | sort -t'|' -k1,1n | cut -d'|' -f2
)

[ -n "$PENDING" ] || { echo "No scaffold scripts found."; exit 0; }

git fetch --all --prune

echo "$PENDING" | while IFS= read -r f; do
  [ -n "$f" ] || continue
  base="$(basename "$f")"
  num="$(echo "$base" | sed -E 's/^scaffold-([0-9]+).*/\1/')"
  title_line="$(grep -E '^##[[:space:]]*Title:' "$f" 2>/dev/null | head -1 || true)"
  slug="$(printf "%s" "$title_line" | sed -E 's/^##[[:space:]]*Title:[[:space:]]*//' | tr '[:upper:] ' '[:lower:]-' | tr -cd 'a-z0-9-')"
  [ -z "$slug" ] && slug="step-${num:-unknown}"
  BR="scaffold/${num}-${slug}"

  git checkout "$BASE"
  git pull --ff-only
  git branch -D "$BR" 2>/dev/null || true
  git checkout -b "$BR" "$BASE"

  LOG="integration-logs/${base%.sh}.log"
  chmod +x "$f"

  if "$f" >"$LOG" 2>&1; then
    echo "✅ $base"
  else
    echo "❌ $base failed. See $LOG"
    git reset --hard
    continue
  fi

  # Move to active/ after successful apply (in the same branch/PR)
  dest="scripts/scaffolds/active/$base"
  mkdir -p "scripts/scaffolds/active"
  git mv "$f" "$dest" 2>/dev/null || mv "$f" "$dest"

  if git status --porcelain | grep -q .; then
    git add -A
    git commit -m "feat(scaffold): apply ${num} (${slug})" -m "Log: ${LOG}"
    git push -u origin "$BR"
    gh pr create --head "$BR" --base "$BASE" \
      --title "Scaffold ${num}: ${slug} → ${BASE}" \
      --body "Applies scaffold ${num} and moves it to active/. See ${LOG}." \
      --label "$LABEL" --label scaffolds || true
  else
    echo "(no changes from $base)"
  fi
done
