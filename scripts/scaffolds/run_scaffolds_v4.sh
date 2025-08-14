#!/opt/homebrew/bin/bash
set -Eeuo pipefail

# Requirements
command -v gh >/dev/null || { echo "gh CLI required"; exit 1; }

# Must run from the integration worktree
TOP="$(git rev-parse --show-toplevel)"; cd "$TOP"
BASE="$(git rev-parse --abbrev-ref HEAD)"
[[ "$BASE" =~ ^integration/ ]] || { echo "Run from an integration/* branch (current: $BASE)"; exit 1; }

# Labels & logs
mkdir -p integration-logs
gh label create scaffold  --color "633974" --description "Scaffold application" 2>/dev/null || true
gh label create scaffolds --color "633974" --description "Scaffold-related"     2>/dev/null || true

# Find scaffold scripts in either location
mapfile -t SCRIPTS < <(
  { ls -1 scripts/scaffolds/scaffold-*.sh 2>/dev/null; ls -1 scripts/scaffold-*.sh 2>/dev/null; } \
  | sort -V | awk '!seen[$0]++'
)
(( ${#SCRIPTS[@]} > 0 )) || { echo "No scaffold scripts found."; exit 1; }

git fetch --all --prune

for f in "${SCRIPTS[@]}"; do
  base="$(basename "$f")"
  num="${base#scaffold-}"; num="${num%.sh}"

  # Slug is optional; tolerate missing Title:
  slug="$(grep -E '^##[[:space:]]*Title:' "$f" | head -1 \
          | sed -E 's/^##[[:space:]]*Title:[[:space:]]*//' \
          | tr '[:upper:][:space:]' '[:lower:]-' \
          | tr -cd 'a-z0-9-')" || true
  [[ -n "${slug:-}" ]] || slug="step-${num}"

  BR="scaffold/${num}-${slug}"

  # Skip if an OPEN PR already exists for this head/base
  if gh pr list --base "$BASE" --head "$BR" --state open --json number --jq '.[0].number' | grep -q .; then
    echo "↷ PR already open for $BR → $BASE, skipping."
    continue
  fi

  git checkout "$BASE"
  git pull --ff-only
  git branch -D "$BR" 2>/dev/null || true
  git checkout -b "$BR" "$BASE"

  # Make sure logs dir survives even after any resets
  mkdir -p integration-logs
  LOG="integration-logs/${base%.sh}.log"

  chmod +x "$f"
  if "$f" >"$LOG" 2>&1; then
    echo "✅ $base"
  else
    echo "❌ $base failed. See $LOG"
    git reset --hard
    git checkout "$BASE"
    continue
  fi

  if [[ -n "$(git status --porcelain)" ]]; then
    git add -A
    git commit -m "feat(scaffold): apply ${num} (${slug})" -m "Log: ${LOG}"
    git push -u origin "$BR"
    gh pr create --head "$BR" --base "$BASE" \
      --title "Scaffold ${num}: ${slug} → ${BASE}" \
      --body "Applies scaffold ${num}. See ${LOG}." \
      --label scaffold --label scaffolds || true
  else
    echo "(no changes from $base)"
    git checkout "$BASE"
  fi
done
