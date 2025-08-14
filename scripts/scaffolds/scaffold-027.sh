#!/usr/bin/env bash
# scripts/scaffolds/scaffold_027.sh
# GitHub “apply scaffold → commit → PR → approve → merge → delete branch” automation
set -euo pipefail

STAMP="$(date +%Y%m%d-%H%M%S)"
bk(){ [[ -f "$1" ]] && { mkdir -p .bak; cp -p "$1" ".bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }
say(){ printf "\033[1;36m%s\033[0m\n" "$*"; }

mkdir -p scripts

# 1) Add a reusable script that applies any scaffold (sh), commits, PRs and merges it.
cat > scripts/gh_scaffold_apply.sh <<'SH'
#!/usr/bin/env bash
# Usage:
#   scripts/gh_scaffold_apply.sh scripts/scaffolds/scaffold_028.sh "feat(ui): overhaul 1"
# Requires:
#   - gh CLI authenticated (gh auth status)
#   - remote "origin" set
#   - default branch main/master detectable
set -euo pipefail
SCRIPT="${1:-}"; MSG="${2:-}"
if [[ -z "$SCRIPT" || -z "$MSG" ]]; then
  echo "Usage: scripts/gh_scaffold_apply.sh <scaffold.sh> <commit message>"; exit 2
fi
if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI 'gh' is required. Install: https://cli.github.com/"; exit 1
fi

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -z "$ROOT" ]] && { echo "Run inside git repo."; exit 1; }
cd "$ROOT"

BASE="$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@' || true)"
[[ -z "$BASE" ]] && BASE="$(git branch -r | grep 'origin/main' >/dev/null && echo main || echo master)"

BR="scaff/$(basename "$SCRIPT" .sh)-$(date +%Y%m%d-%H%M%S)"

echo "→ Base branch: $BASE"
echo "→ New branch: $BR"

git fetch origin "$BASE" --quiet || true
git checkout -B "$BR" "origin/$BASE" 2>/dev/null || git checkout -B "$BR" "$BASE"

# Run scaffold
bash "$SCRIPT"

# Stage, commit
git add -A
if git diff --cached --quiet; then
  echo "No changes from $SCRIPT. Aborting PR."
  git checkout "$BASE"; git branch -D "$BR" || true
  exit 0
fi

git commit -m "$MSG"

# Push and create PR
git push -u origin "$BR"
# PR title/body
TITLE="$MSG"
BODY="Automated application of \`$SCRIPT\`.\n\n- Generated at: $(date -u +%Y-%m-%dT%H:%M:%SZ)\n- Branch: \`$BR\`"

# Create PR; if already exists, reuse
if ! gh pr view "$BR" >/dev/null 2>&1; then
  gh pr create --title "$TITLE" --body "$BODY" --base "$BASE" --head "$BR"
fi

# Approve + merge (squash) + delete branch
gh pr review "$BR" --approve || true
gh pr merge "$BR" --squash --delete-branch --admin || gh pr merge "$BR" --squash --delete-branch || true

# Return to base
git checkout "$BASE"
git pull --rebase
echo "✓ Merged $BR into $BASE."
SH
chmod +x scripts/gh_scaffold_apply.sh

# 2) Makefile convenience
if [[ -f Makefile ]]; then bk Makefile; fi
cat >> Makefile <<'MK'

# === Scaffolds → PR automation ===
# Example:
#   make ship S=scripts/scaffolds/scaffold_028.sh M="feat(ui): overhaul"
ship:
	@bash scripts/gh_scaffold_apply.sh "$(S)" "$(M)"

# Apply multiple scaffolds in order:
#   make ship_many S="scripts/scaffolds/scaffold_022.sh scripts/scaffolds/scaffold_023.sh" M="feat: batch scaffolds"
ship_many:
	@for s in $(S); do echo "==> $$s"; bash scripts/gh_scaffold_apply.sh "$$s" "$(M) - $$(basename $$s)"; done
MK

echo "Done 027.

Now you can:
  make ship S=scripts/scaffolds/scaffold_028.sh M=\"feat(ui): overhaul\"
or batch:
  make ship_many S=\"scripts/scaffolds/scaffold_022.sh scripts/scaffolds/scaffold_023.sh\" M=\"feat: env+llm\"
"
