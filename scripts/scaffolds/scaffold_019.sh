#!/usr/bin/env bash
# scripts/scaffold_019.sh
# Git push + BuildLog automation for second_brain
set -euo pipefail

STAMP="$(date +%Y%m%d-%H%M%S)"
b(){ [[ -f "$1" ]] && { mkdir -p .bak; cp -p "$1" ".bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }

mkdir -p scripts out/build_logs .githooks
touch .gitignore

# 1) BuildLog & commit runner
b scripts/buildlog_commit.sh
cat > scripts/buildlog_commit.sh <<'BASH'
#!/usr/bin/env bash
# Usage:
#   scripts/buildlog_commit.sh -m "feat: thing" [-b main] [--run "make test"] [--no-push]
# Env:
#   OBSIDIAN_VAULT_DIR=~/Obsidian/SecondBrain   # where to mirror BuildLogs (optional)
#   OBSIDIAN_VAULT_GIT=1                        # commit/push in the vault if it's a git repo (optional)
#   BUILDLOG_DIR_IN_REPO=BuildLogs              # where to store BuildLogs in repo (default BuildLogs/)
set -euo pipefail

MSG=""
BRANCH=""
RUN_CMD=""
PUSH=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--message) MSG="${2:-}"; shift 2;;
    -b|--branch)  BRANCH="${2:-}"; shift 2;;
    -r|--run|--capture|--cmd) RUN_CMD="${2:-}"; shift 2;;
    --no-push)    PUSH=0; shift;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

if [[ -z "${MSG}" ]]; then
  echo "Please provide -m|--message" >&2; exit 2
fi

# repo root
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then echo "Not inside a git repo." >&2; exit 1; fi
cd "$ROOT"

# pick branch
CUR_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
TARGET_BRANCH="${BRANCH:-$CUR_BRANCH}"

# preflight info
USER_NAME="$(git config user.name || true)"
USER_EMAIL="$(git config user.email || true)"
REMOTE_URL="$(git config --get remote.origin.url || true)"
HOST="$(hostname -s 2>/dev/null || echo unknown)"
NOW_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PYVER="$(python3 -c 'import sys;print(".".join(map(str,sys.version_info[:3])))' 2>/dev/null || echo none)"
GITVER="$(git --version | awk '{print $3}' 2>/dev/null || echo none)"

# optional code style / checks
if [[ -f Makefile ]]; then
  if grep -qE '^fmt:' Makefile; then
    echo "→ make fmt"; make -s fmt || true
  fi
  if grep -qE '^test:' Makefile; then
    echo "→ make test"; make -s test || true
  fi
fi

# Optional custom command to capture (tests, scaffolds, etc.)
LOGDIR="out/build_logs"
mkdir -p "$LOGDIR"
RUN_LOG="$LOGDIR/run_${NOW_ISO//[:]/-}.log"
{
  echo "\$ git status --porcelain -b"
  git status --porcelain -b
  if [[ -n "$RUN_CMD" ]]; then
    echo
    echo "\$ $RUN_CMD"
    # Prefer 'script' to capture full TTY if available
    if command -v script >/dev/null 2>&1; then
      script -q /dev/null bash -lc "$RUN_CMD" 2>&1
    else
      bash -lc "$RUN_CMD" 2>&1
    fi
  fi
} | tee "$RUN_LOG" >/dev/null

# stage & commit
PRE_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo 'no-commit')"
git add -A
set +e
DIFFSTAT="$(git diff --cached --shortstat)"
set -e
if [[ -z "$DIFFSTAT" ]]; then
  echo "No staged changes. Creating BuildLog anyway." >&2
else
  echo "→ committing: $MSG"
  git commit -m "$MSG"
fi

# ensure branch exists and is current
if [[ "$TARGET_BRANCH" != "$CUR_BRANCH" ]]; then
  git checkout -B "$TARGET_BRANCH"
fi

# push (optional)
PUSHED_SHA=""
if [[ "$PUSH" -eq 1 ]]; then
  echo "→ pushing to $REMOTE_URL ($TARGET_BRANCH)"
  # Set upstream if needed
  set +e
  git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >/dev/null 2>&1
  HAS_UPSTREAM=$?
  set -e
  if [[ $HAS_UPSTREAM -ne 0 ]]; then
    git push -u origin "$TARGET_BRANCH"
  else
    git pull --rebase origin "$TARGET_BRANCH" || true
    git push origin "$TARGET_BRANCH"
  fi
  PUSHED_SHA="$(git rev-parse --short HEAD)"
fi

POST_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo 'no-commit')"

# Build BuildLog.md
REPO_NAME="$(basename -s .git "$(git config --get remote.origin.url 2>/dev/null || echo "$ROOT")" || echo repo)"
BL_DIR="${BUILDLOG_DIR_IN_REPO:-BuildLogs}"
mkdir -p "$BL_DIR"
TITLE_MSG="${MSG//$'\n'/ }"
TITLE_MSG="${TITLE_MSG//\"/\\\"}"

# get changed file list (staged in this commit) if any
set +e
LAST_FILES="$(git show --pretty="" --name-status HEAD 2>/dev/null)"
set -e

BL_FILE="${BL_DIR}/BuildLog_${NOW_ISO//[:]/-}.md"
cat > "$BL_FILE" <<MD
---
type: buildlog
repo: "$REPO_NAME"
branch: "$TARGET_BRANCH"
user: "$(whoami)"
host: "$HOST"
timestamp: "$NOW_ISO"
python: "$PYVER"
git: "$GITVER"
remote: "${REMOTE_URL}"
commit_before: "$PRE_SHA"
commit_after: "$POST_SHA"
changes: "${DIFFSTAT:-none}"
message: "$TITLE_MSG"
---

# Build Log — ${TITLE_MSG}

**When:** $NOW_ISO (UTC)  
**Repo:** $REPO_NAME • **Branch:** \`$TARGET_BRANCH\`  
**Commit:** \`$PRE_SHA\` → \`$POST_SHA\`

## Summary
- Message: **${MSG}**
- Diffstat: ${DIFFSTAT:-_no staged changes_}
- Remote: \`${REMOTE_URL:-none}\`

## Changed files (last commit)
\`\`\`text
${LAST_FILES:-none}
\`\`\`

## git status (pre-commit)
\`\`\`text
$(sed -n '1,100p' "$RUN_LOG")
\`\`\`

## Command output
_Command executed:_ \`${RUN_CMD:-none}\`

\`\`\`text
$(if [[ -n "$RUN_CMD" ]]; then sed -n '101,99999p' "$RUN_LOG"; else echo "(no command captured)"; fi)
\`\`\`

## Notes
- Keep these BuildLogs inside the repo (\`${BL_DIR}/\`) so they show up in Obsidian if your vault is pointed at the repo.
- If your vault lives elsewhere, we can mirror this file (see below).
MD

echo "✓ BuildLog written: $BL_FILE"

# Mirror to Obsidian vault (optional)
VAULT_DIR="${OBSIDIAN_VAULT_DIR:-}"
if [[ -n "$VAULT_DIR" ]]; then
  DEST_DIR="${VAULT_DIR%/}/BuildLogs"
  mkdir -p "$DEST_DIR"
  cp -f "$BL_FILE" "$DEST_DIR/"
  echo "✓ Mirrored to Obsidian: $DEST_DIR/$(basename "$BL_FILE")"
  # Optional: git commit & push inside the vault
  if [[ "${OBSIDIAN_VAULT_GIT:-0}" = "1" && -d "$DEST_DIR/../.git" ]]; then
    ( set -e
      cd "$VAULT_DIR"
      git add "BuildLogs/$(basename "$BL_FILE")"
      git commit -m "BuildLog: ${MSG}" || true
      git pull --rebase || true
      git push || true
    )
    echo "✓ Vault git push complete."
  fi
fi

echo "All done."
BASH
chmod +x scripts/buildlog_commit.sh

# 2) Makefile helpers (commit/push/buildlog)
if [[ -f Makefile ]]; then b Makefile; fi
cat >> Makefile <<'MK'

# === BuildLog & Git helpers ===
buildlog:
	@bash scripts/buildlog_commit.sh -m "$(MSG)" $(ARGS)

commit:
	@git add -A && git commit -m "$(MSG)"

push:
	@git pull --rebase || true && git push

# Examples:
#   make buildlog MSG="feat: add X" ARGS='--run "bash scripts/scaffold_018.sh"'
#   OBSIDIAN_VAULT_DIR=~/Obsidian/SecondBrain make buildlog MSG="chore: tidy" ARGS="--no-push"
MK

# 3) Git hook (post-commit) to auto-create a BuildLog for each commit message
cat > .githooks/post-commit <<'HOOK'
#!/usr/bin/env bash
set -euo pipefail
MSG="$(git log -1 --pretty=%B | head -n1)"
# Avoid infinite loops: do not run if the commit is a BuildLog itself
if echo "$MSG" | grep -qi 'buildlog'; then exit 0; fi
# Only run if our script exists
if [[ -x scripts/buildlog_commit.sh ]]; then
  # no push from hook; skip extra command
  scripts/buildlog_commit.sh -m "$MSG" --no-push || true
fi
HOOK
chmod +x .githooks/post-commit
git config core.hooksPath .githooks || true

# 4) .gitignore niceties
if ! grep -q "^out/" .gitignore; then echo "out/" >> .gitignore; fi
if ! grep -q "^.bak/" .gitignore; then echo ".bak/" >> .gitignore; fi

echo "Done.

Next:
  1) Ensure your remote is set:
       git remote -v
       # if missing:
       # git remote add origin git@github.com:<you>/<repo>.git
       # git push -u origin $(git rev-parse --abbrev-ref HEAD)

  2) Create your first BuildLog:
       bash scripts/buildlog_commit.sh -m \"feat: initial BuildLog wiring\" --run \"bash scripts/scaffold_018.sh\"

     Or with Make:
       make buildlog MSG=\"feat: polish\" ARGS='--run \"pytest -q\"'

  3) (Optional) Mirror BuildLogs to your Obsidian vault:
       export OBSIDIAN_VAULT_DIR=~/Obsidian/SecondBrain
       export OBSIDIAN_VAULT_GIT=1   # if that folder is also a git repo
       bash scripts/buildlog_commit.sh -m \"feat: mirror to vault\"

  4) Auto BuildLogs after each commit:
       # already enabled via .githooks/post-commit
       # to disable: rm .githooks/post-commit or unset hooksPath
"
