#!/usr/bin/env bash
# scripts/scaffolds/scaffold_020.sh
# VERSION endpoint + CI BuildLog + version bump helpers (idempotent)
set -euo pipefail

STAMP="$(date +%Y%m%d-%H%M%S)"
bk(){ [[ -f "$1" ]] && { mkdir -p .bak; cp -p "$1" ".bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }

mkdir -p scripts .github/workflows out/ci_buildlogs

# 0) VERSION file
[[ -f VERSION ]] || { echo "0.1.0" > VERSION; echo "• created VERSION (0.1.0)"; }

# 1) Append /version route to app.py once
if ! grep -q "scaffold_020 additions (VERSION + /version route)" app.py 2>/dev/null; then
  bk app.py
  cat >> app.py <<'PY'

# ==== scaffold_020 additions (VERSION + /version route) ====
import os
from pathlib import Path

try:
    BASE_DIR = settings.base_dir  # from your config.py
except Exception:
    BASE_DIR = Path(__file__).parent

VERSION_FILE = (BASE_DIR / "VERSION")

def _read_version():
    try:
        return VERSION_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return "0.0.0"

@app.get("/version")
def version():
    ver = _read_version()
    commit = os.getenv("APP_BUILD_SHA", "")[:12]
    return {"version": ver, "commit": commit}
PY
fi

# 2) CI BuildLog generator
if [[ ! -f scripts/ci_buildlog.py ]]; then
  cat > scripts/ci_buildlog.py <<'PY'
#!/usr/bin/env python3
import os, subprocess, datetime, pathlib, textwrap
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "out" / "ci_buildlogs"; OUT_DIR.mkdir(parents=True, exist_ok=True)
def run(cmd):
    try:
        return subprocess.run(cmd, shell=True, cwd=ROOT, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()
    except Exception as e:
        return f"(error running {cmd}: {e})"
now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
sha = os.getenv("GITHUB_SHA", run("git rev-parse HEAD")); sha_short = (sha or "")[:12]
ref = os.getenv("GITHUB_REF_NAME", run("git rev-parse --abbrev-ref HEAD"))
repo = os.getenv("GITHUB_REPOSITORY", ROOT.name)
actor = os.getenv("GITHUB_ACTOR",""); runner=os.getenv("RUNNER_NAME","")
python = run("python3 -c 'import sys;print(\".\".join(map(str,sys.version_info[:3])))'")
msg_full = run("git log -1 --pretty=%B") or "(no message)"
msg = msg_full.splitlines()[0]
diffstat = run("git diff --shortstat HEAD~1..HEAD || true")
status = run("git status --porcelain -b")
test_cmd = os.getenv("CI_TEST_CMD","").strip()
test_out = run(test_cmd) if test_cmd else "(no tests run)"
content = f"""---
type: buildlog_ci
repo: "{repo}"
branch: "{ref}"
timestamp: "{now}"
actor: "{actor}"
runner: "{runner}"
commit: "{sha_short}"
message: "{msg.replace('"','\\\"')}"
python: "{python}"
---

# CI Build Log — {msg}

**When:** {now} (UTC)  
**Repo:** {repo} • **Branch:** `{ref}`  
**Commit:** `{sha_short}`

## Diffstat
{diffstat or "(n/a)"}

## git status
{status}

## Test Output
_Command executed:_ `{test_cmd or "(none)"}`
{textwrap.dedent(test_out).strip()}

"""
save_path = os.getenv("SAVE_PATH") or str(OUT_DIR / f"BuildLog_CI_{now.replace(':','-')}.md")
open(save_path,"w",encoding="utf-8").write(content)
print(save_path)
"""
save_path = os.getenv("SAVE_PATH") or str(OUT_DIR / f"BuildLog_CI_{now.replace(':','-')}.md")
open(save_path,"w",encoding="utf-8").write(content)
print(save_path)
PY
  chmod +x scripts/ci_buildlog.py
fi

# 3) GitHub workflow
if [[ ! -f .github/workflows/buildlog.yml ]]; then
  cat > .github/workflows/buildlog.yml <<'YML'
name: CI BuildLog
on:
  push: { branches: ["**"] }
  workflow_dispatch:
    inputs: { test_cmd: { description: "Optional test command", required: false, default: "" } }
permissions: { contents: read }
jobs:
  buildlog:
    runs-on: ubuntu-latest
    env: { APP_BUILD_SHA: ${{ github.sha }}, CI_TEST_CMD: ${{ inputs.test_cmd }} }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - name: Install deps (best-effort)
        run: |
          if [ -f requirements.txt ]; then pip install -r requirements.txt || true; fi
      - name: Optional tests
        if: ${{ inputs.test_cmd != '' }}
        run: |
          set -o pipefail
          bash -lc "${{ inputs.test_cmd }}" | tee out/ci_buildlogs/test_output.txt || true
      - name: Generate CI BuildLog
        run: python scripts/ci_buildlog.py
      - name: Upload BuildLog artifact
        uses: actions/upload-artifact@v4
        with: { name: BuildLog_${{ github.sha }}, path: out/ci_buildlogs/*.md }
YML
fi

# 4) Version bump helper + Makefile targets
if [[ ! -f scripts/bump_version.sh ]]; then
  cat > scripts/bump_version.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-patch}"; FILE="${2:-VERSION}"
[[ -f "$FILE" ]] || echo "0.1.0" > "$FILE"
VER="$(tr -d ' \n\r' < "$FILE")"; IFS='.' read -r MA MI PA <<< "${VER:-0.1.0}"
MA=${MA:-0}; MI=${MI:-1}; PA=${PA:-0}
case "$MODE" in patch) PA=$((PA+1));; minor) MI=$((MI+1)); PA=0;; major) MA=$((MA+1)); MI=0; PA=0;; *) echo "patch|minor|major"; exit 2;; esac
NEW="${MA}.${MI}.${PA}"; echo "$NEW" > "$FILE"; echo "$NEW"
BASH
  chmod +x scripts/bump_version.sh
fi

if [[ -f Makefile ]]; then
  if ! grep -q "Versioning" Makefile; then
    cat >> Makefile <<'MK'

# === Versioning ===
bump-patch: ; @./scripts/bump_version.sh patch
bump-minor: ; @./scripts/bump_version.sh minor
bump-major: ; @./scripts/bump_version.sh major
tag:
	@git add VERSION && git commit -m "chore: bump version to $$(cat VERSION)" || true
	@git tag -a "v$$(cat VERSION)" -m "Release v$$(cat VERSION)"
release: bump-patch tag
MK
  fi
else
  cat > Makefile <<'MK'
PY ?= python3
bump-patch: ; @./scripts/bump_version.sh patch
bump-minor: ; @./scripts/bump_version.sh minor
bump-major: ; @./scripts/bump_version.sh major
tag:
	@git add VERSION && git commit -m "chore: bump version to $$(cat VERSION)" || true
	@git tag -a "v$$(cat VERSION)" -m "Release v$$(cat VERSION)"
release: bump-patch tag
MK
fi

# 5) .gitignore niceties
touch .gitignore
for p in "out/" ".bak/" "BuildLogs/" "out/ci_buildlogs/"; do
  grep -qx "$p" .gitignore || echo "$p" >> .gitignore
done

echo "scaffold_020 complete.

Verify:
  curl -s http://localhost:8084/version
  # Push to GitHub and see CI artifact under Actions → CI BuildLog"
