#!/usr/bin/env bash
# scripts/scaffold_020.sh
# CI BuildLog + VERSION route + bump tools
set -euo pipefail

STAMP="$(date +%Y%m%d-%H%M%S)"
b(){ [[ -f "$1" ]] && { mkdir -p .bak; cp -p "$1" ".bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }

mkdir -p scripts .github/workflows out/ci_buildlogs

# 0) VERSION file (create if missing)
if [[ ! -f VERSION ]]; then
  echo "0.1.0" > VERSION
  echo "• created VERSION (0.1.0)"
fi

# 1) Append /version route to app.py once
if ! grep -q "scaffold_020 additions" app.py 2>/dev/null; then
  b app.py
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
else
  echo "• app.py already contains scaffold_020 additions — skipping append"
fi

# 2) CI BuildLog generator (Python script used in workflow)
b scripts/ci_buildlog.py
cat > scripts/ci_buildlog.py <<'PY'
#!/usr/bin/env python3
"""
Generate a Markdown BuildLog for CI (GitHub Actions).
- Captures environment, commit metadata, diffstat, and (optional) test output.
- Writes to out/ci_buildlogs/BuildLog_CI_<ISO>.md and echoes the path.
Environment:
  CI_TEST_CMD   : optional command to run (e.g., "pytest -q" or "make test")
  SAVE_PATH     : optional output path override
"""
import os, subprocess, sys, datetime, pathlib, textwrap, shlex

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "out" / "ci_buildlogs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run(cmd, check=False, capture=True):
    try:
        if capture:
            return subprocess.run(cmd, shell=True, cwd=ROOT, check=check, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()
        else:
            return subprocess.run(cmd, shell=True, cwd=ROOT, check=check).returncode
    except Exception as e:
        return f"(error running {cmd}: {e})"

def git(cmd):
    return run(f"git {cmd}")

def main():
    now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    sha = os.getenv("GITHUB_SHA", git("rev-parse HEAD"))
    sha_short = sha[:12] if sha else ""
    ref = os.getenv("GITHUB_REF_NAME", git("rev-parse --abbrev-ref HEAD"))
    repo = os.getenv("GITHUB_REPOSITORY", pathlib.Path.cwd().name)
    actor = os.getenv("GITHUB_ACTOR", "")
    runner = os.getenv("RUNNER_NAME", "")
    python = run("python3 -c 'import sys;print(\".\".join(map(str,sys.version_info[:3])))'")
    msg = git("log -1 --pretty=%B").splitlines()[0] if git("log -1 --pretty=%B") else "(no message)"
    diffstat = run("git diff --shortstat HEAD~1..HEAD || true")
    status = git("status --porcelain -b")
    # Optional tests
    test_cmd = os.getenv("CI_TEST_CMD", "").strip()
    test_out = "(no tests run)"
    if test_cmd:
        test_out = run(test_cmd)

    # Build content
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
