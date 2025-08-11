#!/opt/homebrew/bin/bash
set -Eeuo pipefail
BASE="${1:?Usage: 20_copy_from_snapshot.sh <integration-branch> <paths...>}"; shift
(( $# >= 1 )) || { echo "Provide 1+ paths to copy"; exit 1; }
PREMIUM_DIR="${PREMIUM_DIR:?Set PREMIUM_DIR to the premium snapshot path}"
TOP="$(git rev-parse --show-toplevel)"; cd "$TOP"
git fetch --all --prune
git checkout "$BASE"; git pull --ff-only
BR="port/$(date +%Y%m%d-%H%M%S)"; git switch -c "$BR"
EXCLUDES=(--exclude ".git" --exclude ".venv" --exclude "__pycache__" --exclude "backups" --exclude ".DS_Store")
for p in "$@"; do
  SRC="$PREMIUM_DIR/$p"
  if [ -e "$SRC" ]; then
    mkdir -p "$(dirname "$p")" 2>/dev/null || true
    if command -v rsync >/dev/null; then rsync -a "${EXCLUDES[@]}" "$SRC" "$(dirname "$p")/"; else rm -rf "$p"; cp -R "$SRC" "$(dirname "$p")/"; fi
    echo "Copied: $p"
  else
    echo "WARN: $p not found in premium snapshot"
  fi
done
if [ -f "$PREMIUM_DIR/requirements.txt" ]; then
python3 - <<'PY'
import os, pathlib
dst = pathlib.Path("requirements.txt")
src = pathlib.Path(os.environ["PREMIUM_DIR"]) / "requirements.txt"
have=set(); lines=[]
if dst.exists():
  for ln in dst.read_text().splitlines():
    s=ln.strip(); 
    if s and not s.startswith("#"): have.add(s)
    lines.append(ln)
add=[]
for ln in src.read_text().splitlines():
  s=ln.strip()
  if s and not s.startswith("#") and s not in have: add.append(s)
if add:
  lines += add; dst.write_text("\n".join(lines)+ "\n"); print("Merged requirements:", ", ".join(add))
else:
  print("No new requirements")
PY
fi
printf "\n# local\n.DS_Store\n__pycache__/\nbackups/\n*.bundle\n" >> .gitignore || true
git add -A
git commit -m "feat(port): copy from premium snapshot -> $BASE" -m "Paths: $*"
git push -u origin "$BR"
gh pr create --head "$BR" --base "$BASE" --title "Port from premium: $*" --label scaffolds --body "One-way copy from snapshot: $PREMIUM_DIR"
