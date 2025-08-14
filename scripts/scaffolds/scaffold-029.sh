#!/usr/bin/env bash
# scripts/scaffolds/scaffold_029.sh
# DB migrations with PRAGMA user_version, FTS triggers, audit log, Make targets
set -euo pipefail

STAMP="$(date +%Y%m%d-%H%M%S)"
bk(){ [[ -f "$1" ]] && { mkdir -p .bak; cp -p "$1" ".bak/$(basename "$1").$STAMP.bak"; echo "• backup: $1 -> .bak/$(basename "$1").$STAMP.bak"; } || true; }

mkdir -p scripts/db/migrations scripts

# 0) Ensure HTMLResponse import (fix NameError)
if grep -q "from fastapi.responses import" app.py; then
  if ! grep -q "HTMLResponse" app.py; then
    bk app.py
    perl -0777 -pe "s/from fastapi\.responses import ([^\n]+)/from fastapi.responses import \1, HTMLResponse/g" -i app.py
    echo "• added HTMLResponse import to app.py"
  fi
else
  bk app.py
  printf "\nfrom fastapi.responses import HTMLResponse\n" >> app.py
  echo "• appended HTMLResponse import to app.py"
fi

# 1) Migration runner (Python)
cat > scripts/db_migrate.py <<'PY'
#!/usr/bin/env python3
import sqlite3, sys, pathlib, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB = (ROOT / "second_brain.db")
try:
    from config import settings
    DB = settings.db_path
except Exception:
    pass

MIG_DIR = ROOT / "scripts" / "db" / "migrations"

def get_user_version(conn):
    return conn.execute("PRAGMA user_version").fetchone()[0]

def set_user_version(conn, v):
    conn.execute(f"PRAGMA user_version = {int(v)}")

def list_migrations():
    files = sorted(MIG_DIR.glob("[0-9][0-9][0-9]_*.sql"))
    items = []
    for f in files:
        num = int(f.name.split("_", 1)[0])
        items.append((num, f))
    return items

def apply_migration(conn, num, path):
    print(f"==> applying {path.name}")
    with open(path, "r", encoding="utf-8") as fh:
        sql = fh.read()
    conn.executescript("BEGIN;")
    conn.executescript(sql)
    set_user_version(conn, num)
    conn.commit()

def main():
    cmd = sys.argv[1] if len(sys.argv)>1 else "migrate"
    conn = sqlite3.connect(str(DB))
    cur = get_user_version(conn)
    migs = list_migrations()
    if cmd == "status":
        print(f"user_version={cur}")
        for num, f in migs:
            mark = "APPLIED" if num <= cur else "PENDING"
            print(f"{num:03d}  {f.name}  {mark}")
        return
    # migrate
    for num, f in migs:
        if num > cur:
            apply_migration(conn, num, f)
            cur = num
    print(f"OK user_version={cur}")
    conn.close()

if __name__ == "__main__":
    main()
PY
chmod +x scripts/db_migrate.py

# 2) Migrations

# 001: baseline/tolerant create
cat > scripts/db/migrations/001_baseline.sql <<'SQL'
-- 001: ensure core tables & FTS virtual table
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  hashed_password TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT,
  summary TEXT,
  tags TEXT,
  actions TEXT,
  type TEXT,
  timestamp TEXT,
  audio_filename TEXT,
  content TEXT,
  status TEXT DEFAULT 'complete',
  user_id INTEGER,
  FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
  title, summary, tags, actions, content, content='notes', content_rowid='id'
);

CREATE TABLE IF NOT EXISTS sync_status (
  id INTEGER PRIMARY KEY,
  last_sync TEXT
);

-- baseline indexes
CREATE INDEX IF NOT EXISTS idx_notes_user_ts ON notes(user_id, timestamp);

PRAGMA user_version = 1;
SQL

# 002: FTS triggers
cat > scripts/db/migrations/002_fts_triggers.sql <<'SQL'
-- 002: FTS sync triggers for notes <-> notes_fts
DROP TRIGGER IF EXISTS notes_ai_fts;
DROP TRIGGER IF EXISTS notes_au_fts;
DROP TRIGGER IF EXISTS notes_ad_fts;

CREATE TRIGGER notes_ai_fts AFTER INSERT ON notes BEGIN
  INSERT INTO notes_fts(rowid, title, summary, tags, actions, content)
  VALUES (new.id, new.title, new.summary, new.tags, new.actions, new.content);
END;

CREATE TRIGGER notes_au_fts AFTER UPDATE ON notes BEGIN
  DELETE FROM notes_fts WHERE rowid = old.id;
  INSERT INTO notes_fts(rowid, title, summary, tags, actions, content)
  VALUES (new.id, new.title, new.summary, new.tags, new.actions, new.content);
END;

CREATE TRIGGER notes_ad_fts AFTER DELETE ON notes BEGIN
  DELETE FROM notes_fts WHERE rowid = old.id;
END;

PRAGMA user_version = 2;
SQL

# 003: audit log + note triggers
cat > scripts/db/migrations/003_audit_log.sql <<'SQL'
-- 003: audit log for note create/update/delete
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  action TEXT NOT NULL,
  note_id INTEGER,
  user_id INTEGER,
  title_old TEXT,
  title_new TEXT,
  tags_old TEXT,
  tags_new TEXT,
  ts TEXT DEFAULT (datetime('now')),
  meta TEXT
);

DROP TRIGGER IF EXISTS notes_ai_audit;
DROP TRIGGER IF EXISTS notes_au_audit;
DROP TRIGGER IF EXISTS notes_ad_audit;

CREATE TRIGGER notes_ai_audit AFTER INSERT ON notes BEGIN
  INSERT INTO audit_log(action, note_id, user_id, title_new, tags_new, meta)
  VALUES ('note_insert', NEW.id, NEW.user_id, NEW.title, NEW.tags, NULL);
END;

CREATE TRIGGER notes_au_audit AFTER UPDATE ON notes BEGIN
  INSERT INTO audit_log(action, note_id, user_id, title_old, title_new, tags_old, tags_new, meta)
  VALUES ('note_update', NEW.id, NEW.user_id, OLD.title, NEW.title, OLD.tags, NEW.tags, NULL);
END;

CREATE TRIGGER notes_ad_audit AFTER DELETE ON notes BEGIN
  INSERT INTO audit_log(action, note_id, user_id, title_old, tags_old, meta)
  VALUES ('note_delete', OLD.id, OLD.user_id, OLD.title, OLD.tags, NULL);
END;

PRAGMA user_version = 3;
SQL

# 3) Makefile helpers
if [[ -f Makefile ]]; then bk Makefile; fi
cat >> Makefile <<'MK'

# === DB Migrations ===
db-status:
	@python scripts/db_migrate.py status

db-migrate:
	@python scripts/db_migrate.py migrate
MK

echo "Done 029.

Run:
  bash scripts/scaffolds/scaffold_029.sh
  make db-status
  make db-migrate
"
