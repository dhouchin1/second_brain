#!/usr/bin/env python3
"""
sb.py — lightweight CLI for Second Brain
Subcommands:
  - export-json <path>
  - export-mdzip <path>
  - import-json <path>
  - import-md <dir_or_zip>
  - batch-tag (--add TAG | --remove TAG) [--ids "1,2,3" | --fts "query"]
  - rebuild-embeddings [--limit N] [--force]
"""
import argparse, os, sys, sqlite3, json, re, io, zipfile
from datetime import datetime
from typing import List, Iterable, Tuple

try:
    import httpx
except Exception as e:
    print("httpx required; run: pip install -r requirements.txt", file=sys.stderr); sys.exit(1)

DB_DEFAULT = "./notes.db"
EMBED_BASE = os.getenv("OLLAMA_EMBED_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS notes ( id INTEGER PRIMARY KEY AUTOINCREMENT, body TEXT NOT NULL, created_at TEXT NOT NULL );
CREATE TABLE IF NOT EXISTS tags ( id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE );
CREATE TABLE IF NOT EXISTS note_tags ( note_id INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE, tag_id  INTEGER NOT NULL REFERENCES tags(id)  ON DELETE CASCADE, PRIMARY KEY (note_id, tag_id) );
CREATE TABLE IF NOT EXISTS note_embeddings ( note_id INTEGER PRIMARY KEY REFERENCES notes(id) ON DELETE CASCADE, dim INTEGER NOT NULL, vec BLOB NOT NULL );

CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(body, content='notes', content_rowid='id');

CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN INSERT INTO notes_fts(rowid, body) VALUES (new.id, new.body); END;
CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
  INSERT INTO notes_fts(notes_fts, rowid, body) VALUES('delete', old.id, old.body);
  INSERT INTO notes_fts(rowid, body) VALUES (new.id, new.body);
END;
CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN INSERT INTO notes_fts(notes_fts, rowid, body) VALUES('delete', old.id, old.body); END;
"""

def conn(db_path: str):
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA_SQL)
    c.commit()
    return c

def norm_tag(name: str) -> str:
    return re.sub(r"[^a-z0-9\-]", "", (name or "").strip().lower().replace("#","").replace(" ", "-"))

def ensure_tag(c: sqlite3.Connection, name: str) -> int:
    name = norm_tag(name)
    cur = c.execute("INSERT OR IGNORE INTO tags(name) VALUES (?)", (name,))
    if cur.lastrowid: return cur.lastrowid
    return c.execute("SELECT id FROM tags WHERE name=?", (name,)).fetchone()["id"]

def set_note_tags(c: sqlite3.Connection, note_id: int, tags: Iterable[str]):
    c.execute("DELETE FROM note_tags WHERE note_id=?", (note_id,))
    for nm in tags:
        tid = ensure_tag(c, nm)
        c.execute("INSERT OR IGNORE INTO note_tags(note_id, tag_id) VALUES (?,?)", (note_id, tid))

def get_taglist(c: sqlite3.Connection, note_id: int) -> List[str]:
    rows = c.execute("""SELECT t.name FROM tags t JOIN note_tags nt ON nt.tag_id=t.id WHERE nt.note_id=? ORDER BY t.name""", (note_id,)).fetchall()
    return [r["name"] for r in rows]

def embed_text(text: str) -> List[float]:
    payload = {"model": EMBED_MODEL, "prompt": text}
    r = httpx.post(f"{EMBED_BASE}/api/embeddings", json=payload, timeout=30.0)
    r.raise_for_status()
    data = r.json()
    vec = data.get("embedding") or (data.get("data",[{}])[0].get("embedding"))
    if not vec: raise RuntimeError("No embedding returned")
    return vec

def export_json(c: sqlite3.Connection, path: str):
    rows = c.execute("SELECT id, body, created_at FROM notes ORDER BY id ASC").fetchall()
    out = []
    for r in rows:
        tags = get_taglist(c, r["id"])
        out.append({"id": r["id"], "body": r["body"], "created_at": r["created_at"], "tags": tags})
    payload = {"exported_at": datetime.utcnow().isoformat(timespec="seconds")+"Z", "notes": out}
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    with open(path, "wb") as f: f.write(data)
    print(f"wrote {path}")

def _slugify(title: str) -> str:
    s = (title or "").strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s]+", "-", s)
    return s[:60] or "note"

def _title_from_body(body: str) -> str:
    for line in (body or "").splitlines():
        t = line.strip()
        if t: return t
    return "Untitled"

def export_mdzip(c: sqlite3.Connection, path: str):
    rows = c.execute("SELECT id, body, created_at FROM notes ORDER BY id ASC").fetchall()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in rows:
            title = _title_from_body(r["body"])
            name = f'{r["id"]}-{_slugify(title)}.md'
            tags = ", ".join(get_taglist(c, r["id"]))
            content = f"""--- 
id: {r["id"]}
created_at: {r["created_at"]}
tags: [{tags}]
---
{r["body"]}
"""
            zf.writestr(name, content)
    with open(path, "wb") as f: f.write(buf.getvalue())
    print(f"wrote {path}")

def _parse_front_matter(text: str) -> Tuple[dict,str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("\n", 1)[1].split("\n---", 1)
    if len(parts) < 2: return {}, text
    fm_raw, rest = parts[0], parts[1]
    meta = {}
    for line in fm_raw.splitlines():
        if ":" not in line: continue
        k, v = line.split(":", 1)
        meta[k.strip()] = v.strip().strip('"\'')
    # after '---' may have leading newline
    rest = rest.lstrip("\n")
    return meta, rest

def import_json(c: sqlite3.Connection, path: str):
    data = json.load(open(path, "r", encoding="utf-8"))
    notes = data["notes"] if isinstance(data, dict) and "notes" in data else data
    added = 0
    for n in notes:
        body = n.get("body","")
        when = n.get("created_at") or datetime.utcnow().isoformat(timespec="seconds")
        cur = c.execute("INSERT INTO notes(body, created_at) VALUES (?,?)", (body, when))
        nid = cur.lastrowid
        set_note_tags(c, nid, [norm_tag(t) for t in n.get("tags",[]) if t])
        added += 1
    c.commit()
    print(f"imported {added} notes")

def import_md(c: sqlite3.Connection, target: str):
    def iter_files() -> Iterable[Tuple[str, str]]:
        if os.path.isdir(target):
            for root, _dirs, files in os.walk(target):
                for fn in files:
                    if fn.lower().endswith(".md"):
                        p = os.path.join(root, fn)
                        yield fn, open(p, "r", encoding="utf-8").read()
        else:
            with zipfile.ZipFile(target, "r") as zf:
                for zi in zf.infolist():
                    if zi.filename.lower().endswith(".md"):
                        yield os.path.basename(zi.filename), zf.read(zi).decode("utf-8", errors="replace")
    added = 0
    for name, text in iter_files():
        meta, body = _parse_front_matter(text)
        when = meta.get("created_at") or datetime.utcnow().isoformat(timespec="seconds")
        cur = c.execute("INSERT INTO notes(body, created_at) VALUES (?,?)", (body, when))
        nid = cur.lastrowid
        tags_val = meta.get("tags","")
        if isinstance(tags_val, str):
            tags = [norm_tag(x) for x in re.split(r"[,\[\]]", tags_val) if x.strip()]
        elif isinstance(tags_val, list):
            tags = [norm_tag(x) for x in tags_val]
        else:
            tags = []
        set_note_tags(c, nid, tags)
        added += 1
    c.commit()
    print(f"imported {added} notes")

def batch_tag(c: sqlite3.Connection, add: str, remove: str, ids: str, fts: str):
    target_ids: List[int] = []
    if ids:
        target_ids = [int(x) for x in re.split(r"[,\s]+", ids) if x.strip().isdigit()]
    elif fts:
        rows = c.execute("""
          SELECT n.id FROM notes_fts f JOIN notes n ON n.id=f.rowid
          WHERE notes_fts MATCH ? ORDER BY bm5(notes_fts)
        """, (fts,)).fetchall()
        target_ids = [r["id"] for r in rows]
    if not target_ids:
        print("No target notes."); return
    if add:
        tid = ensure_tag(c, add)
        c.executemany("INSERT OR IGNORE INTO note_tags(note_id, tag_id) VALUES (?,?)", [(i, tid) for i in target_ids])
    if remove:
        row = c.execute("SELECT id FROM tags WHERE name=?", (norm_tag(remove),)).fetchone()
        if row:
            c.executemany("DELETE FROM note_tags WHERE note_id=? AND tag_id=?", [(i, row["id"]) for i in target_ids])
    c.commit()
    print(f"Updated {len(target_ids)} notes.")

def rebuild_embeddings(c: sqlite3.Connection, limit: int, force: bool):
    if force:
        rows = c.execute("SELECT id, body FROM notes ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    else:
        rows = c.execute("""
          SELECT n.id, n.body
          FROM notes n LEFT JOIN note_embeddings e ON e.note_id=n.id
          WHERE e.note_id IS NULL
          ORDER BY n.id DESC LIMIT ?
        """, (limit,)).fetchall()
    done=0; err=0
    for r in rows:
        try:
            vec = embed_text(r["body"])
            # store as float32 bytes to match app convention
            import numpy as np
            blob = np.asarray(vec, dtype=np.float32).tobytes()
            c.execute("INSERT INTO note_embeddings(note_id, dim, vec) VALUES (?,?,?) ON CONFLICT(note_id) DO UPDATE SET dim=excluded.dim, vec=excluded.vec",
                      (r["id"], len(vec), blob))
            done+=1
        except Exception as e:
            err+=1
    c.commit()
    print(f"Embeddings: {done} updated ({err} errors)")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=DB_DEFAULT, help="Path to notes.db")
    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("export-json"); s.add_argument("path")
    s = sp.add_parser("export-mdzip"); s.add_argument("path")
    s = sp.add_parser("import-json"); s.add_argument("path")
    s = sp.add_parser("import-md"); s.add_argument("target")

    s = sp.add_parser("batch-tag")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--add", help="Tag to add")
    g.add_argument("--remove", help="Tag to remove")
    s.add_argument("--ids", help="Comma-separated IDs", default="")
    s.add_argument("--fts", help="FTS MATCH query", default="")

    s = sp.add_parser("rebuild-embeddings")
    s.add_argument("--limit", type=int, default=100)
    s.add_argument("--force", action="store_true")

    args = p.parse_args()
    c = conn(args.db)

    if args.cmd == "export-json":
        export_json(c, args.path)
    elif args.cmd == "export-mdzip":
        export_mdzip(c, args.path)
    elif args.cmd == "import-json":
        import_json(c, args.path)
    elif args.cmd == "import-md":
        import_md(c, args.target)
    elif args.cmd == "batch-tag":
        batch_tag(c, getattr(args, "add", ""), getattr(args, "remove", ""), args.ids, args.fts)
    elif args.cmd == "rebuild-embeddings":
        rebuild_embeddings(c, args.limit, args.force)

if __name__ == "__main__":
    main()
