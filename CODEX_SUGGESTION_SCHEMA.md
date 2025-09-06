# Second Brain Database Schema: Current State, Proposal, and Migration Plan

Date: 2025-09-06

This document summarizes the current SQLite schema, proposes a cleaner local‑first PKM schema (Markdown as source of truth; SQLite for metadata + FTS5 + sqlite‑vec), outlines a pragmatic migration plan, and lists first coding tasks to execute.

---

## 1) Current DB Schema (notes.db)

The live database currently mixes “core content,” capture/processing columns on `notes`, search analytics, integrations, and benchmarking. The effective schema (from db/migrations, app initialization, and runtime evolution) includes the following.

### Core Content
- `notes`
  - id INTEGER PK
  - title TEXT, body TEXT, tags TEXT
  - created_at TEXT, updated_at TEXT
  - zettel_id TEXT UNIQUE (nullable)
  - content/summary/timestamp/type
  - actions
  - user_id INTEGER (FK users)
  - file_* capture fields: file_filename, file_type, file_mime_type, file_size, extracted_text, file_metadata
  - web ingestion: source_url, web_metadata, screenshot_path
  - dedupe: content_hash
  - integrations: external_id, external_url, metadata JSON

- `notes_fts` (FTS5; content='notes', rowid='id'): title, body, tags
- `notes_fts5` (legacy enhanced FTS5; not strictly aligned with core)

### Embeddings & Vector Search
- `note_embeddings`
  - note_id INTEGER (FK notes)
  - embedding_model TEXT, embedding BLOB, embedding_dim INTEGER
  - created_at, updated_at

- `embedding_jobs` (status + attempts tracking per note/model)

- `note_vecs` (sqlite‑vec vec0 virtual table; embedding float[384], note_id PK)
  - Includes vec0 metadata tables: `note_vecs_info`, `note_vecs_chunks`, `note_vecs_rowids`, `note_vecs_vector_chunks..`

- `embeddings` (fallback JSON/text embedding store used by services) 
  - note_id, model, embedding_vector (TEXT JSON), created_at

### Users & Auth Adjacent
- `users` (id, username, hashed_password)
- `discord_users` (discord_id PK, user_id, linked_at)

### Search History & Analytics
- `search_history` (per query/user, mode, counts, response time)
- `saved_searches` (user, name, query, filters JSON, timestamps)
- `semantic_search_analytics` (query, query_embedding BLOB, counts, thresholds)
- `search_analytics` (low‑level perf samples: mode, query_hash, times)
- `search_analytics_daily` (aggregated daily stats)
- `unified_search_history` (query + intent + result metrics)

### Jobs, Logs, Rules
- `jobs` (type, status, scheduling, payload JSON)
- `job_logs` (per job log lines)
- `rules` (JSON definitions)

### Integrations
- `integration_sync` (per user/platform target sync runs)
- `integration_config` (per user/platform config key/values)

### Seeding, Audio, System
- `auto_seeding_log` (attempts, counts, status)
- `audio_processing_queue` (note‑level audio processing state)
- `sync_status` (singleton last_sync)

### Benchmarking (Search Performance)
- `benchmark_suites`, `benchmark_results`, `benchmark_baselines`
- `query_performance_trends`, `ab_test_experiments`, `ab_test_results`
- Views: `benchmark_summary`, `recent_search_performance`

### Views
- `notes_with_embeddings` (latest embedding join)
- `embedding_status` (note + latest embedding job + presence)

Notes
- Multi‑tenant migration exists in `migrations/002_add_tenants.sql` (tenants, tenant_memberships, etc.) but those tables are not visible in the current `notes.db` dump, implying not applied to this DB file.
- There are chunk/FTS helper tables in `services/search_index.py` (e.g., `chunk`, `fts_chunk`, `vec_chunk`, `vec_map`) that may exist in other environments or get created on demand; they are not present in this `notes.db` snapshot.

---

## 2) Summary of Current Tables/Columns

High‑level grouping with key columns:

- Content
  - `notes`: title, body, tags, metadata JSON, content_hash, file_* fields, web fields, timestamps, user_id
  - `notes_fts`: FTS5 on (title, body, tags)

- Embeddings / Vector
  - `note_embeddings`: note_id, embedding BLOB, model, dim, timestamps
  - `embedding_jobs`: note_id, model_name, status, attempts, timestamps
  - `note_vecs`: sqlite‑vec vec0 table for ANN search
  - `embeddings`: fallback text JSON vectors per note/model

- Users & Links
  - `users`: username, hashed_password
  - `discord_users`: discord_id, user_id, linked_at

- Search Tracking
  - `search_history`: user_id, query, mode, counts, response_time_ms, created_at
  - `saved_searches`: user_id, name, query, filters JSON, timestamps
  - `semantic_search_analytics`: query, query_embedding, counts, thresholds
  - `search_analytics`: timestamp, search_mode, query_hash, execution_time, result_count, cache flags
  - `search_analytics_daily`: per user/date aggregates
  - `unified_search_history`: query, intent, mode, counts

- Jobs & Rules
  - `jobs`, `job_logs`, `rules`

- Integrations
  - `integration_sync`: platform, target, last/next sync, status
  - `integration_config`: per platform config keys

- System / Pipelines
  - `auto_seeding_log`, `audio_processing_queue`, `sync_status`

- Benchmarking
  - `benchmark_*`, `query_performance_trends`, `ab_test_*` (+ summary views)

---

## 3) Proposed Cleaner Local‑First PKM Schema

Principles
- Markdown files are canonical source. The database is an index/cache for metadata, search, relationships, and activity. 
- Separation of concerns: files + structure (frontmatter, tags, links) → chunks → search (FTS5, vectors) → activity/analytics → integrations/settings.
- Optional features gracefully degrade (e.g., embeddings when sqlite‑vec unavailable).

Core Tables

1) `files`
- id TEXT (stable ID; content‑hash or UUID), PRIMARY KEY
- path TEXT UNIQUE NOT NULL (vault‑relative path to .md)
- title TEXT
- content_hash TEXT NOT NULL
- size_bytes INTEGER
- created_at TEXT, updated_at TEXT (filesystem timestamps)
- author TEXT NULL, source_url TEXT NULL
- status TEXT DEFAULT 'active'  -- active|archived|deleted
- frontmatter_json TEXT DEFAULT '{}'  -- optional cached snapshot for convenience

2) `frontmatter`
- file_id TEXT NOT NULL REFERENCES files(id)
- key TEXT NOT NULL
- value TEXT NOT NULL  -- store as string; app can parse YAML types
- PRIMARY KEY(file_id, key)

3) `tags`
- id INTEGER PK AUTOINCREMENT
- name TEXT UNIQUE NOT NULL

4) `file_tags`
- file_id TEXT REFERENCES files(id) ON DELETE CASCADE
- tag_id INTEGER REFERENCES tags(id) ON DELETE CASCADE
- PRIMARY KEY(file_id, tag_id)

5) `links`
- id INTEGER PK AUTOINCREMENT
- src_file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE
- dst_file_id TEXT NULL REFERENCES files(id)  -- internal wiki link if resolved
- url TEXT NULL  -- external link if not a wiki link
- anchor TEXT NULL  -- heading anchor (if present)
- created_at TEXT DEFAULT CURRENT_TIMESTAMP
- UNIQUE(src_file_id, dst_file_id, url, anchor)

6) `chunks`
- id TEXT PRIMARY KEY  -- deterministic (file_id + ord)
- file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE
- ord INTEGER NOT NULL  -- order within file
- heading TEXT NOT NULL DEFAULT ''
- text TEXT NOT NULL DEFAULT ''
- start_offset INTEGER DEFAULT 0  -- optional char start
- end_offset INTEGER DEFAULT 0    -- optional char end

7) `chunk_fts` (FTS5)
- FTS virtual table using fts5 over (heading, text)
- content='chunks', content_rowid='id'
- Triggers to sync on insert/update/delete of `chunks`

8) `embeddings`
- id INTEGER PK AUTOINCREMENT
- chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE
- model TEXT NOT NULL
- dim INTEGER NOT NULL
- vec_json TEXT NOT NULL  -- fallback when vec0 unavailable
- created_at TEXT DEFAULT CURRENT_TIMESTAMP
- UNIQUE(chunk_id, model)

9) `vec_chunk` (sqlite‑vec, optional)
- VIRTUAL TABLE USING vec0(embedding FLOAT[dim])
- `vec_map` table: (chunk_id TEXT, model TEXT, dim INTEGER, rowid_int INTEGER, PRIMARY KEY(chunk_id, model))

10) `activity`
- id INTEGER PK AUTOINCREMENT
- file_id TEXT NULL REFERENCES files(id)
- type TEXT NOT NULL  -- capture|edit|sync|process|search|export
- actor TEXT NULL  -- user or system
- payload TEXT DEFAULT '{}'  -- JSON details
- created_at TEXT DEFAULT CURRENT_TIMESTAMP

11) `integrations`
- `integration_config` (as today): user/platform/config_key/value
- `integration_sync` (as today): sync runs + status

12) `settings`
- scope TEXT NOT NULL  -- 'app' | 'user:{id}' | 'integration:{name}'
- key TEXT NOT NULL
- value TEXT NOT NULL
- updated_at TEXT DEFAULT CURRENT_TIMESTAMP
- PRIMARY KEY(scope, key)

Compatibility Notes
- Keep `users` (if local auth needed) as is.
- Keep search analytics tables; longer‑term unify `semantic_search_analytics` into a single `search_analytics` with optional embedding metadata.
- Avoid storing full note content in DB; prefer filesystem read. Cache only what’s required for search (chunks) and metadata.

---

## 4) Migration Plan (from current → proposed)

Strategy: Add new tables side‑by‑side, backfill incrementally, switch readers/writers, then retire legacy columns/tables. No downtime needed for a single‑user local DB; ensure idempotency.

Phases
1) Foundations
   - Create: `files`, `frontmatter`, `tags`, `file_tags`, `links`, `chunks`, `chunk_fts`, `embeddings`, `vec_map` (+ optional `vec_chunk`), `activity`, `settings`.
   - Keep current tables untouched.

2) Backfill Files
   - For each row in `notes`:
     - Determine canonical `path`: if existing vault file known, use it; else generate a synthetic path like `db_notes/{id}.md` to be created during export.
     - Compute `content_hash` from `COALESCE(body, content, '')`.
     - Insert into `files` with title, size (if known), source_url, and copy relevant capture/web metadata into `frontmatter` rows (keys: type, tags, actions, audio_filename, file_* fields, web_metadata, metadata, zettel_id, external_{id,url}).
     - Normalize `tags`: split `notes.tags` by comma/space, insert into `tags`, relate in `file_tags`.

3) Derive Links
   - Parse wiki‑style links and external URLs from the Markdown (body/content). Populate `links` with `src_file_id` and either `dst_file_id` (if resolvable) or `url`.

4) Chunking
   - Split each file’s content into `chunks` using existing chunking logic in `services/search_index.py` (heading‑aware, token counts). Insert chunks and rebuild `chunk_fts`.

5) Embeddings
   - Preferred: generate chunk‑level embeddings, insert into `vec_chunk` + `vec_map` when sqlite‑vec is available. Also store fallback in `embeddings` (vec_json) for portability.
   - Transitional: map existing `note_embeddings` to a single chunk per file (ord=0) if re‑embedding is deferred; mark for later refresh.

6) Writers & Readers Switch
   - Update search services to read from `chunk_fts` and chunk‑level embeddings first; fall back to `notes_fts`/note‑level when chunks missing.
   - Update capture/processing to write Markdown to the vault and update `files`/`frontmatter`/`tags` rather than expanding `notes` columns.

7) Decommission Legacy
   - Freeze writes to `notes` non‑core columns (file_*, web_*, metadata on notes, etc.).
   - Optionally keep a minimal `notes` table (id, title, body) for transition or remove entirely once services stop relying on it.
   - Consolidate `embedding_jobs` into general `jobs` or keep as specialized queue; remove `note_embeddings` when chunk embeddings are stable.

8) Validation
   - Run hybrid search parity tests (RRF against `notes_fts` vs `chunk_fts`+embeddings) and benchmark scripts to confirm quality/performance.
   - Verify end‑to‑end flows: capture → vault write → index → search.

Data Safety
- Make a full backup (`.backup` + VACUUM) before applying migrations.
- Migrations should be additive, with clear rollback (drop new tables if needed).

---

## 5) First Coding Tasks

Near‑Term, High‑Value Steps
1) Migrations
   - Write SQL migrations to create: `files`, `frontmatter`, `tags`, `file_tags`, `links`, `chunks`, `chunk_fts` (FTS5 + triggers), `embeddings`, `vec_map` (+ optional `vec_chunk`), `activity`, `settings`.

2) Indexer (Filesystem → DB)
   - Implement a vault indexer that:
     - Walks Markdown files under `vault/` and populates `files` (path, title, content_hash, timestamps).
     - Parses YAML frontmatter into `frontmatter` rows and normalizes tags → `tags`/`file_tags`.
     - Extracts links (wiki + external) into `links`.
     - Chunks content and populates `chunks` + `chunk_fts`.

3) Embedding Pipeline
   - Extend existing search indexer to build chunk‑level embeddings.
   - Prefer sqlite‑vec (`vec_chunk` + `vec_map`); store JSON fallback in `embeddings` for portability.

4) Search Adapter Upgrade
   - Update `services/search_adapter.py` / `unified_search_service` to search over `chunk_fts` and rerank with chunk embeddings; RRF merge with keyword results; aggregate back to files.

5) Capture Path
   - Update capture services to write canonical Markdown files to the vault, then trigger reindex rather than inserting bulky columns into `notes`.

6) Backfill from Current DB
   - Build a one‑shot backfill script to create synthetic Markdown files for `notes` that don’t have vault counterparts (using title/body/tags) and index them into the new schema.

7) Tests
   - Add tests for: migrations, vault indexing, chunking, FTS rebuild, embedding creation, hybrid search parity, and link graph extraction.

8) Cleanup Plan
   - Once reads/writes run entirely through the new schema, plan removal of legacy columns/tables (`notes` expansions, `note_embeddings`) and adjust routers/services accordingly.

---

## Appendix: Supporting Context

- `db/migrations` defines core tables (`notes`, FTS), embeddings, search analytics, and benchmarking. Optional vec migration (`002_vec.sql`) is designed to be skippable.
- `app.py` includes runtime schema alignment that added many capture/web columns to `notes`; this is the primary source of column bloat the proposal addresses.
- `services/search_index.py` already reflects chunk‑based indexing patterns; the proposal formalizes those structures database‑wide.

