Second Brain – Codex Changes (Seeding Improvements)

Date: 2025-09-05

Overview
- Goal: Make vault seeding reliable, idempotent, namespace-consistent, and accurately reported.
- Scope: Adjust detection/clearing SQL, unify default namespace usage, add seed tags on insert, compute true embedding counts, and fix writability check. No behavior changes outside seeding flows.

Changes Implemented

1) Robust seed detection and clearing
- Files:
  - services/vault_seeding_service.py
- What changed:
  - Wrapped conditions with proper parentheses and anchored them to the `user_id` to fix SQL operator-precedence bugs.
  - Switched detection from brittle title/content substrings to tag/namespace-aware queries:
    tags LIKE '%seed%' OR tags LIKE '%ns:<namespace>%', with content/title namespace fallback.
  - Applied the same logic to the clear operation to reliably remove seeded items.
- Why:
  - Existing queries could miscount due to precedence (AND bound to only one clause) and rarely matched real seeded notes.
  - Tagging is a much more reliable marker for “seeded” data.

2) Namespace consistency across the system
- Files:
  - services/vault_seeding_service.py
  - services/vault_seeding_router.py
- What changed:
  - Default `SeedingOptions.namespace` and router `SeedVaultRequest.namespace` now use `settings.auto_seeding_namespace` instead of hard-coded values.
  - Status checks and filesystem paths now reference `settings.auto_seeding_namespace`.
- Why:
  - Prevents mismatches between auto-seeding and service/router defaults (was `.starter_content` vs `.seed_samples`).

3) Accurate embedding counts in service results
- Files:
  - services/vault_seeding_service.py
- What changed:
  - `seed_vault` now computes embeddings created using before/after counts (like the existing alternative path `seed_vault_with_content`), instead of assuming `notes_created`.
- Why:
  - Reported numbers now reflect actual embedding writes (Ollama availability can vary).

4) Idempotent seeding without DB duplication
- Files:
  - scripts/seed_starter_vault.py
- What changed:
  - `write_markdown` returns `(path, wrote_file)` so the pipeline can skip DB inserts when files already exist and we’re not forcing overwrite.
  - When forcing overwrite, delete prior seeded duplicates for the same title/namespace (and user) before inserting.
  - All seeded notes/bookmarks now include tag markers `seed` and `ns:<namespace>` in both Markdown frontmatter and DB `tags`.
- Why:
  - Prevents duplicate seeded rows across re-runs while keeping behavior simple and schema-free.

5) Reliable vault writability check
- Files:
  - services/vault_seeding_router.py
- What changed:
  - Replaced bitmask permission check with `os.access(vault_path, os.W_OK)`.
- Why:
  - Bit flag checks don’t account for the current user’s effective permissions. `os.access` directly answers “can we write here?”.

6) Consistent default embedding model
- Files:
  - services/vault_seeding_service.py
  - services/vault_seeding_router.py
- What changed:
  - Default `embed_model` now uses `settings.auto_seeding_embed_model` instead of a hard-coded value.
- Why:
  - Keeps router requests and service defaults aligned with configuration (and `AutoSeedingService`) to avoid silent mismatches.

Notes & Compatibility
- No schema migration required; all changes rely on existing tables.
- Auto-seeding detection now keys off seed tags; previously seeded content without tags may not be detected until reseeded. The file-existence fallback and namespace-in-content/title remain as secondary signals.
- `write_markdown` signature changed only inside `scripts/seed_starter_vault.py` (its sole caller), so no external impact.

Suggested Follow-ups (optional)
- Consider adding a hidden HTML marker in `content` (e.g., `<!-- seed: id ns:namespace -->`) for future-proof detection.
- If stronger idempotency is desired, add a `seed_uid` column or create a uniqueness constraint via a migration.

7) Align seed embeddings with unified vector table
- Files:
  - scripts/seed_starter_vault.py
- What changed:
  - `ensure_embeddings_schema` now detects `note_vecs` (from db/migrations/002_vec.sql) instead of creating an ad-hoc vec table.
  - `upsert_embeddings` writes to `note_vecs(note_id, embedding)` when available (sqlite-vec), otherwise falls back to JSON in `embeddings`.
- Why:
  - The previous `vec_notes_fts` path didn’t map vectors to notes and didn’t match the rest of the system. Using `note_vecs` integrates seeded vectors with the SearchService’s semantic search.

8) Reuse central Embeddings service (avoid duplication)
- Files:
  - scripts/seed_starter_vault.py
- What changed:
  - Seeding now prefers `services.embeddings.Embeddings` (local-first policy) to generate vectors; only falls back to direct Ollama HTTP when needed. The older `try_ollama_embed` is retained only for dependency testing.
- Why:
  - Prevents duplicated embedding logic and keeps model/provider behavior consistent across the app.

9) Enrich auto_seeding_log with metrics (no migration required)
- Files:
  - services/auto_seeding_service.py
- What changed:
  - `_record_auto_seeding` now opportunistically adds columns `notes_created`, `files_written`, and `embeddings_created` (ALTER TABLE guarded by try/except), and writes these metrics when available from the seeding result.
- Why:
  - Improves observability of auto-seeding outcomes without requiring a standalone migration. Existing tables are upgraded in-place.
10) Optional post-seed search index refresh
- Files:
  - services/vault_seeding_service.py
- What changed:
  - Added `SeedingOptions.refresh_search_indices` (default True). On successful seeding, `_refresh_search_indexes` attempts to rebuild `notes_fts` from `notes` so seeded content is keyword-searchable immediately. It tries both schemas: (title, body, tags) and (title, content, tags), and silently skips if `notes_fts` is absent.
- Why:
  - Provides a best-effort refresh without duplicating broader indexing logic. Keeps UX snappy post-seed.

10a) Environment toggle for index refresh
- Files:
  - config.py, services/vault_seeding_service.py, services/vault_seeding_router.py, services/auto_seeding_service.py, .env
- What changed:
  - Added `AUTO_SEEDING_REFRESH_INDICES` env setting, exposed as `settings.auto_seeding_refresh_indices`.
  - `SeedingOptions.refresh_search_indices` defaults to this setting.
  - Router request accepts `refresh_search_indices` to override per-call.
  - .env includes commented context for safe configuration.
- Why:
  - Makes post-seed index refresh behavior explicit and configurable across environments without code changes.

11) Hidden seed marker in Markdown content
- Files:
  - scripts/seed_starter_vault.py
- What changed:
  - Prepends a hidden HTML comment to each seeded Markdown document body: `<!-- seed:<id> ns:<namespace> -->`.
- Why:
  - Strengthens detection and auditing of seeded content beyond tags. It’s invisible to users, resilient to renames, and lets DB/content checks key on a consistent marker.

12) Capture deduplication (stable content hash)
- Files:
  - services/unified_capture_service.py, config.py, .env
- What changed:
  - Added optional dedup logic in `_save_note`: computes a SHA-256 over normalized (title+content) and, when enabled, reuses an existing note with the same `content_hash` (within a configurable window) by returning its id and touching `updated_at` instead of inserting a new row. New notes persist `content_hash` inside `metadata` for future checks.
  - New settings: `CAPTURE_DEDUP_ENABLED` and `CAPTURE_DEDUP_WINDOW_DAYS` (0 = no time window).
- Why:
  - Prevents accidental duplicates across multiple capture channels without introducing a schema migration. Uses metadata JSON and a LIKE query for compatibility.

13) Diagnostics endpoints and URL-aware dedup
- Files:
  - services/diagnostics_router.py, app.py, services/unified_capture_service.py
- What changed:
  - Added `/api/diagnostics/health` and `/api/diagnostics/search` endpoints. Search diagnostics report presence and row counts of `notes_fts`, `note_vecs`, `embeddings`, plus total notes and whether `SQLITE_VEC_PATH` is set.
  - Extended URL capture to deduplicate by normalized URL (scheme/host lower-cased, fragment stripped, trailing slash normalized, sorted query). If a matching note is found within the configured window, returns that note and touches `updated_at` instead of re-ingesting.
- Why:
  - Quick visibility into index health without manual DB queries. URL dedup prevents duplicate web captures across sources.

14) Admin dashboard diagnostics tile
- Files:
  - templates/dashboard.html
- What changed:
  - Added a lightweight “System Diagnostics” card in the right panel that calls `/api/diagnostics/search` and displays notes count, FTS presence/rows, vector presence/rows, and JSON embeddings presence/rows. Includes a refresh button.
- Why:
  - Surfaces essential search/index health at a glance without leaving the dashboard.

14a) One-click index rebuild from dashboard
- Files:
  - services/diagnostics_router.py, templates/dashboard.html
- What changed:
  - Added `/api/diagnostics/reindex` POST endpoint. Triggers FTS rebuild and, if possible, embeddings rebuild via SearchIndexer. Returns result JSON.
  - Added a “Rebuild Index” button to the diagnostics widget that calls the endpoint and refreshes diagnostics.
- Why:
  - Provides an admin-friendly way to recover or refresh search state without CLI access.
