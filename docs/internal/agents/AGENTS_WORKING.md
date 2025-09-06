# Agents Working Guide

Purpose: Shared plan and live context for all AI agents collaborating on Second Brain. Use this file to understand current state, design choices, and the prioritized roadmap. Keep updates scoped, consistent, and reversible.

## Current State (Sept 2025)

- Schema v2 baseline in SQLite
  - Core table: `notes(id, title, body, tags, created_at, updated_at, …)` from `db/migrations/001_core.sql`.
  - Legacy columns retained for compatibility: `content`, `summary`, `timestamp`, `actions`, etc.
  - FTS5: `notes_fts(title, body, tags)` kept in sync via triggers.
  - Optional vector table `note_vecs` (skipped if `sqlite-vec` not loaded).

- Migration Runner
  - `migrate_db.py` uses `executescript`, strips `#`-style comments, and treats `002_vec.sql` as optional.
  - Migrations apply cleanly on a fresh DB; see pending `002_vec` when `sqlite-vec` isn’t available.

- Real‑time (SSE)
  - Endpoints registered from `realtime_status.py` regardless of enhanced tasks presence.
  - Stream endpoint accepts `?token=` query param (JWT) or cookie session.
  - `/api/sse-token` requires authentication and returns 401 if unauthenticated (by design; can be adjusted).

- Notable recent edits
  - `migrate_db.py` (executescript, comment stripping, optional vec skip)
  - `db/migrations/001_core.sql` (simpler FTS tokenizer)
  - `db/migrations/002_vector_embeddings.sql` (triggers watch `body,title`)
  - `db/migrations/004_search_features.sql` (renamed daily analytics table)
  - `app.py` (compat columns on startup; FTS normalized; SSE registration decoupled)
- `realtime_status.py` (token auth support for stream)
- `scripts/ollama_check.py` (local Ollama diagnostics)
- Seeding: `scripts/seed_starter_vault.py` now inserts notes with body/content and supports per-user seeding

## Why 401 Unauthorized appears

- `/api/sse-token` deliberately requires a logged-in user. When background JS tries to refresh tokens while not authenticated (e.g., on anonymous pages or expired session), it returns 401. This is expected and safe. If noise reduction is desired, we can change it to return `200 { token: "" }` instead.

## Architecture Snapshot

- App Entrypoint: `app.py`
  - Registers routers, SSE endpoints, and startup tasks.
  - Ensures legacy compatibility columns exist.
- Real‑time: `realtime_status.py`
  - `StatusManager` for broadcasting; SSE endpoints for stream/queue.
- Search: `services/search_adapter.py`
  - Runs baseline migrations; supports keyword + optional semantic/hybrid when vectors available.
- Auth: `services/auth_service.py`
- Processing: `tasks.py` (baseline); `tasks_enhanced.py` optional.

## Roadmap (Phased)

1) Schema Alignment (Core)
   - Goal: Move codebase off legacy fields (`content`, `summary`, `timestamp`) to (`body`, `created_at`, `updated_at`).
   - Tasks:
     - Replace reads/writes to legacy fields in services and routes with `body` and timestamps.
     - Adjust analytics and templates that rely on `timestamp` and `summary` to new fields or computed equivalents.
     - Provide a one-time backfill path: set `body = COALESCE(content, '')` where needed.
     - Keep legacy columns temporarily for backward compatibility, then deprecate.

2) Real‑time Consistency (SSE)
   - Goal: Reliable, informative progress streams without auth confusion.
   - Tasks:
     - Add minimal status broadcasting in `tasks.py` using `StatusManager` (emit stage/progress at key steps).
     - Frontend: Ensure EventSource always includes `?token=` if cookie auth is not guaranteed; refresh token on expiry.
     - Option (noise reduction): Make `/api/sse-token` return `200` with `{ token: "" }` when not logged in.
     - Unify queue endpoints (`/api/queue/status` vs `/api/status/queue`) or document their intent and clients.

3) Search Consistency
   - Goal: Ensure search codepaths use `notes.body` and `notes_fts(title, body, tags)` consistently.
   - Tasks:
     - Verify `services/search_adapter.py` CRUD uses `body` and updates `updated_at`.
     - Validate FTS triggers; add rebuild utility for FTS reindex.
     - When `sqlite-vec` available: apply `002_vec.sql` and make hybrid queries prefer ANN when present.

4) Security and Auth
   - Goal: Centralize token strategies and minimize auth surprises.
   - Tasks:
     - Standardize short-lived signed tokens for SSE with configurable TTL.
     - Clarify frontend rules: only call `/api/sse-token` for authenticated sessions.
     - Review CSRF checks and ensure consistent use across write endpoints.

5) Testing + Validation
- Goal: Confidence per change.
- Tasks:
  - Tests: migrations on empty DB; SSE endpoints (authorized/unauthorized + token); search smoke tests.
  - Performance: quick search benchmark run to ensure no regressions.
  - Ollama: run `python scripts/ollama_check.py` to validate local server and models.
  - Seeding: run `python scripts/seed_starter_vault.py --namespace .starter_content --no-embed` for file-only, or without `--no-embed` when Ollama is ready.

6) Documentation + Developer Experience
   - Goal: Clear onboarding for humans and agents.
   - Tasks:
     - Update AGENTS.md to reference schema v2 and this roadmap.
     - Maintain this file with short changelog per phase.

## Implementation Guidance for Sub‑Agents

- Use general‑purpose agent for code searches and mapping legacy field usage across repo.
- Use code‑writer for scoped refactors (e.g., replace legacy fields in a module).
- Keep patches minimal, with clear commit messages. Avoid unrelated changes.

Prompts to use:
- Mapping legacy schema usage
  - "Find all SELECT/INSERT/UPDATE statements referencing `content`, `summary`, or `timestamp` in Python code, list file paths and lines."
- SSE integration
  - "Add stage/progress emits into `tasks.py` at start, mid, and completion of processing using `StatusManager`."
- Search alignment
  - "Ensure all search paths use `notes.body` and FTS over `(title, body, tags)`; provide a `rebuild_fts()` helper if missing."

## Open Decisions

- Silence unauth `/api/sse-token`?
  - Option A (current): 401 when not logged in — explicit and secure.
  - Option B: Return 200 with `{ token: "" }` to reduce log noise.
- Deprecation timeline for legacy columns — after Phase 1 refactor + data backfill.
- Endpoint unification for queue status.

## Quick Commands

- Start app: `uvicorn app:app --reload --port 8082`
- Run migrations: `python3 migrate_db.py`
- Status: `python3 migrate_db.py --status`
- Health: `curl http://localhost:8082/health`
- Ollama diagnostics: `python scripts/ollama_check.py --url http://localhost:11434 --gen-model llama3.2 --embed-model nomic-embed-text`
- Seed vault (files only): `python scripts/seed_starter_vault.py --namespace .starter_content --no-embed`
- Seed vault (with embeddings): `python scripts/seed_starter_vault.py --namespace .starter_content`

## Pointers (for navigation)

- SSE stream endpoint: `realtime_status.py:203`
- SSE token endpoint: `app.py:1377`
- SSE registration on startup: `app.py:72` and `app.py:669`
- Migration runner: `migrate_db.py:1`
- Core migration: `db/migrations/001_core.sql:1`

## Changelog (summary)

- Aligned migrations for clean fresh DB install; optional vec.
- SSE endpoints registered regardless of enhanced tasks module; token param supported.
- Startup now adds legacy columns for compatibility and normalizes FTS schema.
