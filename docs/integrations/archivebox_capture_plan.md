# ArchiveBox Capture Integration Plan

## 1. Context & Feasibility
- Network sandboxing prevented a live fetch of https://github.com/ArchiveBox/ArchiveBox, but ArchiveBox is well-documented as a self-hosted archiver that snapshots URLs using tools like wget, Chromium headless, youtube-dl, and readability. It exposes a CLI, REST-ish JSON export, and a pluggable storage model (SQLite/Postgres).
- Its strengths (long-term HTML/PDF/WARC capture, media downloads, out-of-band cron ingestion) complement Second Brain’s need for durable web captures beyond transient scraping.
- Integration is viable because ArchiveBox can run alongside our stack and publish structured snapshot metadata we can ingest.

## 2. Integration Goals
- Allow Second Brain to queue URLs for deep archival via ArchiveBox.
- Import archival artifacts (HTML, PDF, WARC, screenshots, text) into the existing ingestion pipeline with metadata and content hashes.
- Preserve bidirectional links: notes in Second Brain should reference ArchiveBox snapshot IDs; ArchiveBox should be aware of Second Brain job IDs for traceability.
- Reuse the new ingestion queue (`services/ingestion_queue`) so ArchiveBox runs asynchronously and updates note status when artifacts arrive.

## 3. High-Level Architecture
1. **Archive Job Request**: User submits a URL via capture UI/API → `/capture` detects `archive=true` (or advanced option).
2. **Ingestion Queue**: We enqueue an `ARCHIVE_WEB_CAPTURE` job containing URL, user_id, options.
3. **Archive Worker**: Background worker invokes ArchiveBox CLI/API (`archivebox add <url>`). Worker monitors job until snapshot completed.
4. **Artifact Harvest**: Once complete, worker reads ArchiveBox’s `index.sqlite3` or JSON export to gather paths (HTML, PDF, WARC, text, media).
5. **Second Brain Note Update**: Worker packages key assets (primary HTML text, metadata, screenshot) and calls `FileProcessor`/ingestion helpers to attach to the original note, update status to `complete`, and add links to stored archives.
6. **Storage Strategy**: Optionally copy canonical artifacts into our `uploads/` tree or store pointers to ArchiveBox-managed directories mounted under `/archivebox/snapshots`.

## 4. Implementation Steps

### 4.1 Environment & Deployment
- Add optional `ARCHIVEBOX_ENABLED`, `ARCHIVEBOX_PATH`, `ARCHIVEBOX_DATA_DIR`, `ARCHIVEBOX_API_TOKEN` to `.env`.
- Provide docker-compose profile or systemd instructions for running ArchiveBox side-by-side (reuse their official container).
- Update documentation (`docs/integrations/`) with setup instructions and security considerations (headless browser, storage size).

### 4.2 Application Layer Changes
- **Schemas**: Extend ingestion job payload schema to include `archivebox_snapshot_id`, `url`, `options`, `artifacts`.
- **Job Types**: Add `ARCHIVE_WEB_CAPTURE` enum to `IngestionJobType` (`services/ingestion_queue.py`).
- **Workers**: Create `services/archivebox_worker.py` with:
  - `enqueue_archive_job(url, user_id, note_id, priority)` helper.
  - Worker loop: call ArchiveBox (`subprocess` or HTTP), poll status (`archivebox status --json`), and timeout handling.
  - Artifact extraction: parse snapshot JSON, compute content hash, store canonical text (via readability output) into note; attach files via new `ArchiveArtifact` dataclass.
  - Event hooks: push realtime updates via `services.realtime_events` and notifications.
- **Capture Endpoint**: In `/capture`, when `source_url` present and archive option set, skip direct HTML parsing and enqueue archive job. Immediately return pending status + snapshot token.
- **Unified Capture / Web Ingestion**: Offer `ArchiveBox` as optional backend when deep archival required (e.g., long-form articles, YouTube). Web ingestion service can check `settings.web_capture_archive_with_archivebox`.

### 4.3 Data Import Pipeline
- Use `FileProcessor.process_saved_file` for downloaded artifacts we choose to copy into `uploads/`. For large WARC/ZIP we may only keep metadata + pointer to ArchiveBox storage to avoid duplication.
- Persist metadata in `notes.web_metadata`:
  ```json
  {
    "archivebox": {
      "snapshot_id": "2025-01-01T12:34:56Z",
      "url": "https://example.com",
      "files": {
        "html": "archivebox/snapshots/.../index.html",
        "pdf": "archivebox/snapshots/.../output.pdf",
        "text": "archivebox/snapshots/.../article.txt",
        "screenshot": "uploads/....png"
      },
      "timestamp": "...",
      "status": "complete"
    }
  }
  ```
- Update search indexing to pull in archive text chunks (existing processing pipeline handles `note.content`).

### 4.4 CLI & Automation
- CLI command (`scripts/archivebox_ingest.py`) for backfilling existing ArchiveBox snapshots into Second Brain.
- Cron job or Celery-like worker to process queue entries; leverage existing `tasks.process_ingestion_queue` to branch on job type.

## 5. Security & Resource Considerations
- Sandbox: ArchiveBox spawns browsers (Chromium); ensure it runs with limited privileges and outside web-facing network if sensitive.
- Storage growth: snapshots can be large; provide retention policies (`ARCHIVEBOX_MAX_SNAPSHOTS_PER_NOTE`, cleanup script).
- Concurrency: limit simultaneous ArchiveBox jobs to avoid saturating CPU/IO; maintain queue priority (urgent vs. backlog).
- Authentication: if ArchiveBox served via HTTP API, secure with token/SSH; prefer local socket invocation to reduce attack surface.

## 6. Testing Strategy
- Unit tests mocking `subprocess` responses for ArchiveBox CLI (success, failure, timeout).
- Integration test (behind feature flag) using a lightweight URL + headless stub to verify artifacts imported and note status transitions from `pending` → `complete`.
- Regression tests for ingestion queue ensuring non-archive jobs still flow correctly.

## 7. Rollout Plan
1. Land feature-flagged code paths (ArchiveBox disabled by default).
2. Deploy ArchiveBox in staging; run pilot ingestion on curated URLs.
3. Monitor note processing times, storage usage, and queue depth.
4. Gradually enable for specific capture sources (e.g., manual web captures, high-value sources).
5. Document admin procedures for snapshot pruning and ArchiveBox upgrades.

## 8. Open Questions
- Do we copy all files into Second Brain storage or rely on ArchiveBox paths? (Impacts backup strategy.)
- Should we expose ArchiveBox browsing UI inside Second Brain via iframe/proxy?
- How do we handle private/authenticated URLs (ArchiveBox supports cookies/headers; need secure credential management)?
- Do we dedupe snapshots across users or isolate per-user archives?

## 9. Next Steps
- [ ] Decide on storage strategy (symlink vs. copy) and update plan accordingly.
- [ ] Prototype ArchiveBox worker script with mocked CLI responses.
- [ ] Extend ingestion job enum and worker dispatcher.
- [ ] Update documentation (.env example, deployment guide).

