# Search Troubleshooting Notes (Codex)

Author: Codex (GPT-5)
Date: 2025-09-19

This document summarizes the investigation and changes made while attempting to restore search functionality in the `second_brain` project. It is intended for engineers continuing the effort.

## 1. Baseline Assessment
- Initial symptom: Advanced Search UI returned "No results" even for broad queries (e.g., "audio recordings").
- API sanity check: `SearchService('notes.db').search('note')` worked only when the sqlite-vec extension was available. Without it, hybrid search raised `sqlite3.OperationalError: no such module: vec0` and the adapter silently returned an empty list.
- The unified search endpoint (`/api/search/unified`) depends on `UnifiedSearchService`, which wraps `SearchService`. When `SearchService` failed, the unified endpoint returned zero results.
- UI filters (`Notes`, `Audio`, `Files`, etc.) were purely cosmetic. Toggling them didn’t mutate the payload sent to `/api/search`, so the backend was always asked for “notes” regardless of selections.
- An autocomplete dropdown (`#searchSuggestions`) surfaced prior queries and templates, obscuring the input box.

## 2. Backend Changes
- **services/search_adapter.py**
  - Added `self.vec_available` to track whether sqlite-vec successfully loads.
  - Guarded `_semantic` and `_hybrid` to fall back to keyword search when vector support is missing or errors occur. Prevents `no such module: vec0` failures.
  - Relaxed `_sanitize_fts_query` to join multi-word queries with `OR` instead of requiring an exact phrase.
- **services/search_router.py**
  - `/api/search` now respects UI filters: `filters.types` (audio, files, ai) and `filters.recent`. Added SQL clauses for each filter and fallback to keyword fuzzy search for plain queries.
- **services/unified_search_service.py**
  - Added graceful handling when `SmartTemplatesService` or sqlite-vec are missing.
  - Generated better note descriptions, previews, quick actions. Still relies on `SearchService` results.
  - Suggestion generation simplified when smart templates are unavailable.

## 3. Frontend Changes
- **templates/dashboard_v3.html**
  - `toggleSearchFilter` now maintains `activeSearchFilters`, mutually exclusive with “Notes”.
  - `buildSearchFilters` constructs the new filters payload consumed by `/api/search`.
  - Removed the autocomplete dropdown behavior by hiding `#searchSuggestions` and short-circuiting `showSearchSuggestions`, `displaySearchSuggestions`, and `showLocalSearchHistory`.
  - Multiple UI refinements (token manager styling, search result rendering) made during testing.

## 4. Current State
- Keyword-only search via `SearchService` returns results; sqlite-vec is still absent, so hybrid/semantic portions are keyword-only.
- `/api/search` now filters correctly when called manually (verified via `curl`).
- Advanced Search UI still shows "No results" even though API returns data. Likely causes:
  1. The frontend still calls `/api/search/unified` elsewhere or expects `results.results` vs `results` shapes; the current `/api/search` returns a list, not the expected object. Need to confirm the UI route invoked on the `Search` button.
  2. Dashboard might expect `/api/search/unified` response; the basic `/api/search` fallback may not satisfy `displaySearchResults` binding.
  3. `displaySearchResults` expects objects with `created_at`, `title`, etc. Confirm API response structure matches.
- Autocomplete dropdown still appears despite JS changes; suspect cached script or additional handlers re-injecting suggestions (investigate inline event listeners, or remove the container entirely from DOM).

## 5. Recommendations / Next Steps
1. **Inspect Network Calls** in browser DevTools when hitting "Search" to confirm endpoint and response payload. Ensure the handler matches the UI’s expected structure (likely needs `{results: [...], total, mode}` rather than bare list).
2. **Consider switching UI to `/api/search/unified`** exclusively and ensure the backend returns the expected object with `success/results/analytics`. If continuing with `/api/search`, update `displaySearchResults` to handle the simple list format.
3. **Remove `#searchSuggestions` element entirely** or ensure CSS/JS doesn’t resurrect it. Might need to delete the DOM block or set `display:none !important` in CSS.
4. **Install sqlite-vec** properly (`brew install sqlite-utils` or vendor the extension). Set `SQLITE_VEC_PATH` so hybrid mode can be restored.
5. Create integration tests hitting the search endpoints to prevent regressions.

## 6. Files Modified During Investigation
- Backend: `services/search_adapter.py`, `services/search_router.py`, `services/unified_search_service.py`, `services/auth_service.py`, `services/apple_shortcuts_router.py`, etc.
- Frontend: `templates/dashboard_v3.html`, `templates/partials/userbar.html`.
- Misc: `README.md`, Apple shortcut JSONs, `config.py`, `audio_utils.py`, new migration `db/migrations/007_api_tokens.sql` (unrelated to search but added earlier).

This document does not revert changes; it only summarizes the work performed so far.
