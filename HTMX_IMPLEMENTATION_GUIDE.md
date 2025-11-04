# HTMX Implementation Guide for Second Brain

## Overview

This guide documents the HTMX-based frontend implementation for Second Brain. The implementation uses **100% Python** (FastAPI + Jinja2) with HTMX and Alpine.js for dynamic interactions, requiring **zero build steps or JavaScript bundlers**.

## Architecture

### Technology Stack

- **Backend**: FastAPI (existing)
- **Templates**: Jinja2 (existing)
- **Dynamic Updates**: HTMX (via CDN)
- **Local State**: Alpine.js (via CDN)
- **Styling**: TailwindCSS (via CDN)
- **Build Process**: NONE ✅

### Key Benefits

1. ✅ **No Build Step** - Everything runs directly from CDN or static files
2. ✅ **100% Python** - All logic in Python, minimal JavaScript
3. ✅ **Progressive Enhancement** - Works without JavaScript (forms submit normally)
4. ✅ **Fast Development** - No compile time, instant refresh
5. ✅ **Easy Testing** - Test templates with Python, no JavaScript testing needed
6. ✅ **Maintainability** - Component-based templates, reusable fragments

---

## File Structure

```
second_brain/
├── services/
│   └── htmx_helpers.py              # HTMX utility functions (NEW)
├── templates/
│   ├── base_htmx.html               # Base template with HTMX/Alpine (NEW)
│   ├── pages/
│   │   └── dashboard_htmx.html      # Main dashboard page (NEW)
│   └── components/                  # Reusable HTML fragments (NEW)
│       ├── notes/
│       │   ├── note_card.html       # Single note display
│       │   └── note_list.html       # List with infinite scroll
│       ├── search/
│       │   ├── search_bar.html      # Search input with filters
│       │   └── search_results.html  # Search results fragment
│       └── ui/
│           ├── empty_state.html     # Empty state component
│           ├── loading_spinner.html # Loading indicator
│           └── stats_widget.html    # Stats dashboard
├── static/
│   └── js/
│       └── htmx-helpers.js          # Minimal JavaScript utilities (NEW)
└── app.py                           # FastAPI app (UPDATED with new routes)
```

---

## Implementation Details

### 1. HTMX Helper Service (`services/htmx_helpers.py`)

Utility functions for working with HTMX in FastAPI:

```python
from services.htmx_helpers import (
    is_htmx_request,      # Check if request is from HTMX
    htmx_redirect,        # HTMX-compatible redirect
    htmx_trigger,         # Trigger client-side events
    HTMXResponse          # Enhanced HTML response with helpers
)
```

**Key Functions:**
- `is_htmx_request(request)` - Detects HTMX requests via HX-Request header
- `htmx_trigger(response, event, detail)` - Send events to client
- `HTMXResponse` - Chainable response helpers (.trigger(), .refresh(), etc.)

### 2. Base Template (`templates/base_htmx.html`)

Provides the foundation for all HTMX pages:

**Features:**
- HTMX Core + Extensions (SSE, Response Targets)
- Alpine.js for local component state
- Global toast notification system
- Error handling for failed requests
- Dark mode support
- PWA meta tags

**Alpine.js Global State:**
```javascript
function appState() {
    return {
        toasts: [],           // Toast notifications
        showToast(options),   // Display toast
        removeToast(id),      // Dismiss toast
        handleHtmxResponse()  // Global HTMX error handler
    }
}
```

### 3. Component Templates (`templates/components/`)

Reusable HTML fragments for HTMX swaps:

#### Note Components
- **`note_card.html`** - Single note with actions (edit, delete, duplicate)
- **`note_list.html`** - List of notes with infinite scroll trigger

#### Search Components
- **`search_bar.html`** - Search input with debouncing and filters
- **`search_results.html`** - Search results fragment

#### UI Components
- **`stats_widget.html`** - Dashboard stats (auto-refreshes every 30s)
- **`loading_spinner.html`** - Loading indicator
- **`empty_state.html`** - Empty state with icons

### 4. Main Dashboard (`templates/pages/dashboard_htmx.html`)

The main user interface:

**Key Features:**
1. **Auto-refreshing Stats** - Updates every 30 seconds via HTMX
2. **Quick Capture Form** - Inline note creation with HTMX
3. **Search-as-you-type** - 300ms debounced search
4. **Infinite Scroll** - Lazy-loading notes list
5. **Real-time Updates** - SSE connection for live note updates
6. **Offline Indicator** - Shows when connection is lost

**HTMX Attributes Used:**
```html
hx-get="/api/notes/fragment"           # GET request
hx-post="/api/notes"                   # POST request
hx-delete="/api/notes/{id}"            # DELETE request
hx-trigger="load, every 30s"           # Trigger on load + poll
hx-target="#notes-list"                # Where to swap content
hx-swap="innerHTML"                    # How to swap
hx-confirm="Are you sure?"             # Confirmation dialog
hx-ext="sse"                           # Server-Sent Events
sse-connect="/api/sse/notes"           # SSE endpoint
```

### 5. API Endpoints (app.py)

New HTMX-compatible endpoints that return HTML fragments:

#### Dashboard Route
```python
@app.get("/dashboard/htmx")
def dashboard_htmx(request: Request)
```
- Serves the main HTMX dashboard page
- Requires authentication

#### Fragment Endpoints
```python
@app.get("/api/notes/fragment")
async def get_notes_fragment(request, skip, limit, current_user)
```
- Returns HTML fragment of notes
- Supports pagination for infinite scroll
- Uses `note_list.html` component

```python
@app.get("/api/stats/fragment")
async def get_stats_fragment(request, current_user)
```
- Returns stats widget HTML
- Auto-refreshes every 30 seconds
- Uses `stats_widget.html` component

```python
@app.get("/api/search/fragment")
async def search_fragment(request, q, type, date_range, sort, current_user)
```
- Returns search results HTML
- Supports filters and sorting
- Uses `search_results.html` component

### 6. JavaScript Helpers (`static/js/htmx-helpers.js`)

Minimal JavaScript utilities (< 150 lines):

**Features:**
- Global error handling for HTMX requests
- Toast notification wrapper function
- Form reset on success
- Loading states for buttons
- Development logging

**Usage:**
```javascript
showToast('Note saved!', 'success');
showToast('Error occurred', 'error', 'Details here');
```

---

## How HTMX Works - Examples

### Example 1: Quick Capture Form

**HTML:**
```html
<form hx-post="/api/notes"
      hx-target="#notes-list"
      hx-swap="afterbegin"
      hx-on::after-request="if(event.detail.successful) this.reset()">
    <textarea name="content"></textarea>
    <button type="submit">Capture</button>
</form>
```

**What Happens:**
1. User types note and clicks "Capture"
2. HTMX sends POST to `/api/notes`
3. FastAPI creates note, returns HTML fragment of new note
4. HTMX inserts HTML at top of `#notes-list`
5. Form resets automatically
6. Toast notification shows success

**FastAPI Endpoint:**
```python
@app.post("/api/notes")
async def create_note(request: Request, current_user: User):
    # Create note in database
    note = create_note_in_db(...)

    # Return HTML fragment (not JSON!)
    return templates.TemplateResponse("components/notes/note_card.html", {
        "request": request,
        "note": note
    })
```

### Example 2: Infinite Scroll

**HTML:**
```html
<div hx-get="/api/notes/fragment?skip=20&limit=20"
     hx-trigger="revealed"
     hx-swap="afterend">
    <!-- Loading spinner shown during request -->
    <div class="htmx-indicator">Loading...</div>
</div>
```

**What Happens:**
1. User scrolls to bottom
2. Element becomes visible ("revealed")
3. HTMX fetches next page
4. HTML fragments inserted after current list
5. Process repeats for next page

### Example 3: Search-as-you-type

**HTML:**
```html
<input type="text"
       hx-get="/api/search/fragment"
       hx-trigger="keyup changed delay:300ms"
       hx-target="#search-results">
```

**What Happens:**
1. User types in search box
2. HTMX waits 300ms after last keystroke
3. Sends GET request to `/api/search/fragment?q=query`
4. FastAPI searches database, returns HTML
5. Results appear instantly

---

## Development Workflow

### Starting the Server

```bash
# Start development server
.venv/bin/python -m uvicorn app:app --reload --port 8082

# Visit HTMX dashboard
open http://localhost:8082/dashboard/htmx
```

### Creating New Components

1. **Create component template:**
```bash
touch templates/components/ui/my_component.html
```

2. **Write component HTML:**
```html
{# My Component
Usage: {% include 'components/ui/my_component.html' with data=data %}
#}
<div class="my-component">
    {{ data.title }}
</div>
```

3. **Create API endpoint:**
```python
@app.get("/api/my-component/fragment")
async def get_my_component(request: Request, current_user: User):
    data = get_data_from_db()
    return templates.TemplateResponse("components/ui/my_component.html", {
        "request": request,
        "data": data
    })
```

4. **Use in dashboard:**
```html
<div hx-get="/api/my-component/fragment"
     hx-trigger="load"
     hx-swap="innerHTML">
    Loading...
</div>
```

### Debugging

**Enable HTMX Debug Mode:**
Already enabled in development via `base_htmx.html`:
```html
{% if config.DEBUG %}
<script src="https://unpkg.com/htmx.org@1.9.10/dist/ext/debug.js"></script>
{% endif %}
```

**Check Browser Console:**
- HTMX logs all requests
- JavaScript helpers log errors
- Alpine.js state visible in Vue DevTools

**Check FastAPI Logs:**
```bash
# API requests logged automatically
INFO:     127.0.0.1:52392 - "GET /api/notes/fragment?skip=0&limit=20 HTTP/1.1" 200 OK
```

---

## Testing

### Testing Components

```python
# Test template rendering
from fastapi.testclient import TestClient

def test_note_card_rendering():
    response = client.get("/api/notes/fragment", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert b"note-card" in response.content
```

### Testing HTMX Endpoints

```bash
# Test with curl
curl -H "HX-Request: true" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8082/api/notes/fragment
```

---

## Common Patterns

### Pattern 1: Modal Dialogs

```html
<!-- Trigger -->
<button hx-get="/api/notes/123/edit"
        hx-target="#modal-container"
        hx-swap="innerHTML">
    Edit Note
</button>

<!-- Container -->
<div id="modal-container"></div>
```

### Pattern 2: Optimistic Updates

```html
<button hx-delete="/api/notes/123"
        hx-target="closest .note-card"
        hx-swap="outerHTML swap:300ms">
    Delete
</button>
```

### Pattern 3: Dependent Dropdowns

```html
<select hx-get="/api/subcategories"
        hx-target="#subcategory-select"
        hx-trigger="change">
    <option>Category 1</option>
</select>

<select id="subcategory-select">
    <!-- Populated by HTMX -->
</select>
```

---

## Performance Considerations

1. **Caching** - Add caching headers to fragment endpoints for static data
2. **Pagination** - Use cursor-based pagination for large lists
3. **Debouncing** - Always use `delay:Xms` for search inputs
4. **Lazy Loading** - Use `hx-trigger="revealed"` for infinite scroll
5. **Minimal JavaScript** - Keep custom JS under 200 lines total

---

## Migration from v3 Dashboard

Both dashboards can coexist:

```python
# Existing dashboard
@app.get("/dashboard/v3")
def dashboard_v3(request: Request):
    return templates.TemplateResponse("dashboard_v3.html", {...})

# New HTMX dashboard
@app.get("/dashboard/htmx")
def dashboard_htmx(request: Request):
    return templates.TemplateResponse("pages/dashboard_htmx.html", {...})
```

Users can choose which version to use during the transition period.

---

## Next Steps

### Immediate Improvements

1. **SSE Implementation** - Add Server-Sent Events for real-time updates
2. **Note Editing** - Create inline editing with HTMX
3. **File Upload** - Add drag-and-drop with progress bars
4. **Voice Recording** - Integrate browser recording API

### Future Enhancements

1. **Offline Support** - Service worker for offline capability
2. **Keyboard Shortcuts** - Add power user shortcuts
3. **Bulk Actions** - Multi-select with batch operations
4. **Advanced Filters** - Saved searches and complex filters

---

## Troubleshooting

### HTMX Not Working

1. Check browser console for errors
2. Verify HTMX CDN is loading
3. Check HX-Request header in network tab
4. Enable debug extension

### Templates Not Found

```python
# Verify templates directory structure
templates/
├── base_htmx.html
├── pages/
│   └── dashboard_htmx.html
└── components/
    └── ...
```

### Authentication Issues

```python
# Check current user in endpoint
current_user = get_current_user_silent(request)
if not current_user:
    return RedirectResponse("/login", status_code=302)
```

---

## Resources

- **HTMX Docs**: https://htmx.org/docs/
- **Alpine.js Docs**: https://alpinejs.dev/start-here
- **FastAPI Templates**: https://fastapi.tiangolo.com/advanced/templates/
- **Examples**: Check `templates/pages/dashboard_htmx.html` for patterns

---

## Summary

The HTMX implementation provides a modern, reactive frontend experience while keeping the entire stack in Python. No build process, no complex JavaScript frameworks, just clean HTML templates with dynamic behavior via HTMX attributes.

**Total Lines of Code:**
- Python (htmx_helpers.py): ~200 lines
- Templates (all components): ~800 lines
- JavaScript (htmx-helpers.js): ~150 lines
- **Total**: ~1,150 lines for a complete interactive dashboard

Compare to React equivalent: 3,000+ lines + build config + node_modules 😅
