# Debug HTMX Issues

Help debug HTMX interactions and find why they're not working.

## Instructions

When the user reports an HTMX issue, systematically check:

1. **Browser Console** - Check for JavaScript errors
2. **Network Tab** - Verify requests are being sent
3. **Response Content** - Ensure HTML is being returned
4. **HTMX Headers** - Check HX-Request headers
5. **Target Elements** - Verify selectors exist
6. **Swap Behavior** - Check swap method is correct

## Debugging Checklist

### 1. Check Browser Console

```javascript
// Common errors to look for:
- "HTMX: Cannot find target element"
- "Unexpected token < in JSON"
- CORS errors
- 404/500 errors
```

### 2. Enable HTMX Debug Mode

Already enabled in development via `base_htmx.html`:

```html
<script src="https://unpkg.com/htmx.org@1.9.10/dist/ext/debug.js"></script>
```

Look for HTMX debug logs in console.

### 3. Check Network Requests

Look for:
- Request method (GET, POST, etc.)
- Request URL
- HX-Request header (should be "true")
- HX-Target header
- HX-Trigger header
- Response status code
- Response content type (should be text/html)

### 4. Verify Endpoint Returns HTML

Common mistake: Returning JSON instead of HTML

**❌ Wrong:**
```python
@app.get("/api/notes/fragment")
async def get_notes():
    return {"notes": notes}  # JSON!
```

**✅ Correct:**
```python
@app.get("/api/notes/fragment")
async def get_notes(request: Request):
    return templates.TemplateResponse("components/notes/note_list.html", {
        "request": request,
        "notes": notes
    })
```

### 5. Check Target Element Exists

```javascript
// In browser console:
document.querySelector('#target-id')
// Should return the element, not null
```

### 6. Verify HTMX Attributes

Common issues:

```html
<!-- ❌ Wrong: Missing hx-target -->
<button hx-get="/api/data">Load</button>

<!-- ✅ Correct: Has target -->
<button hx-get="/api/data" hx-target="#results">Load</button>

<!-- ❌ Wrong: Invalid selector -->
<div hx-get="/api/data" hx-target="results"></div>

<!-- ✅ Correct: Valid selector -->
<div hx-get="/api/data" hx-target="#results"></div>
```

## Common Issues & Solutions

### Issue 1: "Nothing happens when I click"

**Check:**
1. Is HTMX loaded? `typeof htmx` in console
2. Does element have hx- attributes?
3. Is there a JavaScript error?
4. Is the endpoint correct?

**Debug:**
```javascript
// Check if HTMX is loaded
console.log('HTMX version:', htmx.version);

// Listen for HTMX events
document.body.addEventListener('htmx:beforeRequest', (e) => {
    console.log('Request starting:', e.detail);
});

document.body.addEventListener('htmx:afterRequest', (e) => {
    console.log('Request finished:', e.detail);
});
```

### Issue 2: "Request sent but nothing updates"

**Check:**
1. Response content (should be HTML)
2. hx-target selector (does it exist?)
3. hx-swap method
4. Response status code (should be 2xx)

**Debug:**
```python
# In your endpoint, log the response
@app.get("/api/test")
async def test(request: Request):
    html = templates.TemplateResponse("test.html", {"request": request})
    print(f"Returning HTML: {html.body[:100]}")  # First 100 chars
    return html
```

### Issue 3: "Page refreshes instead of HTMX swap"

**Cause:** Form is submitting normally, not via HTMX

**Fix:**
```html
<!-- ❌ Wrong: Missing hx-post -->
<form action="/api/submit">
    <button>Submit</button>
</form>

<!-- ✅ Correct: Has hx-post -->
<form hx-post="/api/submit" hx-target="#result">
    <button type="submit">Submit</button>
</form>
```

### Issue 4: "Getting JSON instead of HTML"

**Check endpoint:**
```python
# ❌ Wrong
return JSONResponse({"data": data})

# ✅ Correct
return templates.TemplateResponse("component.html", {
    "request": request,
    "data": data
})
```

### Issue 5: "CORS errors"

**Solution:**
```python
# In app.py, ensure CORS middleware is configured
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Issue 6: "Trigger not working"

**Common triggers:**
```html
<!-- Load on page load -->
<div hx-get="/api/data" hx-trigger="load"></div>

<!-- Every 30 seconds -->
<div hx-get="/api/data" hx-trigger="every 30s"></div>

<!-- On click -->
<button hx-get="/api/data" hx-trigger="click"></button>

<!-- On input change with delay -->
<input hx-get="/api/search" hx-trigger="keyup changed delay:300ms">

<!-- When revealed (infinite scroll) -->
<div hx-get="/api/more" hx-trigger="revealed"></div>

<!-- Multiple triggers -->
<div hx-get="/api/data" hx-trigger="load, every 30s"></div>
```

## Debug Script

Run this in browser console to diagnose issues:

```javascript
// HTMX Debug Helper
function debugHTMX(selector) {
    const el = document.querySelector(selector);

    if (!el) {
        console.error('❌ Element not found:', selector);
        return;
    }

    console.log('✅ Element found:', el);
    console.log('HTMX attributes:', {
        'hx-get': el.getAttribute('hx-get'),
        'hx-post': el.getAttribute('hx-post'),
        'hx-target': el.getAttribute('hx-target'),
        'hx-swap': el.getAttribute('hx-swap'),
        'hx-trigger': el.getAttribute('hx-trigger'),
    });

    // Check if target exists
    const target = el.getAttribute('hx-target');
    if (target) {
        const targetEl = document.querySelector(target);
        console.log('Target element:', targetEl ? '✅ Found' : '❌ Not found');
    }

    // Add event listeners
    el.addEventListener('htmx:beforeRequest', (e) => {
        console.log('🔵 Request starting:', e.detail);
    });

    el.addEventListener('htmx:afterRequest', (e) => {
        console.log('🟢 Request finished:', e.detail);
    });

    el.addEventListener('htmx:responseError', (e) => {
        console.error('🔴 Request error:', e.detail);
    });
}

// Usage: debugHTMX('#my-button')
```

## Testing Endpoints Directly

```bash
# Test with curl
curl -H "HX-Request: true" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8082/api/notes/fragment

# Should return HTML, not JSON
```

## Response Template

After debugging:

**🔍 Debug Results:**

**Issue Found:**
- [Description of the problem]

**Root Cause:**
- [Why it's happening]

**Solution:**
```[language]
[Code fix]
```

**How to verify:**
1. [Step to test]
2. [Expected result]

**Additional Tips:**
- [Helpful advice]
