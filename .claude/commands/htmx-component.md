# Create HTMX Component

Create a new HTMX component for the Second Brain dashboard.

## Instructions

Help the user create a new reusable HTMX component with:

1. **Component template** (Jinja2 HTML)
2. **API endpoint** (FastAPI route)
3. **Integration code** (how to use it)

## Component Structure

```
templates/components/
├── [category]/
│   └── component_name.html
```

## Template Template 😄

```html
{# Component Name

Usage:
    {% include 'components/category/component_name.html' with data=data %}

Props:
    - data: Component data object
    - option1: Optional parameter
#}

<div class="component-container"
     x-data="{ state: 'initial' }"
     hx-get="/api/component/data"
     hx-trigger="load"
     hx-swap="innerHTML">

    <!-- Component content -->
    <div class="component-body">
        {{ data.field }}
    </div>

    <!-- Component actions -->
    <button hx-post="/api/component/action"
            hx-target="closest .component-container"
            hx-swap="outerHTML">
        Action
    </button>
</div>
```

## API Endpoint Template

```python
# In app.py or a router file

@app.get("/api/component/fragment")
async def get_component_fragment(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Return HTML fragment for component"""

    # Get data
    data = get_component_data(current_user.id)

    # Return template
    return templates.TemplateResponse(
        "components/category/component_name.html",
        {
            "request": request,
            "data": data
        }
    )

@app.post("/api/component/action")
async def component_action(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Handle component action"""

    # Process action
    result = process_action()

    # Return updated component
    return templates.TemplateResponse(
        "components/category/component_name.html",
        {
            "request": request,
            "data": result
        }
    )
```

## Usage in Dashboard

```html
<!-- In dashboard_htmx.html or any page -->

<div id="component-container">
    <!-- Load component on page load -->
    <div hx-get="/api/component/fragment"
         hx-trigger="load"
         hx-swap="innerHTML">
        {% include 'components/ui/loading_spinner.html' %}
    </div>
</div>

<!-- Or include directly with data -->
{% include 'components/category/component_name.html' with data=component_data %}
```

## HTMX Patterns

### 1. Auto-Refresh Component
```html
<div hx-get="/api/component/fragment"
     hx-trigger="load, every 30s"
     hx-swap="innerHTML">
    <!-- Auto-refreshes every 30 seconds -->
</div>
```

### 2. Click-to-Edit
```html
<div hx-get="/api/component/edit"
     hx-trigger="click"
     hx-target="this"
     hx-swap="outerHTML">
    Click to edit
</div>
```

### 3. Infinite Scroll
```html
<div hx-get="/api/component/more?page=2"
     hx-trigger="revealed"
     hx-swap="afterend">
    <!-- Loads when scrolled into view -->
</div>
```

### 4. Form Submission
```html
<form hx-post="/api/component/save"
      hx-target="#result"
      hx-swap="innerHTML">
    <input name="field">
    <button type="submit">Save</button>
</form>
```

## Alpine.js Integration

```html
<div x-data="{
        expanded: false,
        count: 0,
        toggleExpanded() {
            this.expanded = !this.expanded
        }
     }">

    <button @click="toggleExpanded()">
        <span x-show="!expanded">Show More</span>
        <span x-show="expanded">Show Less</span>
    </button>

    <div x-show="expanded" x-transition>
        <!-- Expandable content -->
    </div>
</div>
```

## Component Examples

### Tag Cloud Component

```html
{# tags/tag_cloud.html #}
<div class="flex flex-wrap gap-2">
    {% for tag, count in tags %}
    <a href="/search?tag={{ tag }}"
       class="inline-flex items-center px-3 py-1 rounded-full text-sm
              bg-indigo-100 text-indigo-800 hover:bg-indigo-200
              dark:bg-indigo-900 dark:text-indigo-200">
        {{ tag }}
        <span class="ml-1 text-xs opacity-75">{{ count }}</span>
    </a>
    {% endfor %}
</div>
```

### Activity Feed Component

```html
{# activity/feed.html #}
<div class="space-y-4">
    {% for activity in activities %}
    <div class="flex items-start gap-3 p-3 bg-gray-50 dark:bg-gray-800 rounded">
        <div class="flex-shrink-0">
            <span class="text-2xl">{{ activity.icon }}</span>
        </div>
        <div class="flex-1">
            <p class="text-sm font-medium">{{ activity.title }}</p>
            <p class="text-xs text-gray-500">{{ activity.time_ago }}</p>
        </div>
    </div>
    {% endfor %}
</div>
```

## Tips

1. **Keep components small** - Single responsibility
2. **Use TailwindCSS** - Consistent styling
3. **Add loading states** - Better UX
4. **Handle errors** - Show user-friendly messages
5. **Use Alpine.js for local state** - Dropdowns, modals, toggles
6. **Use HTMX for server state** - Data fetching, updates
7. **Add transitions** - x-transition for smooth animations
8. **Document props** - Comment at top of template
9. **Test with data** - Create example data structures
10. **Make it reusable** - Accept parameters via props

## Response Format

After creating a component:

**✅ Component created!**

**Files:**
- `templates/components/category/component_name.html`
- API endpoint in `app.py` (lines X-Y)

**Usage:**
```html
{% include 'components/category/component_name.html' with data=data %}
```

**API Endpoints:**
- `GET /api/component/fragment` - Fetch component
- `POST /api/component/action` - Perform action

**Next Steps:**
1. Add the component to your dashboard
2. Style it with TailwindCSS
3. Test the HTMX interactions
4. Add error handling
