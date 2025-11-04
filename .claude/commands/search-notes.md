# Search Second Brain Notes

Search through all your Second Brain notes using the unified search service.

## Instructions

You are helping the user search their Second Brain knowledge base. Follow these steps:

1. **Ask for search query** if not provided
2. **Use the search service** to find matching notes
3. **Display results** in a clear, organized format
4. **Offer to show details** of specific notes

## Example Usage

```python
# Search for notes
from services.search_adapter import get_search_service

search_service = get_search_service()
results = search_service.search(
    query="user's search query",
    user_id=1,  # Use appropriate user_id
    limit=20
)

# Display results
for result in results:
    print(f"Title: {result['title']}")
    print(f"Content: {result['content'][:200]}...")
    print(f"Tags: {result.get('tags', [])}")
    print(f"Created: {result.get('created_at')}")
    print("---")
```

## Database Access

```python
import sqlite3
from pathlib import Path

db_path = Path("/Users/dhouchin/mvp-setup/second_brain/notes.db")
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Search using FTS5
cursor.execute("""
    SELECT id, title, body, tags, created_at, rank
    FROM notes_fts
    WHERE notes_fts MATCH ?
    ORDER BY rank
    LIMIT 20
""", (query,))

results = cursor.fetchall()
conn.close()
```

## Response Format

Present results like this:

**Found X results for "query":**

1. **Note Title**
   - Tags: #tag1, #tag2
   - Created: 2024-01-15
   - Preview: First 200 characters...

2. **Another Note**
   - Tags: #tag3
   - Created: 2024-01-14
   - Preview: Content preview...

Would you like to see the full content of any note?

## Tips

- Use FTS5 for fast full-text search
- Show top 10-20 results
- Highlight matching terms if possible
- Offer to filter by tags, dates, or type
