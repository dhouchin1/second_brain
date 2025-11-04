# Create Note in Second Brain

Create a new note in the Second Brain database with AI-powered enhancements.

## Instructions

You are helping the user create a new note. Follow these steps:

1. **Get note content** from the user
2. **Generate title** using AI (Ollama) if not provided
3. **Generate tags** automatically based on content
4. **Create summary** using AI
5. **Save to database** with all metadata
6. **Sync to Obsidian vault** if enabled

## Code Template

```python
import sqlite3
from datetime import datetime
from pathlib import Path
from llm_utils import ollama_generate_title, ollama_summarize

# Database connection
db_path = Path("/Users/dhouchin/mvp-setup/second_brain/notes.db")
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Note data
content = "User's note content..."
user_id = 1  # Get from context

# Generate AI enhancements
title = ollama_generate_title(content) if not user_provided_title else user_provided_title
summary = ollama_summarize(content)
tags = extract_tags(content)  # Simple keyword extraction

# Insert note
cursor.execute("""
    INSERT INTO notes (
        user_id, title, content, body, summary, tags,
        type, status, created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (
    user_id,
    title,
    content,
    content,  # body is same as content
    summary,
    ','.join(tags),
    'text',
    'completed',
    datetime.now().isoformat(),
    datetime.now().isoformat()
))

note_id = cursor.lastrowid
conn.commit()
conn.close()

print(f"✅ Created note #{note_id}: {title}")
```

## AI Enhancement Functions

```python
from llm_utils import ollama_generate_title, ollama_summarize

# Generate title
title = ollama_generate_title(content)

# Generate summary
summary = ollama_summarize(content)
```

## Tag Extraction

```python
import re

def extract_tags(text):
    """Extract hashtags and keywords"""
    # Find hashtags
    hashtags = re.findall(r'#(\w+)', text)

    # Extract keywords (simple approach)
    keywords = []
    # Add your keyword extraction logic

    return list(set(hashtags + keywords))[:5]  # Top 5 tags
```

## Response Format

After creating the note:

**✅ Note created successfully!**

- **ID**: #123
- **Title**: [Generated or provided title]
- **Tags**: #tag1, #tag2, #tag3
- **Summary**: [AI-generated summary]
- **Created**: 2024-01-15 14:30

The note has been:
- ✅ Saved to database
- ✅ Indexed for search
- ✅ Synced to Obsidian vault (if enabled)

Would you like to:
1. View the full note
2. Create another note
3. Search for related notes

## Database Schema

```sql
CREATE TABLE notes (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    title TEXT,
    content TEXT,
    body TEXT,
    summary TEXT,
    tags TEXT,
    type TEXT,
    status TEXT,
    created_at TEXT,
    updated_at TEXT
);
```

## Tips

- Always generate a title if not provided
- Generate summary for notes > 100 characters
- Extract up to 5 relevant tags
- Set status to 'completed' for text notes
- Use ISO format for timestamps
