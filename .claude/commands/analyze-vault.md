# Analyze Second Brain Vault

Analyze the user's Second Brain vault to provide insights about their knowledge base.

## Instructions

Provide comprehensive analytics about the user's notes, including:

1. **Basic Statistics**
   - Total notes count
   - Notes by type (text, audio, image, etc.)
   - Recent activity (notes per day/week/month)

2. **Content Analysis**
   - Most common tags
   - Tag cloud / tag frequency
   - Average note length
   - Longest/shortest notes

3. **Temporal Patterns**
   - Notes created over time
   - Most productive days/times
   - Recent vs historical activity

4. **Knowledge Graph Insights**
   - Connected topics (via tags)
   - Isolated notes (no tags)
   - Tag relationships

## Code Template

```python
import sqlite3
from datetime import datetime, timedelta
from collections import Counter
from pathlib import Path

db_path = Path("/Users/dhouchin/mvp-setup/second_brain/notes.db")
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Basic stats
total_notes = cursor.execute("SELECT COUNT(*) FROM notes WHERE user_id = ?", (user_id,)).fetchone()[0]

# Notes by type
cursor.execute("""
    SELECT type, COUNT(*) as count
    FROM notes
    WHERE user_id = ?
    GROUP BY type
    ORDER BY count DESC
""", (user_id,))
types = cursor.fetchall()

# Recent activity (last 30 days)
thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
recent_notes = cursor.execute("""
    SELECT COUNT(*) FROM notes
    WHERE user_id = ? AND created_at > ?
""", (user_id, thirty_days_ago)).fetchone()[0]

# Tag analysis
cursor.execute("SELECT tags FROM notes WHERE user_id = ? AND tags IS NOT NULL", (user_id,))
all_tags = []
for (tags_str,) in cursor.fetchall():
    if tags_str:
        all_tags.extend([t.strip() for t in tags_str.replace('#', '').split(',') if t.strip()])

tag_counts = Counter(all_tags)
top_tags = tag_counts.most_common(10)

# Average note length
cursor.execute("""
    SELECT AVG(LENGTH(content)) as avg_length,
           MIN(LENGTH(content)) as min_length,
           MAX(LENGTH(content)) as max_length
    FROM notes
    WHERE user_id = ?
""", (user_id,))
avg_len, min_len, max_len = cursor.fetchone()

conn.close()
```

## Response Format

Present analytics like this:

---

# 📊 Second Brain Analytics

## 📈 Overview
- **Total Notes**: 1,234
- **Notes (Last 30 Days)**: 89
- **Average Daily**: 3.0 notes/day

## 📝 Notes by Type
1. Text: 856 (69%)
2. Audio: 234 (19%)
3. URL: 89 (7%)
4. Image: 55 (5%)

## 🏷️ Top Tags
1. #work (234 notes)
2. #ideas (189 notes)
3. #learning (156 notes)
4. #productivity (145 notes)
5. #personal (123 notes)

## 📏 Content Stats
- **Average Length**: 342 characters
- **Shortest Note**: 12 characters
- **Longest Note**: 5,678 characters

## 📅 Activity Trends
- **Most Productive Day**: Monday (234 notes)
- **Peak Hour**: 9-10 AM (123 notes)
- **Trend**: ↗️ Increasing (12% vs last month)

## 🔗 Knowledge Graph
- **Connected Topics**: 23 tag clusters
- **Isolated Notes**: 45 (4%)
- **Most Connected Tag**: #work (links to 12 other tags)

---

Would you like me to:
1. Deep dive into a specific tag?
2. Find patterns in your notes?
3. Suggest connections between topics?
4. Export this analysis?

## Visualizations

```python
# Create tag cloud data
tag_cloud = {tag: count for tag, count in top_tags}

# Activity over time
cursor.execute("""
    SELECT DATE(created_at) as date, COUNT(*) as count
    FROM notes
    WHERE user_id = ?
    GROUP BY DATE(created_at)
    ORDER BY date DESC
    LIMIT 30
""", (user_id,))
daily_activity = cursor.fetchall()
```

## Advanced Queries

```python
# Find notes without tags
cursor.execute("""
    SELECT id, title FROM notes
    WHERE user_id = ? AND (tags IS NULL OR tags = '')
    LIMIT 10
""", (user_id,))
untagged = cursor.fetchall()

# Find most edited notes
cursor.execute("""
    SELECT id, title,
           JULIANDAY(updated_at) - JULIANDAY(created_at) as age_days
    FROM notes
    WHERE user_id = ?
    ORDER BY age_days DESC
    LIMIT 10
""", (user_id,))
old_notes = cursor.fetchall()
```

## Tips

- Present data visually with emojis and formatting
- Highlight trends and patterns
- Offer actionable insights
- Suggest improvements (tag untagged notes, etc.)
