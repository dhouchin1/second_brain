from pathlib import Path
from datetime import datetime
import re
from config import settings

def safe_filename(text: str) -> str:
    """Create a safe filename from text"""
    # Remove special characters and spaces
    safe = re.sub(r'[^\w\s-]', '', text.lower())
    safe = re.sub(r'[-\s]+', '-', safe)
    return safe[:50]  # Limit length

def save_markdown(title: str, content: str, filename: str = None, **metadata):
    """Save a note as markdown file"""
    vault_path = settings.vault_path
    vault_path.mkdir(exist_ok=True)
    
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = safe_filename(title)
        filename = f"{timestamp}_{safe_title}.md"
    
    # Build frontmatter
    frontmatter = ["---"]
    frontmatter.append(f"title: {title}")
    frontmatter.append(f"date: {datetime.now().isoformat()}")
    
    for key, value in metadata.items():
        if value:
            frontmatter.append(f"{key}: {value}")
    
    frontmatter.append("---")
    
    # Combine frontmatter and content
    full_content = "\n".join(frontmatter) + "\n\n" + content
    
    # Write to file
    file_path = vault_path / filename
    file_path.write_text(full_content, encoding='utf-8')
    
    return str(file_path)

def export_notes_to_obsidian(user_id: int):
    """Export all user notes to Obsidian vault"""
    from database import get_conn
    
    conn = get_conn()
    c = conn.cursor()
    
    rows = c.execute(
        "SELECT title, content, tags, timestamp FROM notes WHERE user_id = ?",
        (user_id,)
    ).fetchall()
    
    exported = []
    for title, content, tags, timestamp in rows:
        if not title:
            title = "Untitled Note"
        if not content:
            content = ""
        
        filename = save_markdown(
            title=title,
            content=content,
            tags=tags,
            created=timestamp
        )
        exported.append(filename)
    
    conn.close()
    return exported
