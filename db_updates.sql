-- Add to init_db() function in app.py

# Create Discord users table
c.execute('''
    CREATE TABLE IF NOT EXISTS discord_users (
        discord_id INTEGER PRIMARY KEY,
        user_id INTEGER,
        linked_at TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
''')

# Create reminders table  
c.execute('''
    CREATE TABLE IF NOT EXISTS reminders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        note_id INTEGER,
        user_id INTEGER,
        due_date TEXT,
        completed BOOLEAN DEFAULT FALSE,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(note_id) REFERENCES notes(id),
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
''')

# Create enhanced FTS5 table
c.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts5 USING fts5(
        title, content, summary, tags, actions,
        content='notes', content_rowid='id',
        tokenize='porter unicode61'
    )
''')

# Create search analytics
c.execute('''
    CREATE TABLE IF NOT EXISTS search_analytics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        query TEXT,
        results_count INTEGER,
        clicked_result_id INTEGER,
        search_type TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    )
''')
