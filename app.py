from search_engine import EnhancedSearchEngine, SearchResult
from discord_bot import SecondBrainCog
from obsidian_sync import ObsidianSync
import sqlite3
from datetime import datetime, timedelta
from fastapi import (
    FastAPI,
    Request,
    Form,
    UploadFile,
    File,
    Body,
    Query,
    BackgroundTasks,
    Depends,
    HTTPException,
    status,
)
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from collections import defaultdict
import pathlib
from llm_utils import ollama_summarize, ollama_generate_title
from tasks import process_note
from markupsafe import Markup, escape
import re
from config import settings
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel
from markdown_writer import save_markdown, safe_filename
from audio_utils import transcribe_audio
from typing import Optional, List

# ---- FastAPI Setup ----
app = FastAPI()
templates = Jinja2Templates(directory=str(settings.base_dir / "templates"))
app.mount("/static", StaticFiles(directory=str(settings.base_dir / "static")), name="static")

def highlight(text, term):
    if not text or not term:
        return text
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    return Markup(pattern.sub(lambda m: f"<mark>{escape(m.group(0))}</mark>", text))
templates.env.filters['highlight'] = highlight

def get_conn():
    return sqlite3.connect(str(settings.db_path))

def get_last_sync():
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "CREATE TABLE IF NOT EXISTS sync_status (id INTEGER PRIMARY KEY, last_sync TEXT)"
    )
    row = c.execute("SELECT last_sync FROM sync_status WHERE id = 1").fetchone()
    conn.close()
    return row[0] if row else None

def set_last_sync(ts: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "CREATE TABLE IF NOT EXISTS sync_status (id INTEGER PRIMARY KEY, last_sync TEXT)"
    )
    c.execute(
        "INSERT INTO sync_status (id, last_sync) VALUES (1, ?) "
        "ON CONFLICT(id) DO UPDATE SET last_sync=excluded.last_sync",
        (ts,),
    )
    conn.commit()
    conn.close()

def export_notes_to_obsidian(user_id: int):
    conn = get_conn()
    c = conn.cursor()
    rows = c.execute(
        "SELECT title, content, timestamp FROM notes WHERE user_id = ?",
        (user_id,),
    ).fetchall()
    for title, content, ts in rows:
        file_ts = ts.replace(":", "-").replace(" ", "_") if ts else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = f"{file_ts}-{safe_filename(title or 'note')}.md"
        save_markdown(title or "", content or "", fname)
    set_last_sync(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    conn.close()

# Auth setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SECRET_KEY = "super-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class User(BaseModel):
    id: int
    username: str

class UserInDB(User):
    hashed_password: str

# Enhanced data models
class DiscordWebhook(BaseModel):
    note: str
    tags: str = ""
    type: str = "discord"
    discord_user_id: Optional[int] = None
    timestamp: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    filters: Optional[dict] = {}
    limit: int = 20

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(username: str):
    conn = get_conn()
    c = conn.cursor()
    row = c.execute(
        "SELECT id, username, hashed_password FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    conn.close()
    if row:
        return UserInDB(id=row[0], username=row[1], hashed_password=row[2])
    return None

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(token_data.username)
    if user is None:
        raise credentials_exception
    return user

def init_db():
    conn = get_conn()
    c = conn.cursor()
    
    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL
        )
    ''')
    
    # Notes table
    c.execute('''
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            summary TEXT,
            tags TEXT,
            actions TEXT,
            type TEXT,
            timestamp TEXT,
            audio_filename TEXT,
            content TEXT,
            status TEXT DEFAULT 'complete',
            user_id INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    # FTS table
    c.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
            title, summary, tags, actions, content, content='notes', content_rowid='id'
        )
    ''')
    
    # Enhanced FTS5 table
    c.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts5 USING fts5(
            title, content, summary, tags, actions,
            content='notes', content_rowid='id',
            tokenize='porter unicode61'
        )
    ''')
    
    # Discord users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS discord_users (
            discord_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            linked_at TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    # Reminders table
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
    
    # Search analytics
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
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS sync_status (
            id INTEGER PRIMARY KEY,
            last_sync TEXT
        )
    ''')
    
    # Ensure columns exist
    cols = [row[1] for row in c.execute("PRAGMA table_info(notes)")]
    if 'status' not in cols:
        c.execute("ALTER TABLE notes ADD COLUMN status TEXT DEFAULT 'complete'")
        c.execute("UPDATE notes SET status='complete' WHERE status IS NULL")
    if 'user_id' not in cols:
        c.execute("ALTER TABLE notes ADD COLUMN user_id INTEGER")
    if 'actions' not in cols:
        c.execute("ALTER TABLE notes ADD COLUMN actions TEXT")

    # Update FTS if needed
    fts_cols = [row[1] for row in c.execute("PRAGMA table_info(notes_fts)")]
    if 'actions' not in fts_cols:
        c.execute("DROP TABLE IF EXISTS notes_fts")
        c.execute('''
            CREATE VIRTUAL TABLE notes_fts USING fts5(
                title, summary, tags, actions, content, content='notes', content_rowid='id'
            )
        ''')
        rows = c.execute("SELECT id, title, summary, tags, actions, content FROM notes").fetchall()
        c.executemany(
            "INSERT INTO notes_fts(rowid, title, summary, tags, actions, content) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
    
    conn.commit()
    conn.close()

init_db()

def find_related_notes(note_id, tags, user_id, conn):
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    if not tag_list:
        return []
    q = " OR ".join(["tags LIKE ?"] * len(tag_list))
    params = [f"%{tag}%" for tag in tag_list]
    sql = f"SELECT id, title FROM notes WHERE id != ? AND user_id = ? AND ({q}) LIMIT 3"
    params = [note_id, user_id] + params
    rows = conn.execute(sql, params).fetchall()
    return [{"id": row[0], "title": row[1]} for row in rows]

# Auth endpoints
@app.post("/register", response_model=User)
def register(username: str = Form(...), password: str = Form(...)):
    conn = get_conn()
    c = conn.cursor()
    hashed = get_password_hash(password)
    try:
        c.execute(
            "INSERT INTO users (username, hashed_password) VALUES (?, ?)",
            (username, hashed),
        )
        conn.commit()
        user_id = c.lastrowid
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="Username already registered")
    conn.close()
    return User(id=user_id, username=username)

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect username or password",
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Main endpoints (keep existing)
@app.get("/")
def dashboard(
    request: Request,
    q: str = "",
    tag: str = "",
    current_user: User = Depends(get_current_user),
):
    conn = get_conn()
    c = conn.cursor()
    if q:
        rows = c.execute(
            """
            SELECT n.*
            FROM notes_fts fts
            JOIN notes n ON n.id = fts.rowid
            WHERE notes_fts MATCH ? AND n.user_id = ?
            ORDER BY n.timestamp DESC LIMIT 100
        """,
            (q, current_user.id),
        ).fetchall()
    elif tag:
        rows = c.execute(
            "SELECT * FROM notes WHERE tags LIKE ? AND user_id = ? ORDER BY timestamp DESC",
            (f"%{tag}%", current_user.id),
        ).fetchall()
    else:
        rows = c.execute(
            "SELECT * FROM notes WHERE user_id = ? ORDER BY timestamp DESC LIMIT 100",
            (current_user.id,),
        ).fetchall()
    notes = [dict(zip([col[0] for col in c.description], row)) for row in rows]
    notes_by_day = defaultdict(list)
    for note in notes:
        day = note["timestamp"][:10] if note.get("timestamp") else "Unknown"
        notes_by_day[day].append(note)
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "notes_by_day": dict(notes_by_day),
            "q": q,
            "tag": tag,
            "last_sync": get_last_sync(),
        },
    )

# Enhanced Search Endpoint
@app.post("/api/search/enhanced")
async def enhanced_search(
    request: SearchRequest,
    current_user: User = Depends(get_current_user)
):
    """Enhanced search with FTS and semantic similarity"""
    conn = get_conn()
    c = conn.cursor()
    
    # Base FTS search
    base_query = """
        SELECT n.*, rank FROM notes_fts fts
        JOIN notes n ON n.id = fts.rowid
        WHERE notes_fts MATCH ? AND n.user_id = ?
    """
    
    params = [request.query, current_user.id]
    
    # Add filters
    if request.filters:
        if 'type' in request.filters:
            base_query += " AND n.type = ?"
            params.append(request.filters['type'])
        
        if 'tags' in request.filters:
            base_query += " AND n.tags LIKE ?"
            params.append(f"%{request.filters['tags']}%")
        
        if 'date_from' in request.filters:
            base_query += " AND date(n.timestamp) >= date(?)"
            params.append(request.filters['date_from'])
    
    base_query += f" ORDER BY rank LIMIT {request.limit}"
    
    rows = c.execute(base_query, params).fetchall()
    notes = [dict(zip([col[0] for col in c.description], row)) for row in rows]
    
    # Log search
    c.execute(
        "INSERT INTO search_analytics (user_id, query, results_count, search_type) VALUES (?, ?, ?, ?)",
        (current_user.id, request.query, len(notes), "fts")
    )
    
    conn.commit()
    conn.close()
    
    return {
        "results": notes,
        "total": len(notes),
        "query": request.query
    }


# New data models
class DiscordWebhook(BaseModel):
    note: str
    tags: str = ""
    type: str = "discord"
    discord_user_id: int
    timestamp: str

class AppleReminderWebhook(BaseModel):
    title: str
    due_date: Optional[str] = None
    notes: str = ""
    tags: str = "reminder,apple"

class CalendarEvent(BaseModel):
    title: str
    start_date: str
    end_date: str
    description: str = ""
    attendees: List[str] = []

class SearchRequest(BaseModel):
    query: str
    filters: Optional[dict] = {}
    limit: int = 20

# Discord Integration
@app.post("/webhook/discord")
async def webhook_discord(
    data: DiscordWebhook,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user_from_discord)
):
    """Enhanced Discord webhook with user mapping"""
    # Map Discord user to Second Brain user
    conn = get_conn()
    c = conn.cursor()
    
    # Check if Discord user is linked
    discord_link = c.execute(
        "SELECT user_id FROM discord_users WHERE discord_id = ?",
        (data.discord_user_id,)
    ).fetchone()
    
    if not discord_link:
        # Auto-register or return error
        raise HTTPException(
            status_code=401, 
            detail="Discord user not linked. Use !link command first."
        )
    
    user_id = discord_link[0]
    
    # Process note with AI
    result = ollama_summarize(data.note)
    summary = result.get("summary", "")
    ai_tags = result.get("tags", [])
    ai_actions = result.get("actions", [])
    
    # Combine tags
    tag_list = [t.strip() for t in data.tags.split(",") if t.strip()]
    tag_list.extend([t for t in ai_tags if t and t not in tag_list])
    tags = ",".join(tag_list)
    
    # Save note
    c.execute(
        "INSERT INTO notes (title, content, summary, tags, actions, type, timestamp, user_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            data.note[:60] + "..." if len(data.note) > 60 else data.note,
            data.note,
            summary,
            tags,
            "\n".join(ai_actions),
            data.type,
            data.timestamp,
            user_id
        ),
    )
    conn.commit()
    note_id = c.lastrowid
    
    # Update FTS
    c.execute(
        "INSERT INTO notes_fts(rowid, title, summary, tags, actions, content) VALUES (?, ?, ?, ?, ?, ?)",
        (note_id, data.note[:60], summary, tags, "\n".join(ai_actions), data.note),
    )
    conn.commit()
    conn.close()
    
    return {"status": "ok", "note_id": note_id}

# Apple Shortcuts Enhanced
@app.post("/webhook/apple/reminder")
async def create_apple_reminder(
    data: AppleReminderWebhook,
    current_user: User = Depends(get_current_user)
):
    """Create reminder from Apple Shortcuts"""
    conn = get_conn()
    c = conn.cursor()
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create note
    c.execute(
        "INSERT INTO notes (title, content, summary, tags, type, timestamp, user_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            data.title,
            data.notes,
            f"Reminder: {data.title}",
            data.tags,
            "reminder",
            now,
            current_user.id
        ),
    )
    note_id = c.lastrowid
    
    # Create reminder entry
    c.execute(
        "INSERT INTO reminders (note_id, user_id, due_date, completed) VALUES (?, ?, ?, ?)",
        (note_id, current_user.id, data.due_date, False)
    )
    
    conn.commit()
    conn.close()
    
    return {"status": "ok", "reminder_id": note_id}

@app.post("/webhook/apple/calendar")
async def create_calendar_event(
    data: CalendarEvent,
    current_user: User = Depends(get_current_user)
):
    """Create calendar event and meeting note"""
    # This would integrate with Apple Calendar API
    # For now, create a meeting note placeholder
    
    conn = get_conn()
    c = conn.cursor()
    
    meeting_note = f"""
Meeting: {data.title}
Date: {data.start_date} - {data.end_date}
Attendees: {', '.join(data.attendees)}

{data.description}

--- Meeting Notes ---
(This will be filled during/after the meeting)
"""
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute(
        "INSERT INTO notes (title, content, summary, tags, type, timestamp, user_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            f"Meeting: {data.title}",
            meeting_note,
            f"Scheduled meeting: {data.title}",
            "meeting,calendar,scheduled",
            "meeting",
            now,
            current_user.id
        ),
    )
    
    conn.commit()
    conn.close()
    
    return {"status": "ok", "event_created": True}

# Enhanced Search
@app.post("/api/search")
async def enhanced_search(
    request: SearchRequest,
    current_user: User = Depends(get_current_user)
):
    """Advanced search with filters and semantic similarity"""
    conn = get_conn()
    c = conn.cursor()
    
    # Base FTS search
    base_query = """
        SELECT n.*, rank FROM notes_fts fts
        JOIN notes n ON n.id = fts.rowid
        WHERE notes_fts MATCH ? AND n.user_id = ?
    """
    
    params = [request.query, current_user.id]
    
    # Add filters
    if request.filters:
        if 'type' in request.filters:
            base_query += " AND n.type = ?"
            params.append(request.filters['type'])
        
        if 'tags' in request.filters:
            base_query += " AND n.tags LIKE ?"
            params.append(f"%{request.filters['tags']}%")
        
        if 'date_from' in request.filters:
            base_query += " AND date(n.timestamp) >= date(?)"
            params.append(request.filters['date_from'])
    
    base_query += f" ORDER BY rank LIMIT {request.limit}"
    
    rows = c.execute(base_query, params).fetchall()
    notes = [dict(zip([col[0] for col in c.description], row)) for row in rows]
    
    conn.close()
    
    return {
        "results": notes,
        "total": len(notes),
        "query": request.query
    }

# Analytics
@app.get("/api/analytics")
async def get_analytics(current_user: User = Depends(get_current_user)):
    """Get user analytics and insights"""
    conn = get_conn()
    c = conn.cursor()
    
    # Basic stats
    total_notes = c.execute(
        "SELECT COUNT(*) as count FROM notes WHERE user_id = ?",
        (current_user.id,)
    ).fetchone()["count"]
    
    # This week
    this_week = c.execute(
        "SELECT COUNT(*) as count FROM notes WHERE user_id = ? AND date(timestamp) >= date('now', '-7 days')",
        (current_user.id,)
    ).fetchone()["count"]
    
    # By type
    by_type = c.execute(
        "SELECT type, COUNT(*) as count FROM notes WHERE user_id = ? GROUP BY type",
        (current_user.id,)
    ).fetchall()
    
    # Popular tags
    tag_counts = {}
    tag_rows = c.execute(
        "SELECT tags FROM notes WHERE user_id = ? AND tags IS NOT NULL",
        (current_user.id,)
    ).fetchall()
    
    for row in tag_rows:
        tags = row["tags"].split(",")
        for tag in tags:
            tag = tag.strip()
            if tag:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    popular_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    conn.close()
    
    return {
        "total_notes": total_notes,
        "this_week": this_week,
        "by_type": [{"type": row["type"], "count": row["count"]} for row in by_type],
        "popular_tags": [{"name": tag, "count": count} for tag, count in popular_tags]
    }

# Real-time status updates
@app.get("/api/notes/{note_id}/status")
async def get_note_processing_status(
    note_id: int,
    current_user: User = Depends(get_current_user)
):
    """Get real-time processing status"""
    conn = get_conn()
    c = conn.cursor()
    
    note = c.execute(
        "SELECT status, title, summary FROM notes WHERE id = ? AND user_id = ?",
        (note_id, current_user.id)
    ).fetchone()
    
    conn.close()
    
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    
    return {
        "status": note["status"],
        "title": note["title"],
        "summary": note["summary"],
        "progress": 100 if note["status"] == "complete" else 50
    }

# Batch operations
@app.post("/api/notes/batch")
async def batch_operations(
    operations: List[dict],
    current_user: User = Depends(get_current_user)
):
    """Perform batch operations on notes"""
    conn = get_conn()
    c = conn.cursor()
    
    results = []
    
    for op in operations:
        try:
            if op["action"] == "delete":
                c.execute(
                    "DELETE FROM notes WHERE id = ? AND user_id = ?",
                    (op["note_id"], current_user.id)
                )
                c.execute("DELETE FROM notes_fts WHERE rowid = ?", (op["note_id"],))
                results.append({"note_id": op["note_id"], "status": "deleted"})
            
            elif op["action"] == "tag":
                c.execute(
                    "UPDATE notes SET tags = ? WHERE id = ? AND user_id = ?",
                    (op["tags"], op["note_id"], current_user.id)
                )
                results.append({"note_id": op["note_id"], "status": "tagged"})
            
            elif op["action"] == "export":
                # Add to export queue
                results.append({"note_id": op["note_id"], "status": "queued_for_export"})
                
        except Exception as e:
            results.append({"note_id": op.get("note_id"), "status": "error", "error": str(e)})
    
    conn.commit()
    conn.close()
    
    return {"results": results}

# Helper function for Discord user authentication
async def get_current_user_from_discord(discord_id: int = None):
    """Map Discord user to Second Brain user"""
    if not discord_id:
        raise HTTPException(status_code=401, detail="Discord user ID required")
    
    conn = get_conn()
    c = conn.cursor()
    
    link = c.execute(
        "SELECT u.* FROM users u JOIN discord_users du ON u.id = du.user_id WHERE du.discord_id = ?",
        (discord_id,)
    ).fetchone()
    
    conn.close()
    
    if not link:
        raise HTTPException(status_code=401, detail="Discord user not linked")
    
    return User(id=link["id"], username=link["username"])


# Discord Integration
@app.post("/webhook/discord")
async def webhook_discord(
    data: DiscordWebhook,
    current_user: User = Depends(get_current_user)
):
    """Discord webhook endpoint"""
    note = data.note
    tags = data.tags
    note_type = data.type
    
    result = ollama_summarize(note)
    summary = result.get("summary", "")
    ai_tags = result.get("tags", [])
    ai_actions = result.get("actions", [])
    
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    tag_list.extend([t for t in ai_tags if t and t not in tag_list])
    tags = ",".join(tag_list)
    actions = "\n".join(ai_actions)
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO notes (title, content, summary, tags, actions, type, timestamp, user_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            note[:60] + "..." if len(note) > 60 else note,
            note,
            summary,
            tags,
            actions,
            note_type,
            now,
            current_user.id,
        ),
    )
    conn.commit()
    note_id = c.lastrowid
    
    c.execute(
        "INSERT INTO notes_fts(rowid, title, summary, tags, actions, content) VALUES (?, ?, ?, ?, ?, ?)",
        (note_id, note[:60] + "..." if len(note) > 60 else note, summary, tags, actions, note),
    )
    conn.commit()
    conn.close()
    
    return {"status": "ok", "note_id": note_id}

# Keep all existing endpoints (detail, edit, delete, capture, etc.)
@app.get("/detail/{note_id}")
def detail(
    request: Request,
    note_id: int,
    current_user: User = Depends(get_current_user),
):
    conn = get_conn()
    c = conn.cursor()
    row = c.execute(
        "SELECT * FROM notes WHERE id = ? AND user_id = ?",
        (note_id, current_user.id),
    ).fetchone()
    if not row:
        return RedirectResponse("/", status_code=302)
    note = dict(zip([col[0] for col in c.description], row))
    related = find_related_notes(note_id, note.get("tags", ""), current_user.id, conn)
    return templates.TemplateResponse(
        "detail.html",
        {"request": request, "note": note, "related": related}
    )

@app.get("/edit/{note_id}")
def edit_get(
    request: Request,
    note_id: int,
    current_user: User = Depends(get_current_user),
):
    conn = get_conn()
    c = conn.cursor()
    row = c.execute(
        "SELECT * FROM notes WHERE id = ? AND user_id = ?",
        (note_id, current_user.id),
    ).fetchone()
    if not row:
        conn.close()
        return RedirectResponse("/", status_code=302)
    note = dict(zip([col[0] for col in c.description], row))
    conn.close()
    return templates.TemplateResponse(
        "edit.html", {"request": request, "note": note}
    )

@app.post("/edit/{note_id}")
def edit_post(
    request: Request,
    note_id: int,
    content: str = Form(""),
    tags: str = Form(""),
    current_user: User = Depends(get_current_user),
):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "UPDATE notes SET content = ?, tags = ? WHERE id = ? AND user_id = ?",
        (content, tags, note_id, current_user.id),
    )
    row = c.execute(
        "SELECT title, summary, actions FROM notes WHERE id = ? AND user_id = ?",
        (note_id, current_user.id),
    ).fetchone()
    if row:
        title, summary, actions = row
        c.execute("DELETE FROM notes_fts WHERE rowid = ?", (note_id,))
        c.execute(
            "INSERT INTO notes_fts(rowid, title, summary, tags, actions, content) VALUES (?, ?, ?, ?, ?, ?)",
            (note_id, title, summary, tags, actions, content),
        )
    conn.commit()
    conn.close()
    if "application/json" in request.headers.get("accept", ""):
        return {"status": "ok"}
    return RedirectResponse(f"/detail/{note_id}", status_code=302)

@app.post("/delete/{note_id}")
def delete_note(
    request: Request,
    note_id: int,
    current_user: User = Depends(get_current_user),
):
    conn = get_conn()
    c = conn.cursor()
    row = c.execute(
        "SELECT audio_filename FROM notes WHERE id = ? AND user_id = ?",
        (note_id, current_user.id),
    ).fetchone()
    if row and row[0]:
        audio_path = settings.audio_dir / row[0]
        converted = audio_path.with_suffix('.converted.wav')
        transcript = pathlib.Path(str(converted) + '.txt')
        for p in [audio_path, converted, transcript]:
            if p.exists():
                p.unlink()
    c.execute(
        "DELETE FROM notes WHERE id = ? AND user_id = ?",
        (note_id, current_user.id),
    )
    c.execute("DELETE FROM notes_fts WHERE rowid = ?", (note_id,))
    conn.commit()
    conn.close()
    if "application/json" in request.headers.get("accept", ""):
        return {"status": "deleted"}
    return RedirectResponse("/", status_code=302)

@app.get("/audio/{filename}")
def get_audio(filename: str, current_user: User = Depends(get_current_user)):
    conn = get_conn()
    c = conn.cursor()
    row = c.execute(
        "SELECT 1 FROM notes WHERE audio_filename = ? AND user_id = ?",
        (filename, current_user.id),
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Audio not found")
    audio_path = settings.audio_dir / filename
    if audio_path.exists():
        return FileResponse(str(audio_path))
    raise HTTPException(status_code=404, detail="Audio not found")

@app.post("/capture")
async def capture(
    request: Request,
    background_tasks: BackgroundTasks,
    note: str = Form(""),
    tags: str = Form(""),
    file: UploadFile = File(None),
    current_user: User = Depends(get_current_user),
):
    content = note.strip()
    note_type = "note"
    audio_filename = None

    if file:
        settings.audio_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
        safe_name = f"{timestamp}-{file.filename.replace(' ', '_')}"
        audio_path = settings.audio_dir / safe_name
        with open(audio_path, "wb") as out_f:
            out_f.write(await file.read())
        audio_filename = safe_name
        note_type = "audio"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO notes (title, content, summary, tags, actions, type, timestamp, audio_filename, status, user_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "[Processing]",
            content if note_type == "note" else "",
            "",
            tags,
            "",
            note_type,
            now,
            audio_filename,
            "pending",
            current_user.id,
        ),
    )
    conn.commit()
    note_id = c.lastrowid
    conn.close()

    background_tasks.add_task(process_note, note_id)

    if "application/json" in request.headers.get("accept", ""):
        return {"id": note_id, "status": "pending"}
    return RedirectResponse("/", status_code=302)

@app.post("/webhook/apple")
async def webhook_apple(
    data: dict = Body(...),
    current_user: User = Depends(get_current_user),
):
    print("APPLE WEBHOOK RECEIVED:", data)
    note = data.get("note", "")
    tags = data.get("tags", "")
    note_type = data.get("type", "apple")
    result = ollama_summarize(note)
    summary = result.get("summary", "")
    ai_tags = result.get("tags", [])
    ai_actions = result.get("actions", [])
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    tag_list.extend([t for t in ai_tags if t and t not in tag_list])
    tags = ",".join(tag_list)
    actions = "\n".join(ai_actions)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO notes (title, content, summary, tags, actions, type, timestamp, user_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            note[:60] + "..." if len(note) > 60 else note,
            note,
            summary,
            tags,
            actions,
            note_type,
            now,
            current_user.id,
        ),
    )
    conn.commit()
    note_id = c.lastrowid
    c.execute(
        "INSERT INTO notes_fts(rowid, title, summary, tags, actions, content) VALUES (?, ?, ?, ?, ?, ?)",
        (note_id, note[:60] + "..." if len(note) > 60 else note, summary, tags, actions, note),
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}

@app.post("/sync/obsidian")
def sync_obsidian(
    background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user)
):
    background_tasks.add_task(export_notes_to_obsidian, current_user.id)
    return {"status": "queued"}

@app.get("/activity")
def activity_timeline(
    request: Request,
    activity_type: str = Query("all", alias="type"),
    start: str = "",
    end: str = "",
    current_user: User = Depends(get_current_user),
):
    conn = get_conn()
    c = conn.cursor()

    base_query = "SELECT id, summary, type, timestamp FROM notes WHERE user_id = ?"
    conditions = []
    params = [current_user.id]
    if activity_type and activity_type != "all":
        conditions.append("type = ?")
        params.append(activity_type)
    if start:
        conditions.append("date(timestamp) >= date(?)")
        params.append(start)
    if end:
        conditions.append("date(timestamp) <= date(?)")
        params.append(end)
    if conditions:
        base_query += " AND " + " AND ".join(conditions)
    base_query += " ORDER BY timestamp DESC LIMIT 100"

    rows = c.execute(base_query, params).fetchall()
    activities = [
        dict(zip([col[0] for col in c.description], row)) for row in rows
    ]
    conn.close()
    return templates.TemplateResponse(
        "activity_timeline.html",
        {
            "request": request,
            "activities": activities,
            "activity_type": activity_type,
            "start": start,
            "end": end,
        },
    )

@app.get("/status/{note_id}")
def note_status(note_id: int, current_user: User = Depends(get_current_user)):
    conn = get_conn()
    c = conn.cursor()
    row = c.execute(
        "SELECT status FROM notes WHERE id = ? AND user_id = ?",
        (note_id, current_user.id),
    ).fetchone()
    conn.close()
    if not row:
        return {"status": "missing"}
    return {"status": row[0]}

# Enhanced Analytics endpoint
@app.get("/api/analytics")
async def get_analytics(current_user: User = Depends(get_current_user)):
    """Get user analytics and insights"""
    conn = get_conn()
    c = conn.cursor()
    
    # Basic stats
    total_notes = c.execute(
        "SELECT COUNT(*) as count FROM notes WHERE user_id = ?",
        (current_user.id,)
    ).fetchone()[0]
    
    # This week
    this_week = c.execute(
        "SELECT COUNT(*) as count FROM notes WHERE user_id = ? AND date(timestamp) >= date('now', '-7 days')",
        (current_user.id,)
    ).fetchone()[0]
    
    # By type
    by_type_rows = c.execute(
        "SELECT type, COUNT(*) as count FROM notes WHERE user_id = ? GROUP BY type",
        (current_user.id,)
    ).fetchall()
    by_type = [{"type": row[0], "count": row[1]} for row in by_type_rows]
    
    # Popular tags
    tag_counts = {}
    tag_rows = c.execute(
        "SELECT tags FROM notes WHERE user_id = ? AND tags IS NOT NULL",
        (current_user.id,)
    ).fetchall()
    
    for row in tag_rows:
        if row[0]:
            tags = row[0].split(",")
            for tag in tags:
                tag = tag.strip()
                if tag:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    popular_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    conn.close()
    
    return {
        "total_notes": total_notes,
        "this_week": this_week,
        "by_type": by_type,
        "popular_tags": [{"name": tag, "count": count} for tag, count in popular_tags]
    }


# Add these endpoints before the health check
@app.post("/api/search/enhanced")
async def enhanced_search(
    q: str = Query(..., description="Search query"),
    type: str = Query("hybrid", description="Search type: fts, semantic, or hybrid"),
    limit: int = Query(20, description="Number of results"),
    current_user: User = Depends(get_current_user)
):
    search_engine = EnhancedSearchEngine(str(settings.db_path))
    results = search_engine.search(q, current_user.id, limit, type)
    search_engine.log_search(current_user.id, q, len(results), type)
    
    return {
        "query": q,
        "results": [
            {
                "id": r.note_id,
                "title": r.title,
                "summary": r.summary,
                "tags": r.tags,
                "timestamp": r.timestamp,
                "score": r.score,
                "snippet": r.snippet,
                "match_type": r.match_type
            } for r in results
        ],
        "total": len(results),
        "search_type": type
    }

@app.post("/webhook/discord")
async def webhook_discord(
    data: dict = Body(...),
    current_user: User = Depends(get_current_user)
):
    # Discord webhook implementation
    note = data.get("note", "")
    tags = data.get("tags", "")
    note_type = data.get("type", "discord")
    
    result = ollama_summarize(note)
    summary = result.get("summary", "")
    ai_tags = result.get("tags", [])
    ai_actions = result.get("actions", [])
    
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    tag_list.extend([t for t in ai_tags if t and t not in tag_list])
    tags = ",".join(tag_list)
    actions = "\n".join(ai_actions)
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO notes (title, content, summary, tags, actions, type, timestamp, user_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            note[:60] + "..." if len(note) > 60 else note,
            note,
            summary,
            tags,
            actions,
            note_type,
            now,
            current_user.id,
        ),
    )
    conn.commit()
    note_id = c.lastrowid
    
    c.execute(
        "INSERT INTO notes_fts(rowid, title, summary, tags, actions, content) VALUES (?, ?, ?, ?, ?, ?)",
        (note_id, note[:60] + "..." if len(note) > 60 else note, summary, tags, actions, note),
    )
    conn.commit()
    conn.close()
    
    return {"status": "ok", "note_id": note_id}

@app.get("/health")
def health():
    conn = get_conn()
    c = conn.cursor()
    tables = c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    conn.close()
    return {"tables": [t[0] for t in tables]}

# Add this to your app.py - Browser Extension Integration

from typing import Dict, Any
import json
import hashlib
import base64
from urllib.parse import urlparse

class BrowserCapture(BaseModel):
    note: str
    tags: str = ""
    type: str = "browser"
    metadata: Dict[str, Any] = {}

@app.post("/webhook/browser")
async def webhook_browser(
    data: BrowserCapture,
    current_user: User = Depends(get_current_user)
):
    """Enhanced browser capture endpoint with metadata processing"""
    
    # Extract metadata
    metadata = data.metadata
    url = metadata.get('url', '')
    title = metadata.get('title', '')
    capture_type = metadata.get('captureType', 'unknown')
    
    # Enhanced content processing
    content = data.note
    if capture_type == 'page':
        content = f"# {title}\n\nSource: {url}\n\n{content}"
    elif capture_type == 'selection':
        content = f"Selection from: {title}\nURL: {url}\n\n> {content}"
    elif capture_type == 'bookmark':
        content = f"# {title}\n\nURL: {url}\n\n{content}"
    
    # Process with AI
    result = ollama_summarize(content)
    summary = result.get("summary", "")
    ai_tags = result.get("tags", [])
    ai_actions = result.get("actions", [])
    
    # Enhanced tag generation
    tag_list = [t.strip() for t in data.tags.split(",") if t.strip()]
    tag_list.extend([t for t in ai_tags if t and t not in tag_list])
    
    # Add smart tags based on content and URL
    smart_tags = generate_smart_tags(content, url, metadata)
    tag_list.extend([t for t in smart_tags if t not in tag_list])
    
    tags = ",".join(tag_list)
    actions = "\n".join(ai_actions)
    
    # Generate title
    note_title = generate_browser_note_title(title, capture_type, content)
    
    # Save to database
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_conn()
    c = conn.cursor()
    
    c.execute(
        """INSERT INTO notes 
           (title, content, summary, tags, actions, type, timestamp, user_id, metadata) 
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            note_title,
            content,
            summary,
            tags,
            actions,
            data.type,
            now,
            current_user.id,
            json.dumps(metadata)
        ),
    )
    
    conn.commit()
    note_id = c.lastrowid
    
    # Update FTS index
    c.execute(
        "INSERT INTO notes_fts(rowid, title, summary, tags, actions, content) VALUES (?, ?, ?, ?, ?, ?)",
        (note_id, note_title, summary, tags, actions, content),
    )
    
    conn.commit()
    conn.close()
    
    # Optional: Process screenshots or HTML archival
    if metadata.get('html') and capture_type == 'page':
        await save_html_archive(note_id, metadata['html'], url)
    
    return {
        "status": "ok", 
        "note_id": note_id,
        "title": note_title,
        "tags": tag_list,
        "summary": summary
    }

def generate_smart_tags(content: str, url: str, metadata: Dict) -> List[str]:
    """Generate intelligent tags based on content and context"""
    tags = []
    
    # Domain-based tags
    if url:
        try:
            domain = urlparse(url).netloc.replace('www.', '')
            tags.append(domain)
            
            # Special site handling
            if 'github.com' in domain:
                tags.extend(['code', 'development'])
            elif 'stackoverflow.com' in domain:
                tags.extend(['programming', 'qa'])
            elif 'medium.com' in domain or 'blog' in domain:
                tags.extend(['blog', 'article'])
            elif 'youtube.com' in domain:
                tags.extend(['video', 'tutorial'])
            elif 'twitter.com' in domain or 'x.com' in domain:
                tags.extend(['social', 'tweet'])
            elif 'reddit.com' in domain:
                tags.extend(['reddit', 'discussion'])
            elif 'arxiv.org' in domain:
                tags.extend(['research', 'paper'])
            elif 'wikipedia.org' in domain:
                tags.extend(['reference', 'wiki'])
                
        except Exception:
            pass
    
    # Content-based smart tagging
    content_lower = content.lower()
    
    # Technical content
    tech_keywords = {
        'python': ['python', 'programming'],
        'javascript': ['javascript', 'programming', 'web'],
        'react': ['react', 'frontend', 'javascript'],
        'api': ['api', 'development'],
        'docker': ['docker', 'devops'],
        'kubernetes': ['kubernetes', 'devops'],
        'machine learning': ['ml', 'ai'],
        'artificial intelligence': ['ai', 'technology'],
        'blockchain': ['blockchain', 'crypto'],
        'cybersecurity': ['security', 'infosec']
    }
    
    for keyword, related_tags in tech_keywords.items():
        if keyword in content_lower:
            tags.extend(related_tags)
    
    # Content type detection
    if any(word in content_lower for word in ['recipe', 'ingredients', 'cooking']):
        tags.extend(['recipe', 'cooking'])
    elif any(word in content_lower for word in ['workout', 'exercise', 'fitness']):
        tags.extend(['fitness', 'health'])
    elif any(word in content_lower for word in ['tutorial', 'how to', 'guide']):
        tags.extend(['tutorial', 'howto'])
    elif any(word in content_lower for word in ['news', 'breaking', 'report']):
        tags.extend(['news', 'current-events'])
    elif any(word in content_lower for word in ['research', 'study', 'analysis']):
        tags.extend(['research', 'academic'])
    
    # Remove duplicates and return
    return list(set(tags))

def generate_browser_note_title(page_title: str, capture_type: str, content: str) -> str:
    """Generate meaningful titles for browser captures"""
    
    if capture_type == 'selection':
        # Use first few words of selection
        words = content.split()[:8]
        title = ' '.join(words)
        if len(content.split()) > 8:
            title += '...'
        return f"Selection: {title}"
    
    elif capture_type == 'bookmark':
        return f"Bookmark: {page_title}"
    
    elif capture_type == 'page':
        return page_title or "Web Page"
    
    elif capture_type == 'manual':
        # Extract first sentence or line
        first_line = content.split('\n')[0][:60]
        return first_line if first_line else "Manual Note"
    
    else:
        return page_title or "Web Capture"

async def save_html_archive(note_id: int, html_content: str, url: str):
    """Save HTML content for archival (optional feature)"""
    try:
        archive_dir = settings.base_dir / "archives"
        archive_dir.mkdir(exist_ok=True)
        
        # Create filename from note ID and URL hash
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        filename = f"note_{note_id}_{url_hash}.html"
        
        # Save HTML with basic metadata
        html_with_meta = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Archived from {url}</title>
    <meta name="original-url" content="{url}">
    <meta name="archived-date" content="{datetime.now().isoformat()}">
    <meta name="second-brain-note-id" content="{note_id}">
    <style>
        .second-brain-archive-header {{
            background: #f3f4f6;
            padding: 1rem;
            border-bottom: 1px solid #d1d5db;
            font-family: system-ui, -apple-system, sans-serif;
        }}
    </style>
</head>
<body>
    <div class="second-brain-archive-header">
        <p><strong>Archived from:</strong> <a href="{url}">{url}</a></p>
        <p><strong>Saved on:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Second Brain Note ID:</strong> {note_id}</p>
    </div>
    {html_content}
</body>
</html>
"""
        
        archive_path = archive_dir / filename
        archive_path.write_text(html_with_meta, encoding='utf-8')
        
        # Update note metadata to include archive path
        conn = get_conn()
        c = conn.cursor()
        c.execute(
            "UPDATE notes SET archive_path = ? WHERE id = ?",
            (str(archive_path), note_id)
        )
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"Failed to save HTML archive: {e}")

# Add recent captures endpoint for extension
@app.get("/api/captures/recent")
async def get_recent_captures(
    limit: int = Query(10, le=50),
    type: str = Query(None),
    current_user: User = Depends(get_current_user)
):
    """Get recent captures for the browser extension"""
    conn = get_conn()
    c = conn.cursor()
    
    query = "SELECT * FROM notes WHERE user_id = ?"
    params = [current_user.id]
    
    if type:
        query += " AND type = ?"
        params.append(type)
    
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    
    rows = c.execute(query, params).fetchall()
    notes = [dict(zip([col[0] for col in c.description], row)) for row in rows]
    
    # Parse metadata for each note
    for note in notes:
        if note.get('metadata'):
            try:
                note['metadata'] = json.loads(note['metadata'])
            except:
                note['metadata'] = {}
    
    conn.close()
    
    return notes

# Database migration to add metadata and archive_path columns
def add_browser_capture_columns():
    """Add columns for browser capture functionality"""
    conn = get_conn()
    c = conn.cursor()
    
    # Check if columns exist
    columns = [row[1] for row in c.execute("PRAGMA table_info(notes)")]
    
    if 'metadata' not in columns:
        c.execute("ALTER TABLE notes ADD COLUMN metadata TEXT")
    
    if 'archive_path' not in columns:
        c.execute("ALTER TABLE notes ADD COLUMN archive_path TEXT")
    
    conn.commit()
    conn.close()

# Call this in your init_db() function
# add_browser_capture_columns()