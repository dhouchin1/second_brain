# Second Brain - Quick Start Guide

## Prerequisites

- Python 3.9 or higher
- SQLite 3
- (Optional) Ollama for AI features
- (Optional) Whisper.cpp for audio transcription

---

## Setup Instructions

### 1. Navigate to Project Directory

```bash
cd /Users/dhouchin/mvp-setup/second_brain
```

### 2. Check if Virtual Environment Exists

```bash
# Check for .venv directory
ls -la .venv
```

If `.venv` exists, **skip to step 4**. If not, continue to step 3.

### 3. Create Virtual Environment (if needed)

```bash
# Create new virtual environment
python3 -m venv .venv
```

### 4. Activate Virtual Environment

```bash
# Activate venv
source .venv/bin/activate

# You should see (.venv) in your terminal prompt
```

### 5. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

### 6. Verify Environment Variables

Check that `.env` file exists and has required settings:

```bash
# View current settings
cat .env | grep -E "DB_PATH|VAULT_PATH|AUDIO_DIR|OLLAMA"
```

**Required settings:**
- `DB_PATH=./notes.db` - Database file path
- `VAULT_PATH=./vault` - Obsidian vault location
- `AUDIO_DIR=./audio` - Audio files directory
- `SECRET_KEY=<your-secret>` - Already set
- `WEBHOOK_TOKEN=<your-token>` - Already set

### 7. Create Required Directories

```bash
# Create directories if they don't exist
mkdir -p vault audio uploads
```

### 8. Initialize Database

The database will be created automatically on first run. To manually initialize:

```bash
# Check if database exists
ls -la notes.db

# If it doesn't exist, it will be created when you start the app
```

### 9. (Optional) Setup Ollama for AI Features

If you want AI-powered summaries and tagging:

```bash
# Install Ollama (if not installed)
# Visit: https://ollama.com/download

# Start Ollama service
ollama serve

# In another terminal, pull the model
ollama pull llama3.2
```

### 10. Start the Application

```bash
# Make sure venv is activated (you should see (.venv) in prompt)
# If not: source .venv/bin/activate

# Start FastAPI server
python -m uvicorn app:app --reload --port 8082
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8082 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 11. Access the Application

Open your browser and visit:

**HTMX Dashboard (New):**
```
http://localhost:8082/dashboard/htmx
```

**Dashboard v3 (Existing):**
```
http://localhost:8082/dashboard/v3
```

**API Documentation:**
```
http://localhost:8082/docs
```

---

## Complete Startup Commands (Copy-Paste)

If starting fresh or after reboot:

```bash
# Navigate to project
cd /Users/dhouchin/mvp-setup/second_brain

# Activate virtual environment
source .venv/bin/activate

# (Optional) Start Ollama in background
# ollama serve &

# Start the application
python -m uvicorn app:app --reload --port 8082
```

Then visit: **http://localhost:8082/dashboard/htmx**

---

## First-Time User Registration

If you don't have an account yet:

1. Visit: http://localhost:8082/register
2. Create an account with username/password
3. Login at: http://localhost:8082/login
4. You'll be redirected to the dashboard

---

## Stopping the Application

```bash
# In the terminal running uvicorn, press:
CTRL+C

# Deactivate virtual environment (optional)
deactivate
```

---

## Common Issues & Solutions

### Issue: "No module named 'fastapi'"

**Solution:**
```bash
# Make sure venv is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "Address already in use" (port 8082)

**Solution:**
```bash
# Find and kill process using port 8082
lsof -ti:8082 | xargs kill -9

# Or use a different port
python -m uvicorn app:app --reload --port 8083
```

### Issue: Database errors

**Solution:**
```bash
# Remove existing database and let it recreate
rm notes.db

# Restart the application
python -m uvicorn app:app --reload --port 8082
```

### Issue: Ollama connection errors

**Solution:**
```bash
# Start Ollama service
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

### Issue: Templates not found

**Solution:**
```bash
# Verify template structure
ls -R templates/

# Should show:
# templates/
# templates/base_htmx.html
# templates/pages/
# templates/components/
```

---

## Development Workflow

### Daily Startup

```bash
# 1. Navigate to project
cd /Users/dhouchin/mvp-setup/second_brain

# 2. Activate venv
source .venv/bin/activate

# 3. Start server with auto-reload
python -m uvicorn app:app --reload --port 8082

# 4. Open browser to http://localhost:8082/dashboard/htmx
```

### Making Changes

The server will automatically reload when you modify:
- Python files (`.py`)
- Template files (`.html`)
- Static files (`.js`, `.css`)

Just refresh your browser to see changes!

### Running Database Migrations

```bash
# Activate venv
source .venv/bin/activate

# Run migrations
python migrate_db.py
```

### Checking Logs

```bash
# Server logs appear in terminal where uvicorn is running

# For more verbose logging, use:
python -m uvicorn app:app --reload --port 8082 --log-level debug
```

---

## Testing the HTMX Dashboard

Once the app is running, try these features:

1. **Quick Capture**
   - Type a note in the capture form
   - Click "Capture"
   - Watch it appear at the top of the list

2. **Search**
   - Type in the search box
   - Results appear as you type (300ms delay)
   - Try toggling filters

3. **Infinite Scroll**
   - Scroll to the bottom of the notes list
   - More notes load automatically

4. **Note Actions**
   - Click the 3-dot menu on any note
   - Try View, Edit, Duplicate, Delete

5. **Auto-Refresh Stats**
   - Stats at the top refresh every 30 seconds
   - Create a note and watch the count update

---

## Environment Variables Reference

Key variables in `.env`:

```bash
# Database
DB_PATH=./notes.db                      # SQLite database location

# Directories
VAULT_PATH=./vault                      # Obsidian vault
AUDIO_DIR=./audio                       # Audio files
UPLOADS_DIR=./uploads                   # File uploads

# AI Services
OLLAMA_API_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=llama3.2                   # LLM model to use

# Audio Transcription (optional)
WHISPER_CPP_PATH=./whisper.cpp/build/bin/whisper-cli
WHISPER_MODEL_PATH=./whisper.cpp/models/ggml-base.en.bin

# Security
SECRET_KEY=<auto-generated>             # Don't change
WEBHOOK_TOKEN=<auto-generated>          # Don't change

# Discord (optional)
DISCORD_BOT_TOKEN=<your-token>
```

---

## Performance Tips

1. **Development Mode**
   - Use `--reload` for auto-restart on file changes
   - Keep browser DevTools open to see HTMX logs

2. **Production Mode**
   - Remove `--reload` flag
   - Set `DEBUG=False` in `.env`
   - Use `--workers 4` for multiple workers

3. **Database Optimization**
   - Run VACUUM periodically: `sqlite3 notes.db "VACUUM;"`
   - Rebuild search index if needed

---

## Next Steps

- Read `HTMX_IMPLEMENTATION_GUIDE.md` for development patterns
- Check `CLAUDE.md` for project architecture
- Visit `/docs` for interactive API documentation
- Try creating notes via the HTMX dashboard

---

## Getting Help

- **API Docs**: http://localhost:8082/docs
- **HTMX Guide**: `HTMX_IMPLEMENTATION_GUIDE.md`
- **Project Docs**: `CLAUDE.md`
- **GitHub Issues**: [Report bugs](https://github.com/dhouchin1/second_brain/issues)

---

**You're all set! 🚀**

Open http://localhost:8082/dashboard/htmx and start capturing your thoughts!
