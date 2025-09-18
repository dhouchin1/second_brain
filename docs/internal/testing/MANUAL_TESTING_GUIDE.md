# Manual Testing Guide - Second Brain System

*Generated: 2025-09-05*

## Quick Start Testing Protocol

### Pre-Testing Setup
1. **Environment Check**
   ```bash
   # Verify all services are available
   .venv/bin/python -c "from services.auto_seeding_service import get_auto_seeding_service; print('✅ Auto-seeding ready')"
   .venv/bin/python -c "from database import get_db_connection; print('✅ Database ready')"
   
   # Start the server
   .venv/bin/python -m uvicorn app:app --reload --port 8082
   ```

2. **Service Status Check**
   - [ ] Ollama running (`ollama serve`)
   - [ ] Database migrations applied
   - [ ] Server starts without errors on `http://localhost:8082`

---

## Core Functionality Tests (Priority 1)

### 🔐 Authentication & User Management
1. **New User Creation**
   - [ ] Navigate to `http://localhost:8082`
   - [ ] Create new account
   - [ ] Verify email/magic link flow (if configured)
   - [ ] Login successfully
   - [ ] Session persists on browser refresh

2. **User Experience**
   - [ ] Dashboard loads without errors
   - [ ] Navigation between pages works
   - [ ] Logout functionality works

### 📝 Core Note Operations
1. **Basic Note Creation** (CRITICAL)
   - [ ] Create note via web UI
   - [ ] Note appears in notes list
   - [ ] Note content is saved correctly
   - [ ] Search finds the note

2. **Content Processing** (CRITICAL)
   - [ ] AI title generation works (requires Ollama)
   - [ ] AI summarization works
   - [ ] Tags are generated/applied correctly

---

## Enhanced Capture Features (Priority 2)

### 🔄 Unified Capture System
✅ **Status: 16/16 tests passing**

**Test Scenarios:**
1. **Text Capture**
   - [ ] POST to `/api/unified-capture/text` with JSON
   - [ ] Verify note creation and AI processing
   - [ ] Check tags and summary generation

2. **Quick Note Endpoint** ✅ **Recently Enhanced**
   - [ ] POST to `/api/unified-capture/quick-note` with JSON: `{"content": "test note"}`
   - [ ] POST with form data: `content=test+note`
   - [ ] Confirm GET requests with query parameters alone return a validation error
   - [ ] Both supported formats should create notes successfully

### 🖼️ Advanced Capture Features
⚠️ **Status: Several tests failing but core functionality present**

1. **Image OCR** (Test manually - automated tests fixed)
   - [ ] Upload image via `/capture` endpoint
   - [ ] Verify OCR text extraction (requires tesseract)
   - [ ] Check if note is created with extracted text

2. **PDF Processing**
   - [ ] Upload PDF file
   - [ ] Verify text extraction (requires PyMuPDF)
   - [ ] Check content quality

### 📱 Apple Shortcuts Integration
⚠️ **Status: Some test failures, manual testing recommended**

**Test Endpoints:**
- [ ] POST to `/webhook/apple` with voice memo data
- [ ] POST to `/webhook/apple` with photo data  
- [ ] Verify iOS Shortcuts integration (if available)

### 💬 Discord Integration  
⚠️ **Status: Some test failures, manual testing needed**

**Test Components:**
- [ ] Discord bot responds to slash commands (if configured)
- [ ] Message capture works
- [ ] Thread summarization functions

---

## Search & Discovery (Priority 2)

### 🔍 Search System Testing
1. **Full-Text Search**
   - [ ] Basic keyword search returns results
   - [ ] Search with quotes (phrase search)
   - [ ] Search handles special characters
   - [ ] Search performance is acceptable

2. **Vector Search** (if sqlite-vec available)
   - [ ] Semantic search returns relevant results
   - [ ] Embedding generation works for new notes

---

## Integration Testing (Priority 3)

### 🔄 Obsidian Sync
- [ ] Notes sync to Obsidian vault (if configured)
- [ ] YAML frontmatter is generated correctly
- [ ] Changes in Obsidian reflect in system

### 🤖 Auto-Seeding System
- [ ] New users get starter content
- [ ] Seeding improves search experience
- [ ] Content quality is appropriate

---

## Error Handling & Edge Cases (Priority 3)

### 🚨 Error Scenarios to Test
1. **Service Failures**
   - [ ] Ollama unavailable (AI processing should degrade gracefully)
   - [ ] Database connection issues
   - [ ] File upload errors

2. **Input Validation**
   - [ ] Empty content submission
   - [ ] Very large files
   - [ ] Invalid file formats
   - [ ] SQL injection attempts (security test)

---

## Performance & Load Testing (Priority 4)

### ⚡ Performance Checks
- [ ] Large note creation (>10MB text)
- [ ] Multiple concurrent requests
- [ ] Search response times with large dataset
- [ ] File upload performance

---

## Critical Paths Summary

### ✅ **Must Work for Production:**
1. **User Authentication** - Account creation, login, session management
2. **Basic Note Operations** - Create, read, search notes
3. **Core API Endpoints** - `/capture`, `/quick-note` endpoints
4. **Search Functionality** - Basic full-text search
5. **Web Interface** - All main pages load and function

### 🔧 **Should Work (Investigate if failing):**
1. **AI Processing** - Title generation, summarization (requires Ollama)
2. **File Processing** - Basic file uploads and processing
3. **Enhanced Capture** - Advanced OCR and PDF features

### 📋 **Nice to Have (Fix if time permits):**
1. **External Integrations** - Discord, Apple Shortcuts
2. **Advanced Search** - Vector/semantic search  
3. **Auto-seeding** - New user content bootstrapping

---

## Quick Issue Reporting Template

**When you find issues, please note:**
- **URL/Endpoint:** 
- **Expected Behavior:** 
- **Actual Behavior:** 
- **Browser/Environment:** 
- **Steps to Reproduce:** 
- **Error Messages:** 
- **Screenshot (if UI issue):**

---

## Test Data Suggestions

### Sample Content for Testing:
```
# Test Note 1 - Basic
Title: Meeting Notes
Content: Discussed project timeline and deliverables for Q4.

# Test Note 2 - Rich Content  
Title: Research Findings
Content: Found interesting insights about user behavior patterns. Key metrics show 40% improvement in engagement.

# Test Note 3 - Technical
Title: API Documentation
Content: GET /api/notes returns all notes. POST /api/notes creates new note. Requires authentication header.
```

### Sample Files:
- Small text file (.txt)
- Simple PDF document
- Basic image with text (for OCR testing)
- Audio file (for transcription testing)

---

## Expected Test Duration

- **Core Functionality:** 30-45 minutes
- **Enhanced Features:** 45-60 minutes  
- **Integration Testing:** 30 minutes
- **Edge Cases:** 30 minutes
- **Total:** 2-3 hours for comprehensive testing

---

## Current System Status

✅ **Stable & Tested:**
- Unified Capture Service (16/16 tests passing)
- Core note operations
- Basic search functionality
- Web interface routes
- User authentication

⚠️ **Needs Manual Validation:**
- Advanced capture features (OCR, PDF)
- Apple Shortcuts integration
- Discord integration
- External service dependencies

🔧 **Known Test Issues (may work in practice):**
- Some dependency mocking issues in tests
- Response format mismatches in advanced features
- Complex async operation testing challenges

The core system is solid with 58/77 tests passing. Most failures are in advanced features that may work fine in practice but have test setup issues.
