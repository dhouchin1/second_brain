# Manual Testing Checklist - Second Brain

## Overview
This comprehensive manual testing checklist covers all current features in the Second Brain application. Use this for thorough review and audit of functionality.

## Test Environment Setup
- [ ] Verify `.env` file is configured with all required settings
- [ ] Confirm Ollama server is running (`ollama serve`)
- [ ] Verify whisper.cpp is installed and configured
- [ ] Check that the vault directory exists and is accessible
- [ ] Confirm database migrations have been applied
- [ ] Verify all optional dependencies are installed as needed

---

## Core Application (app.py)

### Authentication & User Management
- [ ] Login with valid credentials
- [ ] Login with invalid credentials (should fail gracefully)
- [ ] Magic link authentication flow
- [ ] Session persistence across browser restarts
- [ ] Logout functionality
- [ ] User profile page access

### Main UI Routes
- [ ] Home page loads correctly (`/`)
- [ ] Notes listing page (`/notes`)
- [ ] Individual note view page (`/note/{id}`)
- [ ] Search page functionality (`/search`)
- [ ] Settings page access (`/settings`)

### Legacy API Endpoints
- [ ] `/capture` endpoint with text content
- [ ] `/capture` endpoint with file upload
- [ ] `/webhook/apple` endpoint functionality
- [ ] Basic search API endpoints

---

## Unified Capture Service

### Text Content Capture
- [ ] Simple text note creation via API
- [ ] Text with custom title
- [ ] Text with custom tags
- [ ] Text without title (AI generation)
- [ ] Markdown content preservation
- [ ] Special character handling
- [ ] Unicode/emoji content

### Multi-modal Content Capture
- [ ] Image upload with OCR extraction
- [ ] PDF file upload and text extraction
- [ ] Audio file upload and transcription
- [ ] Voice memo processing with location data
- [ ] URL content extraction and processing
- [ ] Batch processing of multiple items

### Routing and Service Selection
- [ ] Automatic service selection based on content type
- [ ] Fallback handling when preferred service unavailable
- [ ] Error propagation from downstream services
- [ ] Processing statistics tracking
- [ ] Content type validation

### Integration Context Handling
- [ ] Apple Shortcuts integration context
- [ ] Discord context processing
- [ ] Location data preservation
- [ ] User context handling
- [ ] Custom metadata processing

---

## Advanced Capture Service

### OCR and Image Processing
- [ ] Screenshot OCR with Tesseract
- [ ] Image quality settings (low/medium/high)
- [ ] Multiple image format support (PNG, JPG, etc.)
- [ ] OCR language selection
- [ ] Invalid image handling
- [ ] OCR failure graceful degradation

### PDF Processing
- [ ] PDF text extraction with PyMuPDF
- [ ] Multi-page PDF handling
- [ ] Password-protected PDF handling (should fail gracefully)
- [ ] Corrupted PDF file handling
- [ ] Large PDF file processing
- [ ] PDF metadata extraction

### YouTube Integration
- [ ] YouTube URL transcript extraction
- [ ] Multiple language transcript support
- [ ] Video with no available transcript
- [ ] Invalid YouTube URL handling
- [ ] Private video handling
- [ ] Age-restricted content handling

### Bulk Processing
- [ ] Bulk URL processing within limits
- [ ] Bulk processing size limit enforcement
- [ ] Concurrent processing handling
- [ ] Partial failure in bulk operations
- [ ] Progress tracking for bulk operations

---

## Apple Shortcuts Service

### Voice Memo Processing
- [ ] Voice memo with pre-transcribed text
- [ ] Voice memo with raw audio data
- [ ] Voice memo with location data
- [ ] Voice memo with context metadata
- [ ] Missing content handling
- [ ] Audio transcription queue integration

### Photo Processing
- [ ] Photo with OCR extraction
- [ ] Photo with manual description
- [ ] Photo with location data
- [ ] Multiple photo processing
- [ ] Invalid photo data handling

### Web Clip Processing
- [ ] Web page URL processing
- [ ] Web page with custom title
- [ ] Web page with tags
- [ ] Invalid URL handling
- [ ] Timeout handling for slow pages

### Template System
- [ ] Shortcut template generation
- [ ] Custom template creation
- [ ] Template parameter validation
- [ ] Template export functionality

### Integration Features
- [ ] iOS Shortcuts app integration
- [ ] macOS Shortcuts integration
- [ ] Location data capture
- [ ] Contact integration
- [ ] Calendar integration

---

## Discord Service

### Text Note Capture
- [ ] Simple message capture
- [ ] Message with original author attribution
- [ ] Message with Discord metadata
- [ ] Special character and emoji handling
- [ ] Long message content
- [ ] Empty message handling

### Thread Processing
- [ ] Thread summary generation
- [ ] Large thread handling (50+ messages)
- [ ] Empty thread handling
- [ ] Thread with media content
- [ ] Thread with external links
- [ ] Thread pagination

### Discord Context
- [ ] Guild/server information capture
- [ ] Channel information preservation
- [ ] User mention handling
- [ ] Role mention processing
- [ ] Custom emoji processing
- [ ] Message formatting preservation

### Bot Integration
- [ ] Slash command registration
- [ ] Slash command execution
- [ ] Bot permissions validation
- [ ] Multi-guild support
- [ ] Rate limiting compliance
- [ ] Error message handling in Discord

### Statistics and Analytics
- [ ] Usage statistics per guild
- [ ] User activity tracking
- [ ] Command usage analytics
- [ ] Error rate monitoring

---

## Search System

### Full-Text Search (FTS5)
- [ ] Basic keyword search
- [ ] Phrase search with quotes
- [ ] Boolean operators (AND, OR, NOT)
- [ ] Wildcard search patterns
- [ ] Search result ranking (BM25)
- [ ] Search result snippets
- [ ] Special character handling in queries

### Vector Search (sqlite-vec)
- [ ] Semantic similarity search
- [ ] Vector embedding generation
- [ ] Similarity threshold testing
- [ ] Vector index performance
- [ ] Embedding model consistency

### Hybrid Search
- [ ] Combined FTS5 + vector results
- [ ] Reciprocal Rank Fusion (RRF)
- [ ] Result deduplication
- [ ] Ranking adjustment
- [ ] Search performance monitoring

### Search Features
- [ ] Auto-complete suggestions
- [ ] Search history
- [ ] Saved searches
- [ ] Search filters (date, type, tags)
- [ ] Export search results

---

## Database and Storage

### Core Database Operations
- [ ] Note creation and storage
- [ ] Note retrieval by ID
- [ ] Note updates and versioning
- [ ] Note deletion
- [ ] Bulk operations

### Migration System
- [ ] Database migration execution
- [ ] Migration rollback capability
- [ ] Schema version tracking
- [ ] Migration failure recovery

### Search Indexing
- [ ] FTS5 index updates
- [ ] Vector index updates
- [ ] Index rebuild functionality
- [ ] Index consistency validation
- [ ] Orphaned index cleanup

---

## Obsidian Integration

### Vault Synchronization
- [ ] Bi-directional sync with Obsidian vault
- [ ] YAML frontmatter processing
- [ ] File creation in vault
- [ ] File updates from vault
- [ ] File deletion handling

### Metadata Processing
- [ ] YAML frontmatter generation
- [ ] Metadata field mapping
- [ ] Tag synchronization
- [ ] Custom field handling
- [ ] Metadata validation

### File Management
- [ ] Markdown file creation
- [ ] File naming conventions
- [ ] Directory structure handling
- [ ] Attachment file management
- [ ] Conflict resolution

---

## AI Processing (Ollama Integration)

### Title Generation
- [ ] Title generation from content
- [ ] Title quality and relevance
- [ ] Fallback when AI unavailable
- [ ] Custom title preservation
- [ ] Title length limits

### Content Summarization
- [ ] Summary generation from long content
- [ ] Summary quality assessment
- [ ] Summary length control
- [ ] Multi-language content
- [ ] Technical content handling

### Tag Suggestion
- [ ] Automatic tag generation
- [ ] Tag relevance and quality
- [ ] Custom tag preservation
- [ ] Tag deduplication
- [ ] Tag formatting consistency

### AI Service Reliability
- [ ] Ollama service connectivity
- [ ] Model availability checking
- [ ] Timeout handling
- [ ] Graceful degradation when AI fails
- [ ] Error message clarity

---

## Auto-Seeding System

### Content Seeding
- [ ] New user content seeding
- [ ] Seeding status checking
- [ ] Available seed content listing
- [ ] Seeding progress monitoring
- [ ] Seeding completion validation

### Seed Content Quality
- [ ] Starter notes relevance
- [ ] Bookmark collection quality
- [ ] Example content usefulness
- [ ] Search performance improvement
- [ ] Knowledge base bootstrapping

---

## Background Processing

### Audio Processing Queue
- [ ] Audio transcription queue
- [ ] Concurrent processing limits
- [ ] Queue status monitoring
- [ ] Failed job retry logic
- [ ] Processing timeout handling

### Job Management
- [ ] Job creation and scheduling
- [ ] Job status tracking
- [ ] Job error handling
- [ ] Job cleanup and maintenance
- [ ] Job logging and debugging

---

## API and Integration Testing

### REST API Endpoints
- [ ] Authentication header validation
- [ ] Request payload validation
- [ ] Response format consistency
- [ ] Error response format
- [ ] Rate limiting compliance

### Webhook Endpoints
- [ ] Apple Shortcuts webhook
- [ ] Discord webhook
- [ ] Generic webhook processing
- [ ] Webhook security validation
- [ ] Webhook retry handling

### External Service Integration
- [ ] Whisper.cpp integration
- [ ] Vosk fallback integration
- [ ] Tesseract OCR integration
- [ ] PyMuPDF integration
- [ ] YouTube API integration

---

## Performance and Reliability

### System Performance
- [ ] Large file handling
- [ ] Concurrent request handling
- [ ] Memory usage monitoring
- [ ] Database query performance
- [ ] Search response times

### Error Handling
- [ ] Service unavailable scenarios
- [ ] Network timeout handling
- [ ] Disk space limitations
- [ ] Memory limitations
- [ ] Graceful degradation

### Monitoring and Logging
- [ ] Application logging levels
- [ ] Error log capture
- [ ] Performance metrics
- [ ] Health check endpoints
- [ ] Service status monitoring

---

## Security Testing

### Input Validation
- [ ] SQL injection prevention
- [ ] XSS attack prevention
- [ ] File upload validation
- [ ] Path traversal prevention
- [ ] Input sanitization

### Authentication Security
- [ ] Password security
- [ ] Session management
- [ ] Magic link security
- [ ] Token expiration
- [ ] Rate limiting

### Data Protection
- [ ] Sensitive data handling
- [ ] File permission security
- [ ] Database access control
- [ ] API key protection
- [ ] Log data sanitization

---

## Browser and Device Testing

### Web Browser Compatibility
- [ ] Chrome/Chromium latest
- [ ] Firefox latest
- [ ] Safari latest
- [ ] Edge latest
- [ ] Mobile browsers

### Device Testing
- [ ] Desktop web interface
- [ ] Tablet interface
- [ ] Mobile web interface
- [ ] iOS Shortcuts integration
- [ ] macOS Shortcuts integration

---

## Edge Cases and Stress Testing

### Content Edge Cases
- [ ] Very long text content (>10MB)
- [ ] Very large image files
- [ ] Empty or minimal content
- [ ] Malformed content
- [ ] Special character combinations

### System Stress Testing
- [ ] High concurrent user load
- [ ] Large batch processing
- [ ] Extended uptime testing
- [ ] Resource exhaustion scenarios
- [ ] Recovery after system restart

---

## Checklist Completion

### Pre-deployment Validation
- [ ] All critical functionality working
- [ ] No blocking issues identified
- [ ] Performance within acceptable limits
- [ ] Security validations passed
- [ ] Documentation updated

### Post-deployment Monitoring
- [ ] Production health checks
- [ ] Error rate monitoring
- [ ] Performance monitoring
- [ ] User feedback collection
- [ ] Issue tracking setup

---

**Testing Notes:**
- Document any failures with specific steps to reproduce
- Include screenshots for UI-related issues
- Note performance benchmarks for comparison
- Track which features require additional development
- Identify areas needing automated test coverage improvement