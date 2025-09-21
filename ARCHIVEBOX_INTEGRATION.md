# ArchiveBox Integration - Phase 1 Implementation

This document describes the Phase 1 implementation of ArchiveBox integration for Second Brain.

## Overview

Phase 1 provides the foundational components for web archival using ArchiveBox:

- **Environment and Configuration**: Complete configuration system with feature flags
- **Core Service Implementation**: CLI/API wrapper with Docker and local support
- **Integration Queue Extension**: Job queue support for asynchronous archival
- **API Endpoints**: RESTful API for archival operations

## Components Implemented

### 1. Configuration System (`config.py`)

Added comprehensive ArchiveBox settings:

```python
# Core settings
ARCHIVEBOX_ENABLED=false
ARCHIVEBOX_FEATURE_FLAG=false
ARCHIVEBOX_DATA_DIR=./archivebox_data

# Archival options
ARCHIVEBOX_EXTRACT=title,favicon,wget,pdf,screenshot,dom,singlefile
ARCHIVEBOX_TIMEOUT=60
ARCHIVEBOX_RESOLUTION=1440,2000

# Storage strategy
ARCHIVEBOX_STORAGE_STRATEGY=symlink  # or 'copy'
ARCHIVEBOX_AUTO_CLEANUP_DAYS=365
ARCHIVEBOX_MAX_SIZE_MB=500
```

### 2. Docker Compose Setup (`docker-compose.archivebox.yml`)

Standalone ArchiveBox service with:
- Production-ready container configuration
- Resource limits and health checks
- Redis queue integration
- Shared volume support
- Security settings

### 3. Core Service (`services/archivebox_service.py`)

**ArchiveBoxService** provides:
- **Dual execution modes**: Docker containers and local CLI
- **Async operations**: Full async/await support
- **Error handling**: Comprehensive error recovery
- **Storage strategies**: Symlink (efficient) vs copy (portable)
- **Health checking**: Service availability detection
- **Auto-cleanup**: Configurable archive retention

**Key APIs:**
```python
service = ArchiveBoxService()
await service.is_available()  # Check if ArchiveBox is ready
result = await service.archive_url(request)  # Archive a URL
status = await service.get_archive_status(url)  # Check archive status
content = await service.get_archive_content(snapshot_id)  # Retrieve content
await service.cleanup_old_archives()  # Clean old archives
```

### 4. Integration Queue (`services/ingestion_queue.py`)

Extended with:
- **New job type**: `ARCHIVE_WEB_CAPTURE`
- **Payload schema**: `ArchiveWebCapturePayload` with metadata
- **Helper function**: `create_archive_job()` for easy job creation

### 5. API Router (`services/archivebox_router.py`)

RESTful endpoints:
- `POST /api/archivebox/archive` - Archive a URL
- `GET /api/archivebox/archive/{url}` - Get archive status
- `GET /api/archivebox/content/{snapshot_id}` - Retrieve content
- `GET /api/archivebox/content/{snapshot_id}/file/{type}` - Download files
- `POST /api/archivebox/bulk-archive` - Bulk archive URLs
- `GET /api/archivebox/health` - Health check

## Setup Instructions

### 1. Environment Configuration

```bash
# Copy example configuration
cp .env.archivebox.example .env

# Enable ArchiveBox (add to .env)
ARCHIVEBOX_ENABLED=true
ARCHIVEBOX_FEATURE_FLAG=true
```

### 2. Docker Deployment (Recommended)

```bash
# Start ArchiveBox service
docker-compose -f docker-compose.archivebox.yml up -d

# Initialize ArchiveBox data directory
docker exec secondbrain-archivebox archivebox init
```

### 3. Local Installation (Alternative)

```bash
# Install ArchiveBox
pip install archivebox

# Initialize data directory
export ARCHIVEBOX_DATA_DIR=./archivebox_data
archivebox init
```

### 4. Service Integration

The service automatically integrates with Second Brain when enabled:

```python
# In app.py, add to router initialization:
from services.archivebox_router import router as archivebox_router, init_archivebox_router

# Initialize the router
init_archivebox_router(get_conn, get_current_user)

# Include the router
app.include_router(archivebox_router)
```

## Usage Examples

### Archive a URL

```python
from services.archivebox_service import ArchiveRequest, get_archivebox_service

service = get_archivebox_service()
request = ArchiveRequest(
    url="https://example.com",
    extract_types=["title", "pdf", "screenshot"],
    timeout=60
)
result = await service.archive_url(request)
print(f"Archived: {result.success}, Path: {result.archive_path}")
```

### Create Background Job

```python
from services.ingestion_queue import create_archive_job

job = create_archive_job(
    url="https://example.com",
    user_id=user.id,
    extract_types=["title", "screenshot", "pdf"],
    priority=5
)
print(f"Job queued: {job.job_key}")
```

### API Usage

```bash
# Archive a URL
curl -X POST "/api/archivebox/archive" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "async_processing": true}'

# Check status
curl "/api/archivebox/archive/https://example.com"

# Download screenshot
curl "/api/archivebox/content/{snapshot_id}/file/screenshot" \
  -o screenshot.png
```

## Feature Flags & Rollout

The implementation includes comprehensive feature flags:

- **`ARCHIVEBOX_ENABLED`**: Master enable switch
- **`ARCHIVEBOX_FEATURE_FLAG`**: Gradual rollout control
- **Automatic fallback**: Graceful degradation when unavailable

## Security Considerations

- **Authentication required**: For cleanup and bulk operations
- **File access controls**: Restricted to archive directories
- **Resource limits**: Configurable timeouts and size limits
- **Input validation**: URL and parameter sanitization

## Storage Strategies

### Symlink Strategy (Default)
- **Efficient**: No data duplication
- **Fast**: Instant "copying"
- **Limitation**: Requires same filesystem

### Copy Strategy
- **Portable**: Independent archive copies
- **Safer**: No symlink dependencies
- **Cost**: Additional storage usage

## Next Phase Preparation

Phase 1 provides the foundation for:
- **Phase 2**: Web UI integration and user-facing features
- **Phase 3**: Advanced archival workflows and automation
- **Phase 4**: Analytics and archive management

## Architecture Decisions

- **Service-oriented**: Modular, testable design
- **Async-first**: Non-blocking operations
- **Dual execution**: Docker + CLI support for flexibility
- **Configuration-driven**: Environment-based setup
- **Queue integration**: Background processing support
- **Error recovery**: Graceful failure handling

## Monitoring & Debugging

- **Health checks**: Service availability monitoring
- **Logging**: Comprehensive operation logging
- **Job tracking**: Queue-based status monitoring
- **Resource monitoring**: Docker container metrics

This Phase 1 implementation provides a solid foundation for web archival integration while maintaining Second Brain's architectural patterns and quality standards.