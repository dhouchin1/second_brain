# ArchiveBox Integration Plan for Second Brain

## Overview

ArchiveBox is an excellent candidate for enhancing Second Brain's content ingestion capabilities. It provides comprehensive web archiving functionality that would significantly expand our current URL ingestion system beyond basic Playwright extraction.

## What ArchiveBox Provides

### Core Capabilities
- **Multi-format preservation**: HTML, PDF, PNG screenshots, JSON metadata, WARC archives
- **Diverse content sources**: Web pages, social media, YouTube videos, GitHub repos
- **Multiple interfaces**: CLI, Web UI, Python API
- **Self-hosted solution**: Aligns with Second Brain's architecture
- **Durable storage**: Long-term accessible file formats

### Content Types Supported
- Web pages and articles
- Social media posts (Twitter, Reddit, etc.)
- YouTube videos and media content
- GitHub repositories and documentation
- RSS/Atom feeds
- Browser bookmarks and history

## Integration Architecture

### Current State
```
User Input (URL) → URL Ingestion Service → Playwright Extraction → Unified Capture Service → SQLite + Embeddings
```

### Proposed Enhanced Architecture
```
User Input → ArchiveBox Integration Layer → Multiple Preservation Formats → Enhanced Processing → Second Brain Storage
```

### Integration Points

#### 1. **Enhanced URL Ingestion Service**
- Extend existing `services/url_ingestion_service.py`
- Add ArchiveBox as premium archival backend
- Maintain Playwright as fallback for simple extraction

#### 2. **ArchiveBox Service Layer**
```python
services/
├── archivebox_service.py          # Core ArchiveBox integration
├── archivebox_router.py           # API endpoints for archival
└── enhanced_url_ingestion.py      # Updated URL service with ArchiveBox
```

#### 3. **Storage Enhancement**
- **Archival Storage**: `/archival/` directory for ArchiveBox data
- **Metadata Integration**: Enhanced note metadata with archive links
- **Format Variety**: Multiple preservation formats per URL

## Implementation Plan

### Phase 1: Foundation Setup (Week 1)
1. **Install ArchiveBox**
   ```bash
   # Docker-based installation (recommended)
   docker run -v $PWD/archival:/data -p 8000:8000 archivebox/archivebox init
   ```

2. **Basic Service Integration**
   - Create `services/archivebox_service.py`
   - Implement Python API wrapper
   - Add configuration to `config.py`

3. **Environment Configuration**
   ```python
   # config.py additions
   archivebox_enabled: bool = False
   archivebox_data_dir: str = "./archival"
   archivebox_api_url: str = "http://localhost:8000"
   ```

### Phase 2: Core Integration (Week 2)
1. **Enhanced URL Ingestion**
   - Modify existing URL ingestion to use ArchiveBox
   - Implement multi-format processing
   - Add archive metadata to notes

2. **API Endpoints**
   ```python
   # New endpoints in archivebox_router.py
   POST /api/archive/url          # Archive single URL
   POST /api/archive/batch        # Batch archive URLs
   GET  /api/archive/status/{id}  # Check archival status
   GET  /api/archive/formats/{id} # Get available formats
   ```

3. **Database Schema Updates**
   ```sql
   -- Add archive tracking
   ALTER TABLE notes ADD COLUMN archive_id TEXT;
   ALTER TABLE notes ADD COLUMN archive_formats TEXT; -- JSON array
   ALTER TABLE notes ADD COLUMN archive_status TEXT;
   ```

### Phase 3: Advanced Features (Week 3)
1. **Bulk Import Capabilities**
   - Browser bookmarks import
   - Browser history import
   - RSS feed monitoring
   - Social media feed integration

2. **Archive Management UI**
   - Archive status dashboard
   - Format selection interface
   - Bulk operations panel

3. **Enhanced Search Integration**
   - Search across archived content
   - Format-specific search (PDF text, image OCR)
   - Archive timeline visualization

### Phase 4: Advanced Integration (Week 4)
1. **Automated Archival**
   - Background archival queue
   - Periodic re-archival for updated content
   - Archive quality monitoring

2. **Content Enhancement**
   - Extract structured data from archives
   - Generate enhanced summaries from multiple formats
   - Cross-reference archived versions

## Technical Implementation

### Core Service Implementation

```python
# services/archivebox_service.py
class ArchiveBoxService:
    def __init__(self, data_dir: str, api_url: str):
        self.data_dir = Path(data_dir)
        self.api_url = api_url

    async def archive_url(self, url: str, extract_types: List[str] = None) -> dict:
        """Archive URL with specified extraction types"""

    async def get_archive_status(self, archive_id: str) -> dict:
        """Get archival status and available formats"""

    async def get_archive_content(self, archive_id: str, format_type: str) -> bytes:
        """Retrieve archived content in specific format"""

    async def batch_archive(self, urls: List[str]) -> dict:
        """Archive multiple URLs efficiently"""
```

### Enhanced URL Ingestion

```python
# Enhanced url_ingestion_service.py
async def ingest_url_with_archive(self, url: str, user_id: int,
                                 archive_enabled: bool = True) -> dict:
    """Enhanced URL ingestion with ArchiveBox integration"""

    results = {}

    if archive_enabled and settings.archivebox_enabled:
        # Use ArchiveBox for comprehensive archival
        archive_result = await self.archivebox_service.archive_url(url)
        results['archive_id'] = archive_result['id']
        results['formats'] = archive_result['available_formats']

        # Process archived content for Second Brain
        for format_type in ['html', 'pdf', 'json']:
            if format_type in archive_result['available_formats']:
                content = await self._process_archived_format(
                    archive_result['id'], format_type
                )
                results[f'{format_type}_content'] = content
    else:
        # Fallback to existing Playwright extraction
        results = await self._ingest_webpage(url, user_id, source_text)

    return results
```

## Benefits for Second Brain

### 1. **Comprehensive Preservation**
- Multiple redundant formats ensure long-term accessibility
- Better resilience against link rot and content changes
- Higher fidelity content capture

### 2. **Enhanced Content Types**
- Support for multimedia content (videos, images)
- Social media post preservation
- GitHub repository archival
- RSS feed content capture

### 3. **Improved Search & Discovery**
- Search across multiple content formats
- Better text extraction from PDFs and images
- Metadata-rich content indexing

### 4. **Professional Use Cases**
- Research documentation and citation
- Legal evidence preservation
- Journalism source archival
- Content backup and redundancy

### 5. **Integration Advantages**
- Leverages existing URL ingestion infrastructure
- Maintains backward compatibility
- Adds value without disrupting current workflows

## Configuration Options

### Basic Configuration
```python
# config.py
class Settings:
    # ArchiveBox Integration
    archivebox_enabled: bool = False
    archivebox_data_dir: str = "./archival"
    archivebox_docker_image: str = "archivebox/archivebox:latest"
    archivebox_extraction_types: List[str] = ["html", "pdf", "png", "json"]

    # Integration Preferences
    archivebox_primary_backend: bool = False  # Use as primary or secondary
    archivebox_auto_archive: bool = True      # Auto-archive all URLs
    archivebox_batch_size: int = 10           # Batch processing size
```

### Advanced Configuration
```python
class ArchiveBoxConfig:
    # Quality Settings
    pdf_quality: str = "high"
    screenshot_resolution: str = "1920x1080"
    extraction_timeout: int = 300

    # Storage Settings
    max_archive_size: str = "10GB"
    retention_period: int = 365  # days
    compress_archives: bool = True

    # Integration Settings
    sync_with_obsidian: bool = True
    generate_embeddings: bool = True
    extract_structured_data: bool = True
```

## Migration Strategy

### Phase 1: Parallel Deployment
- Run ArchiveBox alongside existing system
- Selective URL archival for testing
- Compare quality and performance

### Phase 2: Gradual Integration
- Enable ArchiveBox for new URLs
- Migrate high-value existing URLs
- User opt-in for enhanced archival

### Phase 3: Full Integration
- Make ArchiveBox primary archival backend
- Maintain Playwright as lightweight fallback
- Complete feature parity

## Resource Requirements

### Infrastructure
- **Docker**: ArchiveBox container (recommended)
- **Storage**: Additional storage for archived content (~100MB per complex page)
- **Memory**: Additional 512MB-1GB RAM for ArchiveBox
- **CPU**: Minimal additional CPU usage

### Dependencies
```bash
# Additional Python packages
pip install archivebox-client  # If available
pip install docker-py          # For Docker management
pip install asyncio-subprocess # For CLI integration
```

### Docker Setup
```yaml
# docker-compose.yml addition
services:
  archivebox:
    image: archivebox/archivebox:latest
    volumes:
      - ./archival:/data
    ports:
      - "8001:8000"  # Avoid conflict with Second Brain
    environment:
      - ADMIN_USERNAME=admin
      - ADMIN_PASSWORD=archivebox
```

## Success Metrics

### Quality Metrics
- Archive success rate (target: >95%)
- Content extraction accuracy
- Format availability and completeness
- Search result relevance improvement

### Performance Metrics
- Archival processing time
- Storage efficiency
- Search query performance
- System resource utilization

### User Experience Metrics
- Feature adoption rate
- User satisfaction with archived content
- Content discovery improvement
- Workflow integration smoothness

## Conclusion

ArchiveBox integration would significantly enhance Second Brain's content ingestion capabilities, providing:

1. **Enterprise-grade archival** with multiple preservation formats
2. **Comprehensive content support** beyond basic web pages
3. **Future-proof storage** in durable, accessible formats
4. **Enhanced search and discovery** across rich content types
5. **Professional workflows** for research, journalism, and documentation

The integration can be implemented gradually, maintaining compatibility with existing systems while adding powerful new capabilities. This positions Second Brain as a comprehensive knowledge management platform capable of preserving and organizing diverse digital content for long-term value.

**Recommendation**: Proceed with Phase 1 implementation to evaluate ArchiveBox's fit within the Second Brain ecosystem.