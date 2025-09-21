"""
URL Archiving Integration

Automatically detects URLs in notes and creates ArchiveBox jobs for archival.
Integrates with the note creation workflow to provide seamless web content preservation.
"""

from __future__ import annotations

import re
import logging
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

from services.ingestion_queue import create_archive_job
from config import settings

logger = logging.getLogger(__name__)

# URL detection patterns
URL_PATTERN = re.compile(
    r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
)

# Common file extensions to skip archiving
SKIP_EXTENSIONS = {
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.zip', '.rar', '.tar', '.gz', '.7z',
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg',
    '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm',
    '.mp3', '.wav', '.flac', '.ogg', '.aac',
    '.exe', '.dmg', '.deb', '.rpm'
}

# Domains to skip archiving
SKIP_DOMAINS = {
    'localhost', '127.0.0.1', '0.0.0.0',
    'github.com',  # Large repos might timeout
    'youtube.com', 'youtu.be',  # Video content better handled differently
}

class URLArchivingIntegration:
    """Handles automatic URL detection and archiving for notes."""

    def __init__(self):
        self.enabled = settings.archivebox_enabled

    def extract_urls_from_note(self, content: str) -> List[str]:
        """Extract all valid URLs from note content."""
        if not content:
            return []

        # Find all URLs in the content
        urls = URL_PATTERN.findall(content)

        # Filter and validate URLs
        valid_urls = []
        for url in urls:
            if self._should_archive_url(url):
                valid_urls.append(url)

        return valid_urls

    def _should_archive_url(self, url: str) -> bool:
        """Determine if a URL should be archived."""
        try:
            parsed = urlparse(url)

            # Skip invalid URLs
            if not parsed.scheme or not parsed.netloc:
                return False

            # Skip non-HTTP(S) URLs
            if parsed.scheme not in ('http', 'https'):
                return False

            # Skip URLs with file extensions we don't want to archive
            path = parsed.path.lower()
            for ext in SKIP_EXTENSIONS:
                if path.endswith(ext):
                    return False

            # Skip certain domains
            domain = parsed.netloc.lower()
            for skip_domain in SKIP_DOMAINS:
                if skip_domain in domain:
                    return False

            # Skip very long URLs (likely not useful web pages)
            if len(url) > 500:
                return False

            return True

        except Exception as e:
            logger.debug(f"Error parsing URL {url}: {e}")
            return False

    def create_archive_jobs_for_note(
        self,
        note_id: int,
        user_id: int,
        content: str,
        priority: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Create ArchiveBox jobs for all URLs found in a note.

        Returns list of job keys created.
        """
        if not self.enabled:
            logger.debug("ArchiveBox integration disabled")
            return []

        urls = self.extract_urls_from_note(content)
        if not urls:
            return []

        job_keys = []
        base_metadata = metadata or {}

        for url in urls:
            try:
                # Create integration metadata
                integration_metadata = {
                    **base_metadata,
                    'source': 'note',
                    'auto_detected': True,
                    'original_note_id': note_id
                }

                # Create archive job
                job = create_archive_job(
                    url=url,
                    user_id=user_id,
                    note_id=note_id,
                    priority=priority,
                    integration_metadata=integration_metadata
                )

                job_keys.append(job.job_key)
                logger.info(f"📦 Created archive job {job.job_key} for URL {url}")

            except Exception as e:
                logger.error(f"Failed to create archive job for {url}: {e}")

        return job_keys

    def update_note_with_archive_links(
        self,
        note_id: int,
        url_to_snapshot_mapping: Dict[str, str]
    ):
        """Update note content to include links to archived versions."""
        try:
            from database import get_db_connection

            conn = get_db_connection()
            cursor = conn.cursor()

            # Get current note content
            cursor.execute("SELECT content FROM notes WHERE id = ?", (note_id,))
            row = cursor.fetchone()
            if not row:
                return

            content = row[0]
            updated_content = content

            # Add archive links for each URL
            for original_url, snapshot_id in url_to_snapshot_mapping.items():
                if snapshot_id:
                    archive_link = f"\n\n📦 **Archived:** [View archived version](/api/archivebox/content/{snapshot_id}/file/html)"
                    # Insert archive link after the original URL
                    updated_content = updated_content.replace(
                        original_url,
                        f"{original_url}{archive_link}"
                    )

            # Update the note if content changed
            if updated_content != content:
                cursor.execute(
                    "UPDATE notes SET content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (updated_content, note_id)
                )
                conn.commit()
                logger.info(f"📝 Updated note {note_id} with archive links")

            conn.close()

        except Exception as e:
            logger.error(f"Failed to update note {note_id} with archive links: {e}")


# Global instance
url_archiving_integration = URLArchivingIntegration()

def integrate_url_archiving_with_note(
    note_id: int,
    user_id: int,
    content: str,
    priority: int = 5,
    metadata: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Convenience function to integrate URL archiving with note creation.

    Returns list of job keys created.
    """
    return url_archiving_integration.create_archive_jobs_for_note(
        note_id, user_id, content, priority, metadata
    )