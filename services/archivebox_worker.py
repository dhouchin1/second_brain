"""
ArchiveBox Background Worker

Processes archive jobs from the ingestion queue, handling URL archival,
content extraction, and integration with the note system.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

from services.archivebox_service import get_archivebox_service, ArchiveRequest
from services.ingestion_queue import (
    ingestion_queue, IngestionJobType, IngestionJobStatus,
    ArchiveWebCapturePayload
)
from graph_memory.service import get_graph_memory_service
from database import get_db_connection
from config import settings

logger = logging.getLogger(__name__)

class ArchiveBoxWorker:
    """Background worker for processing ArchiveBox jobs."""

    def __init__(self):
        self.archivebox_service = get_archivebox_service()
        self.running = False

    async def start(self):
        """Start the worker loop."""
        if not settings.archivebox_enabled:
            logger.info("ArchiveBox worker disabled via configuration")
            return

        if not await self.archivebox_service.is_available():
            logger.warning("ArchiveBox service not available, worker will not start")
            return

        self.running = True
        logger.info("🏗️ ArchiveBox worker started")

        while self.running:
            try:
                # Process one job at a time
                job = ingestion_queue.get_next_job()

                if job and job.job_type == IngestionJobType.ARCHIVE_WEB_CAPTURE.value:
                    await self._process_archive_job(job)
                elif job:
                    # Not our job type, release it back to queue
                    ingestion_queue.update_job_status(job.id, IngestionJobStatus.QUEUED)
                    await asyncio.sleep(1)  # Short wait before next check
                else:
                    # No jobs available, wait before checking again
                    await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Error in ArchiveBox worker loop: {e}")
                await asyncio.sleep(10)  # Back off on errors

    def stop(self):
        """Stop the worker."""
        self.running = False
        logger.info("🛑 ArchiveBox worker stopped")

    async def _process_archive_job(self, job):
        """Process a single archive job."""
        logger.info(f"📦 Processing archive job {job.job_key} for URL")

        try:
            # Parse the job payload
            payload = ArchiveWebCapturePayload(**json.loads(job.payload))

            # Normalise auxiliary fields that may be stored on the job instead of the payload
            user_id = payload.user_id if payload.user_id is not None else job.user_id
            note_id = payload.note_id if payload.note_id is not None else job.note_id
            priority = payload.priority if payload.priority is not None else job.priority
            integration_metadata = payload.integration_metadata or {}

            # Update job status to processing
            ingestion_queue.update_job_status(job.id, IngestionJobStatus.PROCESSING)

            # Create archive request
            archive_request = ArchiveRequest(
                url=payload.url,
                user_id=user_id,
                extract_types=payload.extract_types,
                timeout=payload.timeout,
                only_new=payload.only_new,
                overwrite=payload.overwrite,
                metadata={
                    "job_id": job.id,
                    "job_key": job.job_key,
                    "note_id": note_id,
                    "priority": priority,
                    "storage_strategy": payload.storage_strategy,
                    **integration_metadata
                }
            )

            # Perform the archival
            result = await self.archivebox_service.archive_url(archive_request)

            if result.success:
                # Archive successful - update job and create/update note
                await self._handle_successful_archive(job, payload, result)
                logger.info(f"✅ Archive job {job.job_key} completed successfully")
            else:
                # Archive failed
                await self._handle_failed_archive(job, result)
                logger.error(f"❌ Archive job {job.job_key} failed: {result.message}")

        except Exception as e:
            logger.error(f"💥 Error processing archive job {job.job_key}: {e}")
            ingestion_queue.update_job_status(
                job.id,
                IngestionJobStatus.FAILED,
                error_message=str(e)
            )

    async def _handle_successful_archive(self, job, payload: ArchiveWebCapturePayload, result):
        """Handle successful archive completion."""
        try:
            # Update job status
            ingestion_queue.update_job_status(
                job.id,
                IngestionJobStatus.COMPLETED,
                result_data={
                    "archive_result": result.to_dict(),
                    "snapshot_id": result.snapshot_id,
                    "archive_path": result.archive_path,
                    "title": result.title
                }
            )

            # Create or update note if needed
            integration_metadata = payload.integration_metadata or {}

            if payload.note_id or integration_metadata.get("create_note"):
                note_id, note_content = await self._create_or_update_note(
                    payload,
                    result,
                    user_id=payload.user_id or job.user_id,
                )

                # Index archived content for search
                if note_id:
                    await self._index_archived_content(note_id, result, payload.url)
                    self._maybe_record_graph_memory(
                        note_id=note_id,
                        note_content=note_content,
                        payload=payload,
                        result=result,
                        user_id=payload.user_id or job.user_id,
                    )

        except Exception as e:
            logger.error(f"Error handling successful archive: {e}")

    async def _handle_failed_archive(self, job, result):
        """Handle failed archive."""
        ingestion_queue.update_job_status(
            job.id,
            IngestionJobStatus.FAILED,
            error_message=result.message,
            result_data={"archive_result": result.to_dict()}
        )

    async def _create_or_update_note(
        self,
        payload: ArchiveWebCapturePayload,
        result,
        *,
        user_id: Optional[int] = None,
    ) -> Tuple[Optional[int], str]:
        """Create or update a note with archived content."""

        note_content = ""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Extract content for the note
            note_content = await self._extract_note_content(payload.url, result)

            if payload.note_id:
                # Update existing note
                cursor.execute(
                    """
                    UPDATE notes
                    SET content = ?, archived_url = ?, archive_snapshot_id = ?,
                        archive_path = ?, updated_at = CURRENT_TIMESTAMP,
                        status = 'completed'
                    WHERE id = ?
                    """,
                    (
                        note_content,
                        payload.url,
                        result.snapshot_id,
                        result.archive_path,
                        payload.note_id,
                    ),
                )

                logger.info(f"📝 Updated note {payload.note_id} with archived content")
                note_id = payload.note_id
            else:
                # Create new note
                cursor.execute(
                    """
                    INSERT INTO notes (content, user_id, archived_url, archive_snapshot_id,
                                     archive_path, title, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, 'completed', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    (
                        note_content,
                        user_id,
                        payload.url,
                        result.snapshot_id,
                        result.archive_path,
                        result.title or f"Archived: {payload.url}",
                    ),
                )

                note_id = cursor.lastrowid
                logger.info(f"📝 Created new note {note_id} for archived URL")

            conn.commit()
            conn.close()
            return note_id, note_content

        except Exception as e:
            logger.error(f"Error creating/updating note for archive: {e}")
            return None, note_content

    async def _extract_note_content(self, url: str, result) -> str:
        """Extract meaningful content for a note from archive result."""
        content_parts = [f"# Archived: {result.title or url}", f"\n**Original URL:** {url}"]

        if result.timestamp:
            content_parts.append(f"**Archived:** {result.timestamp}")

        if result.snapshot_id:
            content_parts.append(f"**Archive ID:** {result.snapshot_id}")

        # Try to get readable text content
        try:
            if result.archive_path:
                text_content = await self.archivebox_service.get_archive_content(
                    result.snapshot_id, "text"
                )
                if text_content:
                    # Limit content length and clean it
                    text = text_content.decode('utf-8', errors='ignore')[:2000]
                    content_parts.append(f"\n## Content Summary\n\n{text}")
        except Exception as e:
            logger.debug(f"Could not extract text content: {e}")

        content_parts.append(f"\n## Archive Files")
        content_parts.append(f"- HTML: Available")
        content_parts.append(f"- PDF: Available")
        content_parts.append(f"- Screenshot: Available")

        return "\n".join(content_parts)

    async def _index_archived_content(self, note_id: int, result, url: str):
        """Index archived content for search functionality."""
        try:
            # Trigger reindexing of the note for search
            from services.search_index import SearchIndexer
            indexer = SearchIndexer()

            # Index the note (which includes its content)
            index_result = indexer.index_item(str(note_id))

            if index_result.get("successful_fts", 0) > 0:
                logger.info(f"🔍 Successfully indexed archived note {note_id} for search")
            else:
                logger.warning(f"⚠️ Failed to index archived note {note_id} for search")

            # Optionally, extract and store additional archived content for later use
            full_text_content = await self._extract_full_text_content(result)
            if full_text_content:
                # Store the extracted content as metadata in the database
                await self._store_extracted_content_metadata(note_id, full_text_content, url, result)

        except Exception as e:
            logger.error(f"Failed to index archived content for note {note_id}: {e}")

    async def _store_extracted_content_metadata(self, note_id: int, content: str, url: str, result):
        """Store extracted content metadata for future reference."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Update note metadata with extracted content info
            cursor.execute("SELECT metadata FROM notes WHERE id = ?", (note_id,))
            row = cursor.fetchone()

            if row:
                metadata = {}
                try:
                    metadata = json.loads(row[0]) if row[0] else {}
                except:
                    metadata = {}

                # Add archived content metadata
                metadata.update({
                    "archive_extracted_at": datetime.now().isoformat(),
                    "archive_content_length": len(content),
                    "archive_url": url,
                    "archive_snapshot_id": result.snapshot_id,
                    "archive_content_available": True
                })

                cursor.execute(
                    "UPDATE notes SET metadata = ? WHERE id = ?",
                    (json.dumps(metadata), note_id)
                )
                conn.commit()

            conn.close()
            logger.debug(f"📋 Stored extraction metadata for note {note_id}")

        except Exception as e:
            logger.error(f"Failed to store extraction metadata for note {note_id}: {e}")

    def _maybe_record_graph_memory(
        self,
        *,
        note_id: int,
        note_content: str,
        payload: ArchiveWebCapturePayload,
        result,
        user_id: Optional[int],
    ) -> None:
        """Send the archived document to graph memory when enabled."""

        if not settings.graph_memory_enabled or not settings.graph_memory_extract_archivebox:
            return

        if not note_content:
            return

        checksum = hashlib.sha256(note_content.encode("utf-8")).hexdigest()
        metadata = {
            "snapshot_id": result.snapshot_id,
            "archive_path": result.archive_path,
            "url": payload.url,
        }
        if payload.integration_metadata:
            metadata["integration"] = payload.integration_metadata

        try:
            service = get_graph_memory_service()
            ingest_result = service.ingest_text(
                text=note_content,
                title=result.title or payload.url,
                source_type="archivebox",
                uri=payload.url,
                checksum=checksum,
                path=result.archive_path,
                mime="text/markdown",
                metadata=metadata,
                user_id=user_id,
            )
            if ingest_result.get("facts", 0):
                logger.debug(
                    "Graph memory stored %s facts for archived note %s",
                    ingest_result["facts"],
                    note_id,
                )
        except Exception:
            logger.exception("Graph memory ingestion failed for note %s", note_id)

    async def _extract_full_text_content(self, result) -> str:
        """Extract full readable text content from archived result.

        Prioritizes Readability content (clean, human-readable text) over raw HTML.
        Falls back to other extraction methods if Readability is not available.
        """
        try:
            text_content = ""

            # Prioritized content extraction order (from patch recommendation)
            content_sources = [
                "readability",      # Clean, human-readable text (preferred)
                "mercury",          # Alternative clean text extraction
                "text",             # Plain text extraction
                "htmltotext",       # HTML to text conversion
                "singlefile",       # Single-file HTML (may contain readable content)
            ]

            for source in content_sources:
                try:
                    content = await self.archivebox_service.get_archive_content(
                        result.snapshot_id, source
                    )
                    if content:
                        if isinstance(content, bytes):
                            decoded_content = content.decode('utf-8', errors='ignore')
                        else:
                            decoded_content = str(content)

                        # Clean and validate the content
                        cleaned_content = self._clean_extracted_text(decoded_content)
                        if cleaned_content and len(cleaned_content) > 100:  # Minimum content threshold
                            text_content = cleaned_content
                            logger.debug(f"Successfully extracted text using {source} method")
                            break

                except Exception as e:
                    logger.debug(f"Failed to extract using {source}: {e}")
                    continue

            # If no content found, try to read directly from readability directory
            if not text_content and result.archive_path:
                try:
                    readability_path = Path(result.archive_path) / "readability" / "content.txt"
                    if readability_path.exists():
                        with open(readability_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        cleaned_content = self._clean_extracted_text(content)
                        if cleaned_content and len(cleaned_content) > 100:
                            text_content = cleaned_content
                            logger.debug("Successfully extracted text from readability/content.txt")
                except Exception as e:
                    logger.debug(f"Failed to read from readability directory: {e}")

            return text_content

        except Exception as e:
            logger.error(f"Failed to extract full text content: {e}")
            return ""

    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text content."""
        if not text:
            return ""

        # Basic text cleaning
        import re

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove common artifacts from web pages
        text = re.sub(r'(cookie|privacy policy|terms of service|subscribe|newsletter)', '', text, flags=re.IGNORECASE)

        # Remove very short lines (likely navigation or ads)
        lines = text.split('\n')
        meaningful_lines = [line.strip() for line in lines if len(line.strip()) > 20]

        # Limit content length for search indexing (10KB max)
        cleaned_text = '\n'.join(meaningful_lines)
        if len(cleaned_text) > 10000:
            cleaned_text = cleaned_text[:10000] + "... [content truncated]"

        return cleaned_text.strip()


# Global worker instance
archivebox_worker = ArchiveBoxWorker()

async def start_archivebox_worker():
    """Start the ArchiveBox background worker."""
    await archivebox_worker.start()

def stop_archivebox_worker():
    """Stop the ArchiveBox background worker."""
    archivebox_worker.stop()
