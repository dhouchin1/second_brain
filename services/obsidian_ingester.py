"""
Enhanced Obsidian Vault Ingester

Provides comprehensive Obsidian vault processing with improved text extraction,
frontmatter parsing, and graph memory integration. Based on improvements from
the graph memory patch but integrated with existing system.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from graph_memory.service import get_graph_memory_service
from config import settings

logger = logging.getLogger(__name__)


class ObsidianIngester:
    """Enhanced Obsidian vault ingester with better content processing."""

    def __init__(self):
        self.graph_service = get_graph_memory_service()
        self.md_extensions = {'.md', '.markdown'}

    def ingest_vault(
        self,
        vault_path: Path,
        *,
        glob_pattern: str = "**/*.md",
        max_files: Optional[int] = None,
        force_reingest: bool = False
    ) -> Dict[str, Any]:
        """Ingest an entire Obsidian vault into graph memory.

        Args:
            vault_path: Path to the Obsidian vault
            glob_pattern: Pattern for matching markdown files
            max_files: Maximum number of files to process (None for all)
            force_reingest: Force re-ingestion even if files haven't changed

        Returns:
            Dictionary with ingestion statistics
        """
        vault_path = Path(vault_path)
        if not vault_path.exists():
            raise ValueError(f"Vault path does not exist: {vault_path}")

        if not settings.graph_memory_enabled:
            return {"enabled": False, "processed": 0, "skipped": 0, "errors": 0}

        processed = 0
        skipped = 0
        errors = 0

        try:
            # Find all markdown files in vault
            md_files = list(vault_path.glob(glob_pattern))

            # Filter out hidden files and directories
            md_files = [f for f in md_files if not any(part.startswith('.') for part in f.relative_to(vault_path).parts)]

            # Limit if specified
            if max_files:
                md_files = md_files[:max_files]

            logger.info(f"Processing {len(md_files)} markdown files from vault: {vault_path}")

            for md_file in md_files:
                try:
                    result = self.ingest_file(md_file, vault_path, force_reingest)
                    if result == "processed":
                        processed += 1
                    elif result == "skipped":
                        skipped += 1
                    else:
                        errors += 1

                except Exception as e:
                    logger.error(f"Error processing {md_file}: {e}")
                    errors += 1

        except Exception as e:
            logger.error(f"Error during vault ingestion: {e}")
            return {"enabled": True, "processed": processed, "skipped": skipped, "errors": errors, "failed": True}

        return {
            "enabled": True,
            "processed": processed,
            "skipped": skipped,
            "errors": errors,
            "vault_path": str(vault_path),
            "pattern": glob_pattern,
            "total_files": len(md_files)
        }

    def ingest_file(self, md_path: Path, vault_path: Path, force_reingest: bool = False) -> str:
        """Ingest a single Obsidian markdown file.

        Returns:
            "processed" if file was processed
            "skipped" if file was skipped (unchanged)
            "error" if there was an error
        """
        try:
            # Read file content
            with open(md_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if not content.strip():
                return "skipped"

            # Generate content hash for change detection
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

            # Check if file has changed (unless forcing re-ingestion)
            if not force_reingest and self._file_unchanged(md_path, content_hash):
                return "skipped"

            # Parse frontmatter and body
            frontmatter, body = self._parse_frontmatter(content)

            # Create source record
            relative_path = str(md_path.relative_to(vault_path))
            source_metadata = {
                "frontmatter": frontmatter,
                "vault_path": str(vault_path),
                "relative_path": relative_path,
                "file_path": str(md_path)
            }

            # Ingest into graph memory
            ingest_result = self.graph_service.ingest_text(
                text=body,
                title=frontmatter.get('title', md_path.stem),
                source_type="obsidian",
                uri=relative_path,
                checksum=content_hash,
                path=str(md_path),
                mime="text/markdown",
                metadata=source_metadata
            )

            # Record successful processing
            self._mark_file_processed(md_path, content_hash)

            if ingest_result.get("facts", 0) > 0:
                logger.debug(
                    "Ingested %d facts from %s",
                    ingest_result["facts"],
                    relative_path
                )

            return "processed"

        except Exception as e:
            logger.error(f"Error ingesting file {md_path}: {e}")
            return "error"

    def _parse_frontmatter(self, content: str) -> tuple[Dict[str, Any], str]:
        """Parse YAML frontmatter from markdown content."""
        try:
            if content.lstrip().startswith("---"):
                parts = content.split("\n---\n", 1)
                if len(parts) == 2 and parts[0].strip().startswith("---"):
                    frontmatter_text = parts[0].strip().strip("-")
                    body = parts[1]

                    # Parse YAML
                    try:
                        import yaml
                        frontmatter = yaml.safe_load(frontmatter_text) or {}
                        return frontmatter, body
                    except Exception:
                        # If YAML parsing fails, treat as body
                        return {}, content
            return {}, content
        except Exception:
            return {}, content

    def _file_unchanged(self, md_path: Path, content_hash: str) -> bool:
        """Check if a file has changed since last processing."""
        try:
            # Store processing state in a simple JSON file
            state_dir = md_path.parent / ".secondbrain"
            state_dir.mkdir(exist_ok=True)
            state_file = state_dir / "processing_state.json"

            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
            else:
                state = {}

            last_hash = state.get(str(md_path), {}).get("hash")
            return last_hash == content_hash

        except Exception:
            return False

    def _mark_file_processed(self, md_path: Path, content_hash: str) -> None:
        """Mark a file as processed with its content hash."""
        try:
            state_dir = md_path.parent / ".secondbrain"
            state_dir.mkdir(exist_ok=True)
            state_file = state_dir / "processing_state.json"

            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
            else:
                state = {}

            state[str(md_path)] = {
                "hash": content_hash,
                "last_processed": datetime.now().isoformat()
            }

            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            logger.debug(f"Could not save processing state: {e}")


# Global service instance
_obsidian_ingester: Optional[ObsidianIngester] = None


def get_obsidian_ingester() -> ObsidianIngester:
    """Get global Obsidian ingester instance."""
    global _obsidian_ingester
    if _obsidian_ingester is None:
        _obsidian_ingester = ObsidianIngester()
    return _obsidian_ingester
