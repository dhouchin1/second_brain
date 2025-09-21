"""
Background Ingestion Service

Provides continuous file watching for Obsidian vaults and ArchiveBox snapshots.
Automatically ingests new or modified content into the graph memory system.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Set, Dict, Any

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from graph_memory.service import get_graph_memory_service
from services.obsidian_ingester import get_obsidian_ingester
from config import settings

logger = logging.getLogger(__name__)


class _Debounce:
    """Simple per-path cooldown to avoid duplicate work when editors spam events."""

    def __init__(self, cooldown_s: float = 2.0):
        self.cooldown = cooldown_s
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def ok(self, path: str) -> bool:
        now = time.time()
        with self._lock:
            t = self._last.get(path, 0)
            if now - t >= self.cooldown:
                self._last[path] = now
                return True
            return False


class ObsidianHandler(FileSystemEventHandler):
    """Watch for changes in Obsidian vault files."""

    def __init__(self, ingester, vault_dir: Path):
        super().__init__()
        self.ingester = ingester
        self.vault = Path(vault_dir)
        self.debounce = _Debounce(2.0)
        self.md_extensions = {'.md', '.markdown'}

    def _should_process(self, path: Path) -> bool:
        """Check if file should be processed."""
        # Skip hidden files and directories
        if any(part.startswith('.') for part in path.parts):
            return False

        # Only process markdown files
        return path.suffix.lower() in self.md_extensions and path.is_file()

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if self._should_process(path) and self.debounce.ok(str(path)):
            logger.info(f"📝 New Obsidian file detected: {path}")
            self.ingester.ingest_file(path, self.vault, force_reingest=True)

    def on_modified(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if self._should_process(path) and self.debounce.ok(str(path)):
            logger.info(f"📝 Modified Obsidian file detected: {path}")
            self.ingester.ingest_file(path, self.vault, force_reingest=True)


class ArchiveBoxHandler(FileSystemEventHandler):
    """Watch for changes in ArchiveBox snapshots."""

    def __init__(self, archivebox_service, root: Path):
        super().__init__()
        self.archivebox_service = archivebox_service
        self.root = Path(root)
        self.debounce = _Debounce(5.0)

    def _is_snapshot_dir(self, path: Path) -> Optional[Path]:
        """Check if path represents a snapshot directory."""
        # For a file event (index.html or readability/content.txt), the snapshot dir is its parent or grandparent
        if path.name in {"index.html"} and (path.parent / "index.html").exists():
            return path.parent
        if path.name == "content.txt" and path.parent.name == "readability":
            return path.parent.parent
        # If a directory with index.html appears
        if path.is_dir() and (path / "index.html").exists():
            return path
        return None

    def on_created(self, event):
        path = Path(event.src_path)
        snapshot = self._is_snapshot_dir(path)
        if snapshot and self.debounce.ok(str(snapshot)):
            logger.info(f"📦 New ArchiveBox snapshot detected: {snapshot}")
            # TODO: Implement ArchiveBox snapshot ingestion
            # This would require similar logic to the Obsidian ingester

    def on_modified(self, event):
        path = Path(event.src_path)
        snapshot = self._is_snapshot_dir(path)
        if snapshot and self.debounce.ok(str(snapshot)):
            logger.info(f"📦 Modified ArchiveBox snapshot detected: {snapshot}")
            # TODO: Implement ArchiveBox snapshot ingestion


class BackgroundIngestionService:
    """Manages file system watchers for continuous ingestion."""

    def __init__(
        self,
        obsidian_vault: Optional[Path] = None,
        archivebox_root: Optional[Path] = None
    ):
        self.obsidian_vault = Path(obsidian_vault) if obsidian_vault else None
        self.archivebox_root = Path(archivebox_root) if archivebox_root else None
        self.obsidian_ingester = get_obsidian_ingester()
        self._observers: list[Observer] = []
        self._running = False
        self._watchdog_available = self._check_watchdog()

    def _check_watchdog(self) -> bool:
        """Check if watchdog is available."""
        try:
            from watchdog.observers import Observer
            return True
        except ImportError:
            logger.warning("watchdog package not available, falling back to polling")
            return False

    def _ensure_vault_exists(self, vault_path: Path) -> bool:
        """Ensure vault path exists and is readable."""
        if not vault_path.exists():
            logger.warning(f"Vault path does not exist: {vault_path}")
            return False
        if not vault_path.is_dir():
            logger.warning(f"Vault path is not a directory: {vault_path}")
            return False
        return True

    def start(self) -> None:
        """Start background file watching."""
        if self._running:
            return

        if not settings.graph_memory_enabled:
            logger.info("Graph memory disabled, background ingestion will not start")
            return

        self._running = True

        # Start Obsidian vault watching
        if self.obsidian_vault and self._ensure_vault_exists(self.obsidian_vault):
            if self._watchdog_available:
                self._start_watchdog_observer(
                    self.obsidian_vault,
                    ObsidianHandler(self.obsidian_ingester, self.obsidian_vault)
                )
            else:
                # Start polling fallback
                asyncio.create_task(self._poll_obsidian_vault())

        # Start ArchiveBox watching
        if self.archivebox_root and self._ensure_vault_exists(self.archivebox_root):
            # TODO: Implement ArchiveBox watching
            logger.info(f"ArchiveBox watching configured for: {self.archivebox_root}")
            # This would need a similar handler for ArchiveBox snapshots

        logger.info("🟢 Background ingestion service started")

    def stop(self) -> None:
        """Stop background file watching."""
        for observer in self._observers:
            try:
                observer.stop()
                observer.join(timeout=3)
            except Exception:
                pass
        self._observers.clear()
        self._running = False
        logger.info("🔴 Background ingestion service stopped")

    def _start_watchdog_observer(self, path: Path, handler: FileSystemEventHandler) -> None:
        """Start a watchdog observer for a path."""
        try:
            observer = Observer()
            observer.schedule(handler, str(path), recursive=True)
            observer.start()
            self._observers.append(observer)
            logger.info(f"📺 Started file watching for: {path}")
        except Exception as e:
            logger.error(f"Failed to start file watching for {path}: {e}")

    async def _poll_obsidian_vault(self) -> None:
        """Polling fallback for systems without watchdog."""
        logger.info("🔄 Starting polling-based Obsidian vault monitoring")

        known_files: Dict[str, float] = {}

        while self._running:
            try:
                if self.obsidian_vault and self.obsidian_vault.exists():
                    # Find all markdown files
                    md_files = list(self.obsidian_vault.rglob("*.md"))

                    for md_file in md_files:
                        # Skip hidden files
                        if any(part.startswith('.') for part in md_file.parts):
                            continue

                        # Check if file is new or modified
                        mtime = md_file.stat().st_mtime
                        file_key = str(md_file)

                        if file_key not in known_files or known_files[file_key] < mtime:
                            logger.info(f"📝 Polling detected change in: {md_file}")
                            known_files[file_key] = mtime
                            self.obsidian_ingester.ingest_file(md_file, self.obsidian_vault, force_reingest=True)

                # Wait before next poll
                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Error in polling loop: {e}")
                await asyncio.sleep(30)  # Longer delay on error

    async def rescan_obsidian(self) -> Dict[str, Any]:
        """Manually rescan the entire Obsidian vault."""
        if not self.obsidian_vault:
            return {"error": "No Obsidian vault configured"}

        if not self.obsidian_vault.exists():
            return {"error": f"Obsidian vault does not exist: {self.obsidian_vault}"}

        logger.info(f"🔄 Starting manual rescan of Obsidian vault: {self.obsidian_vault}")
        return self.obsidian_ingester.ingest_vault(self.obsidian_vault, force_reingest=True)

    async def rescan_archivebox(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Manually rescan ArchiveBox snapshots."""
        if not self.archivebox_root:
            return {"error": "No ArchiveBox root configured"}

        if not self.archivebox_root.exists():
            return {"error": f"ArchiveBox root does not exist: {self.archivebox_root}"}

        # TODO: Implement ArchiveBox rescan
        logger.info(f"🔄 Starting manual rescan of ArchiveBox: {self.archivebox_root}")
        return {"message": "ArchiveBox rescan not yet implemented", "archivebox_root": str(self.archivebox_root)}


# Global service instance
_background_ingestion_service: Optional[BackgroundIngestionService] = None


def get_background_ingestion_service() -> BackgroundIngestionService:
    """Get global background ingestion service instance."""
    global _background_ingestion_service
    if _background_ingestion_service is None:
        _background_ingestion_service = BackgroundIngestionService(
            obsidian_vault=settings.obsidian_vault_path,
            archivebox_root=settings.archivebox_data_dir
        )
    return _background_ingestion_service
