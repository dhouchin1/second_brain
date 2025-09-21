"""
ArchiveBox Integration Service

Provides comprehensive web archival capabilities using ArchiveBox.
Integrates with Second Brain's ingestion pipeline for persistent web content storage.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, HttpUrl, validator

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class ArchiveResult:
    """Result of an ArchiveBox archival operation."""

    url: str
    timestamp: str
    title: Optional[str] = None
    archive_path: Optional[str] = None
    snapshot_id: Optional[str] = None
    status: str = "unknown"  # success, failed, skipped, exists
    extractors: Dict[str, Any] = field(default_factory=dict)
    size_bytes: int = 0
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/API responses."""
        return asdict(self)

    @property
    def success(self) -> bool:
        """Check if archival was successful."""
        return self.status == "success"


@dataclass
class ArchiveRequest:
    """Request to archive a URL."""

    url: str
    user_id: Optional[int] = None
    priority: int = 0
    extract_types: Optional[List[str]] = None
    timeout: Optional[int] = None
    only_new: bool = True
    depth: int = 0  # For recursive archiving
    overwrite: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ArchiveBoxConfig(BaseModel):
    """ArchiveBox configuration settings."""

    enabled: bool = False
    data_dir: Path
    docker_image: str = "archivebox/archivebox:latest"
    timeout: int = 60
    only_new: bool = True
    extract_types: str = "title,favicon,wget,pdf,screenshot,dom,singlefile"
    chrome_binary: Optional[str] = None
    resolution: str = "1440,2000"
    storage_strategy: str = "symlink"  # symlink or copy
    feature_flag: bool = False
    auto_cleanup_days: int = 365
    max_size_mb: int = 500

    @validator('data_dir')
    def ensure_data_dir_exists(cls, v):
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def extract_types_list(self) -> List[str]:
        """Get extract types as list."""
        return [t.strip() for t in self.extract_types.split(',') if t.strip()]


class ArchiveBoxService:
    """Service for ArchiveBox integration with CLI and Docker support."""

    def __init__(self, config: Optional[ArchiveBoxConfig] = None):
        self.config = config or self._load_config()
        self._docker_available = None
        self._archivebox_available = None

    def _load_config(self) -> ArchiveBoxConfig:
        """Load configuration from settings."""
        return ArchiveBoxConfig(
            enabled=settings.archivebox_enabled and settings.archivebox_feature_flag,
            data_dir=settings.archivebox_data_dir,
            docker_image=settings.archivebox_docker_image,
            timeout=settings.archivebox_timeout,
            only_new=settings.archivebox_only_new,
            extract_types=settings.archivebox_extract,
            chrome_binary=settings.archivebox_chrome_binary,
            resolution=settings.archivebox_resolution,
            storage_strategy=settings.archivebox_storage_strategy,
            feature_flag=settings.archivebox_feature_flag,
            auto_cleanup_days=settings.archivebox_auto_cleanup_days,
            max_size_mb=settings.archivebox_max_size_mb,
        )

    async def is_available(self) -> bool:
        """Check if ArchiveBox is available and configured."""
        if not self.config.enabled or not self.config.feature_flag:
            return False

        # Check Docker availability
        if await self._check_docker():
            return True

        # Check local ArchiveBox installation
        return await self._check_archivebox_cli()

    async def _check_docker(self) -> bool:
        """Check if Docker is available."""
        if self._docker_available is not None:
            return self._docker_available

        try:
            result = await asyncio.create_subprocess_exec(
                'docker', '--version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()
            self._docker_available = result.returncode == 0
        except Exception:
            self._docker_available = False

        return self._docker_available

    async def _check_archivebox_cli(self) -> bool:
        """Check if ArchiveBox CLI is available."""
        if self._archivebox_available is not None:
            return self._archivebox_available

        try:
            result = await asyncio.create_subprocess_exec(
                'archivebox', '--version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()
            self._archivebox_available = result.returncode == 0
        except Exception:
            self._archivebox_available = False

        return self._archivebox_available

    async def archive_url(self, request: ArchiveRequest) -> ArchiveResult:
        """Archive a single URL."""
        if not await self.is_available():
            raise RuntimeError("ArchiveBox is not available or disabled")

        start_time = datetime.utcnow()

        try:
            # Determine extraction method priority
            if await self._check_docker():
                result = await self._archive_with_docker(request)
            else:
                result = await self._archive_with_cli(request)

            # Calculate duration
            result.duration_seconds = (datetime.utcnow() - start_time).total_seconds()

            # Apply storage strategy
            if result.success and result.archive_path:
                result.archive_path = await self._apply_storage_strategy(result.archive_path)

            logger.info(f"Archived {request.url} in {result.duration_seconds:.2f}s: {result.status}")
            return result

        except Exception as e:
            logger.error(f"Failed to archive {request.url}: {e}")
            return ArchiveResult(
                url=request.url,
                timestamp=start_time.isoformat(),
                status="failed",
                error_message=str(e),
                duration_seconds=(datetime.utcnow() - start_time).total_seconds()
            )

    async def _archive_with_docker(self, request: ArchiveRequest) -> ArchiveResult:
        """Archive URL using Docker container."""
        cmd = [
            'docker', 'run', '--rm',
            '-v', f'{self.config.data_dir}:/data',
            '-e', f'TIMEOUT={request.timeout or self.config.timeout}',
            '-e', f'ONLY_NEW={"True" if request.only_new else "False"}',
            '-e', f'OVERWRITE={"True" if request.overwrite else "False"}',
        ]

        # Add extract types
        if request.extract_types:
            extract_types = ','.join(request.extract_types)
        else:
            extract_types = self.config.extract_types

        for extract_type in extract_types.split(','):
            extract_type = extract_type.strip().upper()
            if extract_type:
                cmd.extend(['-e', f'SAVE_{extract_type}=True'])

        # Add Chrome settings
        if self.config.chrome_binary:
            cmd.extend(['-e', f'CHROME_BINARY={self.config.chrome_binary}'])

        cmd.extend(['-e', f'RESOLUTION={self.config.resolution}'])

        # Add the Docker image and command
        cmd.extend([self.config.docker_image, 'archivebox', 'add', request.url])

        return await self._execute_archivebox_command(cmd, request)

    async def _archive_with_cli(self, request: ArchiveRequest) -> ArchiveResult:
        """Archive URL using local ArchiveBox CLI."""
        # Ensure data directory is set
        env = {
            'ARCHIVEBOX_DATA_DIR': str(self.config.data_dir),
            'TIMEOUT': str(request.timeout or self.config.timeout),
            'ONLY_NEW': 'True' if request.only_new else 'False',
            'OVERWRITE': 'True' if request.overwrite else 'False',
        }

        # Set extract types
        if request.extract_types:
            extract_types = request.extract_types
        else:
            extract_types = self.config.extract_types_list

        for extract_type in extract_types:
            env[f'SAVE_{extract_type.upper()}'] = 'True'

        if self.config.chrome_binary:
            env['CHROME_BINARY'] = self.config.chrome_binary

        env['RESOLUTION'] = self.config.resolution

        cmd = ['archivebox', 'add', request.url]

        return await self._execute_archivebox_command(cmd, request, env=env)

    async def _execute_archivebox_command(
        self,
        cmd: List[str],
        request: ArchiveRequest,
        env: Optional[Dict[str, str]] = None
    ) -> ArchiveResult:
        """Execute ArchiveBox command and parse results."""
        try:
            # Create subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(self.config.data_dir)
            )

            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=request.timeout or self.config.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise RuntimeError(f"Archive operation timed out after {request.timeout or self.config.timeout}s")

            # Parse output
            output = stdout.decode('utf-8', errors='ignore')
            error_output = stderr.decode('utf-8', errors='ignore')

            if process.returncode == 0:
                return await self._parse_archive_success(request.url, output, error_output)
            else:
                return ArchiveResult(
                    url=request.url,
                    timestamp=datetime.utcnow().isoformat(),
                    status="failed",
                    error_message=error_output or "Unknown error"
                )

        except Exception as e:
            return ArchiveResult(
                url=request.url,
                timestamp=datetime.utcnow().isoformat(),
                status="failed",
                error_message=str(e)
            )

    async def _parse_archive_success(
        self,
        url: str,
        stdout: str,
        stderr: str
    ) -> ArchiveResult:
        """Parse successful archive operation output."""
        timestamp = datetime.utcnow().isoformat()

        # Try to extract archive information from output
        title = None
        archive_path = None
        snapshot_id = None
        extractors = {}

        # Parse ArchiveBox output patterns
        lines = stdout.split('\n')
        for line in lines:
            if '√' in line or 'Success' in line.lower():
                # Extract title if available
                if 'title:' in line.lower():
                    title = line.split('title:', 1)[1].strip()

                # Extract snapshot ID pattern
                if 'snapshot' in line.lower() and len(line) > 10:
                    # Look for timestamp-like patterns
                    import re
                    match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', line)
                    if match:
                        snapshot_id = match.group(1)

        # Determine archive path
        if snapshot_id:
            archive_path = str(self.config.data_dir / "archive" / snapshot_id)
        else:
            # Fallback: look for newest directory
            archive_dir = self.config.data_dir / "archive"
            if archive_dir.exists():
                dirs = [d for d in archive_dir.iterdir() if d.is_dir()]
                if dirs:
                    newest_dir = max(dirs, key=lambda x: x.stat().st_mtime)
                    archive_path = str(newest_dir)
                    snapshot_id = newest_dir.name

        # Calculate size if archive path exists
        size_bytes = 0
        if archive_path and Path(archive_path).exists():
            try:
                size_bytes = sum(f.stat().st_size for f in Path(archive_path).rglob('*') if f.is_file())
            except Exception:
                pass

        return ArchiveResult(
            url=url,
            timestamp=timestamp,
            title=title,
            archive_path=archive_path,
            snapshot_id=snapshot_id,
            status="success",
            extractors=extractors,
            size_bytes=size_bytes
        )

    async def _apply_storage_strategy(self, archive_path: str) -> str:
        """Apply configured storage strategy (symlink vs copy)."""
        if self.config.storage_strategy == "symlink":
            # Create symlink in shared location
            shared_dir = self.config.data_dir.parent / "shared_archives"
            shared_dir.mkdir(exist_ok=True)

            archive_name = Path(archive_path).name
            symlink_path = shared_dir / archive_name

            if not symlink_path.exists():
                symlink_path.symlink_to(Path(archive_path))

            return str(symlink_path)

        elif self.config.storage_strategy == "copy":
            # Copy to shared location for portability
            shared_dir = self.config.data_dir.parent / "shared_archives"
            shared_dir.mkdir(exist_ok=True)

            archive_name = Path(archive_path).name
            copy_path = shared_dir / archive_name

            if not copy_path.exists():
                shutil.copytree(archive_path, copy_path)

            return str(copy_path)

        # Default: return original path
        return archive_path

    async def get_archive_status(self, url: str) -> Optional[ArchiveResult]:
        """Get status of an archived URL."""
        if not await self.is_available():
            return None

        try:
            # Check using ArchiveBox list command
            if await self._check_docker():
                cmd = [
                    'docker', 'run', '--rm',
                    '-v', f'{self.config.data_dir}:/data',
                    self.config.docker_image,
                    'archivebox', 'list', '--json', '--filter-type=url', url
                ]
            else:
                cmd = ['archivebox', 'list', '--json', '--filter-type=url', url]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.config.data_dir)
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                output = stdout.decode('utf-8', errors='ignore')
                # Parse JSON output if available
                try:
                    data = json.loads(output)
                    if data and isinstance(data, list) and len(data) > 0:
                        entry = data[0]
                        return ArchiveResult(
                            url=url,
                            timestamp=entry.get('timestamp', ''),
                            title=entry.get('title'),
                            archive_path=entry.get('archive_path'),
                            snapshot_id=entry.get('timestamp'),
                            status="success"
                        )
                except json.JSONDecodeError:
                    pass

            return None

        except Exception as e:
            logger.error(f"Failed to get archive status for {url}: {e}")
            return None

    async def get_archive_content(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Get content from an archived snapshot."""
        if not await self.is_available():
            return None

        try:
            archive_path = self.config.data_dir / "archive" / snapshot_id
            if not archive_path.exists():
                return None

            content = {}

            # Read various content files
            content_files = {
                'title': 'title.txt',
                'html': 'output.html',
                'text': 'output.txt',
                'pdf': 'output.pdf',
                'screenshot': 'screenshot.png'
            }

            for content_type, filename in content_files.items():
                file_path = archive_path / filename
                if file_path.exists():
                    if content_type in ['title', 'html', 'text']:
                        # Text content
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content[content_type] = f.read()
                        except Exception:
                            continue
                    else:
                        # Binary content (just record path)
                        content[content_type + '_path'] = str(file_path)

            return content if content else None

        except Exception as e:
            logger.error(f"Failed to get archive content for {snapshot_id}: {e}")
            return None

    async def cleanup_old_archives(self) -> int:
        """Clean up old archives based on configuration."""
        if not self.config.auto_cleanup_days or not await self.is_available():
            return 0

        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.config.auto_cleanup_days)
            archive_dir = self.config.data_dir / "archive"

            if not archive_dir.exists():
                return 0

            cleaned_count = 0
            for snapshot_dir in archive_dir.iterdir():
                if not snapshot_dir.is_dir():
                    continue

                # Check modification time
                mtime = datetime.fromtimestamp(snapshot_dir.stat().st_mtime)
                if mtime < cutoff_date:
                    shutil.rmtree(snapshot_dir)
                    cleaned_count += 1

            logger.info(f"Cleaned up {cleaned_count} old archives older than {self.config.auto_cleanup_days} days")
            return cleaned_count

        except Exception as e:
            logger.error(f"Failed to cleanup old archives: {e}")
            return 0


# Utility functions for easier integration
async def archive_url_simple(url: str, user_id: Optional[int] = None) -> ArchiveResult:
    """Simple wrapper to archive a URL."""
    service = ArchiveBoxService()
    request = ArchiveRequest(url=url, user_id=user_id)
    return await service.archive_url(request)


async def get_archive_status_simple(url: str) -> Optional[ArchiveResult]:
    """Simple wrapper to check archive status."""
    service = ArchiveBoxService()
    return await service.get_archive_status(url)


# Global service instance
_archivebox_service: Optional[ArchiveBoxService] = None


def get_archivebox_service() -> ArchiveBoxService:
    """Get global ArchiveBox service instance."""
    global _archivebox_service
    if _archivebox_service is None:
        _archivebox_service = ArchiveBoxService()
    return _archivebox_service