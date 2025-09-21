"""Tests for ingestion pipeline components."""

import hashlib
from pathlib import Path

import pytest

from config import settings
from file_processor import FileProcessor
from services.ingestion_queue import IngestionQueue, IngestionJobType, IngestionJobStatus


@pytest.fixture()
def isolated_upload_dirs(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    audio = tmp_path / "audio"
    uploads.mkdir()
    audio.mkdir()

    monkeypatch.setattr(settings, "uploads_dir", uploads)
    monkeypatch.setattr(settings, "audio_dir", audio)
    monkeypatch.setattr(settings, "max_file_size", 10 * 1024 * 1024)
    yield uploads, audio


def test_processed_file_includes_content_hash(isolated_upload_dirs, tmp_path):
    uploads_dir, _ = isolated_upload_dirs
    sample_file = tmp_path / "example.txt"
    sample_file.write_text("hello ingestion pipeline")

    processor = FileProcessor()
    result = processor.process_saved_file(sample_file, "example.txt")

    assert result["success"] is True
    stored_name = result["stored_filename"]
    assert stored_name is not None
    stored_path = uploads_dir / stored_name
    assert stored_path.exists()

    expected_hash = hashlib.sha256("hello ingestion pipeline".encode()).hexdigest()
    assert result["content_hash"] == expected_hash


def test_ingestion_queue_lifecycle(tmp_path):
    db_path = tmp_path / "jobs.db"
    queue = IngestionQueue(str(db_path))

    job = queue.enqueue(
        IngestionJobType.AUDIO_TRANSCRIPTION,
        note_id=42,
        user_id=7,
        payload={"stored_filename": "track.wav"},
        priority=2,
        content_hash="abc123",
    )

    assert job.status == IngestionJobStatus.QUEUED.value
    assert job.progress == 0

    next_job = queue.get_next_job()
    assert next_job is not None
    assert next_job.id == job.id
    assert queue.get_job(job.id).status == IngestionJobStatus.PROCESSING.value

    queue.mark_progress(job.id, 55, "halfway")
    progressed = queue.get_job(job.id)
    assert progressed.progress == 55
    assert progressed.status_detail == "halfway"

    queue.mark_complete(job.id, success=True)
    completed = queue.get_job(job.id)
    assert completed.status == IngestionJobStatus.COMPLETED.value
    assert completed.progress == 100


def test_mark_started_promotes_queued_job(tmp_path):
    db_path = tmp_path / "jobs.db"
    queue = IngestionQueue(str(db_path))

    job = queue.enqueue(IngestionJobType.NOTE_ENRICHMENT, note_id=9, user_id=3)
    queue.mark_started(job.id, "starting")
    started = queue.get_job(job.id)
    assert started.status == IngestionJobStatus.PROCESSING.value
    assert started.status_detail == "starting"
