import sqlite3
import time
from datetime import datetime
from typing import Optional
import asyncio

from llm_utils import ollama_summarize, ollama_generate_title
# Enhanced LLM service with Autom8 integration
try:
    from services.enhanced_llm_service import enhanced_ollama_summarize, enhanced_ollama_generate_title
    _ENHANCED_LLM_AVAILABLE = True
except ImportError:
    _ENHANCED_LLM_AVAILABLE = False
from config import settings
from audio_utils import transcribe_audio
from services.audio_queue import audio_queue
from services.ingestion_queue import (
    ingestion_queue,
    IngestionJobType,
    IngestionJobStatus,
)
try:
    # Optional realtime status broadcasting
    from realtime_status import status_manager  # type: ignore
    _REALTIME = True
except Exception:
    _REALTIME = False

# Import notification functions
try:
    from services.notification_service import (
        notify_processing_started, 
        notify_processing_completed, 
        notify_processing_failed,
        notify_note_created
    )
    from services.websocket_manager import get_connection_manager
    _NOTIFICATIONS_AVAILABLE = True
except ImportError:
    _NOTIFICATIONS_AVAILABLE = False

def _send_notification_sync(coro):
    """Helper to run async notification functions in sync context"""
    if not _NOTIFICATIONS_AVAILABLE:
        return
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create a task
            asyncio.create_task(coro)
        else:
            # If no loop is running, run the coroutine
            loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop, create new one
        try:
            asyncio.run(coro)
        except Exception as e:
            print(f"Failed to send notification: {e}")

def _send_websocket_notification_sync(user_id: int, notification_data: dict):
    """Helper to send WebSocket notification synchronously"""
    if not _NOTIFICATIONS_AVAILABLE:
        return
    try:
        manager = get_connection_manager()
        
        async def send_notification():
            await manager.send_to_user(user_id, {
                'type': 'notification',
                'notification': notification_data,
                'timestamp': datetime.now().isoformat()
            })
        
        _send_notification_sync(send_notification())
    except Exception as e:
        print(f"Failed to send WebSocket notification: {e}")

def _get_user_id_for_note(note_id: int) -> Optional[int]:
    """Get user_id for a note from the database"""
    try:
        conn = get_conn()
        cursor = conn.execute("SELECT user_id FROM notes WHERE id = ?", (note_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    except Exception:
        return None


def get_conn():
    return sqlite3.connect(str(settings.db_path))


def _update_ingestion_progress(note_id: int, progress: int, detail: str) -> None:
    try:
        job = ingestion_queue.get_latest_job_for_note(note_id)
        if job:
            ingestion_queue.mark_started(job.id, detail)
            ingestion_queue.mark_progress(job.id, progress, detail)
    except Exception:
        # Progress updates should never break processing
        pass


def _complete_ingestion_job(note_id: int, success: bool, error: Optional[str] = None) -> None:
    try:
        job = ingestion_queue.get_latest_job_for_note(note_id)
        if job and job.status in (IngestionJobStatus.PROCESSING.value, IngestionJobStatus.QUEUED.value):
            ingestion_queue.mark_complete(job.id, success=success, error=error)
    except Exception:
        pass


def process_note(note_id: int):
    conn: Optional[sqlite3.Connection] = None
    job_success = False
    error_message: Optional[str] = None

    try:
        conn = get_conn()
        c = conn.cursor()
        row = c.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        if not row:
            error_message = "Note not found"
            return

        cols = [d[0] for d in c.description]
        note = dict(zip(cols, row))
        content = note.get("content") or ""
        tags = note.get("tags", "")
        actions = note.get("actions", "")
        note_type = note.get("type", "note")
        audio_filename: Optional[str] = note.get("audio_filename")
        user_id = _get_user_id_for_note(note_id)

        _update_ingestion_progress(note_id, 10, "Preparing note for processing")

        if user_id:
            task_type = "audio transcription" if note_type == "audio" else "note processing"
            _send_notification_sync(notify_processing_started(
                user_id=user_id,
                task_type=task_type,
                task_id=str(note_id)
            ))

        if note_type == "audio" and audio_filename:
            audio_path = settings.audio_dir / audio_filename
            _update_ingestion_progress(note_id, 15, "Queued for transcription")

            try:
                c.execute("UPDATE notes SET status=? WHERE id=?", ("transcribing:0", note_id))
                conn.commit()
                _update_ingestion_progress(note_id, 20, "Starting transcription")
                if _REALTIME:
                    try:
                        loop = asyncio.get_running_loop()
                        asyncio.create_task(status_manager.emit_progress(note_id, "transcribing", 10, "Starting transcription"))
                    except RuntimeError:
                        asyncio.run(status_manager.emit_progress(note_id, "transcribing", 10, "Starting transcription"))
            except Exception:
                pass

            def _on_progress(done: int, total: int):
                pct = 20 + int((done / max(total, 1)) * 60)
                try:
                    c2 = get_conn().cursor()
                    c2.execute("UPDATE notes SET status=? WHERE id=?", (f"transcribing:{pct}", note_id))
                    c2.connection.commit()
                    c2.connection.close()
                    _update_ingestion_progress(note_id, pct, f"Segment {done}/{total}")
                    if _REALTIME:
                        try:
                            loop = asyncio.get_running_loop()
                            asyncio.create_task(status_manager.emit_progress(note_id, "transcribing", pct, f"Segment {done}/{total}"))
                        except RuntimeError:
                            asyncio.run(status_manager.emit_progress(note_id, "transcribing", pct, f"Segment {done}/{total}"))
                except Exception:
                    pass

            transcript, converted_name = transcribe_audio(audio_path, progress_cb=_on_progress)
            if transcript:
                content = transcript
                audio_filename = converted_name
            else:
                content = ""

        if _ENHANCED_LLM_AVAILABLE and settings.autom8_enabled and content:
            try:
                title_result = asyncio.run(enhanced_ollama_generate_title(content))
                title = title_result if title_result else "[No Title]"

                summary_result = asyncio.run(enhanced_ollama_summarize(content))
                result = summary_result
            except Exception as e:
                print(f"Enhanced LLM failed, falling back to Ollama: {e}")
                title = ollama_generate_title(content) if content else "[No Title]"
                result = ollama_summarize(content) if content else {"summary": "", "tags": [], "actions": []}
        else:
            title = ollama_generate_title(content) if content else "[No Title]"
            result = ollama_summarize(content) if content else {"summary": "", "tags": [], "actions": []}

        if not title or title.lower().startswith("untitled"):
            title = content.splitlines()[0][:60] if content else "[No Title]"
        summary = result.get("summary", "")
        ai_tags = result.get("tags", [])
        ai_actions = result.get("actions", [])
        tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
        tag_list.extend([t for t in ai_tags if t and t not in tag_list])
        tags = ",".join(tag_list)

        if ai_actions:
            formatted_actions = []
            for action in ai_actions:
                if isinstance(action, dict):
                    action_text = action.get('action', '')
                    description = action.get('description', '')
                    if action_text and description:
                        formatted_actions.append(f"• {action_text}: {description}")
                    elif action_text:
                        formatted_actions.append(f"• {action_text}")
                elif isinstance(action, str):
                    formatted_actions.append(f"• {action}")
            actions = "\n".join(formatted_actions)
        else:
            actions = ""

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _update_ingestion_progress(note_id, 90, "Saving results")

        c.execute(
            "UPDATE notes SET title=?, content=?, summary=?, tags=?, actions=?, status='complete', timestamp=?, audio_filename=? WHERE id=?",
            (title, content, summary, tags, actions, now, audio_filename, note_id),
        )
        c.execute(
            "INSERT INTO notes_fts(rowid, title, body, tags) VALUES (?, ?, ?, ?)",
            (note_id, title, content, tags),
        )
        conn.commit()

        audio_queue.mark_completed(note_id, success=True)

        job_success = True

        if user_id:
            task_type = "audio transcription" if note_type == "audio" else "note processing"
            result_data = {
                "title": title,
                "content_length": len(content),
                "tags_count": len([t for t in tags.split(",") if t.strip()]),
                "summary": summary[:100] + "..." if len(summary) > 100 else summary,
            }
            _send_notification_sync(notify_processing_completed(
                user_id=user_id,
                task_type=task_type,
                task_id=str(note_id),
                result=result_data,
            ))

    except Exception as exc:
        error_message = str(exc)
        raise
    finally:
        if conn:
            conn.close()
        _complete_ingestion_job(note_id, job_success, error_message)


def process_ingestion_queue():
    """Process queued ingestion jobs sequentially."""

    while True:
        job = ingestion_queue.get_next_job()
        if not job:
            break

        try:
            if job.note_id is None:
                ingestion_queue.mark_complete(job.id, success=False, error="Job missing note_id")
                continue

            job_type = IngestionJobType(job.job_type)
            if job_type in (IngestionJobType.AUDIO_TRANSCRIPTION, IngestionJobType.NOTE_ENRICHMENT):
                process_note(job.note_id)
            else:
                ingestion_queue.mark_complete(job.id, success=False, error=f"Unsupported job type: {job.job_type}")
        except Exception as exc:
            ingestion_queue.mark_complete(job.id, success=False, error=str(exc))



def run_worker(poll_interval: int = 5):
    """Worker that processes notes from FIFO queue with batch processing support"""
    while True:
        # Check if batch processing should be enabled
        if audio_queue.should_enable_batch_processing():
            # Process batch mode
            process_batch()
        else:
            # Process single item from FIFO queue
            next_item = audio_queue.get_next_for_processing()
            if next_item:
                note_id, user_id = next_item
                try:
                    process_note(note_id)
                except Exception as e:
                    print(f"Error processing note {note_id}: {e}")
                    # Mark as failed in queue
                    audio_queue.mark_completed(note_id, success=False)
                    
                    # Send processing failed notification
                    if user_id:
                        _send_notification_sync(notify_processing_failed(
                            user_id=user_id,
                            task_type="note processing",
                            task_id=str(note_id),
                            error=str(e)
                        ))
            else:
                # No items in queue, sleep
                time.sleep(poll_interval)


def process_audio_queue():
    """Process a single queued audio note if available.

    This function is intentionally lightweight so it can be scheduled via
    FastAPI BackgroundTasks. It fetches the next item from the FIFO queue
    and processes it using the existing `process_note` function.
    """
    next_item = audio_queue.get_next_for_processing()
    if not next_item:
        return
    note_id, user_id = next_item
    try:
        process_note(note_id)
    except Exception as e:
        # Ensure we don't leave the item stuck in 'processing'
        audio_queue.mark_completed(note_id, success=False)
        
        # Send processing failed notification
        if user_id:
            _send_notification_sync(notify_processing_failed(
                user_id=user_id,
                task_type="note processing",
                task_id=str(note_id),
                error=str(e)
            ))


def process_batch():
    """Process all queued items in batch mode"""
    conn = get_conn()
    cursor = conn.cursor()
    
    # Get all queued items in FIFO order
    cursor.execute("""
        SELECT q.note_id, q.user_id 
        FROM audio_processing_queue q
        JOIN notes n ON q.note_id = n.id
        WHERE q.status = 'queued'
        ORDER BY q.priority DESC, n.timestamp ASC
    """)
    
    queued_items = cursor.fetchall()
    conn.close()
    
    if not queued_items:
        return
    
    print(f"Starting batch processing of {len(queued_items)} items")
    
    for note_id, user_id in queued_items:
        # Mark as processing (this will be handled by get_next_for_processing)
        next_item = audio_queue.get_next_for_processing()
        if next_item and next_item[0] == note_id:
            try:
                process_note(note_id)
                print(f"Batch processed note {note_id}")
            except Exception as e:
                print(f"Error in batch processing note {note_id}: {e}")
                audio_queue.mark_completed(note_id, success=False)
        
        # Small delay between batch items to prevent system overload
        time.sleep(1)
    
    print(f"Completed batch processing of {len(queued_items)} items")


if __name__ == "__main__":
    run_worker()
