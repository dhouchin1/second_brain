import time
from datetime import datetime
from pathlib import Path
from database import get_conn
from llm_utils import ollama_summarize, ollama_generate_title, ollama_suggest_tags
from audio_utils import transcribe_audio
from config import settings

def process_note(note_id: int):
    """Process a note in the background"""
    conn = get_conn()
    c = conn.cursor()
    
    # Get the note
    row = c.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
    if not row:
        conn.close()
        return
    
    # Process based on type
    cols = [d[0] for d in c.description]
    note = dict(zip(cols, row))
    
    content = note.get("content", "")
    audio_filename = note.get("audio_filename")
    
    # If audio, transcribe it
    if note.get("type") == "audio" and audio_filename:
        audio_path = settings.audio_dir / audio_filename
        if audio_path.exists():
            transcript, _ = transcribe_audio(audio_path)
            if transcript:
                content = transcript
    
    # Generate title and summary
    if content:
        title = ollama_generate_title(content)
        summary = ollama_summarize(content)
        suggested_tags = ollama_suggest_tags(content)
        
        # Update the note
        c.execute(
            """UPDATE notes SET title=?, content=?, summary=?, tags=?, status='complete' 
               WHERE id=?""",
            (title, content, summary, ",".join(suggested_tags), note_id)
        )
        conn.commit()
    
    conn.close()

def run_worker():
    """Run background worker for processing tasks"""
    while True:
        conn = get_conn()
        c = conn.cursor()
        
        # Find pending notes
        row = c.execute(
            "SELECT id FROM notes WHERE status='pending' ORDER BY id LIMIT 1"
        ).fetchone()
        
        if row:
            note_id = row[0]
            print(f"Processing note {note_id}...")
            process_note(note_id)
        else:
            time.sleep(5)
        
        conn.close()

if __name__ == "__main__":
    print("🔄 Starting background worker...")
    run_worker()
