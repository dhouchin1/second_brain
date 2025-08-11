from pathlib import Path
import subprocess
from config import settings

def transcribe_audio(audio_path: Path):
    """Transcribe audio using Whisper.cpp"""
    wav_path = audio_path.with_suffix('.converted.wav')
    
    # Convert to WAV format
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", str(audio_path),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav_path)
    ]
    
    result = subprocess.run(ffmpeg_cmd, capture_output=True)
    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}")
        return "", None
    
    # Run Whisper transcription
    out_txt_path = wav_path.with_suffix('.txt')
    whisper_cmd = [
        str(settings.whisper_cpp_path),
        "-m", str(settings.whisper_model_path),
        "-f", str(wav_path),
        "-otxt",
    ]
    
    try:
        subprocess.run(whisper_cmd, capture_output=True, text=True)
        if out_txt_path.exists():
            return out_txt_path.read_text().strip(), wav_path.name
    except Exception as e:
        print(f"Whisper error: {e}")
    
    return "", wav_path.name
