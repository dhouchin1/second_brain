#!/usr/bin/env python3
"""
Test script for enhanced audio processing improvements.
"""

import tempfile
import subprocess
from pathlib import Path
from audio_utils import _convert_to_wav_16k_mono, _select_best_whisper_model
from config import settings

def test_model_selection():
    """Test that the best available model is selected."""
    print("=== Testing Model Selection ===")

    selected_model = _select_best_whisper_model()
    print(f"Selected model: {selected_model.name}")
    print(f"Model size: {selected_model.stat().st_size // (1024*1024)} MB")

    # Check if we upgraded from base to medium
    if "medium" in selected_model.name:
        print("✅ SUCCESS: Upgraded to medium model for better quality")
        return True
    elif "base" in selected_model.name:
        print("⚠️  FALLBACK: Using base model (medium not available)")
        return True
    else:
        print("❌ ERROR: Unexpected model selected")
        return False

def test_audio_preprocessing():
    """Test audio preprocessing pipeline."""
    print("\n=== Testing Audio Preprocessing ===")

    # Create a simple test audio file (silent)
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        # Generate a 5-second silent audio file for testing
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-t", "5", "-c:a", "pcm_s16le", str(temp_path)
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            print("❌ ERROR: Could not generate test audio file")
            return False

        print(f"Generated test audio: {temp_path} ({temp_path.stat().st_size} bytes)")

        # Test the conversion function
        converted_path = _convert_to_wav_16k_mono(temp_path)

        if converted_path and converted_path.exists():
            print(f"✅ SUCCESS: Audio preprocessing completed")
            print(f"Converted file: {converted_path} ({converted_path.stat().st_size} bytes)")

            # Check for compressed playback file
            playback_path = temp_path.with_suffix('.playback.m4a')
            if playback_path.exists():
                print(f"✅ SUCCESS: Compressed playback file created: {playback_path} ({playback_path.stat().st_size} bytes)")
            else:
                print("⚠️  WARNING: Compressed playback file not found")

            # Cleanup
            temp_path.unlink(missing_ok=True)
            converted_path.unlink(missing_ok=True)
            playback_path.unlink(missing_ok=True)

            return True
        else:
            print("❌ ERROR: Audio preprocessing failed")
            temp_path.unlink(missing_ok=True)
            return False

    except Exception as e:
        print(f"❌ ERROR: Test failed with exception: {e}")
        return False

def test_configuration():
    """Test configuration improvements."""
    print("\n=== Testing Configuration ===")

    print(f"Transcription segment seconds: {settings.transcription_segment_seconds}")
    print(f"Processing timeout: {settings.processing_timeout_seconds}")
    print(f"Whisper model path: {settings.whisper_model_path}")

    # Check if settings are improved
    improvements = []
    if settings.transcription_segment_seconds <= 300:
        improvements.append("✅ Shorter segments for better quality")
    if settings.processing_timeout_seconds >= 3600:
        improvements.append("✅ Extended timeout for long recordings")

    if improvements:
        for improvement in improvements:
            print(improvement)
        return True
    else:
        print("⚠️  No configuration improvements detected")
        return False

def main():
    """Run all tests."""
    print("Testing Enhanced Audio Processing Improvements")
    print("=" * 50)

    tests = [
        test_model_selection,
        test_configuration,
        test_audio_preprocessing,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ ERROR in {test_func.__name__}: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("SUMMARY:")
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 ALL TESTS PASSED - Audio processing improvements are working!")
    else:
        print("⚠️  Some tests failed - check the output above")

    return passed == total

if __name__ == "__main__":
    main()