#!/usr/bin/env python3
"""
Simple audio upload test without authentication
"""
import requests

BASE_URL = "http://localhost:8082"

print("=" * 60)
print("SIMPLE AUDIO UPLOAD TEST (No Auth)")
print("=" * 60)

# Create a minimal valid WAV file
wav_header = bytes([
    0x52, 0x49, 0x46, 0x46,  # "RIFF"
    0x24, 0x00, 0x00, 0x00,  # File size - 8
    0x57, 0x41, 0x56, 0x45,  # "WAVE"
    0x66, 0x6D, 0x74, 0x20,  # "fmt "
    0x10, 0x00, 0x00, 0x00,  # fmt chunk size
    0x01, 0x00,              # Audio format (PCM)
    0x01, 0x00,              # Channels (mono)
    0x40, 0x1F, 0x00, 0x00,  # Sample rate (8000 Hz)
    0x40, 0x1F, 0x00, 0x00,  # Byte rate
    0x01, 0x00,              # Block align
    0x08, 0x00,              # Bits per sample
    0x64, 0x61, 0x74, 0x61,  # "data"
    0x00, 0x00, 0x00, 0x00,  # Data size
])

print("\nTesting audio upload to /webhook/audio...")
files = {
    'file': ('test_audio.wav', wav_header, 'audio/wav')
}
data = {
    'tags': 'test,debug',
    'user_id': '1'  # Using user ID 1 (dan)
}

try:
    response = requests.post(
        f"{BASE_URL}/webhook/audio",
        files=files,
        data=data
    )

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        print("✅ Audio upload successful!")
        result = response.json()
        print(f"   Note ID: {result.get('id', 'N/A')}")
        print(f"   Status: {result.get('status', 'N/A')}")
        print(f"   Message: {result.get('message', 'N/A')}")
        print(f"   Filename: {result.get('filename', 'N/A')}")
    else:
        print(f"❌ Audio upload failed!")
        print(f"   Response: {response.text}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)
