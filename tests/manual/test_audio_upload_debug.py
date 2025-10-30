#!/usr/bin/env python3
"""
Debug script to test audio upload functionality
"""
import requests
import sys

# Configuration
BASE_URL = "http://localhost:8082"
USERNAME = "dan"  # Test username
PASSWORD = "test123"  # Test password (just reset)

def test_audio_upload():
    """Test the complete audio upload flow"""

    print("=" * 60)
    print("AUDIO UPLOAD DIAGNOSTIC TEST")
    print("=" * 60)

    # Step 1: Get CSRF token first
    print("\n1. Getting CSRF token...")
    session = requests.Session()

    # Visit login page to get CSRF token
    login_page = session.get(f"{BASE_URL}/login")
    csrf_token = session.cookies.get('csrf_token')

    if csrf_token:
        print(f"✅ CSRF token obtained: {csrf_token[:20]}...")
    else:
        print("⚠️  No CSRF token found, trying without it...")
        csrf_token = ""

    # Step 2: Login with CSRF token
    print("\n2. Testing login...")

    login_response = session.post(
        f"{BASE_URL}/login",
        data={
            "username": USERNAME,
            "password": PASSWORD,
            "csrf_token": csrf_token
        },
        allow_redirects=False
    )

    if login_response.status_code == 303 or login_response.status_code == 302:
        print("✅ Login successful!")
        print(f"   Session cookies: {list(session.cookies.keys())}")
    else:
        print(f"❌ Login failed with status {login_response.status_code}")
        print(f"   Response: {login_response.text[:500]}")
        return

    # Step 3: Test analytics endpoint
    print("\n3. Testing analytics endpoint...")
    analytics_response = session.get(f"{BASE_URL}/api/analytics")

    if analytics_response.status_code == 200:
        print("✅ Analytics endpoint working!")
        data = analytics_response.json()
        print(f"   Total notes: {data.get('total_notes', 'N/A')}")
        print(f"   This week: {data.get('this_week', 'N/A')}")
    else:
        print(f"❌ Analytics endpoint failed with status {analytics_response.status_code}")
        print(f"   Response: {analytics_response.text[:200]}")

    # Step 4: Test audio upload with a small test file
    print("\n4. Testing audio upload endpoint...")

    # Create a minimal valid audio file (silence)
    # This is a minimal valid WAV file (44 bytes header + 1 second of silence at 8kHz mono)
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

    files = {
        'file': ('test_audio.wav', wav_header, 'audio/wav')
    }
    data = {
        'tags': 'test,debug'
    }

    upload_response = session.post(
        f"{BASE_URL}/webhook/audio",
        files=files,
        data=data
    )

    if upload_response.status_code == 200:
        print("✅ Audio upload successful!")
        result = upload_response.json()
        print(f"   Note ID: {result.get('id', 'N/A')}")
        print(f"   Status: {result.get('status', 'N/A')}")
        print(f"   Message: {result.get('message', 'N/A')}")
    else:
        print(f"❌ Audio upload failed with status {upload_response.status_code}")
        print(f"   Response: {upload_response.text[:500]}")

    # Step 5: Check audio queue status
    print("\n5. Checking audio queue status...")
    queue_response = session.get(f"{BASE_URL}/api/audio-queue/status")

    if queue_response.status_code == 200:
        print("✅ Audio queue accessible!")
        queue_data = queue_response.json()
        print(f"   Queue data: {queue_data}")
    else:
        print(f"⚠️  Audio queue check failed with status {queue_response.status_code}")
        print(f"   Response: {queue_response.text[:200]}")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_audio_upload()
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
