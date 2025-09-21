#!/usr/bin/env python3
"""
Test URL ingestion API with proper authentication
"""

import requests
import json

# First, let's test if we can login and get session cookie
session = requests.Session()

# Login first
login_data = {
    'username': 'user',  # Replace with actual username
    'password': 'password123'  # Replace with actual password
}

print("🔐 Attempting login...")
login_response = session.post('http://localhost:8082/login', data=login_data)
print(f"Login response status: {login_response.status_code}")

if login_response.status_code == 200:
    print("✅ Login successful")

    # Now test the URL ingestion endpoint
    url_data = {
        "url": "https://example.com",
        "context": "Test from API script",
        "auto_title": True,
        "auto_tags": True,
        "auto_summary": True
    }

    print("\n📥 Testing URL ingestion...")
    response = session.post(
        'http://localhost:8082/api/ingest/url',
        headers={'Content-Type': 'application/json'},
        data=json.dumps(url_data)
    )

    print(f"URL ingestion response status: {response.status_code}")
    print(f"Response: {response.text}")

    if response.status_code == 200:
        print("✅ URL ingestion successful!")
    else:
        print(f"❌ URL ingestion failed with status {response.status_code}")

else:
    print(f"❌ Login failed with status {login_response.status_code}")
    print(f"Response: {login_response.text}")