#!/usr/bin/env python3
"""
Test URL ingestion service directly and create a proper note
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.url_ingestion_service import get_url_ingestion_service
from services.unified_capture_service import get_unified_capture_service
from database import get_db_connection

async def test_url_ingestion_service():
    """Test URL ingestion service directly"""
    print("🌐 Testing URL ingestion service...")

    try:
        # Get services
        capture_service = get_unified_capture_service(get_db_connection)
        url_service = get_url_ingestion_service(capture_service)

        test_url = "https://humanlayer.dev/code"
        user_id = 1  # Test user ID

        print(f"📥 Ingesting URL: {test_url}")
        result = await url_service.ingest_url(
            url=test_url,
            user_id=user_id,
            source_text="Test URL ingestion - should extract full content"
        )

        print(f"✅ Success: {result['success']}")

        if result['success']:
            print(f"📝 Note ID: {result.get('note_id')}")
            print(f"📖 Title: {result.get('title')}")
            print(f"🔗 Content Type: {result.get('content_type')}")
            print(f"📊 Word Count: {result.get('word_count')}")
            print(f"🖼️ Screenshot: {result.get('screenshot_path')}")

            # Check the database to see what was actually saved
            note_id = result.get('note_id')
            if note_id:
                import sqlite3
                conn = sqlite3.connect('notes.db')
                cursor = conn.cursor()

                cursor.execute("SELECT title, content, summary, tags FROM notes WHERE id = ?", (note_id,))
                row = cursor.fetchone()

                if row:
                    title, content, summary, tags = row
                    print(f"\n📋 Database Content:")
                    print(f"   Title: {title}")
                    print(f"   Content length: {len(content) if content else 0}")
                    print(f"   Content preview: {content[:200] if content else 'None'}...")
                    print(f"   Summary: {summary}")
                    print(f"   Tags: {tags}")

                conn.close()
        else:
            print(f"❌ Error: {result.get('error')}")

        return result['success']

    except Exception as e:
        print(f"❌ Exception during ingestion: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_url_ingestion_service())
    if success:
        print("\n✅ URL ingestion test passed!")
    else:
        print("\n❌ URL ingestion test failed!")