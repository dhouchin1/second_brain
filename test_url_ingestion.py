#!/usr/bin/env python3
"""
Test script for URL ingestion functionality
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.url_ingestion_service import get_url_ingestion_service
from services.unified_capture_service import get_unified_capture_service
from database import get_db_connection

async def test_url_detection():
    """Test URL detection in text"""
    print("🔍 Testing URL detection...")

    url_service = get_url_ingestion_service()

    test_text = """
    Check out these interesting articles:
    https://arxiv.org/pdf/2403.05530.pdf
    www.example.com/article
    Visit https://github.com/microsoft/playwright for more info
    """

    urls = url_service.detect_urls(test_text)
    print(f"✅ Detected {len(urls)} URLs:")
    for url in urls:
        print(f"   - {url}")

    return urls

async def test_url_ingestion():
    """Test URL ingestion workflow"""
    print("\n📥 Testing URL ingestion...")

    # Test with a simple webpage
    test_url = "https://example.com"

    try:
        capture_service = get_unified_capture_service(get_db_connection)
        url_service = get_url_ingestion_service(capture_service)

        print(f"🌐 Ingesting: {test_url}")
        result = await url_service.ingest_url(
            url=test_url,
            user_id=1,  # Test user ID
            source_text="Test URL ingestion from script"
        )

        if result["success"]:
            print(f"✅ Successfully ingested URL:")
            print(f"   - Note ID: {result.get('note_id')}")
            print(f"   - Title: {result.get('title')}")
            print(f"   - Content Type: {result.get('content_type')}")
            print(f"   - Word Count: {result.get('word_count')}")
        else:
            print(f"❌ Failed to ingest URL: {result.get('error')}")

        return result

    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        return {"success": False, "error": str(e)}

async def test_content_type_detection():
    """Test content type detection"""
    print("\n🔍 Testing content type detection...")

    url_service = get_url_ingestion_service()

    test_urls = [
        "https://example.com",
        "https://arxiv.org/pdf/2403.05530.pdf",
        "https://example.com/image.jpg",
        "https://example.com/document.docx"
    ]

    for url in test_urls:
        content_type = url_service._detect_content_type(url)
        print(f"   {url} -> {content_type}")

async def main():
    """Main test function"""
    print("🚀 Starting URL Ingestion Tests\n")

    try:
        # Test 1: URL Detection
        await test_url_detection()

        # Test 2: Content Type Detection
        await test_content_type_detection()

        # Test 3: URL Ingestion (only if services are available)
        await test_url_ingestion()

        print("\n✅ All tests completed!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())