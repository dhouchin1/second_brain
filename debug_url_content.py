#!/usr/bin/env python3
"""
Debug URL content creation and saving
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from web_extractor import WebContentExtractor
from services.url_ingestion_service import URLIngestionService

async def debug_content_creation():
    """Debug the content creation process"""
    print("🔍 Debugging URL content creation...")

    # Step 1: Test web extraction
    extractor = WebContentExtractor()
    url = "https://humanlayer.dev/code"

    print(f"📥 Extracting content from: {url}")
    extraction_result = await extractor.extract_content(url)

    print(f"✅ Extraction success: {extraction_result.success}")
    print(f"📝 Title: {extraction_result.title}")
    print(f"📄 Content length: {len(extraction_result.content) if extraction_result.content else 0}")
    print(f"📝 Text content length: {len(extraction_result.text_content) if extraction_result.text_content else 0}")

    # Step 2: Test content formatting
    url_service = URLIngestionService(None)  # We'll test without capture service
    source_text = "Test URL ingestion - should extract full content"

    formatted_content = url_service._create_web_note_content(extraction_result, source_text)

    print(f"\n📋 Formatted content length: {len(formatted_content)}")
    print(f"📋 Formatted content preview (first 500 chars):")
    print(formatted_content[:500])
    print("...")

    return formatted_content

if __name__ == "__main__":
    content = asyncio.run(debug_content_creation())