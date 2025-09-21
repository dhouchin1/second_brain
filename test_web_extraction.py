#!/usr/bin/env python3
"""
Test web content extraction directly
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from web_extractor import WebContentExtractor

async def test_web_extraction():
    """Test web content extraction"""
    print("🌐 Testing web content extraction...")

    extractor = WebContentExtractor()

    test_url = "https://humanlayer.dev/code"

    try:
        print(f"📥 Extracting content from: {test_url}")
        result = await extractor.extract_content(test_url)

        print(f"✅ Success: {result.success}")
        print(f"📝 Title: {result.title}")
        print(f"📄 Content length: {len(result.content) if result.content else 0}")
        print(f"📝 Text content length: {len(result.text_content) if result.text_content else 0}")
        print(f"🖼️ Screenshot: {result.screenshot_path}")
        print(f"🔧 Metadata: {result.metadata}")

        if result.error_message:
            print(f"❌ Error: {result.error_message}")

        if result.content:
            print(f"📄 Content preview (first 500 chars):\n{result.content[:500]}...")
        elif result.text_content:
            print(f"📝 Text content preview (first 500 chars):\n{result.text_content[:500]}...")

        return result.success

    except Exception as e:
        print(f"❌ Exception during extraction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_web_extraction())
    if success:
        print("\n✅ Web extraction test passed!")
    else:
        print("\n❌ Web extraction test failed!")