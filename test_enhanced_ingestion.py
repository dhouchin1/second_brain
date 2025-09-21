#!/usr/bin/env python3
"""
Test script for enhanced ingestion capabilities.

Tests the integration of graph memory improvements, enhanced ArchiveBox ingestion,
improved Obsidian processing, and background file watching.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from graph_memory.service import get_graph_memory_service
from services.obsidian_ingester import get_obsidian_ingester
from services.background_ingestion import get_background_ingestion_service
from config import settings


async def test_graph_memory_extraction():
    """Test enhanced graph memory with Ollama-only extraction."""
    print("🧪 Testing enhanced graph memory extraction (Ollama only)...")

    service = get_graph_memory_service()

    # Test text for extraction
    test_text = """
    John Smith works as a software engineer at TechCorp.
    He lives in San Francisco and enjoys hiking.
    Mary Johnson is his manager and leads the development team.
    """

    try:
        result = service.ingest_text(
            text=test_text,
            title="Test Document",
            source_type="test",
            uri="test://document",
            mime="text/plain"
        )

        print(f"✅ Graph memory ingestion successful: {result}")

        # Test context building
        context = service.build_context_for_task("Who works at TechCorp?")
        print(f"✅ Context building successful: {len(context.get('facts', []))} facts found")

        return True

    except Exception as e:
        print(f"❌ Graph memory test failed: {e}")
        return False


async def test_ollama_model_selection():
    """Test Ollama model selection functionality."""
    print("🧪 Testing Ollama model selection...")

    from graph_memory.extractor import GraphFactExtractor

    try:
        extractor = GraphFactExtractor()

        # Test available models
        models = extractor._get_available_ollama_models()
        print(f"✅ Available models: {models}")

        # Test custom models from environment
        import os
        os.environ["OLLAMA_MODELS"] = "custom-model-1,custom-model-2"
        custom_models = extractor._get_available_ollama_models()
        print(f"✅ Custom models: {custom_models}")

        # Clean up
        if "OLLAMA_MODELS" in os.environ:
            del os.environ["OLLAMA_MODELS"]

        return len(models) > 0

    except Exception as e:
        print(f"❌ Ollama model selection test failed: {e}")
        return False


async def test_obsidian_ingester():
    """Test enhanced Obsidian vault ingester."""
    print("🧪 Testing enhanced Obsidian ingester...")

    try:
        ingester = get_obsidian_ingester()

        # Create temporary vault directory
        with tempfile.TemporaryDirectory() as temp_dir:
            vault_path = Path(temp_dir) / "vault"
            vault_path.mkdir()

            # Create test markdown files
            test_file1 = vault_path / "note1.md"
            test_file1.write_text("""---
title: Project Alpha
tags: [project, development]
created: 2024-01-01
---

This is a test note about Project Alpha.
It involves software development and team collaboration.
""")

            test_file2 = vault_path / "note2.md"
            test_file2.write_text("""---
title: Meeting Notes
tags: [meeting, planning]
---

Discussed project timelines and deliverables.
Team members include Alice and Bob.
""")

            # Test vault ingestion
            result = ingester.ingest_vault(vault_path, max_files=5)
            print(f"✅ Obsidian ingester result: {result}")

            return result.get("processed", 0) > 0

    except Exception as e:
        print(f"❌ Obsidian ingester test failed: {e}")
        return False


async def test_background_ingestion_service():
    """Test background ingestion service configuration."""
    print("🧪 Testing background ingestion service...")

    try:
        service = get_background_ingestion_service()

        # Test service configuration
        print(f"✅ Service configured with Obsidian vault: {service.obsidian_vault}")
        print(f"✅ Service configured with ArchiveBox root: {service.archivebox_root}")
        print(f"✅ Watchdog available: {service._watchdog_available}")
        print(f"✅ Service running: {service._running}")

        # Test rescan functions
        obsidian_result = await service.rescan_obsidian()
        print(f"✅ Obsidian rescan result: {obsidian_result}")

        archivebox_result = await service.rescan_archivebox()
        print(f"✅ ArchiveBox rescan result: {archivebox_result}")

        return True

    except Exception as e:
        print(f"❌ Background ingestion service test failed: {e}")
        return False


async def test_archivebox_content_extraction():
    """Test enhanced ArchiveBox content extraction."""
    print("🧪 Testing enhanced ArchiveBox content extraction...")

    try:
        from services.archivebox_worker import ArchiveBoxWorker

        worker = ArchiveBoxWorker()

        # Mock a successful archive result
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.title = "Test Archive"
        mock_result.snapshot_id = "2024-01-01_120000"
        mock_result.archive_path = "/tmp/test_archive"
        mock_result.timestamp = "2024-01-01T12:00:00Z"

        # Test content extraction
        content = await worker._extract_full_text_content(mock_result)
        print(f"✅ ArchiveBox content extraction successful: {len(content)} characters")

        return len(content) > 0

    except Exception as e:
        print(f"❌ ArchiveBox content extraction test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting enhanced ingestion tests...\n")

    tests = [
        ("Graph Memory Extraction", test_graph_memory_extraction),
        ("Ollama Model Selection", test_ollama_model_selection),
        ("Obsidian Ingester", test_obsidian_ingester),
        ("Background Ingestion Service", test_background_ingestion_service),
        ("ArchiveBox Content Extraction", test_archivebox_content_extraction),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)

        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"💥 Test {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if success:
            passed += 1

    print(f"\n📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Enhanced ingestion is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    # Set test configuration
    settings.graph_memory_enabled = True
    settings.graph_memory_extract_on_ingest = True

    # Run tests
    exit_code = asyncio.run(main())
    exit(exit_code)
