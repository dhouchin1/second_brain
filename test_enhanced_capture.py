#!/usr/bin/env python3
"""
Integration test for enhanced capture infrastructure.

Tests the new Phase 1 infrastructure improvements:
- Unified error handling
- Advanced content processing pipeline  
- Enhanced configuration management
- Backward compatibility
"""

import asyncio
import sys
import tempfile
import sqlite3
from pathlib import Path
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from services.unified_capture_service import (
        UnifiedCaptureService, 
        UnifiedCaptureRequest,
        CaptureSourceType,
        CaptureContentType,
        enhanced_capture
    )
    from services.capture_error_handler import get_capture_error_handler
    from services.content_processing_pipeline import get_content_processing_pipeline
    from services.capture_config_manager import get_capture_config_manager
    from config import settings
    import sqlite3
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure you're running this from the project root directory")
    sys.exit(1)


def create_test_database():
    """Create a temporary test database."""
    db_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    db_path = db_file.name
    db_file.close()
    
    # Create basic schema
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            body TEXT,
            tags TEXT,
            metadata TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    ''')
    
    # Optional vector table
    try:
        cursor.execute('''
            CREATE TABLE note_vecs (
                note_id INTEGER PRIMARY KEY,
                embedding TEXT,
                FOREIGN KEY (note_id) REFERENCES notes (id)
            )
        ''')
    except:
        pass  # Vector table is optional
    
    conn.commit()
    conn.close()
    
    return db_path


def get_test_db_connection(db_path):
    """Get database connection for testing."""
    def get_conn():
        return sqlite3.connect(db_path)
    return get_conn


async def test_basic_capture(service, test_name):
    """Test basic capture functionality."""
    logger.info(f"Testing {test_name}...")
    
    request = UnifiedCaptureRequest(
        content_type=CaptureContentType.TEXT,
        source_type=CaptureSourceType.API,
        primary_content="This is a test note for the enhanced capture system. It contains some text that should be processed correctly.",
        metadata={"test": True}
    )
    
    result = await service.unified_capture(request)
    
    if result.success:
        logger.info(f"✅ {test_name} - Success! Note ID: {result.note_id}")
        logger.info(f"   Title: {result.title}")
        logger.info(f"   Tags: {result.tags}")
        logger.info(f"   Processing time: {result.processing_time:.2f}s")
        return True
    else:
        logger.error(f"❌ {test_name} - Failed: {result.error}")
        return False


async def test_enhanced_processing(service, test_name):
    """Test enhanced processing with larger content."""
    logger.info(f"Testing {test_name}...")
    
    # Create content that will trigger enhanced processing
    large_content = """
    # Project Planning Document
    
    ## Overview
    This is a comprehensive project planning document that contains multiple sections and should trigger the enhanced processing pipeline with content chunking and AI processing.
    
    ## Goals and Objectives
    - Implement new capture infrastructure
    - Ensure backward compatibility
    - Improve user experience
    - Add intelligent content processing
    
    ## Technical Requirements
    The system needs to handle various content types including text, images, PDFs, and audio files. It should provide intelligent chunking, quality validation, and error handling.
    
    ## Implementation Plan
    1. Phase 1: Core Infrastructure
       - Error handling system
       - Processing pipeline
       - Configuration management
    
    2. Phase 2: User Interface
       - Enhanced capture forms
       - Progress indicators
       - Configuration UI
    
    3. Phase 3: Advanced Features
       - Smart templates
       - Automation workflows
       - Analytics dashboard
    
    ## Success Criteria
    The system should process content reliably, provide good user feedback, and maintain high quality standards while being performant and scalable.
    """
    
    request = UnifiedCaptureRequest(
        content_type=CaptureContentType.TEXT,  # Use TEXT instead of MARKDOWN
        source_type=CaptureSourceType.API,
        primary_content=large_content,
        metadata={"test": True, "type": "enhanced"},
        processing_priority=2  # High priority to trigger enhanced processing
    )
    
    result = await service.unified_capture(request)
    
    if result.success:
        logger.info(f"✅ {test_name} - Success! Note ID: {result.note_id}")
        logger.info(f"   Title: {result.title}")
        logger.info(f"   Summary: {result.summary[:100] if result.summary else 'None'}...")
        logger.info(f"   Tags: {result.tags}")
        logger.info(f"   Processing time: {result.processing_time:.2f}s")
        logger.info(f"   Warnings: {len(result.warnings)} warnings")
        return True
    else:
        logger.error(f"❌ {test_name} - Failed: {result.error}")
        if result.warnings:
            logger.warning(f"   Warnings: {result.warnings}")
        return False


async def test_error_handling(service, test_name):
    """Test error handling with invalid content."""
    logger.info(f"Testing {test_name}...")
    
    # Test with empty content
    request = UnifiedCaptureRequest(
        content_type=CaptureContentType.TEXT,
        source_type=CaptureSourceType.API,
        primary_content="",  # Empty content should be handled gracefully
        metadata={"test": True, "type": "error_test"}
    )
    
    result = await service.unified_capture(request)
    
    # We expect this to either succeed with a warning or fail gracefully
    if result.success:
        logger.info(f"✅ {test_name} - Handled gracefully! Note ID: {result.note_id}")
        if result.warnings:
            logger.info(f"   Warnings: {result.warnings}")
        return True
    else:
        logger.info(f"✅ {test_name} - Failed gracefully with: {result.error}")
        return True  # Graceful failure is success for this test


async def test_configuration_management(service, test_name):
    """Test configuration management."""
    logger.info(f"Testing {test_name}...")
    
    try:
        # Test getting configuration info
        config_info = service.get_configuration_info()
        
        if "presets" in config_info and "schema" in config_info:
            logger.info(f"✅ {test_name} - Configuration management working!")
            logger.info(f"   Available presets: {len(config_info['presets'])}")
            logger.info(f"   Schema sections: {list(config_info['schema'].keys())}")
            return True
        else:
            logger.error(f"❌ {test_name} - Missing expected configuration data")
            return False
            
    except Exception as e:
        logger.error(f"❌ {test_name} - Exception: {e}")
        return False


async def test_stats_and_monitoring(service, test_name):
    """Test statistics and monitoring capabilities."""
    logger.info(f"Testing {test_name}...")
    
    try:
        # Get basic stats
        stats = service.get_processing_stats()
        logger.info(f"   Basic stats: {stats['total_requests']} requests processed")
        
        # Get enhanced stats
        enhanced_stats = service.get_enhanced_processing_stats()
        logger.info(f"   Enhanced stats available: {'pipeline_stats' in enhanced_stats}")
        
        logger.info(f"✅ {test_name} - Statistics working!")
        return True
        
    except Exception as e:
        logger.error(f"❌ {test_name} - Exception: {e}")
        return False


async def test_convenience_function(db_path, test_name):
    """Test the convenience function for enhanced capture."""
    logger.info(f"Testing {test_name}...")
    
    try:
        def get_conn():
            return sqlite3.connect(db_path)
        
        result = await enhanced_capture(
            content="This is a test using the convenience function.",
            source_type="api",
            content_type="text",
            get_conn_func=get_conn,
            metadata={"convenience_test": True}
        )
        
        if result.success:
            logger.info(f"✅ {test_name} - Success! Note ID: {result.note_id}")
            return True
        else:
            logger.error(f"❌ {test_name} - Failed: {result.error}")
            return False
            
    except Exception as e:
        logger.error(f"❌ {test_name} - Exception: {e}")
        return False


async def test_component_isolation():
    """Test that individual components work in isolation."""
    logger.info("Testing component isolation...")
    
    tests_passed = 0
    total_tests = 3
    
    # Test error handler
    try:
        error_handler = get_capture_error_handler()
        stats = error_handler.get_retry_statistics()
        logger.info("✅ Error handler - Working in isolation")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ Error handler - Exception: {e}")
    
    # Test processing pipeline
    try:
        pipeline = get_content_processing_pipeline()
        pipeline_stats = pipeline.get_processing_statistics()
        logger.info("✅ Processing pipeline - Working in isolation")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ Processing pipeline - Exception: {e}")
    
    # Test config manager
    try:
        config_manager = get_capture_config_manager()
        presets = config_manager.get_available_presets()
        logger.info("✅ Configuration manager - Working in isolation")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ Configuration manager - Exception: {e}")
    
    return tests_passed == total_tests


async def main():
    """Run all tests."""
    logger.info("🚀 Starting Enhanced Capture Infrastructure Tests")
    logger.info("=" * 60)
    
    # Create test database
    db_path = create_test_database()
    logger.info(f"Created test database: {db_path}")
    
    try:
        # Test component isolation first
        component_test = await test_component_isolation()
        
        # Create service instance
        get_conn_func = get_test_db_connection(db_path)
        service = UnifiedCaptureService(get_conn_func)
        
        # Run integration tests
        tests = [
            (test_basic_capture, "Basic Capture"),
            (test_enhanced_processing, "Enhanced Processing"),
            (test_error_handling, "Error Handling"),
            (test_configuration_management, "Configuration Management"), 
            (test_stats_and_monitoring, "Stats and Monitoring"),
        ]
        
        results = []
        if component_test:
            results.append(True)
            logger.info("✅ Component Isolation - All components working")
        else:
            results.append(False)
            logger.error("❌ Component Isolation - Some components failed")
        
        for test_func, test_name in tests:
            try:
                result = await test_func(service, test_name)
                results.append(result)
            except Exception as e:
                logger.error(f"❌ {test_name} - Unexpected exception: {e}")
                results.append(False)
        
        # Test convenience function separately
        convenience_result = await test_convenience_function(db_path, "Convenience Function")
        results.append(convenience_result)
        
        # Print final results
        logger.info("=" * 60)
        logger.info("📊 TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(results)
        total = len(results)
        
        logger.info(f"Tests passed: {passed}/{total}")
        logger.info(f"Success rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED! Enhanced capture infrastructure is working correctly.")
            return_code = 0
        else:
            logger.warning("⚠️  Some tests failed. Check the logs above for details.")
            return_code = 1
        
        # Show some final stats
        try:
            stats = service.get_enhanced_processing_stats()
            logger.info(f"Total requests processed during testing: {stats.get('total_requests', 0)}")
            logger.info(f"Success rate: {stats.get('success_rate', 0):.1f}%")
        except:
            pass
            
    finally:
        # Cleanup test database
        try:
            Path(db_path).unlink()
            logger.info("Cleaned up test database")
        except:
            pass
    
    return return_code


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test runner failed: {e}")
        sys.exit(1)