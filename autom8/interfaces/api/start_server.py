#!/usr/bin/env python3
"""
Startup script for Autom8 Dashboard API Server

This script provides a convenient way to start the FastAPI server with proper
configuration, logging, and error handling.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import uvicorn

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from autom8.config.settings import get_settings
from autom8.interfaces.api.server import app

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)
    
    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("autom8").setLevel(getattr(logging, log_level.upper()))


def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import redis
        logger.info("Redis client available")
    except ImportError:
        logger.warning("Redis client not available - some features may not work")
    
    try:
        import fastapi
        import uvicorn
        logger.info("FastAPI and Uvicorn available")
    except ImportError as e:
        logger.error(f"Required dependency missing: {e}")
        sys.exit(1)


async def check_redis_connection():
    """Check Redis connection status."""
    try:
        from autom8.storage.redis.shared_memory import get_shared_memory
        shared_memory = await get_shared_memory()
        if shared_memory._initialized:
            logger.info("✅ Redis connection successful")
        else:
            logger.warning("⚠️  Redis connection failed - running in degraded mode")
    except Exception as e:
        logger.warning(f"⚠️  Redis check failed: {e} - running in degraded mode")


def main():
    """Main entry point for the server."""
    parser = argparse.ArgumentParser(description="Start Autom8 Dashboard API Server")
    parser.add_argument(
        "--host", 
        default="0.0.0.0", 
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to bind the server to (default: 8000)"
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level", 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)"
    )
    parser.add_argument(
        "--log-file", 
        help="Log file path (optional)"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=1, 
        help="Number of worker processes (default: 1)"
    )
    parser.add_argument(
        "--no-redis-check", 
        action="store_true", 
        help="Skip Redis connection check"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    logger.info("🚀 Starting Autom8 Dashboard API Server")
    logger.info(f"📊 Server will be available at http://{args.host}:{args.port}")
    logger.info(f"🎛️  Dashboard will be available at http://{args.host}:{args.port}/dashboard")
    
    # Check dependencies
    check_dependencies()
    
    # Check Redis connection if not skipped
    if not args.no_redis_check:
        try:
            asyncio.run(check_redis_connection())
        except Exception as e:
            logger.warning(f"Redis check failed: {e}")
    
    # Load settings
    try:
        settings = get_settings()
        logger.info("✅ Configuration loaded successfully")
        logger.info(f"🔧 Debug mode: {settings.debug_mode}")
        logger.info(f"📝 Log level: {settings.log_level}")
    except Exception as e:
        logger.warning(f"⚠️  Configuration load failed: {e} - using defaults")
    
    # Server configuration
    config = {
        "app": "autom8.interfaces.api.server:app",
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level.lower(),
        "reload": args.reload,
        "workers": args.workers if not args.reload else 1,  # Workers don't work with reload
        "access_log": args.log_level.upper() == "DEBUG",
        "server_header": False,  # Don't expose server info
        "date_header": False     # Don't expose date info
    }
    
    logger.info("🔧 Server configuration:")
    for key, value in config.items():
        if key != "app":
            logger.info(f"   {key}: {value}")
    
    try:
        logger.info("🎬 Starting server...")
        uvicorn.run(**config)
    except KeyboardInterrupt:
        logger.info("👋 Server shutdown requested")
    except Exception as e:
        logger.error(f"💥 Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()