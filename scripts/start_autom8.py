#!/usr/bin/env python3
"""
Startup script for Autom8 microservice within Second Brain

This script starts the Autom8 API server as a microservice alongside
Second Brain, providing intelligent AI model routing and cost optimization.
"""

import os
import sys
import subprocess
import time
import signal
import asyncio
import logging
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import settings

logger = logging.getLogger(__name__)

class Autom8Manager:
    """Manager for the Autom8 microservice."""

    def __init__(self, port: int = 8000, host: str = "localhost"):
        self.port = port
        self.host = host
        self.process: Optional[subprocess.Popen] = None
        self.running = False

    def start(self):
        """Start the Autom8 service."""
        logger.info(f"Starting Autom8 service on {self.host}:{self.port}")

        # Check if port is available
        if self._check_port_in_use():
            logger.warning(f"Port {self.port} appears to be in use")
            response = input(f"Port {self.port} is in use. Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return False

        # Set up environment
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root)

        # Start the Autom8 server
        cmd = [
            sys.executable,
            str(project_root / "autom8" / "interfaces" / "api" / "start_server.py"),
            "--host", self.host,
            "--port", str(self.port),
            "--log-level", "INFO"
        ]

        try:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            self.running = True

            logger.info(f"Autom8 service started with PID {self.process.pid}")
            logger.info(f"Dashboard available at: http://{self.host}:{self.port}/dashboard")
            logger.info(f"API documentation at: http://{self.host}:{self.port}/docs")

            return True

        except Exception as e:
            logger.error(f"Failed to start Autom8 service: {e}")
            return False

    def stop(self):
        """Stop the Autom8 service."""
        if self.process and self.running:
            logger.info("Stopping Autom8 service...")
            self.process.terminate()

            try:
                # Wait up to 10 seconds for graceful shutdown
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Autom8 service didn't stop gracefully, killing...")
                self.process.kill()
                self.process.wait()

            self.running = False
            logger.info("Autom8 service stopped")

    def restart(self):
        """Restart the Autom8 service."""
        self.stop()
        time.sleep(2)
        return self.start()

    def status(self):
        """Check service status."""
        if not self.running or not self.process:
            return "stopped"

        if self.process.poll() is None:
            return "running"
        else:
            self.running = False
            return "crashed"

    def _check_port_in_use(self) -> bool:
        """Check if the port is already in use."""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex((self.host, self.port))
                return result == 0
        except Exception:
            return False

    def follow_logs(self):
        """Stream logs from the Autom8 service."""
        if not self.process:
            logger.error("Autom8 service is not running")
            return

        logger.info("Following Autom8 logs (Ctrl+C to stop)...")
        try:
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    print(f"[Autom8] {line.rstrip()}")
        except KeyboardInterrupt:
            logger.info("Stopped following logs")

def setup_signal_handlers(manager: Autom8Manager):
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        manager.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def health_check(manager: Autom8Manager) -> bool:
    """Perform health check on Autom8 service."""
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://{manager.host}:{manager.port}/health",
                timeout=5.0
            )
            return response.status_code == 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Autom8 Microservice Manager")
    parser.add_argument("action", choices=["start", "stop", "restart", "status", "logs"],
                       help="Action to perform")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to run Autom8 on (default: 8000)")
    parser.add_argument("--host", default="localhost",
                       help="Host to bind to (default: localhost)")
    parser.add_argument("--daemon", action="store_true",
                       help="Run in daemon mode (start only)")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    manager = Autom8Manager(port=args.port, host=args.host)
    setup_signal_handlers(manager)

    if args.action == "start":
        if manager.start():
            if args.daemon:
                logger.info("Autom8 service started in daemon mode")
            else:
                logger.info("Autom8 service running. Press Ctrl+C to stop.")
                try:
                    # Wait for the process to finish
                    if manager.process:
                        manager.process.wait()
                except KeyboardInterrupt:
                    logger.info("Stopping service...")
                finally:
                    manager.stop()
        else:
            sys.exit(1)

    elif args.action == "stop":
        manager.stop()

    elif args.action == "restart":
        if manager.restart():
            logger.info("Autom8 service restarted successfully")
        else:
            sys.exit(1)

    elif args.action == "status":
        status = manager.status()
        print(f"Autom8 service status: {status}")

        if status == "running":
            # Perform health check
            healthy = asyncio.run(health_check(manager))
            print(f"Health check: {'✅ Healthy' if healthy else '❌ Unhealthy'}")
            sys.exit(0)
        else:
            sys.exit(1)

    elif args.action == "logs":
        manager.follow_logs()

if __name__ == "__main__":
    main()