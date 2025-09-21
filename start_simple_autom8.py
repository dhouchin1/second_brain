#!/usr/bin/env python3
"""
Simple Autom8 Service Startup Script

Starts the AI routing microservice for Second Brain integration.
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def check_ollama_running():
    """Check if Ollama is running on localhost:11434."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_autom8_service():
    """Start the Autom8 service."""
    print("🔍 Checking dependencies...")

    # Check if Ollama is running
    if not check_ollama_running():
        print("⚠️  Warning: Ollama doesn't appear to be running on localhost:11434")
        print("   Start Ollama first for full functionality")
        print("   The service will still start but AI requests may fail")

    print("🚀 Starting Simple Autom8 AI Router...")

    # Start the service
    try:
        subprocess.run([
            sys.executable, "autom8_service.py"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Service stopped by user")
    except Exception as e:
        print(f"❌ Failed to start service: {e}")
        return False

    return True

if __name__ == "__main__":
    if not Path("autom8_service.py").exists():
        print("❌ autom8_service.py not found!")
        print("   Make sure you're running this from the correct directory")
        sys.exit(1)

    success = start_autom8_service()
    sys.exit(0 if success else 1)