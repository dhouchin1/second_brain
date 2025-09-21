"""
API interface for Autom8 system.

Provides REST API and WebSocket endpoints for external integrations,
including a complete dashboard web server with real-time data streaming.

Components:
- server.py: FastAPI application with REST endpoints and WebSocket handlers
- data_service.py: Data aggregation and caching service
- start_server.py: Production-ready startup script
- static/: Dashboard HTML, CSS, and JavaScript files
"""

from .server import app
from .data_service import DashboardDataService, get_data_service

__all__ = ['app', 'DashboardDataService', 'get_data_service']