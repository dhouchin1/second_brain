# Autom8 Dashboard API Server

FastAPI-based web server providing REST API endpoints and WebSocket connectivity for the Autom8 real-time dashboard.

## Features

- **REST API Endpoints**: Complete set of endpoints for dashboard data
- **Real-time WebSocket Streaming**: Live event and metrics updates
- **Redis Integration**: Direct integration with Autom8's shared memory system
- **EventBus Integration**: Real-time event forwarding from the agent system
- **Built-in Dashboard**: HTML/CSS/JavaScript dashboard for testing and monitoring
- **CORS Support**: Ready for React frontend integration
- **Authentication Framework**: Basic authentication with JWT support ready
- **Performance Optimized**: Caching, connection pooling, and efficient data handling

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Server

Using the startup script (recommended):
```bash
python start_server.py --host 0.0.0.0 --port 8000 --reload
```

Or directly with uvicorn:
```bash
uvicorn autom8.interfaces.api.server:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the Dashboard

- **Dashboard**: http://localhost:8000/dashboard
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## API Endpoints

### System Information
- `GET /api/system/status` - Overall system health and metrics
- `GET /health` - Simple health check

### Agent Management
- `GET /api/agents/list` - Active agents and their status

### Model Performance
- `GET /api/models/status` - Available models and performance metrics

### Analytics
- `GET /api/complexity/stats` - Complexity analysis statistics
- `GET /api/routing/stats` - Model routing distribution and performance
- `GET /api/context/stats` - Context usage and optimization metrics

### WebSocket Endpoints
- `WS /ws/events` - Real-time event streaming
- `WS /ws/metrics` - Real-time system metrics updates

### Connection Information
- `GET /api/websocket/stats` - WebSocket connection statistics

## Configuration

The server uses Autom8's configuration system. Key settings:

```yaml
# autom8.yaml
api:
  host: "0.0.0.0"
  port: 8000
  cors_origins:
    - "http://localhost:3000"
    - "http://127.0.0.1:3000"

shared_memory:
  redis:
    host: "localhost"
    port: 6379

logging:
  level: "INFO"
  file: "./autom8-api.log"
```

## Startup Script Options

```bash
python start_server.py --help
```

Available options:
- `--host` - Host to bind to (default: 0.0.0.0)
- `--port` - Port to bind to (default: 8000)
- `--reload` - Enable auto-reload for development
- `--log-level` - Set log level (DEBUG, INFO, WARNING, ERROR)
- `--log-file` - Write logs to file
- `--workers` - Number of worker processes
- `--no-redis-check` - Skip Redis connection check

## Development

### Running in Development Mode

```bash
python start_server.py --reload --log-level DEBUG
```

### Testing the API

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test system status
curl http://localhost:8000/api/system/status

# Test agents list
curl http://localhost:8000/api/agents/list
```

### WebSocket Testing

You can test WebSocket endpoints using the built-in dashboard or tools like:

```javascript
// Events WebSocket
const eventsWS = new WebSocket('ws://localhost:8000/ws/events');
eventsWS.onmessage = (event) => {
    console.log('Event:', JSON.parse(event.data));
};

// Metrics WebSocket
const metricsWS = new WebSocket('ws://localhost:8000/ws/metrics');
metricsWS.onmessage = (event) => {
    console.log('Metrics:', JSON.parse(event.data));
};
```

## Architecture

### Components

1. **FastAPI Application** (`server.py`)
   - Main application with all endpoints
   - Middleware configuration
   - Authentication handling

2. **Data Service** (`data_service.py`)
   - Data aggregation and transformation
   - Caching layer for performance
   - Integration with Redis and EventBus

3. **WebSocket Manager**
   - Connection management
   - Real-time broadcasting
   - Connection health monitoring

4. **Static Dashboard**
   - HTML/CSS/JavaScript dashboard
   - Chart.js for visualizations
   - Real-time WebSocket integration

### Data Flow

```
Redis/EventBus → Data Service → API Endpoints → Frontend
                              ↓
                         WebSocket Manager → Real-time Updates
```

## Production Deployment

### Using Gunicorn

```bash
gunicorn autom8.interfaces.api.server:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY autom8/ ./autom8/
EXPOSE 8000

CMD ["python", "autom8/interfaces/api/start_server.py", "--host", "0.0.0.0"]
```

### Environment Variables

- `REDIS_URL` - Redis connection URL
- `LOG_LEVEL` - Logging level
- `DEBUG_MODE` - Enable debug mode
- `CORS_ORIGINS` - Allowed CORS origins

## Security

### Current Implementation

- Basic HTTP Bearer token authentication
- CORS configured for common frontend origins
- No sensitive information exposed in responses

### Production Recommendations

1. Implement proper JWT token validation
2. Use HTTPS with proper certificates
3. Configure rate limiting
4. Set up proper CORS origins
5. Add request/response validation
6. Implement audit logging

## Monitoring

The server provides comprehensive monitoring through:

1. **Health Check Endpoint**: `/health`
2. **System Metrics**: Real-time system status
3. **WebSocket Statistics**: Connection monitoring
4. **Structured Logging**: Detailed operation logs
5. **Error Tracking**: Automatic error capture and reporting

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   ```bash
   # Check Redis is running
   redis-cli ping
   
   # Start with Redis check disabled
   python start_server.py --no-redis-check
   ```

2. **Port Already in Use**
   ```bash
   # Use different port
   python start_server.py --port 8001
   ```

3. **Import Errors**
   ```bash
   # Ensure proper Python path
   export PYTHONPATH=/path/to/autom8_workflow:$PYTHONPATH
   ```

4. **WebSocket Connection Issues**
   - Check firewall settings
   - Verify CORS configuration
   - Test with built-in dashboard first

### Debug Mode

Enable detailed logging:
```bash
python start_server.py --log-level DEBUG --log-file api-debug.log
```

## Contributing

1. Follow FastAPI best practices
2. Add proper type hints
3. Include docstrings for all endpoints
4. Add tests for new functionality
5. Update this README for new features

## License

Part of the Autom8 project - see main project LICENSE file.