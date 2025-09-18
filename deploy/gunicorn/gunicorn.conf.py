# Second Brain - Production Gunicorn Configuration
# Optimized for FastAPI with uvicorn workers

import multiprocessing
import os
from pathlib import Path

# Server socket
bind = "127.0.0.1:8082"
backlog = 2048

# Worker processes
workers = (multiprocessing.cpu_count() * 2) + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# Worker timeout and keepalive
timeout = 30
keepalive = 2
graceful_timeout = 30

# Process naming
proc_name = 'secondbrain'

# User and group (run as non-root)
# user = "www-data"
# group = "www-data"

# Directories
# chdir = "/var/www/secondbrain"
tmp_upload_dir = None

# Logging
log_level = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'
accesslog = "/var/log/secondbrain/gunicorn_access.log"
errorlog = "/var/log/secondbrain/gunicorn_error.log"
capture_output = True
enable_stdio_inheritance = True

# Process management
pidfile = "/var/run/secondbrain/gunicorn.pid"
daemon = False  # Don't daemonize when using systemd

# Security
limit_request_line = 8192
limit_request_fields = 100
limit_request_field_size = 8192

# SSL (if terminating SSL at application level instead of nginx)
# keyfile = "/etc/ssl/private/secondbrain.key"
# certfile = "/etc/ssl/certs/secondbrain.crt"

# Preload application for better memory usage
preload_app = True

# Environment variables
raw_env = [
    'ENVIRONMENT=production',
    'LOG_LEVEL=INFO',
]

# Hooks for application lifecycle
def on_starting(server):
    """Called just before the master process is initialized."""
    server.log.info("Starting Second Brain server...")

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    server.log.info("Reloading Second Brain server...")

def when_ready(server):
    """Called just after the server is started."""
    server.log.info("Second Brain server is ready. Listening on %s", server.address)

def worker_int(worker):
    """Called just after a worker exited on SIGINT or SIGQUIT."""
    worker.log.info("Worker %s killed", worker.pid)

def pre_fork(server, worker):
    """Called just before a worker is forked."""
    server.log.info("Forking worker %s", worker.pid)

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    worker.log.info("Worker %s spawned", worker.pid)

def post_worker_init(worker):
    """Called just after a worker has initialized the application."""
    worker.log.info("Worker %s initialized", worker.pid)

def worker_abort(worker):
    """Called when a worker received the SIGABRT signal."""
    worker.log.info("Worker %s aborted", worker.pid)

def pre_exec(server):
    """Called just before a new master process is forked."""
    server.log.info("Forked parent, pre_exec()")

def pre_request(worker, req):
    """Called just before a worker processes the request."""
    worker.log.debug("Processing request: %s %s", req.method, req.path)

def post_request(worker, req, environ, resp):
    """Called after a worker processes the request."""
    worker.log.debug("Request processed: %s %s - %s", req.method, req.path, resp.status)

def child_exit(server, worker):
    """Called just after a worker has been exited."""
    server.log.info("Worker %s exited", worker.pid)

def worker_exit(server, worker):
    """Called just after a worker has been exited."""
    server.log.info("Worker %s finished", worker.pid)

def nworkers_changed(server, new_value, old_value):
    """Called just after num_workers has been changed."""
    server.log.info("Worker count changed from %s to %s", old_value, new_value)

def on_exit(server):
    """Called just before exiting."""
    server.log.info("Shutting down Second Brain server...")

# Custom application configuration
def application(environ, start_response):
    """WSGI application for fallback."""
    status = '200 OK'
    headers = [('Content-Type', 'text/plain')]
    start_response(status, headers)
    return [b'Second Brain is running']