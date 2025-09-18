#!/bin/bash
# Second Brain - Application Deployment Script
# Deploys the application code and manages service lifecycle

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
APP_USER="secondbrain"
APP_GROUP="secondbrain"
APP_DIR="/var/www/secondbrain"
BACKUP_DIR="/var/backups/secondbrain"
DEPLOY_LOG="/var/log/secondbrain/deploy.log"

# Deployment options
ENVIRONMENT="${ENVIRONMENT:-production}"
SKIP_BACKUP="${SKIP_BACKUP:-false}"
SKIP_TESTS="${SKIP_TESTS:-false}"
FORCE_DEPLOY="${FORCE_DEPLOY:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$DEPLOY_LOG"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$DEPLOY_LOG"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$DEPLOY_LOG"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$DEPLOY_LOG"
}

# Check prerequisites
check_prerequisites() {
    log "Checking deployment prerequisites..."
    
    # Check if running as root or with sudo
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root (use sudo)"
    fi
    
    # Check if application user exists
    if ! getent passwd "$APP_USER" > /dev/null; then
        error "Application user '$APP_USER' does not exist. Run setup-server.sh first."
    fi
    
    # Check if application directory exists
    if [[ ! -d "$APP_DIR" ]]; then
        error "Application directory '$APP_DIR' does not exist. Run setup-server.sh first."
    fi
    
    # Check if project directory exists
    if [[ ! -d "$PROJECT_DIR" ]]; then
        error "Project directory '$PROJECT_DIR' not found"
    fi
    
    # Check if environment file exists
    ENV_FILE="/etc/secondbrain/${ENVIRONMENT}.env"
    if [[ ! -f "$ENV_FILE" ]]; then
        error "Environment file '$ENV_FILE' not found. Create it first."
    fi
    
    log "Prerequisites check passed"
}

# Validate configuration
validate_config() {
    log "Validating configuration..."
    
    ENV_FILE="/etc/secondbrain/${ENVIRONMENT}.env"
    
    # Run configuration validator
    if [[ -f "$PROJECT_DIR/deploy/config/config-validator.py" ]]; then
        python3 "$PROJECT_DIR/deploy/config/config-validator.py" "$ENV_FILE"
        if [[ $? -ne 0 ]]; then
            error "Configuration validation failed. Fix errors before deploying."
        fi
    else
        warn "Configuration validator not found, skipping validation"
    fi
    
    log "Configuration validation passed"
}

# Create backup
create_backup() {
    if [[ "$SKIP_BACKUP" == "true" ]]; then
        warn "Skipping backup as requested"
        return
    fi
    
    log "Creating backup..."
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_NAME="secondbrain_${ENVIRONMENT}_${TIMESTAMP}"
    BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"
    
    mkdir -p "$BACKUP_PATH"
    
    # Backup application code (excluding virtual environment and logs)
    if [[ -d "$APP_DIR" ]]; then
        rsync -av --exclude='.venv' --exclude='__pycache__' \
              --exclude='*.log' --exclude='logs' \
              "$APP_DIR/" "$BACKUP_PATH/app/"
    fi
    
    # Backup database
    if [[ -f "$APP_DIR/notes.db" ]]; then
        cp "$APP_DIR/notes.db" "$BACKUP_PATH/"
    fi
    if [[ -f "$APP_DIR/brain.db" ]]; then
        cp "$APP_DIR/brain.db" "$BACKUP_PATH/"
    fi
    
    # Backup configuration
    cp -r /etc/secondbrain "$BACKUP_PATH/config/" 2>/dev/null || true
    
    # Create backup info
    cat > "$BACKUP_PATH/backup_info.txt" << EOF
Backup created: $(date)
Environment: $ENVIRONMENT
Git commit: $(cd "$PROJECT_DIR" && git rev-parse HEAD 2>/dev/null || echo "N/A")
Git branch: $(cd "$PROJECT_DIR" && git branch --show-current 2>/dev/null || echo "N/A")
EOF
    
    # Compress backup
    cd "$BACKUP_DIR"
    tar -czf "${BACKUP_NAME}.tar.gz" "$BACKUP_NAME"
    rm -rf "$BACKUP_NAME"
    
    # Keep only last 10 backups
    ls -t "${BACKUP_DIR}"/secondbrain_*.tar.gz | tail -n +11 | xargs -r rm
    
    log "Backup created: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
}

# Stop services
stop_services() {
    log "Stopping services..."
    
    # Stop application services
    systemctl stop secondbrain-worker || true
    systemctl stop secondbrain || true
    
    # Give services time to stop gracefully
    sleep 5
    
    log "Services stopped"
}

# Deploy application code
deploy_code() {
    log "Deploying application code..."
    
    # Create temporary deployment directory
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT
    
    # Copy project files to temp directory
    rsync -av --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
          --exclude='*.pyc' --exclude='*.log' --exclude='logs' \
          --exclude='audio' --exclude='uploads' --exclude='vault' \
          "$PROJECT_DIR/" "$TEMP_DIR/"
    
    # Set ownership
    chown -R "$APP_USER:$APP_GROUP" "$TEMP_DIR"
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "$APP_DIR/.venv" ]]; then
        log "Creating virtual environment..."
        sudo -u "$APP_USER" python3 -m venv "$APP_DIR/.venv"
    fi
    
    # Install/update Python dependencies
    log "Installing Python dependencies..."
    sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install --upgrade pip setuptools wheel
    sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install -r "$TEMP_DIR/requirements.txt"
    sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install gunicorn uvicorn[standard]
    
    # Sync code to application directory
    log "Syncing application code..."
    rsync -av --delete --exclude='.venv' --exclude='*.db' \
          --exclude='audio' --exclude='uploads' --exclude='vault' \
          --exclude='logs' "$TEMP_DIR/" "$APP_DIR/"
    
    # Ensure correct ownership
    chown -R "$APP_USER:$APP_GROUP" "$APP_DIR"
    
    # Create required directories
    sudo -u "$APP_USER" mkdir -p "$APP_DIR/audio"
    sudo -u "$APP_USER" mkdir -p "$APP_DIR/uploads"
    sudo -u "$APP_USER" mkdir -p "$APP_DIR/vault"
    sudo -u "$APP_USER" mkdir -p "$APP_DIR/static"
    sudo -u "$APP_USER" mkdir -p "$APP_DIR/logs"
    
    log "Application code deployed"
}

# Run database migrations
run_migrations() {
    log "Running database migrations..."
    
    cd "$APP_DIR"
    
    # Set environment
    export ENVIRONMENT="$ENVIRONMENT"
    source "/etc/secondbrain/${ENVIRONMENT}.env"
    
    # Run migrations as application user
    sudo -u "$APP_USER" -E "$APP_DIR/.venv/bin/python" migrate_db.py
    
    log "Database migrations completed"
}

# Install systemd services
install_services() {
    log "Installing systemd services..."
    
    # Copy service files
    cp "$APP_DIR/deploy/systemd/secondbrain.service" /etc/systemd/system/
    cp "$APP_DIR/deploy/systemd/secondbrain-worker.service" /etc/systemd/system/
    
    # Reload systemd
    systemctl daemon-reload
    
    # Enable services
    systemctl enable secondbrain
    systemctl enable secondbrain-worker
    
    log "Systemd services installed and enabled"
}

# Configure NGINX
configure_nginx() {
    log "Configuring NGINX..."
    
    # Copy NGINX configuration files
    cp "$APP_DIR/deploy/nginx/nginx.conf" /etc/nginx/nginx.conf
    cp "$APP_DIR/deploy/nginx/secondbrain.conf" /etc/nginx/sites-available/
    cp "$APP_DIR/deploy/nginx/ssl.conf" /etc/nginx/conf.d/
    cp "$APP_DIR/deploy/nginx/security-headers.conf" /etc/nginx/conf.d/
    cp "$APP_DIR/deploy/nginx/rate-limiting.conf" /etc/nginx/conf.d/
    
    # Enable site
    ln -sf /etc/nginx/sites-available/secondbrain.conf /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default
    
    # Test NGINX configuration
    nginx -t
    
    # Create required directories
    mkdir -p /var/www/secondbrain/static
    mkdir -p /var/www/letsencrypt
    mkdir -p /var/www/secondbrain/error_pages
    
    # Set ownership
    chown -R "$APP_USER:$APP_GROUP" /var/www/secondbrain/static
    
    log "NGINX configured"
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        warn "Skipping tests as requested"
        return
    fi
    
    log "Running tests..."
    
    cd "$APP_DIR"
    export ENVIRONMENT="test"
    
    # Run unit tests
    if [[ -d "tests" ]]; then
        sudo -u "$APP_USER" "$APP_DIR/.venv/bin/python" -m pytest tests/ -v
        if [[ $? -ne 0 ]]; then
            error "Tests failed. Fix issues before deploying."
        fi
    else
        warn "No tests directory found, skipping tests"
    fi
    
    log "Tests passed"
}

# Start services
start_services() {
    log "Starting services..."
    
    # Start application services
    systemctl start secondbrain
    systemctl start secondbrain-worker
    
    # Reload NGINX
    systemctl reload nginx
    
    # Wait for services to start
    sleep 10
    
    # Check service status
    if ! systemctl is-active --quiet secondbrain; then
        error "Main application service failed to start"
    fi
    
    if ! systemctl is-active --quiet secondbrain-worker; then
        error "Worker service failed to start"
    fi
    
    log "Services started successfully"
}

# Health check
health_check() {
    log "Performing health check..."
    
    # Wait for application to be ready
    sleep 15
    
    # Check health endpoint
    MAX_ATTEMPTS=10
    ATTEMPT=1
    
    while [[ $ATTEMPT -le $MAX_ATTEMPTS ]]; do
        if curl -f -s http://localhost:8082/health > /dev/null; then
            log "Health check passed (attempt $ATTEMPT/$MAX_ATTEMPTS)"
            break
        fi
        
        warn "Health check failed (attempt $ATTEMPT/$MAX_ATTEMPTS), retrying..."
        sleep 5
        ((ATTEMPT++))
    done
    
    if [[ $ATTEMPT -gt $MAX_ATTEMPTS ]]; then
        error "Health check failed after $MAX_ATTEMPTS attempts"
    fi
    
    log "Application is healthy"
}

# Post-deployment tasks
post_deploy() {
    log "Running post-deployment tasks..."
    
    # Update search index
    cd "$APP_DIR"
    export ENVIRONMENT="$ENVIRONMENT"
    source "/etc/secondbrain/${ENVIRONMENT}.env"
    
    # Rebuild search index in background
    nohup sudo -u "$APP_USER" -E "$APP_DIR/.venv/bin/python" -c "
from services.search_index import SearchIndexer
indexer = SearchIndexer()
indexer.rebuild_all(embeddings=False)
print('Search index updated')
" >> "$DEPLOY_LOG" 2>&1 &
    
    # Clean up old log files
    find /var/log/secondbrain -name "*.log*" -mtime +30 -delete 2>/dev/null || true
    
    log "Post-deployment tasks completed"
}

# Rollback function
rollback() {
    error "Deployment failed. Initiating rollback..."
    
    # Find latest backup
    LATEST_BACKUP=$(ls -t "${BACKUP_DIR}"/secondbrain_*.tar.gz 2>/dev/null | head -n1)
    
    if [[ -n "$LATEST_BACKUP" ]]; then
        log "Rolling back to: $LATEST_BACKUP"
        
        # Extract backup
        cd "$BACKUP_DIR"
        BACKUP_NAME=$(basename "$LATEST_BACKUP" .tar.gz)
        tar -xzf "$LATEST_BACKUP"
        
        # Restore application code
        rsync -av --delete "$BACKUP_DIR/$BACKUP_NAME/app/" "$APP_DIR/"
        chown -R "$APP_USER:$APP_GROUP" "$APP_DIR"
        
        # Restore database
        if [[ -f "$BACKUP_DIR/$BACKUP_NAME/notes.db" ]]; then
            cp "$BACKUP_DIR/$BACKUP_NAME/notes.db" "$APP_DIR/"
        fi
        
        # Restart services
        systemctl restart secondbrain
        systemctl restart secondbrain-worker
        
        # Clean up
        rm -rf "$BACKUP_DIR/$BACKUP_NAME"
        
        log "Rollback completed"
    else
        warn "No backup found for rollback"
    fi
}

# Main deployment function
main() {
    log "Starting deployment for environment: $ENVIRONMENT"
    
    # Set up error handling
    trap rollback ERR
    
    # Create log file
    mkdir -p "$(dirname "$DEPLOY_LOG")"
    touch "$DEPLOY_LOG"
    chown "$APP_USER:$APP_GROUP" "$DEPLOY_LOG"
    
    # Run deployment steps
    check_prerequisites
    validate_config
    create_backup
    run_tests
    stop_services
    deploy_code
    run_migrations
    install_services
    configure_nginx
    start_services
    health_check
    post_deploy
    
    log "Deployment completed successfully!"
    log "Application is running at: http://localhost:8082"
    log "Deployment log: $DEPLOY_LOG"
    
    # Show service status
    info "Service status:"
    systemctl status secondbrain --no-pager -l
    systemctl status secondbrain-worker --no-pager -l
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        rollback
        ;;
    "health-check")
        health_check
        ;;
    *)
        echo "Usage: $0 [deploy|rollback|health-check]"
        exit 1
        ;;
esac