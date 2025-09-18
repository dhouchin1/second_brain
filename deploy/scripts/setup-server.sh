#!/bin/bash
# Second Brain - Initial Server Setup Script
# Sets up a fresh Ubuntu/Debian server for Second Brain deployment

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
APP_USER="secondbrain"
APP_GROUP="secondbrain"
APP_DIR="/var/www/secondbrain"
LOG_FILE="/var/log/secondbrain-setup.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Update system packages
update_system() {
    log "Updating system packages..."
    apt-get update
    apt-get upgrade -y
    apt-get install -y \
        curl \
        wget \
        git \
        build-essential \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        sqlite3 \
        nginx \
        supervisor \
        redis-server \
        ffmpeg \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        pkg-config \
        libssl-dev \
        libffi-dev \
        cmake \
        make \
        gcc \
        g++ \
        ufw \
        certbot \
        python3-certbot-nginx \
        htop \
        ncdu \
        tree \
        unzip
    
    log "System packages updated successfully"
}

# Create application user and directories
setup_user_and_dirs() {
    log "Setting up application user and directories..."
    
    # Create user and group
    if ! getent group "$APP_GROUP" > /dev/null 2>&1; then
        groupadd -r "$APP_GROUP"
    fi
    
    if ! getent passwd "$APP_USER" > /dev/null 2>&1; then
        useradd -r -g "$APP_GROUP" -s /bin/bash -d "$APP_DIR" "$APP_USER"
    fi
    
    # Create directory structure
    mkdir -p "$APP_DIR"
    mkdir -p /var/log/secondbrain
    mkdir -p /var/run/secondbrain
    mkdir -p /var/cache/secondbrain
    mkdir -p /etc/secondbrain
    mkdir -p /var/backups/secondbrain
    
    # Set ownership and permissions
    chown -R "$APP_USER:$APP_GROUP" "$APP_DIR"
    chown -R "$APP_USER:$APP_GROUP" /var/log/secondbrain
    chown -R "$APP_USER:$APP_GROUP" /var/run/secondbrain
    chown -R "$APP_USER:$APP_GROUP" /var/cache/secondbrain
    chown -R root:root /etc/secondbrain
    chmod 755 /etc/secondbrain
    
    log "User and directories created successfully"
}

# Install Python dependencies
setup_python() {
    log "Setting up Python environment..."
    
    # Ensure pip is up to date
    python3 -m pip install --upgrade pip setuptools wheel
    
    # Install system-wide packages
    pip3 install supervisor gunicorn
    
    log "Python environment setup complete"
}

# Build and install whisper.cpp
install_whisper_cpp() {
    log "Installing whisper.cpp..."
    
    cd /tmp
    if [ -d "whisper.cpp" ]; then
        rm -rf whisper.cpp
    fi
    
    git clone https://github.com/ggerganov/whisper.cpp.git
    cd whisper.cpp
    
    mkdir build
    cd build
    cmake ..
    make -j$(nproc)
    
    # Install binary
    cp bin/whisper /usr/local/bin/
    chmod +x /usr/local/bin/whisper
    
    # Create models directory and download base model
    mkdir -p /opt/models
    cd /tmp/whisper.cpp
    bash ./models/download-ggml-model.sh base.en
    cp models/ggml-base.en.bin /opt/models/
    
    # Set permissions
    chown -R "$APP_USER:$APP_GROUP" /opt/models
    
    log "whisper.cpp installed successfully"
}

# Setup Ollama
install_ollama() {
    log "Installing Ollama..."
    
    # Download and install Ollama
    curl -fsSL https://ollama.ai/install.sh | sh
    
    # Create ollama service user
    if ! getent passwd ollama > /dev/null 2>&1; then
        useradd -r -s /bin/false -d /usr/share/ollama ollama
    fi
    
    # Create systemd service
    cat > /etc/systemd/system/ollama.service << EOF
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="OLLAMA_HOST=127.0.0.1"

[Install]
WantedBy=default.target
EOF
    
    systemctl daemon-reload
    systemctl enable ollama
    systemctl start ollama
    
    # Wait for service to start
    sleep 10
    
    # Pull default model
    sudo -u ollama ollama pull llama3.2
    
    log "Ollama installed and configured"
}

# Configure Redis
configure_redis() {
    log "Configuring Redis..."
    
    # Backup original config
    cp /etc/redis/redis.conf /etc/redis/redis.conf.backup
    
    # Configure Redis for production
    sed -i 's/^# maxmemory <bytes>/maxmemory 512mb/' /etc/redis/redis.conf
    sed -i 's/^# maxmemory-policy noeviction/maxmemory-policy allkeys-lru/' /etc/redis/redis.conf
    sed -i 's/^save/#save/' /etc/redis/redis.conf
    echo 'save 900 1' >> /etc/redis/redis.conf
    echo 'save 300 10' >> /etc/redis/redis.conf
    echo 'save 60 10000' >> /etc/redis/redis.conf
    
    systemctl enable redis-server
    systemctl restart redis-server
    
    log "Redis configured successfully"
}

# Configure firewall
setup_firewall() {
    log "Configuring firewall..."
    
    # Reset UFW to default
    ufw --force reset
    
    # Set default policies
    ufw default deny incoming
    ufw default allow outgoing
    
    # Allow SSH (be careful not to lock yourself out)
    ufw allow ssh
    ufw allow 22/tcp
    
    # Allow HTTP and HTTPS
    ufw allow 80/tcp
    ufw allow 443/tcp
    
    # Allow specific application ports (only on localhost)
    ufw allow from 127.0.0.1 to any port 8082
    ufw allow from 127.0.0.1 to any port 6379
    ufw allow from 127.0.0.1 to any port 11434
    
    # Enable firewall
    ufw --force enable
    
    log "Firewall configured successfully"
}

# Configure system limits
configure_limits() {
    log "Configuring system limits..."
    
    # Set file limits for the application user
    cat >> /etc/security/limits.conf << EOF

# Second Brain application limits
$APP_USER soft nofile 65536
$APP_USER hard nofile 65536
$APP_USER soft nproc 4096
$APP_USER hard nproc 4096
EOF
    
    # Configure systemd limits
    mkdir -p /etc/systemd/system.conf.d
    cat > /etc/systemd/system.conf.d/secondbrain-limits.conf << EOF
[Manager]
DefaultLimitNOFILE=65536
DefaultLimitNPROC=4096
EOF
    
    log "System limits configured"
}

# Setup log rotation
setup_log_rotation() {
    log "Setting up log rotation..."
    
    cat > /etc/logrotate.d/secondbrain << EOF
/var/log/secondbrain/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0644 $APP_USER $APP_GROUP
    postrotate
        systemctl reload secondbrain || true
    endscript
}

/var/log/nginx/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 0644 www-data adm
    postrotate
        systemctl reload nginx || true
    endscript
}
EOF
    
    log "Log rotation configured"
}

# Create basic monitoring script
create_monitoring() {
    log "Setting up basic monitoring..."
    
    cat > /usr/local/bin/secondbrain-monitor << 'EOF'
#!/bin/bash
# Basic monitoring script for Second Brain

CHECK_URL="http://localhost:8082/health"
LOG_FILE="/var/log/secondbrain/monitor.log"

# Check if service is running
if ! systemctl is-active --quiet secondbrain; then
    echo "$(date): Service is down, attempting restart..." >> "$LOG_FILE"
    systemctl restart secondbrain
    sleep 30
fi

# Check if endpoint is responding
if ! curl -f -s "$CHECK_URL" > /dev/null; then
    echo "$(date): Health check failed, attempting restart..." >> "$LOG_FILE"
    systemctl restart secondbrain
fi
EOF
    
    chmod +x /usr/local/bin/secondbrain-monitor
    
    # Add to crontab for regular checks
    (crontab -l 2>/dev/null; echo "*/5 * * * * /usr/local/bin/secondbrain-monitor") | crontab -
    
    log "Monitoring script created"
}

# Main setup function
main() {
    log "Starting Second Brain server setup..."
    
    # Create log file
    touch "$LOG_FILE"
    
    check_root
    update_system
    setup_user_and_dirs
    setup_python
    install_whisper_cpp
    install_ollama
    configure_redis
    setup_firewall
    configure_limits
    setup_log_rotation
    create_monitoring
    
    log "Server setup completed successfully!"
    log "Next steps:"
    log "1. Deploy your application code to $APP_DIR"
    log "2. Copy and configure environment files in /etc/secondbrain/"
    log "3. Install systemd services"
    log "4. Configure NGINX"
    log "5. Set up SSL certificates"
    
    info "Setup log saved to: $LOG_FILE"
    info "Application directory: $APP_DIR"
    info "Application user: $APP_USER"
}

# Run main function
main "$@"