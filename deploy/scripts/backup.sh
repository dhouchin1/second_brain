#!/bin/bash
# Second Brain - Backup Script
# Creates comprehensive backups of application data, database, and configuration

set -euo pipefail

# Configuration
APP_DIR="/var/www/secondbrain"
BACKUP_BASE_DIR="/var/backups/secondbrain"
LOG_FILE="/var/log/secondbrain/backup.log"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
BACKUP_TYPE="${BACKUP_TYPE:-full}"  # full, data, config

# Remote backup settings (optional)
REMOTE_BACKUP_ENABLED="${REMOTE_BACKUP_ENABLED:-false}"
REMOTE_BACKUP_HOST="${REMOTE_BACKUP_HOST:-}"
REMOTE_BACKUP_USER="${REMOTE_BACKUP_USER:-}"
REMOTE_BACKUP_PATH="${REMOTE_BACKUP_PATH:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

# Show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t TYPE          Backup type: full, data, config (default: full)"
    echo "  -r DAYS          Retention period in days (default: 30)"
    echo "  -o DIRECTORY     Output directory (default: $BACKUP_BASE_DIR)"
    echo "  -c               Compress backup"
    echo "  -e               Encrypt backup (requires gpg)"
    echo "  -u               Upload to remote server"
    echo "  -v               Verbose output"
    echo "  -h               Show this help message"
    echo ""
    echo "Backup Types:"
    echo "  full             Complete backup (data + config + logs)"
    echo "  data             Data only (databases, uploads, vault)"
    echo "  config           Configuration only (settings, certificates)"
    echo ""
    echo "Examples:"
    echo "  $0                           # Full backup with defaults"
    echo "  $0 -t data -c                # Compressed data-only backup"
    echo "  $0 -t full -c -e -u          # Full backup, compressed, encrypted, uploaded"
    exit 1
}

# Parse command line arguments
COMPRESS=false
ENCRYPT=false
UPLOAD=false
VERBOSE=false
OUTPUT_DIR="$BACKUP_BASE_DIR"

while getopts "t:r:o:ceuvh" opt; do
    case $opt in
        t)
            BACKUP_TYPE="$OPTARG"
            ;;
        r)
            RETENTION_DAYS="$OPTARG"
            ;;
        o)
            OUTPUT_DIR="$OPTARG"
            ;;
        c)
            COMPRESS=true
            ;;
        e)
            ENCRYPT=true
            ;;
        u)
            UPLOAD=true
            ;;
        v)
            VERBOSE=true
            ;;
        h)
            usage
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
    esac
done

# Validate backup type
case $BACKUP_TYPE in
    full|data|config)
        ;;
    *)
        error "Invalid backup type: $BACKUP_TYPE"
        ;;
esac

# Check prerequisites
check_prerequisites() {
    log "Checking backup prerequisites..."
    
    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root (use sudo)"
    fi
    
    # Check if application directory exists
    if [[ ! -d "$APP_DIR" ]]; then
        error "Application directory not found: $APP_DIR"
    fi
    
    # Create backup directory
    mkdir -p "$OUTPUT_DIR"
    
    # Check encryption requirements
    if [[ "$ENCRYPT" == "true" ]] && ! command -v gpg &> /dev/null; then
        error "GPG is required for encryption but not installed"
    fi
    
    # Check remote backup requirements
    if [[ "$UPLOAD" == "true" ]]; then
        if [[ -z "$REMOTE_BACKUP_HOST" || -z "$REMOTE_BACKUP_USER" ]]; then
            error "Remote backup requires REMOTE_BACKUP_HOST and REMOTE_BACKUP_USER"
        fi
        if ! command -v rsync &> /dev/null; then
            error "rsync is required for remote backup but not installed"
        fi
    fi
    
    log "Prerequisites check completed"
}

# Create backup manifest
create_manifest() {
    local backup_dir="$1"
    local manifest_file="$backup_dir/manifest.txt"
    
    cat > "$manifest_file" << EOF
Second Brain Backup Manifest
Created: $(date)
Backup Type: $BACKUP_TYPE
Hostname: $(hostname)
System: $(uname -a)
Git Commit: $(cd "$APP_DIR" 2>/dev/null && git rev-parse HEAD 2>/dev/null || echo "N/A")
Git Branch: $(cd "$APP_DIR" 2>/dev/null && git branch --show-current 2>/dev/null || echo "N/A")

Backup Contents:
EOF
    
    # List backup contents
    find "$backup_dir" -type f ! -name "manifest.txt" | while read -r file; do
        relative_path="${file#$backup_dir/}"
        file_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "unknown")
        echo "  $relative_path (${file_size} bytes)" >> "$manifest_file"
    done
    
    # Add checksums
    echo "" >> "$manifest_file"
    echo "File Checksums (SHA256):" >> "$manifest_file"
    find "$backup_dir" -type f ! -name "manifest.txt" -exec sha256sum {} \; | \
        sed "s|$backup_dir/||" >> "$manifest_file"
}

# Backup databases
backup_databases() {
    local backup_dir="$1"
    
    log "Backing up databases..."
    
    mkdir -p "$backup_dir/databases"
    
    # SQLite databases
    if [[ -f "$APP_DIR/notes.db" ]]; then
        # Create consistent backup using SQLite backup API
        sqlite3 "$APP_DIR/notes.db" ".backup '$backup_dir/databases/notes.db'"
        log "Backed up notes.db"
    fi
    
    if [[ -f "$APP_DIR/brain.db" ]]; then
        sqlite3 "$APP_DIR/brain.db" ".backup '$backup_dir/databases/brain.db'"
        log "Backed up brain.db"
    fi
    
    # Dump database schemas and metadata
    if [[ -f "$APP_DIR/notes.db" ]]; then
        sqlite3 "$APP_DIR/notes.db" ".schema" > "$backup_dir/databases/notes_schema.sql"
        sqlite3 "$APP_DIR/notes.db" "PRAGMA integrity_check;" > "$backup_dir/databases/notes_integrity.txt"
    fi
    
    if [[ -f "$APP_DIR/brain.db" ]]; then
        sqlite3 "$APP_DIR/brain.db" ".schema" > "$backup_dir/databases/brain_schema.sql"
        sqlite3 "$APP_DIR/brain.db" "PRAGMA integrity_check;" > "$backup_dir/databases/brain_integrity.txt"
    fi
}

# Backup application data
backup_data() {
    local backup_dir="$1"
    
    log "Backing up application data..."
    
    # Vault directory
    if [[ -d "$APP_DIR/vault" ]]; then
        rsync -av "$APP_DIR/vault/" "$backup_dir/vault/" ${VERBOSE:+--progress}
        log "Backed up vault directory"
    fi
    
    # Audio files
    if [[ -d "$APP_DIR/audio" ]]; then
        rsync -av "$APP_DIR/audio/" "$backup_dir/audio/" ${VERBOSE:+--progress}
        log "Backed up audio directory"
    fi
    
    # Uploads
    if [[ -d "$APP_DIR/uploads" ]]; then
        rsync -av "$APP_DIR/uploads/" "$backup_dir/uploads/" ${VERBOSE:+--progress}
        log "Backed up uploads directory"
    fi
    
    # Static files (user-generated)
    if [[ -d "$APP_DIR/static/user" ]]; then
        rsync -av "$APP_DIR/static/user/" "$backup_dir/static/user/" ${VERBOSE:+--progress}
        log "Backed up user static files"
    fi
}

# Backup configuration
backup_config() {
    local backup_dir="$1"
    
    log "Backing up configuration..."
    
    mkdir -p "$backup_dir/config"
    
    # Environment configuration
    if [[ -d "/etc/secondbrain" ]]; then
        cp -r /etc/secondbrain "$backup_dir/config/"
        log "Backed up environment configuration"
    fi
    
    # NGINX configuration
    mkdir -p "$backup_dir/config/nginx"
    cp /etc/nginx/nginx.conf "$backup_dir/config/nginx/" 2>/dev/null || true
    cp -r /etc/nginx/sites-available "$backup_dir/config/nginx/" 2>/dev/null || true
    cp -r /etc/nginx/conf.d "$backup_dir/config/nginx/" 2>/dev/null || true
    
    # Systemd services
    mkdir -p "$backup_dir/config/systemd"
    cp /etc/systemd/system/secondbrain*.service "$backup_dir/config/systemd/" 2>/dev/null || true
    
    # SSL certificates
    if [[ -d "/etc/letsencrypt" ]]; then
        mkdir -p "$backup_dir/config/ssl"
        rsync -av /etc/letsencrypt/ "$backup_dir/config/ssl/letsencrypt/" ${VERBOSE:+--progress}
        log "Backed up SSL certificates"
    fi
    
    # Cron jobs
    crontab -l > "$backup_dir/config/crontab.txt" 2>/dev/null || echo "No crontab" > "$backup_dir/config/crontab.txt"
}

# Backup logs
backup_logs() {
    local backup_dir="$1"
    
    log "Backing up logs..."
    
    if [[ -d "/var/log/secondbrain" ]]; then
        # Only backup recent logs (last 7 days) to save space
        mkdir -p "$backup_dir/logs"
        find /var/log/secondbrain -name "*.log*" -mtime -7 -exec cp {} "$backup_dir/logs/" \;
        log "Backed up recent application logs"
    fi
    
    # System logs related to our application
    mkdir -p "$backup_dir/logs/system"
    journalctl -u secondbrain --since "7 days ago" > "$backup_dir/logs/system/secondbrain.log" 2>/dev/null || true
    journalctl -u secondbrain-worker --since "7 days ago" > "$backup_dir/logs/system/secondbrain-worker.log" 2>/dev/null || true
    journalctl -u nginx --since "7 days ago" > "$backup_dir/logs/system/nginx.log" 2>/dev/null || true
}

# Perform backup
perform_backup() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_name="secondbrain_${BACKUP_TYPE}_${timestamp}"
    local backup_dir="$OUTPUT_DIR/$backup_name"
    
    log "Starting $BACKUP_TYPE backup: $backup_name"
    
    # Create backup directory
    mkdir -p "$backup_dir"
    
    # Perform backup based on type
    case $BACKUP_TYPE in
        full)
            backup_databases "$backup_dir"
            backup_data "$backup_dir"
            backup_config "$backup_dir"
            backup_logs "$backup_dir"
            ;;
        data)
            backup_databases "$backup_dir"
            backup_data "$backup_dir"
            ;;
        config)
            backup_config "$backup_dir"
            ;;
    esac
    
    # Create manifest
    create_manifest "$backup_dir"
    
    # Calculate backup size
    backup_size=$(du -sh "$backup_dir" | cut -f1)
    log "Backup completed: $backup_size"
    
    # Compress if requested
    if [[ "$COMPRESS" == "true" ]]; then
        log "Compressing backup..."
        cd "$OUTPUT_DIR"
        tar -czf "${backup_name}.tar.gz" "$backup_name"
        rm -rf "$backup_name"
        backup_dir="${OUTPUT_DIR}/${backup_name}.tar.gz"
        compressed_size=$(du -sh "$backup_dir" | cut -f1)
        log "Backup compressed: $compressed_size"
    fi
    
    # Encrypt if requested
    if [[ "$ENCRYPT" == "true" ]]; then
        log "Encrypting backup..."
        gpg --cipher-algo AES256 --compress-algo 1 --symmetric --output "${backup_dir}.gpg" "$backup_dir"
        rm "$backup_dir"
        backup_dir="${backup_dir}.gpg"
        log "Backup encrypted"
    fi
    
    # Upload to remote server if requested
    if [[ "$UPLOAD" == "true" ]]; then
        upload_backup "$backup_dir"
    fi
    
    log "Backup saved: $backup_dir"
    echo "$backup_dir"
}

# Upload backup to remote server
upload_backup() {
    local backup_file="$1"
    local filename=$(basename "$backup_file")
    
    log "Uploading backup to remote server..."
    
    if rsync -avz --progress "$backup_file" "${REMOTE_BACKUP_USER}@${REMOTE_BACKUP_HOST}:${REMOTE_BACKUP_PATH}/" ${VERBOSE:+--progress}; then
        log "Backup uploaded successfully"
        
        # Remove local copy if upload successful and requested
        if [[ "${REMOVE_AFTER_UPLOAD:-false}" == "true" ]]; then
            rm "$backup_file"
            log "Local backup removed after successful upload"
        fi
    else
        error "Failed to upload backup to remote server"
    fi
}

# Clean up old backups
cleanup_old_backups() {
    log "Cleaning up old backups (keeping last $RETENTION_DAYS days)..."
    
    local deleted_count=0
    
    # Find and delete old backups
    while IFS= read -r -d '' file; do
        rm "$file"
        ((deleted_count++))
        log "Deleted old backup: $(basename "$file")"
    done < <(find "$OUTPUT_DIR" -name "secondbrain_*" -type f -mtime +$RETENTION_DAYS -print0)
    
    log "Cleaned up $deleted_count old backup files"
}

# Verify backup integrity
verify_backup() {
    local backup_file="$1"
    
    log "Verifying backup integrity..."
    
    # For compressed files
    if [[ "$backup_file" == *.tar.gz ]]; then
        if tar -tzf "$backup_file" > /dev/null; then
            log "Backup archive integrity verified"
        else
            error "Backup archive integrity check failed"
        fi
    fi
    
    # For encrypted files
    if [[ "$backup_file" == *.gpg ]]; then
        if gpg --list-packets "$backup_file" > /dev/null 2>&1; then
            log "Backup encryption integrity verified"
        else
            error "Backup encryption integrity check failed"
        fi
    fi
}

# Generate backup report
generate_report() {
    local backup_file="$1"
    local report_file="/var/log/secondbrain/backup-report-$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" << EOF
Second Brain Backup Report
Generated: $(date)
Backup Type: $BACKUP_TYPE
Backup File: $backup_file
Backup Size: $(du -sh "$backup_file" | cut -f1)

System Information:
Hostname: $(hostname)
System: $(uname -a)
Disk Space: $(df -h "$OUTPUT_DIR" | tail -n1)

Backup Settings:
Compression: $COMPRESS
Encryption: $ENCRYPT
Remote Upload: $UPLOAD
Retention: $RETENTION_DAYS days

$(if [[ -f "${backup_file%.*}/manifest.txt" ]]; then
    echo "Backup Contents:"
    cat "${backup_file%.*}/manifest.txt"
fi)

EOF
    
    log "Backup report generated: $report_file"
}

# Main backup function
main() {
    log "Starting Second Brain backup (type: $BACKUP_TYPE)"
    
    # Create log file
    mkdir -p "$(dirname "$LOG_FILE")"
    touch "$LOG_FILE"
    
    # Check prerequisites
    check_prerequisites
    
    # Perform backup
    backup_file=$(perform_backup)
    
    # Verify backup
    verify_backup "$backup_file"
    
    # Generate report
    generate_report "$backup_file"
    
    # Clean up old backups
    cleanup_old_backups
    
    log "Backup process completed successfully!"
    log "Backup saved: $backup_file"
    
    # Show backup summary
    info "Backup Summary:"
    info "- Type: $BACKUP_TYPE"
    info "- Size: $(du -sh "$backup_file" | cut -f1)"
    info "- Location: $backup_file"
    info "- Retention: $RETENTION_DAYS days"
    
    if [[ "$UPLOAD" == "true" ]]; then
        info "- Uploaded to: ${REMOTE_BACKUP_HOST}:${REMOTE_BACKUP_PATH}/"
    fi
}

# Create log file directory
mkdir -p "$(dirname "$LOG_FILE")"

# Run main function
main