#!/bin/bash
# Second Brain - Production Health Check Script
# Comprehensive health monitoring for all system components

set -euo pipefail

# Configuration
HEALTH_ENDPOINT="http://localhost:8082/health"
LOG_FILE="/var/log/secondbrain/health-check.log"
ALERT_EMAIL="${ALERT_EMAIL:-}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"
MAX_RESPONSE_TIME=5000  # milliseconds
MIN_DISK_SPACE=10       # percentage
MAX_MEMORY_USAGE=90     # percentage
MAX_CPU_USAGE=80        # percentage

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Status tracking
OVERALL_STATUS="healthy"
FAILED_CHECKS=()
WARNING_CHECKS=()

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
    WARNING_CHECKS+=("$1")
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
    FAILED_CHECKS+=("$1")
    OVERALL_STATUS="unhealthy"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

# Check service status
check_service_status() {
    log "Checking service status..."
    
    local services=("secondbrain" "secondbrain-worker" "nginx" "redis-server")
    
    for service in "${services[@]}"; do
        if systemctl is-active --quiet "$service"; then
            log "$service: Running"
        else
            error "$service: Not running"
        fi
    done
}

# Check application health endpoint
check_application_health() {
    log "Checking application health endpoint..."
    
    local start_time=$(date +%s%3N)
    local response_code
    local response_time
    
    if response_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 --max-time 30 "$HEALTH_ENDPOINT" 2>/dev/null); then
        local end_time=$(date +%s%3N)
        response_time=$((end_time - start_time))
        
        if [[ "$response_code" == "200" ]]; then
            log "Health endpoint: OK (${response_time}ms)"
            
            if [[ $response_time -gt $MAX_RESPONSE_TIME ]]; then
                warn "Health endpoint response time too slow: ${response_time}ms (max: ${MAX_RESPONSE_TIME}ms)"
            fi
        else
            error "Health endpoint returned HTTP $response_code"
        fi
    else
        error "Health endpoint unreachable"
    fi
}

# Check database connectivity
check_database() {
    log "Checking database connectivity..."
    
    local app_dir="/var/www/secondbrain"
    
    if [[ -f "$app_dir/notes.db" ]]; then
        if sqlite3 "$app_dir/notes.db" "SELECT 1;" > /dev/null 2>&1; then
            log "Database: Accessible"
            
            # Check database integrity
            local integrity_check=$(sqlite3 "$app_dir/notes.db" "PRAGMA integrity_check;" 2>/dev/null | head -1)
            if [[ "$integrity_check" == "ok" ]]; then
                log "Database integrity: OK"
            else
                error "Database integrity check failed: $integrity_check"
            fi
            
            # Check database size and growth
            local db_size=$(du -h "$app_dir/notes.db" | cut -f1)
            log "Database size: $db_size"
            
        else
            error "Database not accessible"
        fi
    else
        error "Database file not found: $app_dir/notes.db"
    fi
}

# Check Redis connectivity
check_redis() {
    log "Checking Redis connectivity..."
    
    if redis-cli ping | grep -q "PONG"; then
        log "Redis: Connected"
        
        # Check Redis memory usage
        local redis_memory=$(redis-cli info memory | grep "used_memory_human:" | cut -d: -f2 | tr -d '\r')
        log "Redis memory usage: $redis_memory"
        
        # Check Redis key count
        local key_count=$(redis-cli dbsize | tr -d '\r')
        log "Redis keys: $key_count"
        
    else
        error "Redis not responding"
    fi
}

# Check Ollama service
check_ollama() {
    log "Checking Ollama service..."
    
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        log "Ollama: Running"
        
        # Check available models
        local models=$(curl -s http://localhost:11434/api/tags | jq -r '.models[]?.name' 2>/dev/null | wc -l)
        if [[ $models -gt 0 ]]; then
            log "Ollama models available: $models"
        else
            warn "No Ollama models found"
        fi
    else
        warn "Ollama service not responding (optional service)"
    fi
}

# Check system resources
check_system_resources() {
    log "Checking system resources..."
    
    # CPU usage
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 | cut -d' ' -f1)
    if [[ -n "$cpu_usage" ]]; then
        cpu_usage=${cpu_usage%.*}  # Remove decimal part
        log "CPU usage: ${cpu_usage}%"
        if [[ $cpu_usage -gt $MAX_CPU_USAGE ]]; then
            warn "High CPU usage: ${cpu_usage}% (max: ${MAX_CPU_USAGE}%)"
        fi
    fi
    
    # Memory usage
    local memory_info=$(free | grep Mem)
    local total_mem=$(echo $memory_info | awk '{print $2}')
    local used_mem=$(echo $memory_info | awk '{print $3}')
    local memory_percent=$((used_mem * 100 / total_mem))
    
    log "Memory usage: ${memory_percent}% ($(echo $memory_info | awk '{print $3/1024/1024}' | cut -d. -f1)GB/$(echo $memory_info | awk '{print $2/1024/1024}' | cut -d. -f1)GB)"
    
    if [[ $memory_percent -gt $MAX_MEMORY_USAGE ]]; then
        warn "High memory usage: ${memory_percent}% (max: ${MAX_MEMORY_USAGE}%)"
    fi
    
    # Disk space
    while IFS= read -r disk_info; do
        local usage_percent=$(echo "$disk_info" | awk '{print $5}' | sed 's/%//')
        local mount_point=$(echo "$disk_info" | awk '{print $6}')
        local available=$(echo "$disk_info" | awk '{print $4}')
        
        log "Disk usage $mount_point: ${usage_percent}% (${available} available)"
        
        if [[ $usage_percent -gt $((100 - MIN_DISK_SPACE)) ]]; then
            warn "Low disk space on $mount_point: ${usage_percent}% used"
        fi
    done < <(df -h | grep -E '^/dev/')
}

# Check log files
check_logs() {
    log "Checking log files..."
    
    local log_dirs=("/var/log/secondbrain" "/var/log/nginx")
    
    for log_dir in "${log_dirs[@]}"; do
        if [[ -d "$log_dir" ]]; then
            # Check for recent errors
            local error_count=$(find "$log_dir" -name "*.log" -mtime -1 -exec grep -i "error" {} \; 2>/dev/null | wc -l)
            local warning_count=$(find "$log_dir" -name "*.log" -mtime -1 -exec grep -i "warning" {} \; 2>/dev/null | wc -l)
            
            log "Recent errors in $log_dir: $error_count"
            log "Recent warnings in $log_dir: $warning_count"
            
            if [[ $error_count -gt 100 ]]; then
                warn "High error count in $log_dir: $error_count errors in last 24h"
            fi
            
            # Check log file sizes
            local large_logs=$(find "$log_dir" -name "*.log" -size +100M 2>/dev/null | wc -l)
            if [[ $large_logs -gt 0 ]]; then
                warn "Large log files found in $log_dir: $large_logs files > 100MB"
            fi
        fi
    done
}

# Check SSL certificates
check_ssl_certificates() {
    log "Checking SSL certificates..."
    
    local cert_dir="/etc/letsencrypt/live"
    
    if [[ -d "$cert_dir" ]]; then
        for domain_dir in "$cert_dir"/*; do
            if [[ -d "$domain_dir" && -f "$domain_dir/cert.pem" ]]; then
                local domain=$(basename "$domain_dir")
                local expiry_date=$(openssl x509 -in "$domain_dir/cert.pem" -noout -enddate | cut -d= -f2)
                local expiry_timestamp=$(date -d "$expiry_date" +%s)
                local current_timestamp=$(date +%s)
                local days_until_expiry=$(( (expiry_timestamp - current_timestamp) / 86400 ))
                
                log "SSL certificate $domain expires in $days_until_expiry days"
                
                if [[ $days_until_expiry -lt 30 ]]; then
                    warn "SSL certificate $domain expires soon: $days_until_expiry days"
                elif [[ $days_until_expiry -lt 7 ]]; then
                    error "SSL certificate $domain expires very soon: $days_until_expiry days"
                fi
            fi
        done
    else
        info "No SSL certificates found (not an error for HTTP-only deployments)"
    fi
}

# Check network connectivity
check_network() {
    log "Checking network connectivity..."
    
    # Check if we can reach external services
    local external_hosts=("8.8.8.8" "1.1.1.1")
    
    for host in "${external_hosts[@]}"; do
        if ping -c 1 -W 5 "$host" >/dev/null 2>&1; then
            log "Network connectivity to $host: OK"
        else
            warn "Network connectivity to $host: Failed"
        fi
    done
    
    # Check if critical ports are listening
    local ports=("80:nginx" "443:nginx" "8082:app" "6379:redis")
    
    for port_info in "${ports[@]}"; do
        local port=$(echo "$port_info" | cut -d: -f1)
        local service=$(echo "$port_info" | cut -d: -f2)
        
        if netstat -tuln | grep ":$port " >/dev/null 2>&1; then
            log "Port $port ($service): Listening"
        else
            error "Port $port ($service): Not listening"
        fi
    done
}

# Check backup status
check_backups() {
    log "Checking backup status..."
    
    local backup_dir="/var/backups/secondbrain"
    
    if [[ -d "$backup_dir" ]]; then
        local latest_backup=$(find "$backup_dir" -name "secondbrain_*" -type f | sort | tail -n1)
        
        if [[ -n "$latest_backup" ]]; then
            local backup_age=$(( ($(date +%s) - $(stat -c %Y "$latest_backup")) / 86400 ))
            log "Latest backup: $(basename "$latest_backup") ($backup_age days old)"
            
            if [[ $backup_age -gt 7 ]]; then
                warn "Latest backup is $backup_age days old"
            fi
        else
            warn "No backups found in $backup_dir"
        fi
    else
        warn "Backup directory not found: $backup_dir"
    fi
}

# Send alert notification
send_alert() {
    local message="$1"
    local severity="$2"  # info, warning, error
    
    # Email notification
    if [[ -n "$ALERT_EMAIL" ]] && command -v mail >/dev/null 2>&1; then
        echo "$message" | mail -s "Second Brain Health Check - $severity" "$ALERT_EMAIL"
    fi
    
    # Slack notification
    if [[ -n "$SLACK_WEBHOOK" ]] && command -v curl >/dev/null 2>&1; then
        local color="good"
        [[ "$severity" == "warning" ]] && color="warning"
        [[ "$severity" == "error" ]] && color="danger"
        
        curl -s -X POST -H 'Content-type: application/json' \
            --data "{\"attachments\":[{\"color\":\"$color\",\"text\":\"$message\"}]}" \
            "$SLACK_WEBHOOK" >/dev/null || true
    fi
}

# Generate health report
generate_report() {
    local report_file="/var/log/secondbrain/health-report-$(date +%Y%m%d_%H%M%S).json"
    
    # Create JSON report
    cat > "$report_file" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "hostname": "$(hostname)",
    "overall_status": "$OVERALL_STATUS",
    "failed_checks": $(printf '%s\n' "${FAILED_CHECKS[@]}" | jq -R . | jq -s . 2>/dev/null || echo '[]'),
    "warning_checks": $(printf '%s\n' "${WARNING_CHECKS[@]}" | jq -R . | jq -s . 2>/dev/null || echo '[]'),
    "system_info": {
        "uptime": "$(uptime -p 2>/dev/null || uptime)",
        "load_average": "$(uptime | awk -F'load average:' '{print $2}')",
        "memory": $(free -b | awk 'NR==2{printf "{\"total\":%s,\"used\":%s,\"free\":%s}", $2,$3,$4}'),
        "disk_space": $(df / | awk 'NR==2{printf "{\"size\":\"%s\",\"used\":\"%s\",\"available\":\"%s\",\"use_percent\":\"%s\"}", $2,$3,$4,$5}')
    }
}
EOF
    
    log "Health report saved: $report_file"
}

# Main health check function
main() {
    log "Starting Second Brain health check..."
    
    # Create log file
    mkdir -p "$(dirname "$LOG_FILE")"
    touch "$LOG_FILE"
    
    # Run all health checks
    check_service_status
    check_application_health
    check_database
    check_redis
    check_ollama
    check_system_resources
    check_logs
    check_ssl_certificates
    check_network
    check_backups
    
    # Generate report
    generate_report
    
    # Summary
    log "Health check completed"
    info "Overall status: $OVERALL_STATUS"
    
    if [[ ${#FAILED_CHECKS[@]} -gt 0 ]]; then
        error "Failed checks: ${#FAILED_CHECKS[@]}"
        printf '%s\n' "${FAILED_CHECKS[@]}" | while read -r check; do
            error "  - $check"
        done
        
        # Send alert for critical failures
        local alert_message="Second Brain Health Check FAILED on $(hostname) at $(date)\n\nFailed checks:\n$(printf '• %s\n' "${FAILED_CHECKS[@]}")"
        send_alert "$alert_message" "error"
        
        exit 1
    fi
    
    if [[ ${#WARNING_CHECKS[@]} -gt 0 ]]; then
        warn "Warning checks: ${#WARNING_CHECKS[@]}"
        printf '%s\n' "${WARNING_CHECKS[@]}" | while read -r check; do
            warn "  - $check"
        done
        
        # Send alert for warnings
        local alert_message="Second Brain Health Check WARNINGS on $(hostname) at $(date)\n\nWarnings:\n$(printf '• %s\n' "${WARNING_CHECKS[@]}")"
        send_alert "$alert_message" "warning"
    else
        log "All checks passed successfully!"
        
        # Send success notification (optional)
        if [[ "${NOTIFY_ON_SUCCESS:-false}" == "true" ]]; then
            local success_message="Second Brain Health Check PASSED on $(hostname) at $(date)\n\nAll systems operational."
            send_alert "$success_message" "info"
        fi
    fi
}

# Handle different modes
case "${1:-check}" in
    "check")
        main
        ;;
    "status")
        echo "Overall Status: $OVERALL_STATUS"
        echo "Failed Checks: ${#FAILED_CHECKS[@]}"
        echo "Warning Checks: ${#WARNING_CHECKS[@]}"
        ;;
    *)
        echo "Usage: $0 [check|status]"
        exit 1
        ;;
esac