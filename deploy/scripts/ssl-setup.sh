#!/bin/bash
# Second Brain - SSL Certificate Setup Script
# Sets up Let's Encrypt SSL certificates with automatic renewal

set -euo pipefail

# Configuration
DOMAIN_NAME="${DOMAIN_NAME:-}"
EMAIL="${EMAIL:-}"
WEBROOT="/var/www/letsencrypt"
NGINX_CONFIG="/etc/nginx/sites-available/secondbrain.conf"
LOG_FILE="/var/log/secondbrain/ssl-setup.log"

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
    echo "Usage: $0 -d DOMAIN_NAME -e EMAIL [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d DOMAIN_NAME    Domain name for SSL certificate (required)"
    echo "  -e EMAIL         Email address for Let's Encrypt registration (required)"
    echo "  -t               Test mode (use Let's Encrypt staging server)"
    echo "  -f               Force certificate renewal"
    echo "  -h               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -d secondbrain.example.com -e admin@example.com"
    echo "  $0 -d secondbrain.example.com -e admin@example.com -t"
    exit 1
}

# Parse command line arguments
STAGING=false
FORCE=false

while getopts "d:e:tfh" opt; do
    case $opt in
        d)
            DOMAIN_NAME="$OPTARG"
            ;;
        e)
            EMAIL="$OPTARG"
            ;;
        t)
            STAGING=true
            ;;
        f)
            FORCE=true
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

# Validate required parameters
if [[ -z "$DOMAIN_NAME" || -z "$EMAIL" ]]; then
    error "Domain name and email are required. Use -h for help."
fi

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    error "This script must be run as root (use sudo)"
fi

# Check prerequisites
check_prerequisites() {
    log "Checking SSL setup prerequisites..."
    
    # Check if certbot is installed
    if ! command -v certbot &> /dev/null; then
        error "Certbot is not installed. Install it with: apt-get install certbot python3-certbot-nginx"
    fi
    
    # Check if NGINX is installed and running
    if ! command -v nginx &> /dev/null; then
        error "NGINX is not installed"
    fi
    
    if ! systemctl is-active --quiet nginx; then
        error "NGINX is not running. Start it with: systemctl start nginx"
    fi
    
    # Check if domain resolves to this server
    SERVER_IP=$(curl -s ifconfig.me || curl -s icanhazip.com || echo "unknown")
    DOMAIN_IP=$(dig +short "$DOMAIN_NAME" | head -n1)
    
    if [[ "$SERVER_IP" != "unknown" && -n "$DOMAIN_IP" ]]; then
        if [[ "$SERVER_IP" != "$DOMAIN_IP" ]]; then
            warn "Domain $DOMAIN_NAME resolves to $DOMAIN_IP, but server IP is $SERVER_IP"
            warn "Make sure DNS is properly configured before proceeding"
            read -p "Continue anyway? [y/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    else
        warn "Could not verify domain DNS resolution"
    fi
    
    log "Prerequisites check completed"
}

# Prepare NGINX for ACME challenge
prepare_nginx() {
    log "Preparing NGINX for ACME challenge..."
    
    # Create webroot directory
    mkdir -p "$WEBROOT"
    chmod 755 "$WEBROOT"
    chown www-data:www-data "$WEBROOT"
    
    # Create temporary NGINX configuration for HTTP validation
    TEMP_CONFIG="/etc/nginx/sites-available/temp-ssl-setup"
    
    cat > "$TEMP_CONFIG" << EOF
server {
    listen 80;
    listen [::]:80;
    server_name $DOMAIN_NAME www.$DOMAIN_NAME;
    
    location /.well-known/acme-challenge/ {
        root $WEBROOT;
        try_files \$uri =404;
    }
    
    location / {
        return 301 https://\$server_name\$request_uri;
    }
}
EOF
    
    # Enable temporary configuration
    ln -sf "$TEMP_CONFIG" /etc/nginx/sites-enabled/temp-ssl-setup
    
    # Disable main site temporarily to avoid conflicts
    if [[ -L /etc/nginx/sites-enabled/secondbrain.conf ]]; then
        rm /etc/nginx/sites-enabled/secondbrain.conf
    fi
    
    # Test and reload NGINX
    nginx -t
    systemctl reload nginx
    
    log "NGINX prepared for ACME challenge"
}

# Generate DH parameters
generate_dhparam() {
    log "Generating Diffie-Hellman parameters (this may take a while)..."
    
    if [[ ! -f /etc/ssl/certs/dhparam.pem ]]; then
        openssl dhparam -out /etc/ssl/certs/dhparam.pem 2048
        chmod 644 /etc/ssl/certs/dhparam.pem
        log "DH parameters generated"
    else
        log "DH parameters already exist"
    fi
}

# Obtain SSL certificate
obtain_certificate() {
    log "Obtaining SSL certificate from Let's Encrypt..."
    
    # Build certbot command
    CERTBOT_CMD="certbot certonly --webroot --webroot-path=$WEBROOT"
    CERTBOT_CMD="$CERTBOT_CMD --email $EMAIL --agree-tos --no-eff-email"
    CERTBOT_CMD="$CERTBOT_CMD -d $DOMAIN_NAME -d www.$DOMAIN_NAME"
    
    if [[ "$STAGING" == "true" ]]; then
        CERTBOT_CMD="$CERTBOT_CMD --staging"
        warn "Using Let's Encrypt staging server (certificates will not be trusted)"
    fi
    
    if [[ "$FORCE" == "true" ]]; then
        CERTBOT_CMD="$CERTBOT_CMD --force-renewal"
    fi
    
    # Run certbot
    if $CERTBOT_CMD; then
        log "SSL certificate obtained successfully"
    else
        error "Failed to obtain SSL certificate"
    fi
    
    # Verify certificate files exist
    CERT_DIR="/etc/letsencrypt/live/$DOMAIN_NAME"
    if [[ ! -f "$CERT_DIR/fullchain.pem" || ! -f "$CERT_DIR/privkey.pem" ]]; then
        error "Certificate files not found in $CERT_DIR"
    fi
    
    log "Certificate files verified"
}

# Update NGINX configuration for SSL
update_nginx_ssl() {
    log "Updating NGINX configuration for SSL..."
    
    # Remove temporary configuration
    rm -f /etc/nginx/sites-enabled/temp-ssl-setup
    rm -f /etc/nginx/sites-available/temp-ssl-setup
    
    # Update domain name in the main configuration
    if [[ -f "$NGINX_CONFIG" ]]; then
        # Create backup
        cp "$NGINX_CONFIG" "${NGINX_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
        
        # Update domain names
        sed -i "s/secondbrain\.example\.com/$DOMAIN_NAME/g" "$NGINX_CONFIG"
        sed -i "s/www\.secondbrain\.example\.com/www.$DOMAIN_NAME/g" "$NGINX_CONFIG"
        
        # Update SSL certificate paths
        sed -i "s|/etc/letsencrypt/live/secondbrain\.example\.com|/etc/letsencrypt/live/$DOMAIN_NAME|g" "$NGINX_CONFIG"
    else
        error "Main NGINX configuration file not found: $NGINX_CONFIG"
    fi
    
    # Enable the main site
    ln -sf "$NGINX_CONFIG" /etc/nginx/sites-enabled/secondbrain.conf
    
    # Test NGINX configuration
    if nginx -t; then
        systemctl reload nginx
        log "NGINX configuration updated and reloaded"
    else
        error "NGINX configuration test failed"
    fi
}

# Set up automatic renewal
setup_auto_renewal() {
    log "Setting up automatic certificate renewal..."
    
    # Create renewal script
    cat > /usr/local/bin/renew-secondbrain-ssl << 'EOF'
#!/bin/bash
# Automatic SSL certificate renewal for Second Brain

LOG_FILE="/var/log/secondbrain/ssl-renewal.log"

# Renew certificates
/usr/bin/certbot renew --quiet >> "$LOG_FILE" 2>&1

# Reload NGINX if certificates were renewed
if [[ $? -eq 0 ]]; then
    /bin/systemctl reload nginx >> "$LOG_FILE" 2>&1
    echo "$(date): SSL certificates checked and renewed if necessary" >> "$LOG_FILE"
else
    echo "$(date): SSL certificate renewal check failed" >> "$LOG_FILE"
fi
EOF
    
    chmod +x /usr/local/bin/renew-secondbrain-ssl
    
    # Add to crontab for automatic renewal (runs twice daily)
    CRON_JOB="0 */12 * * * /usr/local/bin/renew-secondbrain-ssl"
    
    # Check if cron job already exists
    if ! crontab -l 2>/dev/null | grep -q "renew-secondbrain-ssl"; then
        (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
        log "Automatic renewal cron job added"
    else
        log "Automatic renewal cron job already exists"
    fi
}

# Test SSL configuration
test_ssl() {
    log "Testing SSL configuration..."
    
    # Wait for NGINX to apply changes
    sleep 5
    
    # Test HTTPS connection
    if curl -f -s "https://$DOMAIN_NAME" > /dev/null; then
        log "HTTPS connection successful"
    else
        warn "HTTPS connection test failed - this might be normal if the application isn't running yet"
    fi
    
    # Test HTTP redirect
    HTTP_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "http://$DOMAIN_NAME" || echo "000")
    if [[ "$HTTP_RESPONSE" == "301" ]]; then
        log "HTTP to HTTPS redirect working correctly"
    else
        warn "HTTP to HTTPS redirect not working (got HTTP $HTTP_RESPONSE)"
    fi
    
    # Check certificate validity
    CERT_EXPIRY=$(openssl x509 -in "/etc/letsencrypt/live/$DOMAIN_NAME/cert.pem" -noout -dates | grep "notAfter" | cut -d= -f2)
    log "Certificate expires: $CERT_EXPIRY"
}

# Generate SSL security report
generate_report() {
    log "Generating SSL security report..."
    
    REPORT_FILE="/var/log/secondbrain/ssl-report-$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$REPORT_FILE" << EOF
Second Brain SSL Setup Report
Generated: $(date)
Domain: $DOMAIN_NAME
Email: $EMAIL
Staging Mode: $STAGING

Certificate Information:
$(openssl x509 -in "/etc/letsencrypt/live/$DOMAIN_NAME/cert.pem" -text -noout | head -20)

Certificate Validity:
$(openssl x509 -in "/etc/letsencrypt/live/$DOMAIN_NAME/cert.pem" -noout -dates)

NGINX SSL Configuration Test:
$(nginx -t 2>&1)

Automatic Renewal Setup:
$(crontab -l | grep renew-secondbrain-ssl || echo "Not found")

Next Steps:
1. Test your website at: https://$DOMAIN_NAME
2. Run SSL Labs test: https://www.ssllabs.com/ssltest/analyze.html?d=$DOMAIN_NAME
3. Monitor renewal logs: /var/log/secondbrain/ssl-renewal.log

EOF
    
    log "SSL report generated: $REPORT_FILE"
    
    # Show important information
    info "SSL Setup Summary:"
    info "- Domain: $DOMAIN_NAME"
    info "- Certificate expires: $(openssl x509 -in "/etc/letsencrypt/live/$DOMAIN_NAME/cert.pem" -noout -dates | grep "notAfter" | cut -d= -f2)"
    info "- Auto-renewal: Enabled (runs twice daily)"
    info "- Test your site: https://$DOMAIN_NAME"
    info "- SSL Labs test: https://www.ssllabs.com/ssltest/analyze.html?d=$DOMAIN_NAME"
}

# Main SSL setup function
main() {
    log "Starting SSL setup for domain: $DOMAIN_NAME"
    
    # Create log file
    mkdir -p "$(dirname "$LOG_FILE")"
    touch "$LOG_FILE"
    
    # Run setup steps
    check_prerequisites
    generate_dhparam
    prepare_nginx
    obtain_certificate
    update_nginx_ssl
    setup_auto_renewal
    test_ssl
    generate_report
    
    log "SSL setup completed successfully!"
    log "Your site should now be available at: https://$DOMAIN_NAME"
    
    if [[ "$STAGING" == "true" ]]; then
        warn "Remember: You used staging certificates. Run again without -t for production certificates."
    fi
}

# Create log file directory
mkdir -p "$(dirname "$LOG_FILE")"

# Run main function
main