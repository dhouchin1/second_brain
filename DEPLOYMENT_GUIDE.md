# Second Brain Production Deployment Guide

## Overview

This guide covers deploying Second Brain in a production environment with security, performance, and reliability best practices.

## Prerequisites

- Docker and Docker Compose
- Linux server (Ubuntu 20.04+ recommended)
- Domain name (optional, for HTTPS)
- SSL certificates (for HTTPS deployment)

## Quick Start

### 1. Environment Setup

Create production environment file:

```bash
cp .env .env.production
```

Edit `.env.production` with production values:

```env
# Database
DATABASE_URL=sqlite:///./data/notes.db

# Security (IMPORTANT: Change these!)
SECRET_KEY=your-super-secret-key-here
WEBHOOK_TOKEN=your-webhook-token-here
AUTOM8_JWT_SECRET=your-autom8-secret-here

# Paths
VAULT_PATH=/app/data/vault
AUDIO_DIR=/app/data/audio
UPLOADS_DIR=/app/data/uploads

# Production settings
ENVIRONMENT=production
DEBUG=false

# Optional: External services
REDIS_URL=redis://redis:6379/0
```

### 2. Deploy with Docker Compose

```bash
# Create data directories
sudo mkdir -p /opt/second-brain/{data,logs,ssl}
sudo chown -R $USER:$USER /opt/second-brain

# Copy project files
cp -r . /opt/second-brain/
cd /opt/second-brain

# Start services
docker-compose -f docker-compose.production.yml up -d
```

### 3. Verify Deployment

```bash
# Check service status
docker-compose -f docker-compose.production.yml ps

# Check logs
docker-compose -f docker-compose.production.yml logs second-brain

# Test health endpoint
curl http://localhost/health
```

## Architecture

### Services

1. **second-brain**: Main application (FastAPI)
2. **autom8-service**: AI routing microservice
3. **nginx**: Reverse proxy and load balancer
4. **redis**: Caching and session storage (optional)
5. **prometheus**: Monitoring (optional)

### Network Flow

```
Internet → Nginx (80/443) → Second Brain (8082)
                         → Autom8 Service (8000)
```

## Security Configuration

### 1. Environment Variables

**Critical security variables that MUST be changed:**

```env
SECRET_KEY=<generate-strong-256-bit-key>
WEBHOOK_TOKEN=<generate-random-token>
AUTOM8_JWT_SECRET=<generate-different-strong-key>
```

Generate secure keys:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 2. Firewall Configuration

```bash
# UFW (Ubuntu Firewall)
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### 3. SSL/HTTPS Setup

For production with HTTPS:

1. Obtain SSL certificates (Let's Encrypt recommended):
```bash
sudo apt install certbot
sudo certbot certonly --standalone -d your-domain.com
```

2. Copy certificates to SSL directory:
```bash
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem /opt/second-brain/ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem /opt/second-brain/ssl/key.pem
```

3. Update nginx configuration to enable HTTPS blocks

## Performance Optimization

### 1. Resource Limits

Adjust Docker resource limits in `docker-compose.production.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 4G      # Increase for large datasets
      cpus: '2.0'     # Increase for high traffic
```

### 2. Database Optimization

For high-volume deployments, consider:

1. **External Database**: Replace SQLite with PostgreSQL
2. **Database Backups**: Implement regular backup strategy
3. **Index Optimization**: Monitor query performance

### 3. Caching Strategy

Enable Redis for improved performance:

1. Uncomment Redis service in docker-compose
2. Configure application to use Redis for sessions
3. Enable API response caching

## Monitoring and Logging

### 1. Application Monitoring

Access monitoring endpoints:
- Health check: `GET /api/system/health`
- Error summary: `GET /api/system/errors`
- Metrics: `GET /api/system/metrics`

### 2. Log Management

Logs are stored in:
- Application logs: `/opt/second-brain/logs/`
- Nginx logs: `/var/log/nginx/`
- Container logs: `docker-compose logs`

Configure log rotation:
```bash
sudo logrotate -d /etc/logrotate.d/second-brain
```

### 3. Prometheus Monitoring (Optional)

If enabled, access Prometheus at: `http://your-server:9090`

Key metrics to monitor:
- Response times
- Error rates
- Memory usage
- CPU utilization
- Database performance

## Backup Strategy

### 1. Database Backup

```bash
# Daily backup script
#!/bin/bash
BACKUP_DIR="/opt/second-brain/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
sqlite3 /opt/second-brain/data/notes.db ".backup $BACKUP_DIR/notes_$DATE.db"

# Keep last 30 days
find $BACKUP_DIR -name "notes_*.db" -mtime +30 -delete
```

### 2. Full System Backup

```bash
# Backup entire data directory
tar -czf /backup/second-brain-$DATE.tar.gz /opt/second-brain/data/
```

## Maintenance

### 1. Updates

```bash
# Update application
cd /opt/second-brain
git pull origin main
docker-compose -f docker-compose.production.yml build
docker-compose -f docker-compose.production.yml up -d
```

### 2. Database Migrations

```bash
# Run migrations
docker-compose -f docker-compose.production.yml exec second-brain python migrate_db.py
```

### 3. Health Checks

Regular health check script:
```bash
#!/bin/bash
HEALTH=$(curl -s http://localhost/health | jq -r '.status')
if [ "$HEALTH" != "healthy" ]; then
    echo "Alert: Second Brain unhealthy"
    # Send notification (email, Slack, etc.)
fi
```

## Troubleshooting

### Common Issues

1. **Permission Errors**
   ```bash
   sudo chown -R 1000:1000 /opt/second-brain/data/
   ```

2. **Memory Issues**
   ```bash
   # Check memory usage
   docker stats
   # Increase memory limits in docker-compose.yml
   ```

3. **Database Corruption**
   ```bash
   # Check database integrity
   sqlite3 /opt/second-brain/data/notes.db "PRAGMA integrity_check;"
   ```

### Debug Mode

Enable debug logging:
```bash
# Add to .env.production
DEBUG=true
LOG_LEVEL=debug
```

## Production Checklist

- [ ] Changed all default secrets
- [ ] Configured firewall
- [ ] Set up SSL certificates
- [ ] Configured backup strategy
- [ ] Set up monitoring
- [ ] Tested disaster recovery
- [ ] Configured log rotation
- [ ] Set up health check monitoring
- [ ] Documented deployment-specific configurations

## Support

For deployment issues:
1. Check application logs
2. Verify environment configuration
3. Test individual services
4. Check network connectivity
5. Review security settings

---

**Security Note**: This deployment includes comprehensive security measures, but additional hardening may be required based on your specific environment and compliance requirements.