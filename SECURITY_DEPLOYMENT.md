# Security Deployment Guide for Second Brain

## Overview
This document outlines the security configuration and best practices for deploying Second Brain in production environments.

## Security Architecture

### Authentication & Authorization
- **Session-based Authentication**: FastAPI sessions with secure JWT tokens
- **Multi-layer Authorization**: User dependencies on all protected endpoints
- **Webhook Token Authentication**: Bearer token validation for external integrations
- **Rate Limiting**: Comprehensive rate limiting using slowapi across all services

### Network Security

#### CORS Configuration
- **Main Application**: Restricted to specific origins (localhost for development)
- **Autom8 Service**: Secured CORS with explicit origin allowlist
- **No Wildcard Origins**: All wildcard (`*`) CORS policies have been removed

#### Security Headers
Both main app and Autom8 service implement comprehensive security headers:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'`
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`

### Rate Limiting

#### Main Application (app.py)
- Global rate limiting via slowapi middleware
- Per-endpoint limits based on functionality
- Automatic rate limit exception handling

#### Autom8 Service (autom8_service.py)
- **Generate endpoint**: 10 requests/minute (AI generation is expensive)
- **Status endpoints**: 30 requests/minute (monitoring data)
- **System status**: 60 requests/minute (health checks)
- Uses client IP address for rate limiting key

### Input Validation & Sanitization

#### API Endpoints
- Pydantic models for request validation
- Type checking and constraint validation
- SQL injection protection via parameterized queries
- File upload size limits and type validation

#### Webhook Security
- Token-based authentication for external webhooks
- Request size limitations
- Content-type validation

## Environment Security

### Required Environment Variables
```bash
# Core Security
SECRET_KEY=<strong-random-256-bit-key>
WEBHOOK_TOKEN=<webhook-authentication-token>

# Autom8 Security (if using)
AUTOM8_JWT_SECRET=<different-strong-random-key>
AUTOM8_REQUIRE_AUTH=true
AUTOM8_RATE_LIMIT=true
```

### Secrets Management
- **Never commit secrets to version control**
- Use strong, randomly generated keys (minimum 256 bits)
- Rotate secrets regularly
- Store secrets in secure environment variable management

## Deployment Checklist

### Pre-deployment Security Audit
- [ ] All secrets are environment variables (not hardcoded)
- [ ] CORS is configured for production domains only
- [ ] Rate limiting is enabled and tested
- [ ] Security headers are implemented
- [ ] Input validation is comprehensive
- [ ] Database access uses parameterized queries
- [ ] File upload limits are configured appropriately

### Production Configuration
- [ ] Use HTTPS/TLS for all communications
- [ ] Configure reverse proxy (nginx/Apache) with security headers
- [ ] Enable fail2ban or similar for brute force protection
- [ ] Set up monitoring and alerting for security events
- [ ] Configure log aggregation for security analysis
- [ ] Implement database backups with encryption

### Infrastructure Security
- [ ] Server OS is hardened and updated
- [ ] Firewall rules restrict unnecessary access
- [ ] Database is on private network
- [ ] File storage has appropriate permissions
- [ ] Regular security updates are applied

## Monitoring & Alerting

### Security Event Monitoring
- Failed authentication attempts
- Rate limit violations
- Unusual API usage patterns
- File upload anomalies
- Database query failures

### Log Analysis
- Centralized logging for security events
- Regular review of access logs
- Automated alerting for suspicious patterns
- Log retention policies for compliance

## Incident Response

### Security Incident Procedures
1. **Immediate Response**
   - Identify and contain the threat
   - Preserve evidence for analysis
   - Notify relevant stakeholders

2. **Investigation**
   - Analyze logs and system state
   - Determine scope and impact
   - Document findings

3. **Recovery**
   - Patch vulnerabilities
   - Restore from clean backups if needed
   - Update security measures

4. **Post-Incident**
   - Conduct lessons learned review
   - Update security procedures
   - Implement additional safeguards

## Security Testing

### Regular Security Assessments
- Vulnerability scanning
- Penetration testing
- Code security reviews
- Dependency vulnerability checks

### Automated Security Testing
- SAST (Static Application Security Testing)
- DAST (Dynamic Application Security Testing)
- Dependency vulnerability scanning
- Container security scanning (if containerized)

## Compliance Considerations

### Data Protection
- Encryption at rest and in transit
- Data minimization principles
- User consent and data rights
- Regular data retention review

### Privacy Requirements
- GDPR compliance (if applicable)
- Data processing transparency
- User rights implementation
- Privacy impact assessments

## Security Updates

### Regular Maintenance
- Keep all dependencies updated
- Apply security patches promptly
- Review and update security configurations
- Conduct regular security training

### Change Management
- Security review for all code changes
- Secure deployment procedures
- Rollback plans for security issues
- Documentation of security changes

---

## Quick Security Verification

### Test Rate Limiting
```bash
# Test Autom8 service rate limits
for i in {1..15}; do
  curl -X GET "http://localhost:8000/api/models/status" -w "%{http_code}\n" -o /dev/null -s
done
# Should see 200s then 429s after limit exceeded
```

### Verify Security Headers
```bash
curl -I "http://localhost:8000/health"
# Should show security headers in response
```

### Check CORS Configuration
```bash
curl -H "Origin: http://malicious-site.com" \
     -H "Access-Control-Request-Method: POST" \
     -X OPTIONS "http://localhost:8000/api/generate"
# Should reject unknown origins
```

---

**Last Updated**: September 2025
**Review Schedule**: Quarterly security review required