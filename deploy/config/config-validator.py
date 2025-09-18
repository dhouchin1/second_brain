#!/usr/bin/env python3
"""
Second Brain - Production Configuration Validator

This script validates production environment configuration files to ensure
all required settings are present and properly configured before deployment.
"""

import os
import sys
import re
import urllib.parse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates Second Brain production configuration."""
    
    REQUIRED_VARS = {
        # Core application
        'ENVIRONMENT': str,
        'SECRET_KEY': str,
        'DOMAIN_NAME': str,
        'BASE_URL': str,
        
        # Database
        'DATABASE_URL': str,
        
        # Cache and messaging
        'REDIS_URL': str,
        'CELERY_BROKER_URL': str,
        'CELERY_RESULT_BACKEND': str,
        
        # File paths
        'VAULT_PATH': str,
        'AUDIO_DIR': str,
        'UPLOAD_DIR': str,
        
        # AI services
        'OLLAMA_API_URL': str,
        'OLLAMA_MODEL': str,
        'WHISPER_CPP_PATH': str,
        'WHISPER_MODEL_PATH': str,
        
        # Security
        'JWT_SECRET_KEY': str,
        'SESSION_SECRET_KEY': str,
    }
    
    OPTIONAL_VARS = {
        # External integrations
        'DISCORD_TOKEN': str,
        'SMTP_HOST': str,
        'SMTP_USERNAME': str,
        'SMTP_PASSWORD': str,
        
        # SSL
        'LETSENCRYPT_EMAIL': str,
        
        # Monitoring
        'SENTRY_DSN': str,
        'GRAFANA_PASSWORD': str,
        'POSTGRES_PASSWORD': str,
    }
    
    SECURITY_REQUIREMENTS = {
        'SECRET_KEY': {
            'min_length': 32,
            'no_default_values': ['change-this', 'secret-key', 'your-secret'],
        },
        'JWT_SECRET_KEY': {
            'min_length': 32,
            'no_default_values': ['change-this', 'jwt-secret'],
        },
        'SESSION_SECRET_KEY': {
            'min_length': 32,
            'no_default_values': ['change-this', 'session-secret'],
        },
    }
    
    def __init__(self, config_file: str):
        """Initialize validator with configuration file path."""
        self.config_file = Path(config_file)
        self.config = {}
        self.errors = []
        self.warnings = []
        
    def load_config(self) -> bool:
        """Load environment configuration from file."""
        try:
            if not self.config_file.exists():
                self.errors.append(f"Configuration file not found: {self.config_file}")
                return False
                
            with open(self.config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    if '=' in line:
                        key, value = line.split('=', 1)
                        self.config[key.strip()] = value.strip()
                        
            logger.info(f"Loaded {len(self.config)} configuration variables")
            return True
            
        except Exception as e:
            self.errors.append(f"Failed to load configuration: {str(e)}")
            return False
    
    def validate_required_vars(self) -> None:
        """Validate that all required variables are present."""
        for var_name, var_type in self.REQUIRED_VARS.items():
            if var_name not in self.config:
                self.errors.append(f"Missing required variable: {var_name}")
            elif not self.config[var_name]:
                self.errors.append(f"Empty required variable: {var_name}")
    
    def validate_security_settings(self) -> None:
        """Validate security-related configuration."""
        for var_name, requirements in self.SECURITY_REQUIREMENTS.items():
            if var_name not in self.config:
                continue
                
            value = self.config[var_name]
            
            # Check minimum length
            min_length = requirements.get('min_length', 0)
            if len(value) < min_length:
                self.errors.append(
                    f"{var_name} must be at least {min_length} characters long"
                )
            
            # Check for default/insecure values
            no_defaults = requirements.get('no_default_values', [])
            for default in no_defaults:
                if default.lower() in value.lower():
                    self.errors.append(
                        f"{var_name} contains insecure default value: {default}"
                    )
    
    def validate_urls(self) -> None:
        """Validate URL format and accessibility."""
        url_vars = [
            'BASE_URL', 'OLLAMA_API_URL', 'DATABASE_URL', 
            'REDIS_URL', 'CELERY_BROKER_URL', 'CELERY_RESULT_BACKEND'
        ]
        
        for var_name in url_vars:
            if var_name not in self.config:
                continue
                
            url = self.config[var_name]
            
            try:
                parsed = urllib.parse.urlparse(url)
                if not parsed.scheme:
                    self.errors.append(f"{var_name} missing scheme: {url}")
                if not parsed.netloc and not url.startswith('sqlite'):
                    self.errors.append(f"{var_name} missing host: {url}")
                    
            except Exception as e:
                self.errors.append(f"Invalid URL format for {var_name}: {str(e)}")
    
    def validate_paths(self) -> None:
        """Validate file and directory paths."""
        path_vars = [
            'VAULT_PATH', 'AUDIO_DIR', 'UPLOAD_DIR', 'STATIC_DIR', 'LOG_DIR',
            'WHISPER_CPP_PATH', 'WHISPER_MODEL_PATH'
        ]
        
        for var_name in path_vars:
            if var_name not in self.config:
                continue
                
            path_str = self.config[var_name]
            path = Path(path_str)
            
            # Check for absolute paths in production
            if self.config.get('ENVIRONMENT') == 'production':
                if not path.is_absolute():
                    self.warnings.append(
                        f"{var_name} should use absolute path in production: {path_str}"
                    )
            
            # Check if executable files exist (for specific paths)
            if var_name in ['WHISPER_CPP_PATH'] and path.exists():
                if not os.access(path, os.X_OK):
                    self.errors.append(f"{var_name} is not executable: {path_str}")
    
    def validate_environment_specific(self) -> None:
        """Validate environment-specific settings."""
        environment = self.config.get('ENVIRONMENT', '').lower()
        
        if environment == 'production':
            # Production-specific validations
            if self.config.get('DEBUG', 'false').lower() == 'true':
                self.errors.append("DEBUG should be false in production")
                
            if self.config.get('LOG_LEVEL', '').upper() not in ['INFO', 'WARNING', 'ERROR']:
                self.warnings.append("Consider using INFO, WARNING, or ERROR log level in production")
                
            # SSL should be enabled
            if self.config.get('SSL_ENABLED', 'false').lower() != 'true':
                self.warnings.append("SSL should be enabled in production")
                
            # Security headers should be enabled
            if self.config.get('SECURITY_HEADERS_ENABLED', 'false').lower() != 'true':
                self.warnings.append("Security headers should be enabled in production")
        
        elif environment == 'staging':
            # Staging-specific validations
            if not self.config.get('DOMAIN_NAME', '').startswith('staging'):
                self.warnings.append("Staging domain should typically include 'staging'")
    
    def validate_integrations(self) -> None:
        """Validate external service integrations."""
        # Discord integration
        if self.config.get('DISCORD_ENABLED', 'false').lower() == 'true':
            if not self.config.get('DISCORD_TOKEN'):
                self.errors.append("DISCORD_TOKEN required when Discord is enabled")
            if not self.config.get('DISCORD_GUILD_ID'):
                self.errors.append("DISCORD_GUILD_ID required when Discord is enabled")
        
        # Email integration
        if self.config.get('EMAIL_ENABLED', 'false').lower() == 'true':
            email_vars = ['SMTP_HOST', 'SMTP_USERNAME', 'SMTP_PASSWORD', 'EMAIL_FROM']
            for var in email_vars:
                if not self.config.get(var):
                    self.errors.append(f"{var} required when email is enabled")
        
        # SSL configuration
        if self.config.get('SSL_ENABLED', 'false').lower() == 'true':
            if not self.config.get('LETSENCRYPT_EMAIL'):
                self.warnings.append("LETSENCRYPT_EMAIL recommended for SSL setup")
    
    def validate_resource_limits(self) -> None:
        """Validate resource limit configurations."""
        numeric_vars = {
            'MAX_CONTENT_LENGTH': (1048576, 1073741824),  # 1MB to 1GB
            'GUNICORN_WORKERS': (1, 32),
            'GUNICORN_TIMEOUT': (10, 300),
            'CELERY_WORKER_CONCURRENCY': (1, 16),
            'DATABASE_POOL_SIZE': (1, 100),
        }
        
        for var_name, (min_val, max_val) in numeric_vars.items():
            if var_name not in self.config:
                continue
                
            try:
                value = int(self.config[var_name])
                if not (min_val <= value <= max_val):
                    self.warnings.append(
                        f"{var_name} should be between {min_val} and {max_val}, got {value}"
                    )
            except ValueError:
                self.errors.append(f"{var_name} must be a valid integer")
    
    def validate_consistency(self) -> None:
        """Validate configuration consistency."""
        # Check URL consistency
        base_url = self.config.get('BASE_URL', '')
        domain_name = self.config.get('DOMAIN_NAME', '')
        
        if base_url and domain_name:
            if domain_name not in base_url:
                self.warnings.append(
                    f"BASE_URL ({base_url}) should contain DOMAIN_NAME ({domain_name})"
                )
        
        # Check Redis consistency
        redis_url = self.config.get('REDIS_URL', '')
        celery_broker = self.config.get('CELERY_BROKER_URL', '')
        
        if redis_url and celery_broker and redis_url != celery_broker:
            self.warnings.append(
                "REDIS_URL and CELERY_BROKER_URL should typically be the same"
            )
    
    def validate_all(self) -> Tuple[bool, Dict[str, Any]]:
        """Run all validations and return results."""
        if not self.load_config():
            return False, {'errors': self.errors, 'warnings': self.warnings}
        
        # Run all validation checks
        self.validate_required_vars()
        self.validate_security_settings()
        self.validate_urls()
        self.validate_paths()
        self.validate_environment_specific()
        self.validate_integrations()
        self.validate_resource_limits()
        self.validate_consistency()
        
        success = len(self.errors) == 0
        
        return success, {
            'errors': self.errors,
            'warnings': self.warnings,
            'config_vars': len(self.config),
            'environment': self.config.get('ENVIRONMENT', 'unknown')
        }
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Print validation results in a formatted way."""
        print("\n" + "=" * 70)
        print("SECOND BRAIN CONFIGURATION VALIDATION RESULTS")
        print("=" * 70)
        
        print(f"Environment: {results['environment']}")
        print(f"Configuration variables loaded: {results['config_vars']}")
        
        if results['errors']:
            print(f"\n❌ ERRORS ({len(results['errors'])})")
            for error in results['errors']:
                print(f"   • {error}")
        
        if results['warnings']:
            print(f"\n⚠️  WARNINGS ({len(results['warnings'])})")
            for warning in results['warnings']:
                print(f"   • {warning}")
        
        if not results['errors'] and not results['warnings']:
            print("\n✅ CONFIGURATION VALID - No errors or warnings found!")
        elif not results['errors']:
            print(f"\n✅ CONFIGURATION VALID - {len(results['warnings'])} warnings found")
        else:
            print(f"\n❌ CONFIGURATION INVALID - {len(results['errors'])} errors found")
        
        print("=" * 70)


def main():
    """Main entry point for the configuration validator."""
    if len(sys.argv) != 2:
        print("Usage: python config-validator.py <config-file>")
        print("Example: python config-validator.py production.env")
        sys.exit(1)
    
    config_file = sys.argv[1]
    validator = ConfigValidator(config_file)
    
    success, results = validator.validate_all()
    validator.print_results(results)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()