"""
Configuration-Driven Database Setup for Autom8.

This module provides a configuration-based approach to setting up and
managing the Autom8 database system. It integrates with the Autom8
settings system to provide seamless database configuration from
YAML files, environment variables, and programmatic settings.

Key Features:
- Integration with Autom8Settings for unified configuration
- Support for different deployment environments (dev, test, prod)
- Automatic configuration validation and optimization
- Environment-specific database settings
- Configuration templates and presets
- Validation and recommendation system
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator

from autom8.config.settings import Autom8Settings
from autom8.storage.sqlite.database_setup import DatabaseSetup, DatabaseSetupConfig
from autom8.storage.sqlite.vector_manager import VectorSearchConfig
from autom8.storage.sqlite.connection_manager import RetryConfig
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class DatabaseEnvironmentConfig(BaseModel):
    """Environment-specific database configuration."""

    # Basic settings
    db_path: str = Field("./autom8.db", description="Database file path")
    enable_wal: bool = Field(True, description="Enable WAL journal mode")
    enable_foreign_keys: bool = Field(True, description="Enable foreign key constraints")

    # Performance settings
    cache_size: int = Field(10000, ge=1000, le=100000, description="Cache size in pages")
    page_size: int = Field(4096, description="Database page size")
    auto_vacuum: bool = Field(True, description="Enable auto vacuum")

    # Vector search settings
    embedding_model: str = Field("all-MiniLM-L6-v2", description="Default embedding model")
    embedding_dimensions: int = Field(384, ge=128, le=2048, description="Embedding vector dimensions")
    similarity_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Default similarity threshold")
    max_search_results: int = Field(50, ge=1, le=1000, description="Maximum search results")

    # Connection settings
    max_connections: int = Field(20, ge=1, le=100, description="Maximum database connections")
    connection_timeout: float = Field(30.0, ge=1.0, le=300.0, description="Connection timeout in seconds")
    retry_attempts: int = Field(3, ge=1, le=10, description="Connection retry attempts")

    # Maintenance settings
    auto_cleanup_days: int = Field(30, ge=1, le=365, description="Days to keep old records")
    health_check_interval: float = Field(300.0, ge=60.0, le=3600.0, description="Health check interval in seconds")

    @field_validator('db_path')
    @classmethod
    def validate_db_path(cls, v):
        """Validate database path."""
        path = Path(v)
        # Ensure parent directory can be created
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Cannot create database directory: {e}")
        return str(path)


class DatabaseConfigManager:
    """Manages database configuration across different environments."""

    # Predefined environment configurations
    ENVIRONMENT_CONFIGS = {
        'development': DatabaseEnvironmentConfig(
            db_path="./dev_autom8.db",
            cache_size=5000,
            max_connections=10,
            auto_cleanup_days=7,
            health_check_interval=60.0
        ),
        'testing': DatabaseEnvironmentConfig(
            db_path=":memory:",  # In-memory for tests
            cache_size=1000,
            max_connections=5,
            auto_cleanup_days=1,
            health_check_interval=30.0,
            enable_wal=False  # WAL not supported in memory
        ),
        'production': DatabaseEnvironmentConfig(
            db_path="/var/lib/autom8/autom8.db",
            cache_size=20000,
            max_connections=50,
            auto_cleanup_days=90,
            health_check_interval=300.0,
            page_size=8192  # Larger page size for production
        )
    }

    def __init__(self, autom8_settings: Autom8Settings = None):
        """Initialize with Autom8 settings."""
        self.settings = autom8_settings or Autom8Settings()
        self.current_config = None

    def get_environment(self) -> str:
        """Determine current environment."""
        # Check environment variable first
        env = os.getenv('AUTOM8_ENV', '').lower()
        if env in self.ENVIRONMENT_CONFIGS:
            return env

        # Check debug mode
        if self.settings.debug_mode:
            return 'development'

        # Default to production
        return 'production'

    def load_config_from_settings(self, environment: str = None) -> DatabaseEnvironmentConfig:
        """Load database configuration from Autom8 settings."""
        if environment is None:
            environment = self.get_environment()

        # Start with environment defaults
        base_config = self.ENVIRONMENT_CONFIGS.get(
            environment,
            self.ENVIRONMENT_CONFIGS['production']
        ).model_copy()

        # Override with settings from autom8.yaml and environment variables
        config_dict = base_config.model_dump()

        # Map Autom8Settings to database config
        setting_mappings = {
            'sqlite_path': 'db_path',
            # Add other mappings as needed
        }

        for setting_key, config_key in setting_mappings.items():
            if hasattr(self.settings, setting_key):
                value = getattr(self.settings, setting_key)
                if value is not None:
                    config_dict[config_key] = value

        # Create updated config
        self.current_config = DatabaseEnvironmentConfig(**config_dict)
        return self.current_config

    def load_config_from_file(self, config_path: str) -> DatabaseEnvironmentConfig:
        """Load configuration from YAML file."""
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Database config file not found: {config_path}")

        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        # Extract database section
        db_config = config_data.get('database', {})

        self.current_config = DatabaseEnvironmentConfig(**db_config)
        return self.current_config

    def save_config_to_file(self, config_path: str, config: DatabaseEnvironmentConfig = None):
        """Save configuration to YAML file."""
        config = config or self.current_config
        if config is None:
            raise ValueError("No configuration to save")

        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        config_data = {
            'database': config.model_dump()
        }

        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)

        logger.info(f"Database configuration saved to {config_path}")

    def validate_config(self, config: DatabaseEnvironmentConfig = None) -> Dict[str, Any]:
        """Validate database configuration and return recommendations."""
        config = config or self.current_config
        if config is None:
            raise ValueError("No configuration to validate")

        validation_report = {
            'valid': True,
            'warnings': [],
            'recommendations': [],
            'errors': []
        }

        # Check database path
        if config.db_path != ":memory:":
            db_path = Path(config.db_path)
            if not db_path.parent.exists():
                try:
                    db_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    validation_report['errors'].append(f"Cannot create database directory: {e}")
                    validation_report['valid'] = False

        # Performance recommendations
        if config.cache_size < 5000:
            validation_report['recommendations'].append(
                "Consider increasing cache_size to at least 5000 for better performance"
            )

        if config.page_size < 4096:
            validation_report['warnings'].append(
                "Page size below 4096 may impact performance"
            )

        # Connection settings validation
        if config.max_connections > 50:
            validation_report['warnings'].append(
                "High connection count may impact SQLite performance"
            )

        # Vector search validation
        if config.embedding_dimensions not in [384, 512, 768, 1536]:
            validation_report['warnings'].append(
                f"Embedding dimension {config.embedding_dimensions} is not a common size"
            )

        # Environment-specific checks
        environment = self.get_environment()

        if environment == 'production':
            if config.db_path.startswith('./'):
                validation_report['warnings'].append(
                    "Production database should use absolute path"
                )

            if config.auto_cleanup_days < 30:
                validation_report['recommendations'].append(
                    "Consider longer retention period for production (30+ days)"
                )

        return validation_report

    def create_setup_config(self, config: DatabaseEnvironmentConfig = None) -> DatabaseSetupConfig:
        """Create DatabaseSetupConfig from environment config."""
        config = config or self.current_config
        if config is None:
            raise ValueError("No configuration available")

        return DatabaseSetupConfig(
            db_path=config.db_path,
            embedding_model=config.embedding_model,
            embedding_dimensions=config.embedding_dimensions,
            enable_wal=config.enable_wal,
            enable_foreign_keys=config.enable_foreign_keys,
            cache_size=config.cache_size,
            similarity_threshold=config.similarity_threshold,
            auto_vacuum=config.auto_vacuum,
            page_size=config.page_size
        )

    def create_vector_config(self, config: DatabaseEnvironmentConfig = None) -> VectorSearchConfig:
        """Create VectorSearchConfig from environment config."""
        config = config or self.current_config
        if config is None:
            raise ValueError("No configuration available")

        return VectorSearchConfig(
            embedding_model=config.embedding_model,
            embedding_dimensions=config.embedding_dimensions,
            similarity_threshold=config.similarity_threshold,
            max_search_results=config.max_search_results,
            auto_cleanup_days=config.auto_cleanup_days
        )

    def create_retry_config(self, config: DatabaseEnvironmentConfig = None) -> RetryConfig:
        """Create RetryConfig from environment config."""
        config = config or self.current_config
        if config is None:
            raise ValueError("No configuration available")

        return RetryConfig(
            max_attempts=config.retry_attempts,
            base_delay_seconds=1.0,
            max_delay_seconds=min(config.connection_timeout / 2, 30.0)
        )


async def setup_database_from_config(
    config_path: str = None,
    environment: str = None,
    autom8_settings: Autom8Settings = None,
    validate_only: bool = False,
    reset_if_exists: bool = False
) -> Dict[str, Any]:
    """
    Set up database using configuration-driven approach.

    Args:
        config_path: Path to YAML configuration file
        environment: Environment name (dev, test, prod)
        autom8_settings: Autom8Settings instance
        validate_only: Only validate configuration, don't set up database
        reset_if_exists: Reset database if it exists

    Returns:
        Setup result dictionary
    """
    logger.info("Starting configuration-driven database setup...")

    # Initialize config manager
    config_manager = DatabaseConfigManager(autom8_settings)

    # Load configuration
    if config_path:
        db_config = config_manager.load_config_from_file(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        db_config = config_manager.load_config_from_settings(environment)
        logger.info(f"Loaded configuration for environment: {config_manager.get_environment()}")

    # Validate configuration
    validation_report = config_manager.validate_config(db_config)

    if validate_only:
        return {
            'validation_report': validation_report,
            'configuration': db_config.model_dump()
        }

    if not validation_report['valid']:
        logger.error("Configuration validation failed")
        return {
            'success': False,
            'validation_report': validation_report,
            'error': "Configuration validation failed"
        }

    # Log warnings and recommendations
    for warning in validation_report['warnings']:
        logger.warning(f"Config warning: {warning}")

    for recommendation in validation_report['recommendations']:
        logger.info(f"Config recommendation: {recommendation}")

    # Create setup configuration
    setup_config = config_manager.create_setup_config(db_config)

    # Set up database
    try:
        database_setup = DatabaseSetup(setup_config)

        # Reset if requested
        if reset_if_exists and Path(db_config.db_path).exists():
            logger.info("Resetting existing database...")
            await database_setup.reset_database()

        # Run setup
        setup_success = await database_setup.setup_complete_database()

        if setup_success:
            logger.info("Database setup completed successfully")

            # Perform post-setup validation
            post_validation = await database_setup.validate_setup()

            return {
                'success': True,
                'database_path': db_config.db_path,
                'environment': config_manager.get_environment(),
                'configuration': db_config.model_dump(),
                'validation_report': validation_report,
                'post_setup_validation': post_validation
            }
        else:
            logger.error("Database setup failed")
            return {
                'success': False,
                'error': "Database setup failed",
                'validation_report': validation_report
            }

    except Exception as e:
        logger.error(f"Database setup error: {e}")
        return {
            'success': False,
            'error': str(e),
            'validation_report': validation_report
        }


def generate_config_template(output_path: str, environment: str = 'production'):
    """Generate a configuration template file."""
    config_manager = DatabaseConfigManager()

    # Get base config for environment
    base_config = config_manager.ENVIRONMENT_CONFIGS.get(
        environment,
        config_manager.ENVIRONMENT_CONFIGS['production']
    )

    # Create template with comments
    template_data = {
        'database': {
            '# Database file path': None,
            'db_path': base_config.db_path,

            '# Performance settings': None,
            'cache_size': base_config.cache_size,
            'page_size': base_config.page_size,
            'auto_vacuum': base_config.auto_vacuum,

            '# Vector search settings': None,
            'embedding_model': base_config.embedding_model,
            'embedding_dimensions': base_config.embedding_dimensions,
            'similarity_threshold': base_config.similarity_threshold,
            'max_search_results': base_config.max_search_results,

            '# Connection settings': None,
            'max_connections': base_config.max_connections,
            'connection_timeout': base_config.connection_timeout,
            'retry_attempts': base_config.retry_attempts,

            '# Maintenance settings': None,
            'auto_cleanup_days': base_config.auto_cleanup_days,
            'health_check_interval': base_config.health_check_interval
        }
    }

    # Remove comment entries (they're just for organization)
    clean_data = {
        'database': {k: v for k, v in template_data['database'].items() if v is not None}
    }

    # Write template
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write(f"# Autom8 Database Configuration Template\n")
        f.write(f"# Environment: {environment}\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
        yaml.dump(clean_data, f, default_flow_style=False, indent=2)

    logger.info(f"Configuration template generated: {output_path}")


async def main():
    """Main function for CLI usage."""
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Autom8 Database Configuration Setup")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--environment", choices=['development', 'testing', 'production'],
                       help="Environment preset")
    parser.add_argument("--validate-only", action="store_true", help="Only validate configuration")
    parser.add_argument("--reset", action="store_true", help="Reset database if exists")
    parser.add_argument("--generate-template", help="Generate config template at path")
    parser.add_argument("--template-env", default="production",
                       choices=['development', 'testing', 'production'],
                       help="Environment for template generation")

    args = parser.parse_args()

    # Generate template if requested
    if args.generate_template:
        generate_config_template(args.generate_template, args.template_env)
        return

    # Setup database
    result = await setup_database_from_config(
        config_path=args.config,
        environment=args.environment,
        validate_only=args.validate_only,
        reset_if_exists=args.reset
    )

    print(json.dumps(result, indent=2))

    # Exit with appropriate code
    if result.get('success', False) or args.validate_only:
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())