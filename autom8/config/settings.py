"""
Configuration Management System for Autom8.

Loads settings from autom8.yaml, environment variables, and provides
a unified configuration interface with validation and defaults.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

import logging

logger = logging.getLogger(__name__)


class Autom8Settings(BaseSettings):
    """Main Autom8 configuration settings."""
    
    # Core settings
    debug_mode: bool = Field(False, description="Enable debug mode", alias="DEBUG_MODE")
    
    # Model routing
    prefer_local: bool = Field(True, description="Prefer local models when possible")
    local_quality_threshold: float = Field(0.85, ge=0, le=1, description="Accept local models with this quality threshold")
    
    # Context management
    default_max_tokens: int = Field(500, gt=0, description="Default maximum context tokens")
    hard_limit: int = Field(2000, gt=0, description="Hard limit for context tokens")
    always_preview: bool = Field(True, description="Always show context preview")
    
    # Shared memory
    redis_host: str = Field("localhost", description="Redis host", alias="REDIS_HOST")
    redis_port: int = Field(6379, description="Redis port", alias="REDIS_PORT")
    redis_url: Optional[str] = Field(None, description="Redis connection URL", alias="REDIS_URL")
    
    sqlite_path: str = Field("./autom8.db", description="SQLite database path")
    
    # API configuration
    ollama_host: str = Field("http://localhost:11434", description="Ollama host URL", alias="OLLAMA_HOST")
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key", alias="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key", alias="OPENAI_API_KEY")
    
    # Cost controls
    allow_cloud: bool = Field(False, description="Allow cloud model usage")
    daily_limit: float = Field(1.00, ge=0, description="Daily spending limit in USD")
    monthly_limit: float = Field(10.00, ge=0, description="Monthly spending limit in USD")
    
    # Logging
    log_level: str = Field("INFO", description="Logging level", alias="LOG_LEVEL")
    log_file: Optional[str] = Field("./autom8.log", description="Log file path")
    log_structured: bool = Field(True, description="Use structured logging")
    log_performance: bool = Field(True, description="Log performance metrics")
    log_file_enabled: bool = Field(True, description="Enable file logging")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields from YAML
    )
    
    def get_data_path(self) -> Path:
        """Get the data directory path for storing files."""
        # Use XDG Base Directory specification or fallback to current directory
        if "XDG_DATA_HOME" in os.environ:
            base_path = Path(os.environ["XDG_DATA_HOME"]) / "autom8"
        elif "HOME" in os.environ:
            base_path = Path(os.environ["HOME"]) / ".local" / "share" / "autom8"
        else:
            base_path = Path(".") / ".autom8"
        
        # Create directory if it doesn't exist
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path


def yaml_settings_source(settings: BaseSettings) -> Dict[str, Any]:
    """Load settings from autom8.yaml file."""
    yaml_file = Path("autom8.yaml")
    
    if not yaml_file.exists():
        logger.debug("autom8.yaml not found, using defaults")
        return {}
    
    try:
        with open(yaml_file, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        
        if yaml_data is None:
            return {}
        
        logger.info(f"Loaded configuration from {yaml_file}")
        return flatten_config(yaml_data)
        
    except Exception as e:
        logger.error(f"Error loading autom8.yaml: {e}")
        return {}


def flatten_config(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested configuration for pydantic."""
    result = {}
    
    for key, value in data.items():
        full_key = f"{prefix}_{key}" if prefix else key
        
        if isinstance(value, dict):
            result.update(flatten_config(value, full_key))
        else:
            result[full_key] = value
    
    return result


# Global settings instance
_settings: Optional[Autom8Settings] = None


def get_settings() -> Autom8Settings:
    """Get global settings instance."""
    global _settings
    
    if _settings is None:
        # Load from YAML first, then environment
        yaml_config = yaml_settings_source(None)
        
        # Create settings with YAML config as defaults
        _settings = Autom8Settings(**yaml_config)
        
        logger.info("Configuration loaded successfully")
    
    return _settings


def reload_settings() -> Autom8Settings:
    """Reload settings from configuration sources."""
    global _settings
    _settings = None
    return get_settings()


def create_default_config(path: Optional[Path] = None) -> None:
    """Create a default configuration file."""
    if path is None:
        path = Path("autom8.yaml")
    
    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    default_config = {
        "routing": {
            "thresholds": {
                "trivial": 0.2,
                "simple": 0.4,
                "moderate": 0.6,
                "complex": 0.8,
                "frontier": 1.0
            },
            "prefer_local": True,
            "local_quality_threshold": 0.85
        },
        "context": {
            "default_max_tokens": 500,
            "hard_limit": 2000,
            "always_preview": True,
            "allow_editing": True
        },
        "shared_memory": {
            "redis": {
                "host": "localhost",
                "port": 6379
            },
            "sqlite": {
                "path": "./autom8.db"
            }
        },
        "cost_controls": {
            "allow_cloud": False,
            "daily_limit": 1.00,
            "monthly_limit": 10.00
        },
        "logging": {
            "level": "INFO",
            "file": "./autom8.log",
            "structured": True
        }
    }
    
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Created default configuration at {path}")


# Initialize settings on module import - disabled for now
# try:
#     get_settings()
# except Exception as e:
#     logger.warning(f"Failed to initialize settings: {e}")
#     # Create default config if none exists
#     if not Path("autom8.yaml").exists():
#         create_default_config()