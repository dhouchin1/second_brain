"""
Configuration Management Module

Handles settings with Pydantic BaseSettings,
environment variable loading, and YAML configuration.
"""

from autom8.config.settings import Autom8Settings, get_settings

__all__ = [
    "Settings",
    "get_settings",
    "Autom8Config",
]