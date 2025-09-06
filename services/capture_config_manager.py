# ──────────────────────────────────────────────────────────────────────────────
# File: services/capture_config_manager.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Enhanced Configuration Management for Capture Operations

Provides configuration presets, dynamic validation, smart defaults, and 
content-type specific settings for the Second Brain capture system.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


class ConfigPreset(Enum):
    """Configuration preset levels."""
    BASIC = "basic"
    ADVANCED = "advanced"
    POWER_USER = "power_user"
    CUSTOM = "custom"


class SourceType(Enum):
    """Capture source types for configuration."""
    WEB_UI = "web_ui"
    DISCORD = "discord"
    APPLE_SHORTCUTS = "apple_shortcuts"
    API = "api"
    WEBHOOK = "webhook"
    BULK_UPLOAD = "bulk_upload"
    MOBILE = "mobile"


class ContentTypeConfig(Enum):
    """Content types for specific configurations."""
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    PDF = "pdf"
    VIDEO = "video"
    URL = "url"
    CODE = "code"
    MARKDOWN = "markdown"
    EMAIL = "email"


@dataclass
class ProcessingOptions:
    """Processing options for capture operations."""
    # AI Processing
    enable_ai_processing: bool = True
    enable_summarization: bool = True
    enable_title_generation: bool = True
    enable_tag_generation: bool = True
    enable_action_extraction: bool = True
    ai_model_preference: str = "local"  # local, cloud, hybrid
    
    # Content Processing
    enable_chunking: bool = True
    chunking_strategy: str = "adaptive"  # fixed, semantic, hierarchical, adaptive
    max_chunk_size: int = 8000
    min_chunk_size: int = 500
    chunk_overlap: int = 200
    
    # Embeddings
    enable_embeddings: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    embed_chunks_separately: bool = True
    
    # OCR and Extraction
    enable_ocr: bool = True
    ocr_language: str = "en"
    ocr_quality: str = "high"  # low, medium, high
    
    # Quality and Validation
    enable_quality_validation: bool = True
    min_quality_threshold: float = 0.3
    content_deduplication: bool = True
    dedup_window_days: int = 30
    
    # Performance
    processing_timeout_seconds: int = 300
    processing_priority: int = 1  # 1=normal, 2=high, 3=urgent
    enable_background_processing: bool = True
    
    # Storage
    compress_large_content: bool = False
    archive_old_content: bool = False
    retention_days: int = 0  # 0 = unlimited


@dataclass
class SecurityOptions:
    """Security-related configuration options."""
    require_authentication: bool = True
    allowed_file_types: Set[str] = field(default_factory=lambda: {
        'txt', 'md', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'wav', 'mp3', 'ogg'
    })
    max_file_size_mb: int = 25
    scan_for_sensitive_data: bool = True
    encrypt_at_rest: bool = False
    audit_logging: bool = True
    rate_limit_per_hour: int = 100


@dataclass
class UserPreferences:
    """User-specific preferences for capture operations."""
    default_tags: List[str] = field(default_factory=list)
    preferred_format: str = "markdown"  # markdown, text, html
    auto_categorize: bool = True
    notification_settings: Dict[str, bool] = field(default_factory=lambda: {
        'processing_complete': True,
        'errors': True,
        'warnings': False
    })
    privacy_mode: bool = False
    language: str = "en"
    timezone: str = "UTC"


@dataclass
class CaptureConfig:
    """Complete configuration for capture operations."""
    preset: ConfigPreset = ConfigPreset.BASIC
    processing: ProcessingOptions = field(default_factory=ProcessingOptions)
    security: SecurityOptions = field(default_factory=SecurityOptions)
    user_preferences: UserPreferences = field(default_factory=UserPreferences)
    source_specific: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    content_type_specific: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"


class ConfigValidator:
    """Validates configuration settings and dependencies."""
    
    def __init__(self):
        """Initialize the validator with validation rules."""
        self.validation_rules = {
            'max_chunk_size': lambda x: 100 <= x <= 50000,
            'min_chunk_size': lambda x: 10 <= x <= 5000,
            'chunk_overlap': lambda x: 0 <= x <= 1000,
            'processing_timeout_seconds': lambda x: 30 <= x <= 3600,
            'max_file_size_mb': lambda x: 1 <= x <= 100,
            'rate_limit_per_hour': lambda x: 1 <= x <= 1000,
            'min_quality_threshold': lambda x: 0.0 <= x <= 1.0,
            'dedup_window_days': lambda x: 0 <= x <= 365,
        }
        
        self.dependency_rules = [
            # If chunking is disabled, embedding chunks separately makes no sense
            (lambda cfg: not cfg.processing.enable_chunking and cfg.processing.embed_chunks_separately,
             "Cannot embed chunks separately when chunking is disabled"),
            
            # Quality validation requires minimum thresholds
            (lambda cfg: cfg.processing.enable_quality_validation and cfg.processing.min_quality_threshold <= 0,
             "Quality validation requires a minimum quality threshold > 0"),
            
            # Chunk overlap cannot be larger than chunk size
            (lambda cfg: cfg.processing.chunk_overlap >= cfg.processing.max_chunk_size,
             "Chunk overlap cannot be larger than maximum chunk size"),
            
            # Minimum chunk size should be reasonable compared to maximum
            (lambda cfg: cfg.processing.min_chunk_size >= cfg.processing.max_chunk_size * 0.8,
             "Minimum chunk size is too close to maximum chunk size"),
        ]
    
    def validate_config(self, config: CaptureConfig) -> Tuple[bool, List[str]]:
        """Validate a configuration and return issues found."""
        issues = []
        
        # Validate individual fields
        for field_name, validator in self.validation_rules.items():
            if hasattr(config.processing, field_name):
                value = getattr(config.processing, field_name)
                if not validator(value):
                    issues.append(f"Invalid value for {field_name}: {value}")
            elif hasattr(config.security, field_name):
                value = getattr(config.security, field_name)
                if not validator(value):
                    issues.append(f"Invalid value for {field_name}: {value}")
        
        # Validate dependencies
        for dependency_check, error_message in self.dependency_rules:
            if dependency_check(config):
                issues.append(error_message)
        
        # Validate file types
        allowed_extensions = {
            'txt', 'md', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 
            'wav', 'mp3', 'ogg', 'flac', 'mp4', 'avi', 'mov'
        }
        invalid_types = config.security.allowed_file_types - allowed_extensions
        if invalid_types:
            issues.append(f"Unsupported file types: {invalid_types}")
        
        return len(issues) == 0, issues
    
    def suggest_fixes(self, config: CaptureConfig, issues: List[str]) -> Dict[str, Any]:
        """Suggest fixes for configuration issues."""
        suggestions = {}
        
        for issue in issues:
            if "chunk_overlap" in issue and "larger than maximum" in issue:
                suggestions['processing.chunk_overlap'] = min(
                    config.processing.chunk_overlap, 
                    config.processing.max_chunk_size // 4
                )
            
            elif "minimum chunk size" in issue:
                suggestions['processing.min_chunk_size'] = min(
                    config.processing.min_chunk_size,
                    config.processing.max_chunk_size // 2
                )
            
            elif "quality threshold" in issue:
                suggestions['processing.min_quality_threshold'] = 0.3
            
            elif "embed chunks separately" in issue:
                suggestions['processing.embed_chunks_separately'] = False
        
        return suggestions


class ConfigPresetManager:
    """Manages configuration presets and smart defaults."""
    
    def __init__(self):
        """Initialize with predefined presets."""
        self.presets = {
            ConfigPreset.BASIC: self._create_basic_preset(),
            ConfigPreset.ADVANCED: self._create_advanced_preset(),
            ConfigPreset.POWER_USER: self._create_power_user_preset()
        }
        
        # Content-type specific defaults
        self.content_type_defaults = {
            ContentTypeConfig.TEXT: {
                'enable_chunking': True,
                'chunking_strategy': 'semantic',
                'enable_summarization': True
            },
            ContentTypeConfig.AUDIO: {
                'enable_chunking': False,
                'enable_summarization': True,
                'processing_timeout_seconds': 600
            },
            ContentTypeConfig.IMAGE: {
                'enable_chunking': False,
                'enable_ocr': True,
                'ocr_quality': 'high'
            },
            ContentTypeConfig.PDF: {
                'enable_chunking': True,
                'chunking_strategy': 'hierarchical',
                'max_chunk_size': 10000
            },
            ContentTypeConfig.CODE: {
                'enable_chunking': True,
                'chunking_strategy': 'semantic',
                'enable_summarization': False,  # Code summaries often not helpful
                'enable_tag_generation': True
            },
            ContentTypeConfig.URL: {
                'enable_chunking': True,
                'chunking_strategy': 'adaptive',
                'enable_summarization': True,
                'processing_timeout_seconds': 120
            }
        }
        
        # Source-specific defaults
        self.source_defaults = {
            SourceType.MOBILE: {
                'processing_priority': 2,  # Higher priority for mobile
                'enable_background_processing': True,
                'processing_timeout_seconds': 180  # Shorter timeout for mobile
            },
            SourceType.DISCORD: {
                'enable_summarization': True,
                'enable_action_extraction': True,
                'auto_categorize': True
            },
            SourceType.BULK_UPLOAD: {
                'processing_priority': 1,  # Normal priority for bulk
                'enable_background_processing': True,
                'processing_timeout_seconds': 600  # Longer timeout for bulk
            }
        }
    
    def _create_basic_preset(self) -> CaptureConfig:
        """Create basic configuration preset."""
        return CaptureConfig(
            preset=ConfigPreset.BASIC,
            processing=ProcessingOptions(
                enable_ai_processing=True,
                enable_summarization=True,
                enable_title_generation=True,
                enable_tag_generation=False,  # Keep it simple
                enable_action_extraction=False,
                enable_chunking=False,  # No chunking for basic users
                enable_embeddings=True,
                enable_ocr=True,
                enable_quality_validation=False,
                content_deduplication=True,
                processing_timeout_seconds=120
            ),
            security=SecurityOptions(
                max_file_size_mb=10,  # Smaller limit for basic
                rate_limit_per_hour=50
            ),
            user_preferences=UserPreferences(
                auto_categorize=True,
                notification_settings={'processing_complete': True, 'errors': True, 'warnings': False}
            )
        )
    
    def _create_advanced_preset(self) -> CaptureConfig:
        """Create advanced configuration preset."""
        return CaptureConfig(
            preset=ConfigPreset.ADVANCED,
            processing=ProcessingOptions(
                enable_ai_processing=True,
                enable_summarization=True,
                enable_title_generation=True,
                enable_tag_generation=True,
                enable_action_extraction=True,
                enable_chunking=True,
                chunking_strategy="adaptive",
                enable_embeddings=True,
                embed_chunks_separately=True,
                enable_ocr=True,
                enable_quality_validation=True,
                content_deduplication=True,
                processing_timeout_seconds=300
            ),
            security=SecurityOptions(
                max_file_size_mb=25,
                rate_limit_per_hour=100,
                scan_for_sensitive_data=True
            ),
            user_preferences=UserPreferences(
                auto_categorize=True,
                notification_settings={'processing_complete': True, 'errors': True, 'warnings': True}
            )
        )
    
    def _create_power_user_preset(self) -> CaptureConfig:
        """Create power user configuration preset."""
        return CaptureConfig(
            preset=ConfigPreset.POWER_USER,
            processing=ProcessingOptions(
                enable_ai_processing=True,
                enable_summarization=True,
                enable_title_generation=True,
                enable_tag_generation=True,
                enable_action_extraction=True,
                enable_chunking=True,
                chunking_strategy="hierarchical",
                max_chunk_size=10000,
                enable_embeddings=True,
                embed_chunks_separately=True,
                enable_ocr=True,
                ocr_quality="high",
                enable_quality_validation=True,
                min_quality_threshold=0.5,
                content_deduplication=True,
                processing_timeout_seconds=600,
                enable_background_processing=True,
                compress_large_content=True
            ),
            security=SecurityOptions(
                max_file_size_mb=50,
                rate_limit_per_hour=200,
                scan_for_sensitive_data=True,
                audit_logging=True
            ),
            user_preferences=UserPreferences(
                auto_categorize=True,
                notification_settings={'processing_complete': True, 'errors': True, 'warnings': True}
            )
        )
    
    def get_preset(self, preset: ConfigPreset) -> CaptureConfig:
        """Get a configuration preset."""
        return self.presets[preset]
    
    def create_optimized_config(
        self,
        base_preset: ConfigPreset,
        source_type: Optional[SourceType] = None,
        content_type: Optional[ContentTypeConfig] = None,
        user_overrides: Optional[Dict[str, Any]] = None
    ) -> CaptureConfig:
        """Create an optimized configuration based on context."""
        # Start with base preset
        config = self.get_preset(base_preset)
        
        # Apply content-type specific optimizations
        if content_type and content_type in self.content_type_defaults:
            content_defaults = self.content_type_defaults[content_type]
            for key, value in content_defaults.items():
                if hasattr(config.processing, key):
                    setattr(config.processing, key, value)
            
            # Store content-type specific settings
            config.content_type_specific[content_type.value] = content_defaults
        
        # Apply source-specific optimizations
        if source_type and source_type in self.source_defaults:
            source_defaults = self.source_defaults[source_type]
            for key, value in source_defaults.items():
                if hasattr(config.processing, key):
                    setattr(config.processing, key, value)
                elif hasattr(config.user_preferences, key):
                    setattr(config.user_preferences, key, value)
            
            # Store source-specific settings
            config.source_specific[source_type.value] = source_defaults
        
        # Apply user overrides
        if user_overrides:
            config = self._apply_user_overrides(config, user_overrides)
        
        # Update timestamp
        config.updated_at = datetime.now()
        
        return config
    
    def _apply_user_overrides(self, config: CaptureConfig, overrides: Dict[str, Any]) -> CaptureConfig:
        """Apply user-specific configuration overrides."""
        for key, value in overrides.items():
            # Handle nested keys like 'processing.enable_ai_processing'
            if '.' in key:
                section, field = key.split('.', 1)
                if hasattr(config, section):
                    section_obj = getattr(config, section)
                    if hasattr(section_obj, field):
                        setattr(section_obj, field, value)
            else:
                # Top-level keys
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return config


class CaptureConfigManager:
    """Main configuration manager for capture operations."""
    
    def __init__(self, config_storage_path: Optional[str] = None):
        """Initialize configuration manager."""
        self.validator = ConfigValidator()
        self.preset_manager = ConfigPresetManager()
        self.storage_path = Path(config_storage_path or settings.base_dir / "config" / "capture_configs")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory config cache
        self.config_cache: Dict[str, CaptureConfig] = {}
        self.cache_ttl = timedelta(hours=1)
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Configuration change listeners
        self.change_listeners: List[Callable[[str, CaptureConfig], None]] = []
    
    def get_config_for_operation(
        self,
        user_id: Optional[str] = None,
        source_type: Optional[str] = None,
        content_type: Optional[str] = None,
        preset: Optional[str] = None
    ) -> CaptureConfig:
        """
        Get optimized configuration for a specific capture operation.
        
        Args:
            user_id: User identifier for personalized settings
            source_type: Source of the capture (discord, mobile, etc.)
            content_type: Type of content being captured
            preset: Specific preset to use (basic, advanced, power_user)
            
        Returns:
            CaptureConfig: Optimized configuration for the operation
        """
        # Generate cache key
        cache_key = f"{user_id or 'default'}:{source_type or 'any'}:{content_type or 'any'}:{preset or 'auto'}"
        
        # Check cache first
        if self._is_cached_config_valid(cache_key):
            return self.config_cache[cache_key]
        
        # Determine preset
        if preset:
            try:
                config_preset = ConfigPreset(preset)
            except ValueError:
                logger.warning(f"Invalid preset '{preset}', using ADVANCED")
                config_preset = ConfigPreset.ADVANCED
        else:
            # Auto-determine preset based on context
            config_preset = self._determine_optimal_preset(user_id, source_type, content_type)
        
        # Map string types to enums
        source_enum = None
        if source_type:
            try:
                source_enum = SourceType(source_type)
            except ValueError:
                logger.warning(f"Unknown source type: {source_type}")
        
        content_enum = None
        if content_type:
            try:
                content_enum = ContentTypeConfig(content_type)
            except ValueError:
                logger.warning(f"Unknown content type: {content_type}")
        
        # Load user-specific overrides
        user_overrides = self._load_user_overrides(user_id) if user_id else {}
        
        # Create optimized configuration
        config = self.preset_manager.create_optimized_config(
            base_preset=config_preset,
            source_type=source_enum,
            content_type=content_enum,
            user_overrides=user_overrides
        )
        
        # Validate configuration
        is_valid, issues = self.validator.validate_config(config)
        if not is_valid:
            logger.warning(f"Configuration validation failed: {issues}")
            
            # Try to auto-fix issues
            suggestions = self.validator.suggest_fixes(config, issues)
            if suggestions:
                config = self._apply_auto_fixes(config, suggestions)
                logger.info(f"Applied auto-fixes: {list(suggestions.keys())}")
        
        # Cache the configuration
        self.config_cache[cache_key] = config
        self.cache_timestamps[cache_key] = datetime.now()
        
        return config
    
    def save_user_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any],
        merge_with_existing: bool = True
    ) -> bool:
        """Save user-specific configuration preferences."""
        try:
            user_config_file = self.storage_path / f"user_{user_id}.json"
            
            existing_prefs = {}
            if merge_with_existing and user_config_file.exists():
                with open(user_config_file, 'r') as f:
                    existing_prefs = json.load(f)
            
            # Merge preferences
            if merge_with_existing:
                existing_prefs.update(preferences)
                final_prefs = existing_prefs
            else:
                final_prefs = preferences
            
            # Add metadata
            final_prefs['_metadata'] = {
                'updated_at': datetime.now().isoformat(),
                'version': '1.0.0'
            }
            
            # Save to file
            with open(user_config_file, 'w') as f:
                json.dump(final_prefs, f, indent=2, default=str)
            
            # Invalidate cache for this user
            self._invalidate_user_cache(user_id)
            
            # Notify listeners
            for listener in self.change_listeners:
                try:
                    # Create a dummy config for the listener
                    dummy_config = self.get_config_for_operation(user_id=user_id)
                    listener(user_id, dummy_config)
                except Exception as e:
                    logger.warning(f"Config change listener failed: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save user preferences for {user_id}: {e}")
            return False
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user-specific configuration preferences."""
        return self._load_user_overrides(user_id)
    
    def validate_configuration(self, config: CaptureConfig) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate a configuration and return validation results."""
        is_valid, issues = self.validator.validate_config(config)
        suggestions = self.validator.suggest_fixes(config, issues) if issues else {}
        
        return is_valid, issues, suggestions
    
    def get_available_presets(self) -> List[Dict[str, Any]]:
        """Get information about available configuration presets."""
        presets = []
        
        for preset_enum in ConfigPreset:
            if preset_enum == ConfigPreset.CUSTOM:
                continue
                
            config = self.preset_manager.get_preset(preset_enum)
            presets.append({
                'id': preset_enum.value,
                'name': preset_enum.value.replace('_', ' ').title(),
                'description': self._get_preset_description(preset_enum),
                'features': self._get_preset_features(config)
            })
        
        return presets
    
    def get_configuration_schema(self) -> Dict[str, Any]:
        """Get the configuration schema for UI generation."""
        return {
            'processing': {
                'enable_ai_processing': {'type': 'boolean', 'default': True, 'description': 'Enable AI-powered content enhancement'},
                'enable_summarization': {'type': 'boolean', 'default': True, 'description': 'Generate content summaries'},
                'enable_chunking': {'type': 'boolean', 'default': True, 'description': 'Split large content into chunks'},
                'chunking_strategy': {
                    'type': 'enum', 
                    'options': ['fixed', 'semantic', 'hierarchical', 'adaptive'],
                    'default': 'adaptive',
                    'description': 'Strategy for content chunking'
                },
                'max_chunk_size': {'type': 'integer', 'min': 100, 'max': 50000, 'default': 8000, 'description': 'Maximum chunk size in characters'},
                'enable_embeddings': {'type': 'boolean', 'default': True, 'description': 'Generate semantic embeddings'},
                'processing_timeout_seconds': {'type': 'integer', 'min': 30, 'max': 3600, 'default': 300, 'description': 'Processing timeout'}
            },
            'security': {
                'max_file_size_mb': {'type': 'integer', 'min': 1, 'max': 100, 'default': 25, 'description': 'Maximum file upload size'},
                'allowed_file_types': {'type': 'array', 'items': 'string', 'description': 'Allowed file extensions'},
                'rate_limit_per_hour': {'type': 'integer', 'min': 1, 'max': 1000, 'default': 100, 'description': 'API rate limit per hour'}
            },
            'user_preferences': {
                'auto_categorize': {'type': 'boolean', 'default': True, 'description': 'Automatically categorize content'},
                'preferred_format': {
                    'type': 'enum',
                    'options': ['markdown', 'text', 'html'],
                    'default': 'markdown',
                    'description': 'Preferred output format'
                }
            }
        }
    
    def register_change_listener(self, listener: Callable[[str, CaptureConfig], None]):
        """Register a listener for configuration changes."""
        self.change_listeners.append(listener)
    
    def _is_cached_config_valid(self, cache_key: str) -> bool:
        """Check if cached configuration is still valid."""
        if cache_key not in self.config_cache:
            return False
        
        if cache_key not in self.cache_timestamps:
            return False
        
        return datetime.now() - self.cache_timestamps[cache_key] < self.cache_ttl
    
    def _determine_optimal_preset(
        self, 
        user_id: Optional[str], 
        source_type: Optional[str], 
        content_type: Optional[str]
    ) -> ConfigPreset:
        """Determine the optimal preset based on context."""
        # Load user preferences to determine their typical usage pattern
        if user_id:
            user_prefs = self._load_user_overrides(user_id)
            
            # If user has many custom settings, assume they're a power user
            if len(user_prefs) > 10:
                return ConfigPreset.POWER_USER
            
            # Check for advanced features usage
            advanced_features = ['enable_chunking', 'embed_chunks_separately', 'enable_quality_validation']
            if any(user_prefs.get(f'processing.{feature}', False) for feature in advanced_features):
                return ConfigPreset.ADVANCED
        
        # Source-based heuristics
        if source_type in ['bulk_upload', 'api']:
            return ConfigPreset.ADVANCED  # These typically need more processing power
        
        # Content-type based heuristics
        if content_type in ['pdf', 'code', 'video']:
            return ConfigPreset.ADVANCED  # These content types benefit from advanced processing
        
        # Default to advanced for a good balance
        return ConfigPreset.ADVANCED
    
    def _load_user_overrides(self, user_id: str) -> Dict[str, Any]:
        """Load user-specific configuration overrides from storage."""
        user_config_file = self.storage_path / f"user_{user_id}.json"
        
        if not user_config_file.exists():
            return {}
        
        try:
            with open(user_config_file, 'r') as f:
                config_data = json.load(f)
            
            # Remove metadata
            config_data.pop('_metadata', None)
            
            return config_data
        except Exception as e:
            logger.error(f"Failed to load user config for {user_id}: {e}")
            return {}
    
    def _apply_auto_fixes(self, config: CaptureConfig, suggestions: Dict[str, Any]) -> CaptureConfig:
        """Apply automatic fixes to configuration issues."""
        for key, value in suggestions.items():
            if '.' in key:
                section, field = key.split('.', 1)
                if hasattr(config, section):
                    section_obj = getattr(config, section)
                    if hasattr(section_obj, field):
                        setattr(section_obj, field, value)
        
        return config
    
    def _invalidate_user_cache(self, user_id: str):
        """Invalidate all cached configurations for a user."""
        keys_to_remove = [k for k in self.config_cache.keys() if k.startswith(f"{user_id}:")]
        for key in keys_to_remove:
            self.config_cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
    
    def _get_preset_description(self, preset: ConfigPreset) -> str:
        """Get description for a configuration preset."""
        descriptions = {
            ConfigPreset.BASIC: "Simple configuration with essential features enabled. Best for casual users.",
            ConfigPreset.ADVANCED: "Balanced configuration with most features enabled. Best for regular users who want smart processing.",
            ConfigPreset.POWER_USER: "Full-featured configuration with advanced processing options. Best for users who need maximum capabilities."
        }
        return descriptions.get(preset, "Custom configuration")
    
    def _get_preset_features(self, config: CaptureConfig) -> List[str]:
        """Get list of enabled features for a configuration."""
        features = []
        
        if config.processing.enable_ai_processing:
            features.append("AI Processing")
        if config.processing.enable_chunking:
            features.append("Content Chunking")
        if config.processing.enable_embeddings:
            features.append("Semantic Search")
        if config.processing.enable_ocr:
            features.append("OCR")
        if config.processing.content_deduplication:
            features.append("Deduplication")
        if config.processing.enable_quality_validation:
            features.append("Quality Validation")
        
        return features


# Global configuration manager instance
_config_manager: Optional[CaptureConfigManager] = None


def get_capture_config_manager() -> CaptureConfigManager:
    """Get the global capture configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = CaptureConfigManager()
    return _config_manager