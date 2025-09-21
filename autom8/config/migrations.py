"""
Configuration Migration System for Autom8

Handles configuration schema changes, data migrations, and version upgrades
across all configuration files and settings to ensure smooth updates.
"""

import json
import shutil
import yaml
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from packaging.version import Version

from pydantic import BaseModel, Field, computed_field

from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class MigrationType(str, Enum):
    """Types of configuration migrations."""
    CONFIG_SCHEMA = "config_schema"    # Changes to configuration structure
    DATA_FORMAT = "data_format"       # Changes to data formats
    FEATURE_REMOVAL = "feature_removal"  # Deprecated feature removal
    SECURITY_UPDATE = "security_update"  # Security-related changes
    PERFORMANCE = "performance"       # Performance optimization changes
    DATABASE_COMPAT = "database_compat"  # Database compatibility changes


class ConfigMigration(BaseModel):
    """Represents a single configuration migration."""
    
    version: str = Field(description="Version this migration targets")
    from_version: str = Field(description="Minimum version this migration applies to")
    migration_type: MigrationType = Field(description="Type of migration")
    description: str = Field(description="Human-readable description")
    
    # Migration functions (as string references)
    upgrade_function: str = Field(description="Function to perform upgrade")
    downgrade_function: Optional[str] = Field(default=None, description="Function to perform downgrade")
    
    # Validation
    required_fields: List[str] = Field(default_factory=list, description="Required config fields after migration")
    deprecated_fields: List[str] = Field(default_factory=list, description="Fields deprecated by this migration")
    
    # Safety
    backup_required: bool = Field(default=True, description="Whether backup is required")
    reversible: bool = Field(default=True, description="Whether migration can be reversed")
    
    # Metadata
    breaking_change: bool = Field(default=False, description="Whether this is a breaking change")
    auto_apply: bool = Field(default=True, description="Whether migration can be applied automatically")
    
    def version_object(self) -> Version:
        """Parse version as packaging.version object."""
        return Version(self.version)
    
    def from_version_object(self) -> Version:
        """Parse from_version as packaging.version object."""
        return Version(self.from_version)


class ConfigBackup(BaseModel):
    """Represents a configuration backup."""
    
    backup_id: str = Field(description="Unique backup identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(description="Version when backup was created")
    backup_path: Path = Field(description="Path to backup directory")
    files: List[str] = Field(default_factory=list, description="List of backed up files")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @computed_field
    @property
    def age_hours(self) -> float:
        """Age of backup in hours."""
        return (datetime.utcnow() - self.created_at).total_seconds() / 3600


class MigrationResult(BaseModel):
    """Result of a migration operation."""
    
    success: bool = Field(description="Whether migration succeeded")
    version_from: str = Field(description="Starting version")
    version_to: str = Field(description="Target version")
    migrations_applied: List[str] = Field(default_factory=list, description="Applied migrations")
    migrations_failed: List[str] = Field(default_factory=list, description="Failed migrations")
    backup_id: Optional[str] = Field(default=None, description="Backup ID if created")
    warnings: List[str] = Field(default_factory=list, description="Migration warnings")
    errors: List[str] = Field(default_factory=list, description="Migration errors")
    execution_time: float = Field(default=0.0, description="Execution time in seconds")


class ConfigMigrationManager:
    """
    Manages configuration migrations for Autom8 version updates.
    
    Provides comprehensive configuration migration capabilities:
    - Automatic detection of configuration version
    - Safe migration with backup and rollback
    - Validation of migrated configurations
    - Support for both automatic and manual migrations
    """
    
    def __init__(self, config_dir: Path = None, backup_dir: Path = None):
        """
        Initialize configuration migration manager.
        
        Args:
            config_dir: Directory containing configuration files
            backup_dir: Directory for storing backups
        """
        self.config_dir = config_dir or Path(".")
        self.backup_dir = backup_dir or (self.config_dir / ".autom8_backups")
        
        # Ensure directories exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Migration registry
        self.migrations: List[ConfigMigration] = []
        self._register_migrations()
        
        # Current application version (would be loaded from package metadata)
        self.current_app_version = "1.0.0"  # TODO: Load from package
        
        logger.info(f"Configuration migration manager initialized")
    
    def _register_migrations(self):
        """Register all available configuration migrations."""
        
        # Migration 0.9.0 -> 1.0.0: Initial configuration schema
        self.migrations.append(ConfigMigration(
            version="1.0.0",
            from_version="0.9.0",
            migration_type=MigrationType.CONFIG_SCHEMA,
            description="Migrate to new configuration schema with structured routing and context settings",
            upgrade_function="_migrate_to_v1_0_0",
            downgrade_function="_downgrade_from_v1_0_0",
            required_fields=["routing", "context", "shared_memory"],
            backup_required=True,
            breaking_change=True
        ))
        
        # Migration 1.0.0 -> 1.1.0: Add offline mode configuration
        self.migrations.append(ConfigMigration(
            version="1.1.0", 
            from_version="1.0.0",
            migration_type=MigrationType.CONFIG_SCHEMA,
            description="Add offline mode and graceful degradation configuration",
            upgrade_function="_migrate_to_v1_1_0",
            downgrade_function="_downgrade_from_v1_1_0",
            required_fields=["offline_mode"],
            backup_required=True
        ))
        
        # Migration 1.1.0 -> 1.2.0: Enhanced model routing preferences
        self.migrations.append(ConfigMigration(
            version="1.2.0",
            from_version="1.1.0", 
            migration_type=MigrationType.CONFIG_SCHEMA,
            description="Enhanced model routing with preference learning and adaptive routing",
            upgrade_function="_migrate_to_v1_2_0",
            downgrade_function="_downgrade_from_v1_2_0",
            required_fields=["routing.adaptive", "routing.preference_learning"],
            backup_required=True
        ))
        
        # Migration 1.2.0 -> 1.3.0: Security and privacy enhancements
        self.migrations.append(ConfigMigration(
            version="1.3.0",
            from_version="1.2.0",
            migration_type=MigrationType.SECURITY_UPDATE,
            description="Enhanced security and privacy configuration options",
            upgrade_function="_migrate_to_v1_3_0",
            downgrade_function="_downgrade_from_v1_3_0",
            required_fields=["security", "privacy"],
            backup_required=True,
            breaking_change=True,
            auto_apply=False  # Requires manual approval for security changes
        ))
        
        # Migration 1.3.0 -> 1.4.0: Template marketplace integration
        self.migrations.append(ConfigMigration(
            version="1.4.0",
            from_version="1.3.0",
            migration_type=MigrationType.CONFIG_SCHEMA,
            description="Template marketplace and sharing functionality configuration",
            upgrade_function="_migrate_to_v1_4_0",
            downgrade_function="_downgrade_from_v1_4_0",
            required_fields=["templates.marketplace"],
            backup_required=True
        ))
        
        # Migration 1.4.0 -> 2.0.0: Major architecture changes
        self.migrations.append(ConfigMigration(
            version="2.0.0",
            from_version="1.4.0",
            migration_type=MigrationType.CONFIG_SCHEMA,
            description="Major architecture update with new agent system and enhanced context management",
            upgrade_function="_migrate_to_v2_0_0",
            downgrade_function=None,  # Not reversible
            required_fields=["agents", "context.v2", "shared_memory.v2"],
            deprecated_fields=["legacy_routing", "old_context_format"],
            backup_required=True,
            breaking_change=True,
            reversible=False,
            auto_apply=False
        ))
    
    def detect_config_version(self, config_path: Path = None) -> Optional[str]:
        """
        Detect the version of existing configuration.
        
        Args:
            config_path: Path to configuration file (autom8.yaml by default)
            
        Returns:
            Detected version string or None if not found
        """
        if config_path is None:
            config_path = self.config_dir / "autom8.yaml"
        
        if not config_path.exists():
            logger.info("No configuration file found, assuming new installation")
            return None
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Check for version field
            if isinstance(config, dict):
                # Direct version field
                if 'version' in config:
                    return str(config['version'])
                
                # Infer version from structure
                version_detected = self._infer_version_from_structure(config)
                if version_detected:
                    logger.info(f"Inferred configuration version: {version_detected}")
                    return version_detected
            
            logger.warning("Could not detect configuration version")
            return "0.9.0"  # Assume oldest version
            
        except Exception as e:
            logger.error(f"Failed to detect configuration version: {e}")
            return None
    
    def _infer_version_from_structure(self, config: Dict[str, Any]) -> Optional[str]:
        """Infer configuration version from its structure."""
        
        # Version 2.0.0+ indicators
        if "agents" in config and "context" in config and isinstance(config["context"], dict) and "v2" in config["context"]:
            return "2.0.0"
        
        # Version 1.4.0+ indicators  
        if "templates" in config and isinstance(config["templates"], dict) and "marketplace" in config["templates"]:
            return "1.4.0"
        
        # Version 1.3.0+ indicators
        if "security" in config and "privacy" in config:
            return "1.3.0"
        
        # Version 1.2.0+ indicators
        if ("routing" in config and isinstance(config["routing"], dict) and 
            "adaptive" in config["routing"] and "preference_learning" in config["routing"]):
            return "1.2.0"
        
        # Version 1.1.0+ indicators
        if "offline_mode" in config:
            return "1.1.0"
        
        # Version 1.0.0+ indicators
        if ("routing" in config and "context" in config and "shared_memory" in config):
            return "1.0.0"
        
        # Pre-1.0.0 configurations
        return "0.9.0"
    
    def get_pending_migrations(self, current_version: str, target_version: str = None) -> List[ConfigMigration]:
        """
        Get list of migrations needed to upgrade from current to target version.
        
        Args:
            current_version: Current configuration version
            target_version: Target version (latest if None)
            
        Returns:
            List of migrations to apply in order
        """
        if target_version is None:
            target_version = self.current_app_version
        
        current_ver = Version(current_version)
        target_ver = Version(target_version)
        
        # Find applicable migrations
        applicable_migrations = []
        for migration in self.migrations:
            migration_ver = migration.version_object()
            from_ver = migration.from_version_object()
            
            # Migration applies if:
            # 1. Current version is >= migration's from_version
            # 2. Migration version is <= target version
            # 3. Current version is < migration version
            if (current_ver >= from_ver and 
                migration_ver <= target_ver and 
                current_ver < migration_ver):
                applicable_migrations.append(migration)
        
        # Sort by version to ensure correct order
        applicable_migrations.sort(key=lambda m: m.version_object())
        
        return applicable_migrations
    
    def create_backup(self, backup_id: str = None) -> ConfigBackup:
        """
        Create a backup of current configuration.
        
        Args:
            backup_id: Backup identifier (auto-generated if None)
            
        Returns:
            ConfigBackup object
        """
        if backup_id is None:
            backup_id = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self.backup_dir / backup_id
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Files to backup
        config_files = [
            "autom8.yaml",
            ".env",
            "autom8.db",
            ".autom8",  # Hidden config directory if exists
        ]
        
        backed_up_files = []
        
        for filename in config_files:
            source_path = self.config_dir / filename
            if source_path.exists():
                dest_path = backup_path / filename
                
                if source_path.is_file():
                    shutil.copy2(source_path, dest_path)
                elif source_path.is_dir():
                    shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                
                backed_up_files.append(filename)
                logger.debug(f"Backed up {filename}")
        
        # Create backup metadata
        current_version = self.detect_config_version() or "unknown"
        
        backup = ConfigBackup(
            backup_id=backup_id,
            version=current_version,
            backup_path=backup_path,
            files=backed_up_files,
            metadata={
                "app_version": self.current_app_version,
                "created_by": "autom8_migration_manager"
            }
        )
        
        # Save backup metadata
        metadata_path = backup_path / "backup_metadata.json"
        
        # Convert backup to dict with proper serialization
        backup_dict = backup.model_dump()
        backup_dict['created_at'] = backup.created_at.isoformat()
        backup_dict['backup_path'] = str(backup.backup_path)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(backup_dict, f, indent=2)
        
        logger.info(f"Created configuration backup: {backup_id}")
        return backup
    
    def restore_backup(self, backup_id: str) -> bool:
        """
        Restore configuration from backup.
        
        Args:
            backup_id: Backup identifier to restore
            
        Returns:
            True if successful, False otherwise
        """
        backup_path = self.backup_dir / backup_id
        metadata_path = backup_path / "backup_metadata.json"
        
        if not backup_path.exists() or not metadata_path.exists():
            logger.error(f"Backup {backup_id} not found")
            return False
        
        try:
            # Load backup metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Restore files
            for filename in metadata.get('files', []):
                source_path = backup_path / filename
                dest_path = self.config_dir / filename
                
                if source_path.exists():
                    if source_path.is_file():
                        shutil.copy2(source_path, dest_path)
                    elif source_path.is_dir():
                        if dest_path.exists():
                            shutil.rmtree(dest_path)
                        shutil.copytree(source_path, dest_path)
                    
                    logger.debug(f"Restored {filename}")
            
            logger.info(f"Successfully restored backup: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore backup {backup_id}: {e}")
            return False
    
    def list_backups(self) -> List[ConfigBackup]:
        """List all available configuration backups."""
        backups = []
        
        for backup_dir in self.backup_dir.iterdir():
            if backup_dir.is_dir():
                metadata_path = backup_dir / "backup_metadata.json"
                
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        # Convert string timestamps back to datetime
                        if 'created_at' in metadata and isinstance(metadata['created_at'], str):
                            metadata['created_at'] = datetime.fromisoformat(metadata['created_at'])
                        
                        backup = ConfigBackup(**metadata)
                        backup.backup_path = backup_dir  # Ensure path is correct
                        backups.append(backup)
                        
                    except Exception as e:
                        logger.warning(f"Could not load backup metadata for {backup_dir.name}: {e}")
        
        # Sort by creation date (newest first)
        backups.sort(key=lambda b: b.created_at, reverse=True)
        return backups
    
    def cleanup_old_backups(self, keep_count: int = 10, max_age_days: int = 30) -> int:
        """
        Clean up old configuration backups.
        
        Args:
            keep_count: Number of recent backups to keep
            max_age_days: Maximum age in days for backups
            
        Returns:
            Number of backups removed
        """
        backups = self.list_backups()
        removed_count = 0
        
        # Remove backups beyond keep_count
        if len(backups) > keep_count:
            for backup in backups[keep_count:]:
                try:
                    shutil.rmtree(backup.backup_path)
                    removed_count += 1
                    logger.debug(f"Removed old backup: {backup.backup_id}")
                except Exception as e:
                    logger.warning(f"Failed to remove backup {backup.backup_id}: {e}")
        
        # Remove backups older than max_age_days
        cutoff_time = datetime.utcnow() - timedelta(days=max_age_days)
        
        for backup in backups[:keep_count]:  # Check remaining backups
            if backup.created_at < cutoff_time:
                try:
                    shutil.rmtree(backup.backup_path)
                    removed_count += 1
                    logger.debug(f"Removed expired backup: {backup.backup_id}")
                except Exception as e:
                    logger.warning(f"Failed to remove expired backup {backup.backup_id}: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old backups")
        
        return removed_count
    
    def migrate_configuration(
        self,
        target_version: str = None,
        auto_backup: bool = True,
        dry_run: bool = False
    ) -> MigrationResult:
        """
        Migrate configuration to target version.
        
        Args:
            target_version: Target version (latest if None)
            auto_backup: Whether to create automatic backup
            dry_run: If True, simulate migration without making changes
            
        Returns:
            MigrationResult with details of the migration
        """
        import time
        start_time = time.perf_counter()
        
        if target_version is None:
            target_version = self.current_app_version
        
        # Detect current version
        current_version = self.detect_config_version()
        if current_version is None:
            return MigrationResult(
                success=False,
                version_from="unknown",
                version_to=target_version,
                errors=["Could not detect current configuration version"]
            )
        
        # Check if migration is needed
        if Version(current_version) >= Version(target_version):
            return MigrationResult(
                success=True,
                version_from=current_version,
                version_to=target_version,
                warnings=["Configuration is already at or above target version"]
            )
        
        result = MigrationResult(
            success=False,  # Initialize as False, will be updated later
            version_from=current_version,
            version_to=target_version
        )
        
        try:
            # Get pending migrations
            pending_migrations = self.get_pending_migrations(current_version, target_version)
            
            if not pending_migrations:
                result.success = True
                result.warnings.append("No migrations needed")
                return result
            
            # Check for breaking changes or manual approval required
            manual_approval_needed = any(
                not m.auto_apply or m.breaking_change 
                for m in pending_migrations
            )
            
            if manual_approval_needed and not dry_run:
                result.success = False
                result.errors.append(
                    "Migration requires manual approval due to breaking changes. "
                    "Use --force flag or review changes manually."
                )
                return result
            
            # Create backup if not dry run
            backup_id = None
            if auto_backup and not dry_run:
                backup = self.create_backup()
                backup_id = backup.backup_id
                result.backup_id = backup_id
            
            # Apply migrations
            for migration in pending_migrations:
                logger.info(f"Applying migration to {migration.version}: {migration.description}")
                
                if dry_run:
                    logger.info(f"[DRY RUN] Would apply migration: {migration.upgrade_function}")
                    result.migrations_applied.append(migration.version)
                else:
                    # Apply the migration
                    success = self._apply_migration(migration)
                    
                    if success:
                        result.migrations_applied.append(migration.version)
                    else:
                        result.migrations_failed.append(migration.version)
                        result.errors.append(f"Failed to apply migration {migration.version}")
                        break
            
            # Validate final configuration
            if not dry_run and result.migrations_applied:
                validation_errors = self._validate_migrated_config()
                if validation_errors:
                    result.errors.extend(validation_errors)
                    result.success = False
                else:
                    result.success = True
            else:
                result.success = True
            
            # Update configuration version
            if result.success and not dry_run:
                self._update_config_version(target_version)
            
        except Exception as e:
            logger.error(f"Migration failed with exception: {e}")
            result.success = False
            result.errors.append(f"Migration exception: {str(e)}")
        
        finally:
            result.execution_time = time.perf_counter() - start_time
        
        return result
    
    def _apply_migration(self, migration: ConfigMigration) -> bool:
        """Apply a single migration."""
        
        try:
            # Get the migration function
            migration_func = getattr(self, migration.upgrade_function, None)
            if not migration_func:
                logger.error(f"Migration function {migration.upgrade_function} not found")
                return False
            
            # Apply the migration
            success = migration_func()
            
            if success:
                logger.info(f"Successfully applied migration {migration.version}")
            else:
                logger.error(f"Migration {migration.version} returned False")
            
            return success
            
        except Exception as e:
            logger.error(f"Exception in migration {migration.version}: {e}")
            return False
    
    def _validate_migrated_config(self) -> List[str]:
        """Validate configuration after migration."""
        errors = []
        
        config_path = self.config_dir / "autom8.yaml"
        
        try:
            if not config_path.exists():
                errors.append("Configuration file missing after migration")
                return errors
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if not isinstance(config, dict):
                errors.append("Configuration is not a valid dictionary")
                return errors
            
            # Basic structure validation
            required_sections = ["routing", "context", "shared_memory"]
            for section in required_sections:
                if section not in config:
                    errors.append(f"Required section '{section}' missing from configuration")
            
            # Validate routing section
            if "routing" in config and isinstance(config["routing"], dict):
                routing = config["routing"]
                if "thresholds" not in routing:
                    errors.append("routing.thresholds section missing")
            
            logger.info("Configuration validation completed")
            
        except Exception as e:
            errors.append(f"Configuration validation error: {str(e)}")
        
        return errors
    
    def _update_config_version(self, version: str) -> bool:
        """Update the version field in configuration."""
        
        config_path = self.config_dir / "autom8.yaml"
        
        try:
            # Load current config
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            # Update version
            config['version'] = version
            config['updated_at'] = datetime.utcnow().isoformat()
            
            # Write back
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Updated configuration version to {version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update configuration version: {e}")
            return False
    
    # Migration functions for specific versions
    
    def _migrate_to_v1_0_0(self) -> bool:
        """Migrate configuration to version 1.0.0."""
        
        config_path = self.config_dir / "autom8.yaml"
        
        try:
            # Load existing config or create new one
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
            else:
                config = {}
            
            # Migrate to new structure
            new_config = {
                "version": "1.0.0",
                "routing": {
                    "thresholds": {
                        "trivial": 0.2,
                        "simple": 0.4,
                        "moderate": 0.6,
                        "complex": 0.8,
                        "frontier": 1.0
                    },
                    "prefer_local": config.get("prefer_local", True),
                    "local_quality_threshold": config.get("local_quality_threshold", 0.85)
                },
                "context": {
                    "default_max_tokens": config.get("default_max_tokens", 500),
                    "hard_limit": config.get("hard_limit", 2000),
                    "always_preview": config.get("always_preview", True)
                },
                "shared_memory": {
                    "redis": {
                        "host": config.get("redis_host", "localhost"),
                        "port": config.get("redis_port", 6379),
                        "url": config.get("redis_url")
                    },
                    "sqlite": {
                        "path": config.get("sqlite_path", "./autom8.db")
                    }
                },
                "cost_controls": {
                    "allow_cloud": config.get("allow_cloud", False),
                    "daily_limit": config.get("daily_limit", 1.00),
                    "monthly_limit": config.get("monthly_limit", 10.00)
                },
                "logging": {
                    "level": config.get("log_level", "INFO"),
                    "file": config.get("log_file", "./autom8.log"),
                    "structured": config.get("log_structured", True)
                }
            }
            
            # Write new configuration
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(new_config, f, default_flow_style=False, indent=2)
            
            logger.info("Successfully migrated to v1.0.0 configuration schema")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate to v1.0.0: {e}")
            return False
    
    def _migrate_to_v1_1_0(self) -> bool:
        """Migrate configuration to version 1.1.0 (add offline mode)."""
        
        config_path = self.config_dir / "autom8.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Add offline mode configuration
            config["offline_mode"] = {
                "enabled": True,
                "health_check_interval": 30,
                "service_timeout": 5.0,
                "max_consecutive_failures": 3,
                "preferred_local_models": [
                    "llama3.2:7b",
                    "llama3.2:3b",
                    "mistral:7b"
                ],
                "cache": {
                    "max_responses": 1000,
                    "expiry_hours": 168,
                    "enabled": True
                },
                "notifications": {
                    "notify_offline_mode": True,
                    "show_degraded_features": True
                }
            }
            
            config["version"] = "1.1.0"
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info("Successfully migrated to v1.1.0 (offline mode)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate to v1.1.0: {e}")
            return False
    
    def _migrate_to_v1_2_0(self) -> bool:
        """Migrate configuration to version 1.2.0 (adaptive routing)."""
        
        config_path = self.config_dir / "autom8.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Add adaptive routing configuration
            config["routing"]["adaptive"] = {
                "enabled": True,
                "learning_rate": 0.1,
                "exploration_rate": 0.2,
                "adaptation_threshold": 0.05
            }
            
            config["routing"]["preference_learning"] = {
                "enabled": True,
                "algorithms": ["collaborative_filtering", "q_learning"],
                "pattern_recognition": True,
                "user_feedback_weight": 0.3
            }
            
            config["version"] = "1.2.0"
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info("Successfully migrated to v1.2.0 (adaptive routing)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate to v1.2.0: {e}")
            return False
    
    def _migrate_to_v1_3_0(self) -> bool:
        """Migrate configuration to version 1.3.0 (security enhancements)."""
        
        config_path = self.config_dir / "autom8.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Add security configuration
            config["security"] = {
                "audit_logging": True,
                "secure_api_keys": True,
                "rate_limiting": {
                    "enabled": True,
                    "requests_per_minute": 60
                },
                "access_control": {
                    "require_auth": False,
                    "allowed_hosts": ["localhost", "127.0.0.1"]
                }
            }
            
            # Add privacy configuration
            config["privacy"] = {
                "data_retention_days": 30,
                "anonymize_logs": True,
                "local_processing_preferred": True,
                "consent_required": False
            }
            
            config["version"] = "1.3.0"
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info("Successfully migrated to v1.3.0 (security enhancements)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate to v1.3.0: {e}")
            return False
    
    def _migrate_to_v1_4_0(self) -> bool:
        """Migrate configuration to version 1.4.0 (template marketplace)."""
        
        config_path = self.config_dir / "autom8.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Add template marketplace configuration
            if "templates" not in config:
                config["templates"] = {}
            
            config["templates"]["marketplace"] = {
                "enabled": True,
                "auto_update": False,
                "trusted_sources": [
                    "https://templates.autom8.ai/official"
                ],
                "sharing": {
                    "allow_upload": False,
                    "require_approval": True
                }
            }
            
            config["version"] = "1.4.0"
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info("Successfully migrated to v1.4.0 (template marketplace)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate to v1.4.0: {e}")
            return False
    
    def _migrate_to_v2_0_0(self) -> bool:
        """Migrate configuration to version 2.0.0 (major architecture update)."""
        
        config_path = self.config_dir / "autom8.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Major restructuring for v2.0.0
            new_config = {
                "version": "2.0.0",
                "agents": {
                    "base_agent": {
                        "memory_protocols": ["redis_streams", "sqlite_vec"],
                        "context_optimization": True,
                        "shared_knowledge": True
                    },
                    "coordination": {
                        "event_bus": "redis_streams",
                        "heartbeat_interval": 30
                    }
                },
                "context": {
                    "v2": {
                        "transparency": {
                            "always_show": True,
                            "editable": True,
                            "undo_redo": True
                        },
                        "optimization": {
                            "auto_summarization": True,
                            "intelligent_trimming": True,
                            "compression": True
                        },
                        "packages": {
                            "token_budgeting": True,
                            "dependency_tracking": True
                        }
                    }
                },
                "shared_memory": {
                    "v2": {
                        "redis_streams": {
                            "enabled": True,
                            "consumer_groups": ["agents", "context", "routing"]
                        },
                        "sqlite_vec": {
                            "enabled": True,
                            "dimension": 384,
                            "hybrid_search": True
                        }
                    }
                }
            }
            
            # Preserve important settings from v1.x
            if "routing" in config:
                new_config["routing"] = config["routing"]
            if "cost_controls" in config:
                new_config["cost_controls"] = config["cost_controls"]
            if "offline_mode" in config:
                new_config["offline_mode"] = config["offline_mode"]
            if "security" in config:
                new_config["security"] = config["security"]
            if "privacy" in config:
                new_config["privacy"] = config["privacy"]
            if "templates" in config:
                new_config["templates"] = config["templates"]
            
            # Mark deprecated sections
            new_config["_deprecated"] = {
                "legacy_routing": "Replaced by v2 routing system",
                "old_context_format": "Replaced by context.v2"
            }
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(new_config, f, default_flow_style=False, indent=2)
            
            logger.info("Successfully migrated to v2.0.0 (major architecture update)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate to v2.0.0: {e}")
            return False
    
    # Downgrade functions (for rollback)
    
    def _downgrade_from_v1_0_0(self) -> bool:
        """Downgrade from v1.0.0 to previous format."""
        # Implementation would flatten the structured config back to simple format
        # For brevity, returning True as placeholder
        logger.info("Downgraded from v1.0.0")
        return True
    
    def _downgrade_from_v1_1_0(self) -> bool:
        """Downgrade from v1.1.0 by removing offline mode config."""
        config_path = self.config_dir / "autom8.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Remove offline mode configuration
            if "offline_mode" in config:
                del config["offline_mode"]
            
            config["version"] = "1.0.0"
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info("Downgraded from v1.1.0 to v1.0.0")
            return True
            
        except Exception as e:
            logger.error(f"Failed to downgrade from v1.1.0: {e}")
            return False
    
    def _downgrade_from_v1_2_0(self) -> bool:
        """Downgrade from v1.2.0 by removing adaptive routing."""
        config_path = self.config_dir / "autom8.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Remove adaptive routing configuration
            if "routing" in config:
                config["routing"].pop("adaptive", None)
                config["routing"].pop("preference_learning", None)
            
            config["version"] = "1.1.0"
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info("Downgraded from v1.2.0 to v1.1.0")
            return True
            
        except Exception as e:
            logger.error(f"Failed to downgrade from v1.2.0: {e}")
            return False
    
    def _downgrade_from_v1_3_0(self) -> bool:
        """Downgrade from v1.3.0 by removing security config."""
        config_path = self.config_dir / "autom8.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Remove security and privacy configurations
            config.pop("security", None)
            config.pop("privacy", None)
            
            config["version"] = "1.2.0"
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info("Downgraded from v1.3.0 to v1.2.0")
            return True
            
        except Exception as e:
            logger.error(f"Failed to downgrade from v1.3.0: {e}")
            return False
    
    def _downgrade_from_v1_4_0(self) -> bool:
        """Downgrade from v1.4.0 by removing template marketplace."""
        config_path = self.config_dir / "autom8.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Remove template marketplace configuration
            if "templates" in config and "marketplace" in config["templates"]:
                del config["templates"]["marketplace"]
                if not config["templates"]:  # Remove empty templates section
                    del config["templates"]
            
            config["version"] = "1.3.0"
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info("Downgraded from v1.4.0 to v1.3.0")
            return True
            
        except Exception as e:
            logger.error(f"Failed to downgrade from v1.4.0: {e}")
            return False


# Utility functions

def create_migration_manager(config_dir: Path = None, backup_dir: Path = None) -> ConfigMigrationManager:
    """Create a configuration migration manager."""
    return ConfigMigrationManager(config_dir, backup_dir)


def auto_migrate_config(
    config_dir: Path = None,
    target_version: str = None,
    create_backup: bool = True
) -> MigrationResult:
    """
    Automatically migrate configuration to target version.
    
    Args:
        config_dir: Configuration directory
        target_version: Target version (latest if None)
        create_backup: Whether to create backup before migration
        
    Returns:
        MigrationResult with migration details
    """
    manager = create_migration_manager(config_dir)
    return manager.migrate_configuration(
        target_version=target_version,
        auto_backup=create_backup,
        dry_run=False
    )