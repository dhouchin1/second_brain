#!/usr/bin/env python3
"""
Configuration Migration CLI for Autom8

Command-line interface for managing configuration migrations,
backups, and version upgrades.
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from autom8.config.migrations import (
    ConfigMigrationManager, create_migration_manager, auto_migrate_config
)

console = Console()


@click.group()
@click.option('--config-dir', type=click.Path(exists=True, path_type=Path),
              help='Configuration directory path')
@click.option('--backup-dir', type=click.Path(path_type=Path),
              help='Backup directory path')
@click.pass_context
def config_migrate(ctx, config_dir: Optional[Path], backup_dir: Optional[Path]):
    """Autom8 Configuration Migration Tool
    
    Manage configuration migrations, backups, and version upgrades.
    """
    ctx.ensure_object(dict)
    ctx.obj['config_dir'] = config_dir or Path('.')
    ctx.obj['backup_dir'] = backup_dir
    ctx.obj['manager'] = create_migration_manager(config_dir, backup_dir)


@config_migrate.command()
@click.pass_context
def status(ctx):
    """Show current configuration version and migration status."""
    
    manager: ConfigMigrationManager = ctx.obj['manager']
    
    console.print(Panel.fit(
        "[bold blue]📋 Configuration Migration Status[/bold blue]",
        border_style="blue"
    ))
    
    # Detect current version
    current_version = manager.detect_config_version()
    target_version = manager.current_app_version
    
    # Create status table
    table = Table(title="Configuration Status", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Current Version", current_version or "Not detected")
    table.add_row("Target Version", target_version)
    table.add_row("Config Directory", str(ctx.obj['config_dir']))
    table.add_row("Backup Directory", str(manager.backup_dir))
    
    if current_version:
        pending_migrations = manager.get_pending_migrations(current_version, target_version)
        table.add_row("Pending Migrations", str(len(pending_migrations)))
        
        if pending_migrations:
            migration_versions = ", ".join([m.version for m in pending_migrations])
            table.add_row("Migration Versions", migration_versions)
    
    console.print(table)
    
    # Show pending migrations details
    if current_version:
        pending_migrations = manager.get_pending_migrations(current_version, target_version)
        
        if pending_migrations:
            console.print("\n[bold yellow]📝 Pending Migrations:[/bold yellow]")
            
            for migration in pending_migrations:
                status_icon = "⚠️" if migration.breaking_change else "📦"
                auto_icon = "🤖" if migration.auto_apply else "👤"
                
                console.print(
                    f"{status_icon} {auto_icon} v{migration.version}: {migration.description}"
                )
                
                if migration.breaking_change:
                    console.print("   [red]⚠️  Breaking change - requires manual approval[/red]")
                
                if not migration.auto_apply:
                    console.print("   [yellow]👤 Manual approval required[/yellow]")
        else:
            console.print("\n[green]✅ Configuration is up to date[/green]")


@config_migrate.command()
@click.option('--target-version', help='Target version to migrate to')
@click.option('--dry-run', is_flag=True, help='Simulate migration without making changes')
@click.option('--no-backup', is_flag=True, help='Skip automatic backup creation')
@click.option('--force', is_flag=True, help='Force migration even if manual approval is required')
@click.pass_context
def migrate(ctx, target_version: Optional[str], dry_run: bool, no_backup: bool, force: bool):
    """Migrate configuration to target version."""
    
    manager: ConfigMigrationManager = ctx.obj['manager']
    
    if dry_run:
        console.print(Panel.fit(
            "[bold yellow]🔍 Configuration Migration (Dry Run)[/bold yellow]",
            border_style="yellow"
        ))
    else:
        console.print(Panel.fit(
            "[bold green]🚀 Configuration Migration[/bold green]",
            border_style="green"
        ))
    
    # Get current version
    current_version = manager.detect_config_version()
    if not current_version:
        console.print("[red]❌ Could not detect current configuration version[/red]")
        sys.exit(1)
    
    target = target_version or manager.current_app_version
    
    # Show what will be done
    pending_migrations = manager.get_pending_migrations(current_version, target)
    
    if not pending_migrations:
        console.print("[green]✅ Configuration is already up to date[/green]")
        return
    
    console.print(f"[cyan]📊 Migration Plan: v{current_version} → v{target}[/cyan]")
    console.print(f"[cyan]📁 Config Directory: {ctx.obj['config_dir']}[/cyan]")
    
    # Show migration details
    for i, migration in enumerate(pending_migrations, 1):
        status_icon = "⚠️" if migration.breaking_change else "📦"
        auto_icon = "🤖" if migration.auto_apply else "👤"
        
        console.print(f"{i}. {status_icon} {auto_icon} v{migration.version}: {migration.description}")
    
    # Check for breaking changes or manual approval
    breaking_changes = [m for m in pending_migrations if m.breaking_change]
    manual_approval = [m for m in pending_migrations if not m.auto_apply]
    
    if (breaking_changes or manual_approval) and not force and not dry_run:
        console.print("\n[yellow]⚠️  This migration requires manual approval due to:[/yellow]")
        
        if breaking_changes:
            console.print("  • Breaking changes that may affect existing functionality")
        if manual_approval:
            console.print("  • Security or configuration changes that require review")
        
        console.print("\n[yellow]Use --force to proceed or --dry-run to preview changes[/yellow]")
        return
    
    # Confirm migration
    if not dry_run and not force:
        if not click.confirm(f"\nProceed with migration from v{current_version} to v{target}?"):
            console.print("[yellow]Migration cancelled[/yellow]")
            return
    
    # Perform migration
    with console.status("[bold green]Performing migration...") as status:
        result = manager.migrate_configuration(
            target_version=target,
            auto_backup=not no_backup,
            dry_run=dry_run
        )
    
    # Show results
    if result.success:
        console.print(f"\n[green]✅ Migration completed successfully[/green]")
        
        if result.migrations_applied:
            console.print(f"[green]📦 Applied migrations: {', '.join(result.migrations_applied)}[/green]")
        
        if result.backup_id:
            console.print(f"[blue]💾 Backup created: {result.backup_id}[/blue]")
        
        if result.warnings:
            console.print("\n[yellow]⚠️  Warnings:[/yellow]")
            for warning in result.warnings:
                console.print(f"   • {warning}")
    
    else:
        console.print(f"\n[red]❌ Migration failed[/red]")
        
        if result.errors:
            console.print("\n[red]Errors:[/red]")
            for error in result.errors:
                console.print(f"   • {error}")
        
        if result.backup_id:
            console.print(f"\n[blue]💾 Backup available for rollback: {result.backup_id}[/blue]")
            console.print(f"[blue]Use: autom8 config-migrate restore {result.backup_id}[/blue]")
    
    console.print(f"\n[dim]⏱️  Execution time: {result.execution_time:.2f}s[/dim]")


@config_migrate.command()
@click.option('--backup-id', help='Custom backup identifier')
@click.pass_context
def backup(ctx, backup_id: Optional[str]):
    """Create a configuration backup."""
    
    manager: ConfigMigrationManager = ctx.obj['manager']
    
    console.print(Panel.fit(
        "[bold blue]💾 Creating Configuration Backup[/bold blue]",
        border_style="blue"
    ))
    
    with console.status("[bold blue]Creating backup...") as status:
        backup = manager.create_backup(backup_id)
    
    console.print(f"[green]✅ Backup created successfully[/green]")
    console.print(f"[blue]📦 Backup ID: {backup.backup_id}[/blue]")
    console.print(f"[blue]📁 Backup Path: {backup.backup_path}[/blue]")
    console.print(f"[blue]📄 Files: {', '.join(backup.files)}[/blue]")
    console.print(f"[blue]🕐 Created: {backup.created_at.strftime('%Y-%m-%d %H:%M:%S')}[/blue]")


@config_migrate.command()
@click.pass_context
def list_backups(ctx):
    """List all available configuration backups."""
    
    manager: ConfigMigrationManager = ctx.obj['manager']
    backups = manager.list_backups()
    
    if not backups:
        console.print("[yellow]📭 No backups found[/yellow]")
        return
    
    console.print(Panel.fit(
        f"[bold blue]💾 Configuration Backups ({len(backups)} found)[/bold blue]",
        border_style="blue"
    ))
    
    # Create backups table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Backup ID", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Age", style="yellow")
    table.add_column("Files", style="blue")
    table.add_column("Size", style="dim")
    
    for backup in backups:
        # Calculate backup size
        total_size = 0
        for file_path in backup.backup_path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        size_str = f"{total_size // 1024}KB" if total_size > 0 else "Unknown"
        age_str = f"{backup.age_hours:.1f}h" if backup.age_hours < 24 else f"{backup.age_hours // 24:.0f}d"
        
        table.add_row(
            backup.backup_id,
            backup.version,
            age_str,
            str(len(backup.files)),
            size_str
        )
    
    console.print(table)


@config_migrate.command()
@click.argument('backup_id')
@click.option('--force', is_flag=True, help='Force restore without confirmation')
@click.pass_context
def restore(ctx, backup_id: str, force: bool):
    """Restore configuration from backup."""
    
    manager: ConfigMigrationManager = ctx.obj['manager']
    
    console.print(Panel.fit(
        f"[bold yellow]🔄 Restoring Configuration from Backup[/bold yellow]",
        border_style="yellow"
    ))
    
    # Find backup
    backups = manager.list_backups()
    backup = next((b for b in backups if b.backup_id == backup_id), None)
    
    if not backup:
        console.print(f"[red]❌ Backup '{backup_id}' not found[/red]")
        return
    
    # Show backup details
    console.print(f"[blue]📦 Backup ID: {backup.backup_id}[/blue]")
    console.print(f"[blue]📅 Created: {backup.created_at.strftime('%Y-%m-%d %H:%M:%S')}[/blue]")
    console.print(f"[blue]🏷️  Version: {backup.version}[/blue]")
    console.print(f"[blue]📄 Files: {', '.join(backup.files)}[/blue]")
    
    # Confirm restore
    if not force:
        console.print("\n[yellow]⚠️  This will overwrite current configuration files[/yellow]")
        if not click.confirm("Proceed with restore?"):
            console.print("[yellow]Restore cancelled[/yellow]")
            return
    
    # Perform restore
    with console.status("[bold yellow]Restoring backup...") as status:
        success = manager.restore_backup(backup_id)
    
    if success:
        console.print(f"\n[green]✅ Configuration restored successfully from {backup_id}[/green]")
        console.print(f"[green]📁 Configuration reverted to version {backup.version}[/green]")
    else:
        console.print(f"\n[red]❌ Failed to restore backup {backup_id}[/red]")


@config_migrate.command()
@click.option('--keep', default=10, help='Number of recent backups to keep')
@click.option('--max-age', default=30, help='Maximum age in days for backups')
@click.option('--dry-run', is_flag=True, help='Show what would be cleaned up')
@click.pass_context
def cleanup(ctx, keep: int, max_age: int, dry_run: bool):
    """Clean up old configuration backups."""
    
    manager: ConfigMigrationManager = ctx.obj['manager']
    
    console.print(Panel.fit(
        "[bold blue]🧹 Backup Cleanup[/bold blue]",
        border_style="blue"
    ))
    
    if dry_run:
        console.print("[yellow]🔍 Dry run - showing what would be cleaned up[/yellow]")
    
    backups = manager.list_backups()
    
    if not backups:
        console.print("[green]✅ No backups to clean up[/green]")
        return
    
    console.print(f"[blue]📊 Found {len(backups)} backups[/blue]")
    console.print(f"[blue]📋 Cleanup policy: Keep {keep} recent, max age {max_age} days[/blue]")
    
    if dry_run:
        # Simulate cleanup
        to_remove = []
        
        # Check count limit
        if len(backups) > keep:
            to_remove.extend(backups[keep:])
        
        # Check age limit
        from datetime import datetime, timedelta
        cutoff_time = datetime.utcnow() - timedelta(days=max_age)
        
        for backup in backups[:keep]:
            if backup.created_at < cutoff_time:
                to_remove.append(backup)
        
        if to_remove:
            console.print(f"\n[yellow]📋 Would remove {len(to_remove)} backups:[/yellow]")
            for backup in to_remove:
                age_str = f"{backup.age_hours:.1f}h" if backup.age_hours < 24 else f"{backup.age_hours // 24:.0f}d"
                console.print(f"   • {backup.backup_id} (age: {age_str})")
        else:
            console.print(f"\n[green]✅ No backups need cleanup[/green]")
    
    else:
        # Perform actual cleanup
        with console.status("[bold blue]Cleaning up backups...") as status:
            removed_count = manager.cleanup_old_backups(keep, max_age)
        
        if removed_count > 0:
            console.print(f"\n[green]✅ Cleaned up {removed_count} old backups[/green]")
        else:
            console.print(f"\n[green]✅ No backups needed cleanup[/green]")


@config_migrate.command()
@click.pass_context
def validate(ctx):
    """Validate current configuration."""
    
    manager: ConfigMigrationManager = ctx.obj['manager']
    
    console.print(Panel.fit(
        "[bold blue]🔍 Configuration Validation[/bold blue]",
        border_style="blue"
    ))
    
    config_path = manager.config_dir / "autom8.yaml"
    
    if not config_path.exists():
        console.print("[red]❌ Configuration file not found[/red]")
        return
    
    # Validate configuration
    errors = manager._validate_migrated_config()
    
    if not errors:
        console.print("[green]✅ Configuration is valid[/green]")
        
        # Show current version
        current_version = manager.detect_config_version()
        if current_version:
            console.print(f"[blue]🏷️  Version: {current_version}[/blue]")
    else:
        console.print(f"[red]❌ Configuration validation failed ({len(errors)} errors)[/red]")
        
        for error in errors:
            console.print(f"   • {error}")


if __name__ == '__main__':
    config_migrate()