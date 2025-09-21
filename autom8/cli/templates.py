"""
Template Management CLI Commands

This module provides comprehensive CLI commands for template management,
including creation, editing, testing, composition, and analytics.
"""

import asyncio
import json
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Confirm, Prompt
from rich.layout import Layout
from rich.columns import Columns
from rich.syntax import Syntax

from autom8.core.templates import (
    get_template_manager,
    TemplateManager,
)
from autom8.models.template import (
    ContextTemplate,
    TemplateType,
    TemplateStatus,
    TemplateVariable,
    TemplateSource,
    TemplateMetadata,
    VariableType,
)
from autom8.cli.preview import ContextPreviewManager, PreviewFormat
from autom8.utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.group()
def templates():
    """Manage context templates for reusable patterns."""
    pass


@templates.command()
@click.option('--type', 'template_type', type=click.Choice(['context', 'source', 'query', 'optimization', 'workflow', 'agent']), default='context', help='Template type')
@click.option('--status', type=click.Choice(['draft', 'active', 'deprecated', 'archived']), help='Filter by status')
@click.option('--author', help='Filter by author')
@click.option('--tags', help='Filter by tags (comma-separated)')
@click.option('--limit', default=50, help='Maximum number of templates to show')
@click.option('--format', 'list_format', type=click.Choice(['table', 'detailed', 'json']), default='table', help='Output format')
def list(template_type: Optional[str], status: Optional[str], author: Optional[str], 
         tags: Optional[str], limit: int, list_format: str):
    """List available templates with filtering options."""
    
    console.print(f"[bold blue]Template Library[/bold blue]\n")
    
    async def list_templates():
        try:
            manager = await get_template_manager()
            
            # Parse filters
            template_type_enum = TemplateType(template_type) if template_type else None
            status_enum = TemplateStatus(status) if status else None
            tags_list = [tag.strip() for tag in tags.split(',')] if tags else None
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Loading templates...", total=None)
                
                templates = await manager.list_templates(
                    template_type=template_type_enum,
                    status=status_enum,
                    created_by=author,
                    tags=tags_list,
                    limit=limit
                )
                
                progress.update(task, description="Templates loaded!")
            
            if not templates:
                console.print("[yellow]No templates found matching the criteria.[/yellow]")
                console.print("\n[dim]Try creating your first template with:[/dim]")
                console.print("[dim]  autom8 templates create[/dim]")
                return
            
            if list_format == 'json':
                output = [template.dict() for template in templates]
                console.print(json.dumps(output, indent=2, default=str))
                return
            
            elif list_format == 'detailed':
                for template in templates:
                    _display_template_detail(template)
                    console.print()
                return
            
            # Table format (default)
            table = Table(title=f"Templates ({len(templates)} found)")
            table.add_column("ID", style="cyan", width=25)
            table.add_column("Title", style="green", width=30)
            table.add_column("Type", style="blue", width=12)
            table.add_column("Status", style="yellow", width=10)
            table.add_column("Variables", style="dim", justify="right", width=9)
            table.add_column("Sources", style="dim", justify="right", width=8)
            table.add_column("Usage", style="magenta", justify="right", width=8)
            table.add_column("Updated", style="dim", width=12)
            
            for template in templates:
                status_style = {
                    'active': 'green',
                    'draft': 'yellow', 
                    'deprecated': 'orange1',
                    'archived': 'red'
                }.get(template.status.value, 'white')
                
                updated_str = template.updated_at.strftime("%Y-%m-%d") if template.updated_at else "N/A"
                
                table.add_row(
                    template.template_id,
                    template.metadata.title[:27] + "..." if len(template.metadata.title) > 30 else template.metadata.title,
                    template.type.value,
                    f"[{status_style}]{template.status.value}[/{status_style}]",
                    str(len(template.variables)),
                    str(len(template.sources)),
                    str(template.metadata.usage_count),
                    updated_str
                )
            
            console.print(table)
            
            # Show summary
            console.print(f"\n[dim]Showing {len(templates)} templates")
            if len(templates) == limit:
                console.print(f"[dim]Use --limit to see more results")
            
        except Exception as e:
            console.print(f"[red]Error listing templates: {e}[/red]")
    
    asyncio.run(list_templates())


@templates.command()
@click.argument('template_id')
@click.option('--format', 'show_format', type=click.Choice(['detailed', 'json', 'yaml']), default='detailed', help='Output format')
@click.option('--include-analytics', is_flag=True, help='Include usage analytics')
def show(template_id: str, show_format: str, include_analytics: bool):
    """Show detailed information about a template."""
    
    console.print(f"[bold blue]Template Details: {template_id}[/bold blue]\n")
    
    async def show_template():
        try:
            manager = await get_template_manager()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Loading template...", total=None)
                
                template = await manager.get_template(template_id)
                
                if not template:
                    console.print(f"[red]Template '{template_id}' not found.[/red]")
                    return
                
                analytics = None
                if include_analytics:
                    progress.update(task, description="Loading analytics...")
                    analytics = await manager.get_template_analytics(template_id)
                
                progress.update(task, description="Template loaded!")
            
            if show_format == 'json':
                output = template.dict()
                if analytics:
                    output['analytics'] = analytics
                console.print(json.dumps(output, indent=2, default=str))
                return
            
            elif show_format == 'yaml':
                output = template.dict()
                if analytics:
                    output['analytics'] = analytics
                console.print(yaml.dump(output, default_flow_style=False))
                return
            
            # Detailed format (default)
            _display_template_detail(template, analytics)
            
        except Exception as e:
            console.print(f"[red]Error showing template: {e}[/red]")
    
    asyncio.run(show_template())


@templates.command()
@click.argument('template_id')
@click.option('--title', help='Template title')
@click.option('--description', help='Template description')
@click.option('--type', 'template_type', type=click.Choice(['context', 'source', 'query', 'optimization', 'workflow', 'agent']), default='context', help='Template type')
@click.option('--category', help='Template category')
@click.option('--tags', help='Tags (comma-separated)')
@click.option('--interactive', '-i', is_flag=True, help='Interactive template creation')
def create(template_id: str, title: Optional[str], description: Optional[str], 
           template_type: str, category: Optional[str], tags: Optional[str], 
           interactive: bool):
    """Create a new template."""
    
    console.print(f"[bold blue]Create Template: {template_id}[/bold blue]\n")
    
    async def create_template():
        try:
            manager = await get_template_manager()
            
            # Check if template exists
            existing = await manager.get_template(template_id)
            if existing:
                console.print(f"[red]Template '{template_id}' already exists.[/red]")
                if not Confirm.ask("Do you want to overwrite it?"):
                    return
            
            # Get template information
            if interactive or not all([title, description]):
                title = title or Prompt.ask("Template title")
                description = description or Prompt.ask("Template description", default="")
                category = category or Prompt.ask("Category", default="general")
                tags_input = tags or Prompt.ask("Tags (comma-separated)", default="")
            else:
                tags_input = tags or ""
            
            # Parse inputs
            tags_list = [tag.strip() for tag in tags_input.split(',')] if tags_input else []
            
            # Create metadata
            metadata = TemplateMetadata(
                title=title,
                description=description,
                category=category,
                tags=tags_list
            )
            
            # Start with basic template
            template_sources = []
            template_variables = []
            
            if interactive:
                # Interactive template building
                template_sources, template_variables = await _interactive_template_builder()
            else:
                # Create a basic template with a query source
                console.print("[dim]Creating basic template with query source...[/dim]")
                template_sources.append(TemplateSource(
                    type="query",
                    content_template="{{query}}",
                    priority=100
                ))
                template_variables.append(TemplateVariable(
                    name="query",
                    type=VariableType.STRING,
                    description="The main query or task",
                    required=True
                ))
            
            # Create template
            template = await manager.create_template(
                template_id=template_id,
                template_type=TemplateType(template_type),
                metadata=metadata,
                variables=template_variables,
                sources=template_sources
            )
            
            console.print(f"[green]✓ Template '{template_id}' created successfully![/green]")
            console.print(f"[dim]Type: {template.type.value} | Variables: {len(template.variables)} | Sources: {len(template.sources)}[/dim]")
            
            # Ask if user wants to test the template
            if Confirm.ask("Would you like to test the template now?"):
                await _test_template_interactive(manager, template_id)
            
        except Exception as e:
            console.print(f"[red]Error creating template: {e}[/red]")
    
    asyncio.run(create_template())


@templates.command()
@click.argument('template_id')
@click.option('--variables', help='Variables as JSON string')
@click.option('--with-approval', is_flag=True, help='Use approval workflow (recommended)')
@click.option('--dry-run', is_flag=True, help='Test without executing')
@click.option('--output-format', type=click.Choice(['preview', 'context', 'json']), default='preview', help='Output format')
def execute(template_id: str, variables: Optional[str], with_approval: bool, dry_run: bool, output_format: str):
    """Execute a template with provided variables."""
    
    console.print(f"[bold blue]Execute Template: {template_id}[/bold blue]\n")
    
    async def execute_template():
        try:
            manager = await get_template_manager()
            
            # Parse variables
            variables_dict = {}
            if variables:
                try:
                    variables_dict = json.loads(variables)
                except json.JSONDecodeError:
                    console.print("[red]Invalid JSON for variables.[/red]")
                    return
            
            # Get template to check required variables
            template = await manager.get_template(template_id)
            if not template:
                console.print(f"[red]Template '{template_id}' not found.[/red]")
                return
            
            # Check for missing required variables
            required_vars = [var.name for var in template.variables if var.required and var.default_value is None]
            missing_vars = [var for var in required_vars if var not in variables_dict]
            
            if missing_vars:
                console.print(f"[yellow]Missing required variables: {', '.join(missing_vars)}[/yellow]")
                
                # Interactive variable input
                for var_name in missing_vars:
                    var_def = next((v for v in template.variables if v.name == var_name), None)
                    if var_def:
                        value = _prompt_for_variable(var_def)
                        variables_dict[var_name] = value
            
            # Choose execution path based on approval workflow
            if with_approval:
                # Use the approval workflow for better user experience
                from autom8.cli.preview import ContextPreviewManager, ApprovalDecision, PreviewFormat
                
                preview_manager = ContextPreviewManager(console)
                if not await preview_manager.initialize():
                    console.print("[red]Failed to initialize preview manager[/red]")
                    return
                
                # Request approval for template execution
                decision, final_preview = await preview_manager.request_template_approval(
                    template_id=template_id,
                    variables=variables_dict,
                    preview_format=PreviewFormat(output_format) if output_format == 'preview' else PreviewFormat.DETAILED
                )
                
                # Handle approval decision
                if decision == ApprovalDecision.APPROVE:
                    console.print(f"\n[green]✓ Template context approved![/green]")
                    if final_preview:
                        console.print(f"[dim]Final context: {final_preview.total_tokens} tokens, ${final_preview.cost_estimate:.4f}[/dim]")
                        
                        # Export approved context if requested
                        if not dry_run:
                            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                            export_filename = f"template_{template_id}_context_{timestamp}.txt"
                            
                            # Here you would typically send to model - for now we export
                            context_package = preview_manager.inspector.export_context_package(
                                final_preview, "default"
                            )
                            with open(export_filename, 'w') as f:
                                f.write(context_package)
                            console.print(f"[green]Context exported to {export_filename}[/green]")
                
                elif decision == ApprovalDecision.CANCEL:
                    console.print(f"[yellow]Template execution cancelled[/yellow]")
                else:
                    console.print(f"[yellow]Template execution result: {decision.value}[/yellow]")
            
            else:
                # Direct execution without approval workflow
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Executing template...", total=None)
                    
                    if output_format == 'preview':
                        # Execute to context preview
                        success, result = await manager.execute_template_to_context_preview(
                            template_id=template_id,
                            variables=variables_dict
                        )
                        
                        progress.update(task, description="Template executed!")
                        
                        if success:
                            console.print(f"[green]✓ Template executed successfully![/green]\n")
                            
                            # Display as context preview
                            preview_manager = ContextPreviewManager(console)
                            preview_manager._display_preview_detailed(result)
                        else:
                            console.print(f"[red]✗ Template execution failed: {result}[/red]")
                    
                    else:
                        # Execute template directly
                        result = await manager.execute_template(
                            template_id=template_id,
                            variables=variables_dict,
                            dry_run=dry_run
                        )
                        
                        progress.update(task, description="Template executed!")
                        
                        if result.success:
                            console.print(f"[green]✓ Template executed successfully![/green]")
                            console.print(f"[dim]Rendered {len(result.rendered_sources)} sources, {result.total_tokens} tokens[/dim]\n")
                            
                            if output_format == 'json':
                                output = {
                                    "success": True,
                                    "sources": result.rendered_sources,
                                    "tokens": result.total_tokens,
                                    "render_time_ms": result.render_time_ms,
                                    "warnings": result.warnings
                                }
                                console.print(json.dumps(output, indent=2))
                            
                            elif output_format == 'context':
                                # Display rendered sources
                                for i, source in enumerate(result.rendered_sources):
                                    console.print(f"[bold cyan]Source {i+1} ({source['type']}):[/bold cyan]")
                                    console.print(source['content'])
                                    console.print()
                        else:
                            console.print(f"[red]✗ Template execution failed: {result.error_message}[/red]")
                            if result.validation_errors:
                                console.print(f"[red]Validation errors: {', '.join(result.validation_errors)}[/red]")
            
        except Exception as e:
            console.print(f"[red]Error executing template: {e}[/red]")
    
    asyncio.run(execute_template())


@templates.command()
@click.argument('template_id')
@click.option('--test-file', type=click.Path(exists=True), help='JSON file with test cases')
@click.option('--generate-tests', is_flag=True, help='Auto-generate test cases')
@click.option('--include-edge-cases', is_flag=True, help='Include edge case tests')
def test(template_id: str, test_file: Optional[str], generate_tests: bool, include_edge_cases: bool):
    """Test a template with various inputs."""
    
    console.print(f"[bold blue]Test Template: {template_id}[/bold blue]\n")
    
    async def test_template():
        try:
            manager = await get_template_manager()
            
            # Get template
            template = await manager.get_template(template_id)
            if not template:
                console.print(f"[red]Template '{template_id}' not found.[/red]")
                return
            
            # Get test cases
            test_cases = []
            
            if test_file:
                with open(test_file, 'r') as f:
                    test_data = json.load(f)
                    test_cases = test_data.get('test_cases', [])
            
            elif generate_tests:
                console.print("[dim]Generating test cases...[/dim]")
                validator = manager.validator
                test_cases = await validator.create_test_suite(template, include_edge_cases)
            
            else:
                # Interactive test creation
                test_cases = await _create_interactive_tests(template)
            
            if not test_cases:
                console.print("[yellow]No test cases to run.[/yellow]")
                return
            
            # Run tests
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
            ) as progress:
                task = progress.add_task("Running tests...", total=len(test_cases))
                
                validator = manager.validator
                results = await validator.run_template_tests(template, test_cases)
                
                progress.update(task, completed=len(test_cases))
            
            # Display results
            _display_test_results(results)
            
        except Exception as e:
            console.print(f"[red]Error testing template: {e}[/red]")
    
    asyncio.run(test_template())


@templates.command()
@click.argument('template_id')
@click.option('--include-warnings', is_flag=True, help='Include warnings in validation')
@click.option('--fix-issues', is_flag=True, help='Attempt to fix common issues')
def validate(template_id: str, include_warnings: bool, fix_issues: bool):
    """Validate template structure and content."""
    
    console.print(f"[bold blue]Validate Template: {template_id}[/bold blue]\n")
    
    async def validate_template():
        try:
            manager = await get_template_manager()
            
            template = await manager.get_template(template_id)
            if not template:
                console.print(f"[red]Template '{template_id}' not found.[/red]")
                return
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Validating template...", total=None)
                
                validator = manager.validator
                results = await validator.validate_template_detailed(
                    template, include_warnings=include_warnings
                )
                
                progress.update(task, description="Validation complete!")
            
            # Display validation results
            _display_validation_results(results)
            
            if fix_issues and results['issues']:
                if Confirm.ask("Attempt to fix common issues?"):
                    await _fix_template_issues(manager, template, results)
            
        except Exception as e:
            console.print(f"[red]Error validating template: {e}[/red]")
    
    asyncio.run(validate_template())


@templates.command()
@click.argument('template_ids', nargs=-1, required=True)
@click.option('--output-id', required=True, help='ID for the composed template')
@click.option('--strategy', type=click.Choice(['append', 'replace', 'merge']), default='append', help='Merge strategy')
@click.option('--conflict-resolution', type=click.Choice(['latest', 'priority', 'error']), default='latest', help='Conflict resolution')
def compose(template_ids: tuple, output_id: str, strategy: str, conflict_resolution: str):
    """Compose multiple templates into a new template."""
    
    console.print(f"[bold blue]Compose Templates[/bold blue]\n")
    console.print(f"Input templates: {', '.join(template_ids)}")
    console.print(f"Output template: {output_id}")
    console.print(f"Strategy: {strategy}, Conflicts: {conflict_resolution}\n")
    
    async def compose_templates():
        try:
            manager = await get_template_manager()
            
            # Check if output template already exists
            existing = await manager.get_template(output_id)
            if existing:
                if not Confirm.ask(f"Template '{output_id}' exists. Overwrite?"):
                    return
            
            # Compose templates
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Composing templates...", total=None)
                
                composed = await manager.compose_templates(
                    base_template_ids=list(template_ids),
                    variables={},
                    merge_strategy=strategy,
                    conflict_resolution=conflict_resolution
                )
                
                # Update ID and store
                composed.template_id = output_id
                success = await manager.storage.store_template(composed)
                
                progress.update(task, description="Templates composed!")
            
            if success:
                console.print(f"[green]✓ Successfully composed templates into '{output_id}'[/green]")
                console.print(f"[dim]Variables: {len(composed.variables)}, Sources: {len(composed.sources)}[/dim]")
                
                if Confirm.ask("Show the composed template?"):
                    _display_template_detail(composed)
            else:
                console.print(f"[red]✗ Failed to store composed template[/red]")
            
        except Exception as e:
            console.print(f"[red]Error composing templates: {e}[/red]")
    
    asyncio.run(compose_templates())


@templates.command()
@click.argument('pattern_name', type=click.Choice(['code_review', 'documentation', 'bug_analysis']))
@click.argument('template_id')
@click.option('--customizations', help='JSON string with customizations')
def from_pattern(pattern_name: str, template_id: str, customizations: Optional[str]):
    """Create a template from a predefined pattern."""
    
    console.print(f"[bold blue]Create Template from Pattern: {pattern_name}[/bold blue]\n")
    
    async def create_from_pattern():
        try:
            manager = await get_template_manager()
            
            # Parse customizations
            customizations_dict = {}
            if customizations:
                try:
                    customizations_dict = json.loads(customizations)
                except json.JSONDecodeError:
                    console.print("[red]Invalid JSON for customizations.[/red]")
                    return
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Creating template from pattern...", total=None)
                
                template = await manager.composer.create_template_from_pattern(
                    pattern_name=pattern_name,
                    template_id=template_id,
                    customizations=customizations_dict
                )
                
                # Store the template
                success = await manager.storage.store_template(template)
                
                progress.update(task, description="Template created!")
            
            if success:
                console.print(f"[green]✓ Template '{template_id}' created from pattern '{pattern_name}'[/green]")
                _display_template_detail(template)
                
                if Confirm.ask("Test the template?"):
                    await _test_template_interactive(manager, template_id)
            else:
                console.print(f"[red]✗ Failed to store template[/red]")
            
        except Exception as e:
            console.print(f"[red]Error creating template from pattern: {e}[/red]")
    
    asyncio.run(create_from_pattern())


@templates.command()
@click.argument('template_id', required=False)
@click.option('--days', default=30, help='Days to include in analytics')
@click.option('--format', 'analytics_format', type=click.Choice(['summary', 'detailed', 'json']), default='summary', help='Report format')
def analytics(template_id: Optional[str], days: int, analytics_format: str):
    """Show template usage analytics and insights."""
    
    title = f"Template Analytics: {template_id}" if template_id else "System Analytics"
    console.print(f"[bold blue]{title}[/bold blue]\n")
    
    async def show_analytics():
        try:
            manager = await get_template_manager()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Generating analytics...", total=None)
                
                report = await manager.analytics.generate_analytics_report(
                    template_id=template_id,
                    time_range_days=days
                )
                
                progress.update(task, description="Analytics generated!")
            
            if analytics_format == 'json':
                console.print(json.dumps(report, indent=2, default=str))
                return
            
            # Display formatted analytics
            _display_analytics_report(report, analytics_format == 'detailed')
            
        except Exception as e:
            console.print(f"[red]Error generating analytics: {e}[/red]")
    
    asyncio.run(show_analytics())


@templates.command()
@click.argument('template_id')
@click.option('--format', type=click.Choice(['json', 'yaml']), default='json', help='Export format')
@click.option('--include-metadata', is_flag=True, default=True, help='Include metadata')
@click.option('--output-file', help='Output file path')
def export(template_id: str, format: str, include_metadata: bool, output_file: Optional[str]):
    """Export a template to file."""
    
    console.print(f"[bold blue]Export Template: {template_id}[/bold blue]\n")
    
    async def export_template():
        try:
            manager = await get_template_manager()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Exporting template...", total=None)
                
                exported_data = await manager.export_template(
                    template_id=template_id,
                    format=format,
                    include_metadata=include_metadata
                )
                
                progress.update(task, description="Template exported!")
            
            # Determine output file
            if not output_file:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                output_file = f"{template_id}_export_{timestamp}.{format}"
            
            # Write to file
            with open(output_file, 'w') as f:
                if format == 'json':
                    if isinstance(exported_data, dict):
                        json.dump(exported_data, f, indent=2, default=str)
                    else:
                        f.write(exported_data)
                else:
                    f.write(exported_data)
            
            console.print(f"[green]✓ Template exported to {output_file}[/green]")
            
        except Exception as e:
            console.print(f"[red]Error exporting template: {e}[/red]")
    
    asyncio.run(export_template())


@templates.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--format', type=click.Choice(['json', 'yaml']), default='json', help='Import format')
@click.option('--merge-strategy', type=click.Choice(['replace', 'skip', 'error']), default='replace', help='How to handle existing templates')
def import_template(file_path: str, format: str, merge_strategy: str):
    """Import a template from file."""
    
    console.print(f"[bold blue]Import Template: {file_path}[/bold blue]\n")
    
    async def import_template_file():
        try:
            manager = await get_template_manager()
            
            # Read file
            with open(file_path, 'r') as f:
                if format == 'json':
                    template_data = json.load(f)
                else:
                    template_data = yaml.safe_load(f)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Importing template...", total=None)
                
                template = await manager.import_template(
                    template_data=template_data,
                    format=format,
                    merge_strategy=merge_strategy
                )
                
                progress.update(task, description="Template imported!")
            
            console.print(f"[green]✓ Template '{template.template_id}' imported successfully[/green]")
            _display_template_detail(template)
            
        except Exception as e:
            console.print(f"[red]Error importing template: {e}[/red]")
    
    asyncio.run(import_template_file())


# Helper functions

def _display_template_detail(template: ContextTemplate, analytics: Optional[Dict] = None):
    """Display detailed template information."""
    
    # Basic info panel
    info_content = f"""
[bold]ID:[/bold] {template.template_id}
[bold]Type:[/bold] {template.type.value}
[bold]Status:[/bold] {template.status.value}
[bold]Category:[/bold] {template.metadata.category or 'N/A'}
[bold]Author:[/bold] {template.metadata.author or 'N/A'}
[bold]Version:[/bold] {template.metadata.version}
[bold]Created:[/bold] {template.created_at.strftime('%Y-%m-%d %H:%M') if template.created_at else 'N/A'}
[bold]Updated:[/bold] {template.updated_at.strftime('%Y-%m-%d %H:%M') if template.updated_at else 'N/A'}
"""
    
    console.print(Panel(info_content.strip(), title=template.metadata.title, border_style="blue"))
    
    # Description
    if template.metadata.description:
        console.print(f"\n[bold]Description:[/bold]\n{template.metadata.description}")
    
    # Tags
    if template.metadata.tags:
        console.print(f"\n[bold]Tags:[/bold] {', '.join(template.metadata.tags)}")
    
    # Variables
    if template.variables:
        console.print(f"\n[bold]Variables ({len(template.variables)}):[/bold]")
        var_table = Table()
        var_table.add_column("Name", style="cyan")
        var_table.add_column("Type", style="green")
        var_table.add_column("Required", style="yellow")
        var_table.add_column("Default", style="dim")
        var_table.add_column("Description", style="white")
        
        for var in template.variables:
            var_table.add_row(
                var.name,
                var.type.value,
                "Yes" if var.required else "No",
                str(var.default_value) if var.default_value is not None else "",
                var.description
            )
        
        console.print(var_table)
    
    # Sources
    if template.sources:
        console.print(f"\n[bold]Sources ({len(template.sources)}):[/bold]")
        for i, source in enumerate(template.sources):
            console.print(f"  {i+1}. [cyan]{source.type}[/cyan] (priority: {source.priority})")
            if source.condition:
                console.print(f"     [dim]Condition: {source.condition}[/dim]")
            content_preview = source.content_template[:100] + "..." if len(source.content_template) > 100 else source.content_template
            console.print(f"     [dim]{content_preview}[/dim]")
    
    # Composition
    if template.composition:
        console.print(f"\n[bold]Composition:[/bold]")
        console.print(f"  Base templates: {', '.join(template.composition.base_templates)}")
        console.print(f"  Merge strategy: {template.composition.merge_strategy}")
    
    # Analytics
    if analytics:
        console.print(f"\n[bold]Usage Analytics:[/bold]")
        real_time = analytics.get('real_time', {})
        console.print(f"  Executions: {real_time.get('total_executions', 0)}")
        console.print(f"  Success rate: {real_time.get('success_rate', 0):.1%}")
        console.print(f"  Avg quality: {real_time.get('avg_quality_score', 0):.2f}")


def _display_validation_results(results: Dict[str, Any]):
    """Display template validation results."""
    
    summary = results['summary']
    issues = results['issues']
    
    # Overall status
    status_color = "green" if summary['is_valid'] else "red"
    status_text = "Valid ✓" if summary['is_valid'] else "Invalid ✗"
    
    console.print(f"[bold]Validation Status:[/bold] [{status_color}]{status_text}[/{status_color}]")
    console.print(f"[bold]Total Issues:[/bold] {summary['total_issues']} (Errors: {summary['errors']}, Warnings: {summary['warnings']}, Info: {summary['info']})")
    
    if issues:
        console.print(f"\n[bold]Issues Found:[/bold]")
        
        # Group by severity
        errors = [i for i in issues if i.get('severity') == 'error']
        warnings = [i for i in issues if i.get('severity') == 'warning']
        info = [i for i in issues if i.get('severity') == 'info']
        
        for severity, issue_list in [("Errors", errors), ("Warnings", warnings), ("Info", info)]:
            if issue_list:
                color = {"Errors": "red", "Warnings": "yellow", "Info": "blue"}[severity]
                console.print(f"\n[bold {color}]{severity}:[/bold {color}]")
                for issue in issue_list:
                    field_info = f" ({issue['field']})" if issue.get('field') else ""
                    console.print(f"  [{color}]•[/{color}] {issue['message']}{field_info}")


def _display_test_results(results: Dict[str, Any]):
    """Display template test results."""
    
    console.print(f"[bold]Test Results for {results['template_id']}[/bold]")
    
    success_color = "green" if results['success_rate'] == 1.0 else "yellow" if results['success_rate'] > 0.5 else "red"
    console.print(f"[bold]Success Rate:[/bold] [{success_color}]{results['success_rate']:.1%}[/{success_color}] ({results['passed_tests']}/{results['total_tests']} passed)")
    
    if results['test_cases']:
        console.print(f"\n[bold]Test Cases:[/bold]")
        
        table = Table()
        table.add_column("Test", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Description", style="white")
        table.add_column("Errors", style="red")
        
        for test_case in results['test_cases']:
            status = "✓ Pass" if test_case['passed'] else "✗ Fail"
            status_style = "green" if test_case['passed'] else "red"
            
            errors_text = ""
            if test_case.get('errors'):
                errors_text = "; ".join(test_case['errors'][:2])  # Show first 2 errors
                if len(test_case['errors']) > 2:
                    errors_text += "..."
            
            table.add_row(
                test_case['test_id'],
                f"[{status_style}]{status}[/{status_style}]",
                test_case['description'],
                errors_text
            )
        
        console.print(table)


def _display_analytics_report(report: Dict[str, Any], detailed: bool = False):
    """Display analytics report."""
    
    summary = report.get('summary', {})
    
    if 'template_id' in report:
        # Single template report
        console.print(f"[bold]Template Report[/bold]")
        console.print(f"Usage level: {summary.get('usage_level', 'unknown')}")
        console.print(f"Performance: {summary.get('performance_rating', 'unknown')}")
        console.print(f"Quality score: {summary.get('quality_score', 0):.2f}")
        console.print(f"Health: {summary.get('health_status', 'unknown')}")
    else:
        # System report
        console.print(f"[bold]System Report[/bold]")
        console.print(f"Templates: {summary.get('total_templates', 0)}")
        console.print(f"Total executions: {summary.get('total_executions', 0)}")
        console.print(f"Success rate: {summary.get('system_success_rate', 0):.1%}")
        
        if detailed:
            details = report.get('details', {})
            
            # Popular templates
            popular = details.get('popular_templates', [])
            if popular:
                console.print(f"\n[bold]Most Popular Templates:[/bold]")
                pop_table = Table()
                pop_table.add_column("Template", style="cyan")
                pop_table.add_column("Executions", style="yellow", justify="right")
                pop_table.add_column("Success Rate", style="green", justify="right")
                
                for template in popular[:10]:
                    pop_table.add_row(
                        template['template_id'],
                        str(template['total_executions']),
                        f"{template['success_rate']:.1%}"
                    )
                
                console.print(pop_table)
            
            # Recommendations
            recommendations = details.get('recommendations', [])
            if recommendations:
                console.print(f"\n[bold]Recommendations:[/bold]")
                for rec in recommendations:
                    priority_color = {"high": "red", "medium": "yellow", "low": "blue"}.get(rec['priority'], "white")
                    console.print(f"  [{priority_color}]•[/{priority_color}] {rec['title']}: {rec['description']}")


async def _interactive_template_builder() -> tuple:
    """Interactive template builder."""
    
    console.print("[bold]Interactive Template Builder[/bold]\n")
    
    sources = []
    variables = []
    
    # Add variables
    console.print("[bold]Define Variables:[/bold]")
    while True:
        if not Confirm.ask("Add a variable?" if not variables else "Add another variable?"):
            break
        
        var_name = Prompt.ask("Variable name")
        var_type = Prompt.ask(
            "Variable type",
            choices=[t.value for t in VariableType],
            default="string"
        )
        var_desc = Prompt.ask("Description", default="")
        var_required = Confirm.ask("Required?", default=True)
        var_default = Prompt.ask("Default value (or press Enter for none)", default="") or None
        
        variables.append(TemplateVariable(
            name=var_name,
            type=VariableType(var_type),
            description=var_desc,
            required=var_required,
            default_value=var_default
        ))
    
    # Add sources
    console.print("\n[bold]Define Sources:[/bold]")
    while True:
        if not Confirm.ask("Add a source?" if not sources else "Add another source?"):
            break
        
        source_type = Prompt.ask(
            "Source type",
            choices=["query", "reference", "memory", "context", "retrieved"],
            default="reference"
        )
        
        content_template = Prompt.ask("Content template")
        priority = int(Prompt.ask("Priority (0-100)", default="50"))
        
        source_id_template = Prompt.ask("Source ID template (optional)", default="") or None
        condition = Prompt.ask("Condition (optional)", default="") or None
        
        sources.append(TemplateSource(
            type=source_type,
            content_template=content_template,
            priority=priority,
            source_id_template=source_id_template,
            condition=condition
        ))
    
    return sources, variables


def _prompt_for_variable(var_def: TemplateVariable) -> Any:
    """Prompt user for variable value based on variable definition."""
    
    prompt_text = f"{var_def.name}"
    if var_def.description:
        prompt_text += f" ({var_def.description})"
    
    if var_def.type == VariableType.BOOLEAN:
        return Confirm.ask(prompt_text)
    elif var_def.type == VariableType.ENUM and var_def.allowed_values:
        return Prompt.ask(prompt_text, choices=[str(v) for v in var_def.allowed_values])
    elif var_def.type == VariableType.NUMBER:
        value = Prompt.ask(prompt_text)
        try:
            return float(value) if '.' in value else int(value)
        except ValueError:
            console.print("[red]Invalid number, using 0[/red]")
            return 0
    else:
        return Prompt.ask(prompt_text)


async def _test_template_interactive(manager: TemplateManager, template_id: str):
    """Interactive template testing."""
    
    template = await manager.get_template(template_id)
    if not template:
        return
    
    console.print(f"\n[bold]Test Template: {template_id}[/bold]")
    
    # Get variable values
    variables = {}
    for var in template.variables:
        if var.required and var.default_value is None:
            value = _prompt_for_variable(var)
            variables[var.name] = value
        elif Confirm.ask(f"Set value for optional variable '{var.name}'?"):
            value = _prompt_for_variable(var)
            variables[var.name] = value
    
    # Execute template
    try:
        result = await manager.execute_template(template_id, variables, dry_run=True)
        
        if result.success:
            console.print(f"[green]✓ Template test successful![/green]")
            console.print(f"[dim]Generated {len(result.rendered_sources)} sources, {result.total_tokens} tokens[/dim]")
            
            if Confirm.ask("Show rendered sources?"):
                for i, source in enumerate(result.rendered_sources):
                    console.print(f"\n[cyan]Source {i+1} ({source['type']}):[/cyan]")
                    console.print(source['content'][:300] + "..." if len(source['content']) > 300 else source['content'])
        else:
            console.print(f"[red]✗ Template test failed: {result.error_message}[/red]")
    
    except Exception as e:
        console.print(f"[red]Error testing template: {e}[/red]")


async def _create_interactive_tests(template: ContextTemplate) -> List[Dict[str, Any]]:
    """Create test cases interactively."""
    
    test_cases = []
    
    console.print(f"\n[bold]Create Test Cases[/bold]")
    
    while True:
        if not Confirm.ask("Create a test case?" if not test_cases else "Create another test case?"):
            break
        
        test_desc = Prompt.ask("Test description")
        
        # Get variables for this test
        variables = {}
        for var in template.variables:
            if var.required:
                value = _prompt_for_variable(var)
                variables[var.name] = value
        
        # Simple test case (just check execution)
        test_case = {
            "description": test_desc,
            "variables": variables,
            "expected": {
                "assertions": {
                    "should_succeed": True,
                    "source_count": {"operator": ">=", "value": 1}
                }
            }
        }
        
        test_cases.append(test_case)
    
    return test_cases


async def _fix_template_issues(manager: TemplateManager, template: ContextTemplate, validation_results: Dict[str, Any]):
    """Attempt to fix common template issues."""
    
    issues = validation_results['issues']
    fixed_issues = []
    
    for issue in issues:
        if issue.get('severity') != 'error':
            continue
        
        # Try to fix common issues
        message = issue['message']
        
        if "missing" in message.lower() and "variable" in message.lower():
            # Add missing variable
            if "query" in message:
                template.variables.append(TemplateVariable(
                    name="query",
                    type=VariableType.STRING,
                    description="The main query or task",
                    required=True
                ))
                fixed_issues.append("Added missing 'query' variable")
        
        # Add more fix patterns as needed
    
    if fixed_issues:
        # Update template
        success = await manager.update_template(template.template_id, {
            "variables": template.variables,
            "sources": template.sources
        })
        
        if success:
            console.print(f"[green]✓ Fixed {len(fixed_issues)} issues:[/green]")
            for fix in fixed_issues:
                console.print(f"  • {fix}")
        else:
            console.print(f"[red]✗ Failed to save fixes[/red]")
    else:
        console.print(f"[yellow]No automatic fixes available[/yellow]")