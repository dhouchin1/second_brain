"""
CLI Context Preview and Approval Workflow

This module implements comprehensive context preview and approval workflow for the CLI,
building on the interactive editing system to provide users with transparent control
over what context is sent to models.

Features:
- Interactive context preview with Rich formatting
- Approval workflow with approve/edit/cancel options
- Token counting and cost estimation display
- User preference tracking for learning
- Integration with existing context editing system
- History of approval decisions
"""

import asyncio
import json
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.layout import Layout
from rich.live import Live
from rich.tree import Tree
from rich.text import Text
from rich.rule import Rule

from autom8.models.context import (
    ContextPreview,
    ContextSource,
    ContextSourceType,
    ContextWarning,
    ContextOptimization,
)
from autom8.core.context.inspector import ContextInspector
from autom8.core.context.editor import ContextEditSession, ValidationLevel
from autom8.utils.logging import get_logger
from autom8.config.settings import get_settings

logger = get_logger(__name__)


class ApprovalDecision(str, Enum):
    """User approval decisions"""
    APPROVE = "approve"
    EDIT = "edit"
    CANCEL = "cancel"
    OPTIMIZE = "optimize"
    SAVE_DRAFT = "save_draft"


class PreviewFormat(str, Enum):
    """Context preview display formats"""
    COMPACT = "compact"
    DETAILED = "detailed"
    MINIMAL = "minimal"
    FULL = "full"


@dataclass
class ApprovalSession:
    """Tracks an approval session with user preferences"""
    session_id: str
    preview: ContextPreview
    decision: Optional[ApprovalDecision] = None
    decision_time: Optional[datetime] = None
    edit_session_id: Optional[str] = None
    user_feedback: Optional[str] = None
    cost_accepted: bool = False
    tokens_accepted: bool = False
    approval_speed: Optional[float] = None  # seconds to decision
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "session_id": self.session_id,
            "preview_summary": {
                "total_tokens": self.preview.total_tokens,
                "cost_estimate": self.preview.cost_estimate,
                "source_count": len(self.preview.sources),
                "warnings": len(self.preview.warnings),
                "model_target": self.preview.model_target
            },
            "decision": self.decision.value if self.decision else None,
            "decision_time": self.decision_time.isoformat() if self.decision_time else None,
            "edit_session_id": self.edit_session_id,
            "user_feedback": self.user_feedback,
            "cost_accepted": self.cost_accepted,
            "tokens_accepted": self.tokens_accepted,
            "approval_speed": self.approval_speed
        }


class UserPreferences:
    """Tracks user preferences for approval decisions"""
    
    def __init__(self, preferences_file: Optional[str] = None):
        self.preferences_file = preferences_file or str(
            Path.home() / ".autom8" / "approval_preferences.json"
        )
        self.preferences = self._load_preferences()
    
    def _load_preferences(self) -> Dict[str, Any]:
        """Load user preferences from file"""
        try:
            if Path(self.preferences_file).exists():
                with open(self.preferences_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load preferences: {e}")
        
        return {
            "auto_approve_under_tokens": None,
            "auto_approve_under_cost": None,
            "default_format": PreviewFormat.DETAILED.value,
            "show_warnings_first": True,
            "show_optimization_suggestions": True,
            "approval_history": [],
            "trusted_models": [],
            "cost_thresholds": {},
            "token_thresholds": {}
        }
    
    def save_preferences(self):
        """Save preferences to file"""
        try:
            Path(self.preferences_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.preferences_file, 'w') as f:
                json.dump(self.preferences, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save preferences: {e}")
    
    def record_approval_session(self, session: ApprovalSession):
        """Record an approval session for learning"""
        self.preferences["approval_history"].append(session.to_dict())
        
        # Keep only last 100 sessions
        if len(self.preferences["approval_history"]) > 100:
            self.preferences["approval_history"] = self.preferences["approval_history"][-100:]
        
        self.save_preferences()
    
    def should_auto_approve(self, preview: ContextPreview) -> bool:
        """Check if context should be auto-approved based on preferences"""
        auto_token_limit = self.preferences.get("auto_approve_under_tokens")
        auto_cost_limit = self.preferences.get("auto_approve_under_cost")
        
        if auto_token_limit and preview.total_tokens <= auto_token_limit:
            return True
        
        if auto_cost_limit and preview.cost_estimate <= auto_cost_limit:
            return True
        
        return False
    
    def get_suggested_action(self, preview: ContextPreview) -> Optional[str]:
        """Get suggested action based on user history"""
        recent_sessions = self.preferences["approval_history"][-10:]
        
        if not recent_sessions:
            return None
        
        # Look for patterns in similar contexts
        similar_sessions = [
            s for s in recent_sessions
            if abs(s["preview_summary"]["total_tokens"] - preview.total_tokens) < 200
        ]
        
        if similar_sessions:
            most_common_decision = max(
                set(s["decision"] for s in similar_sessions if s["decision"]),
                key=lambda d: sum(1 for s in similar_sessions if s["decision"] == d),
                default=None
            )
            return most_common_decision
        
        return None


class ContextPreviewManager:
    """
    Manages context preview and approval workflow for CLI operations.
    
    This class orchestrates the entire approval process, from initial preview
    display through user interaction and final approval or cancellation.
    """
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.inspector = ContextInspector()
        self.user_preferences = UserPreferences()
        self.current_session: Optional[ApprovalSession] = None
        
    async def initialize(self) -> bool:
        """Initialize the preview manager"""
        try:
            await self.inspector.initialize()
            logger.info("Context preview manager initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize preview manager: {e}")
            return False
    
    async def request_approval(
        self,
        query: str,
        agent_id: str = "cli-user",
        context_sources: Optional[List[ContextSource]] = None,
        model_target: Optional[str] = None,
        complexity_score: Optional[float] = None,
        preview_format: PreviewFormat = PreviewFormat.DETAILED,
        auto_approve: bool = False
    ) -> Tuple[ApprovalDecision, Optional[ContextPreview]]:
        """
        Request user approval for context before model execution.
        
        Args:
            query: The user query
            agent_id: ID of the requesting agent
            context_sources: Additional context sources
            model_target: Target model for cost estimation
            complexity_score: Complexity score for the query
            preview_format: How to display the preview
            auto_approve: Whether to auto-approve based on preferences
            
        Returns:
            Tuple of (decision, final_preview)
        """
        start_time = datetime.utcnow()
        
        # Create context preview
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Analyzing context...", total=None)
            
            preview = await self.inspector.preview(
                query=query,
                agent_id=agent_id,
                context_sources=context_sources,
                model_target=model_target,
                complexity_score=complexity_score
            )
            
            progress.update(task, description="Analysis complete!")
        
        # Create approval session
        session_id = f"approval_{datetime.utcnow().timestamp()}"
        self.current_session = ApprovalSession(
            session_id=session_id,
            preview=preview
        )
        
        # Check for auto-approval
        if auto_approve and self.user_preferences.should_auto_approve(preview):
            self.console.print("[green]✓ Auto-approved based on your preferences[/green]")
            self.current_session.decision = ApprovalDecision.APPROVE
            self.current_session.decision_time = datetime.utcnow()
            self.current_session.approval_speed = (datetime.utcnow() - start_time).total_seconds()
            self.user_preferences.record_approval_session(self.current_session)
            return ApprovalDecision.APPROVE, preview
        
        # Display context preview
        self._display_context_preview(preview, preview_format)
        
        # Get user decision
        decision, final_preview = await self._get_user_decision(preview, start_time)
        
        # Record session
        self.current_session.decision = decision
        self.current_session.decision_time = datetime.utcnow()
        self.current_session.approval_speed = (datetime.utcnow() - start_time).total_seconds()
        self.user_preferences.record_approval_session(self.current_session)
        
        return decision, final_preview
    
    async def request_template_approval(
        self,
        template_id: str,
        variables: Dict[str, Any],
        agent_id: str = "template-user",
        model_target: Optional[str] = None,
        preview_format: PreviewFormat = PreviewFormat.DETAILED,
        auto_approve: bool = False
    ) -> Tuple[ApprovalDecision, Optional[ContextPreview]]:
        """
        Request approval for template execution with context preview.
        
        This method integrates templates with the existing approval workflow,
        allowing users to preview what context will be generated from a template
        before execution.
        """
        from autom8.core.templates import get_template_manager
        
        start_time = datetime.utcnow()
        
        self.console.print(f"[bold blue]Template Execution Preview[/bold blue]")
        self.console.print(f"Template: [cyan]{template_id}[/cyan]")
        
        # Show variables being used
        if variables:
            vars_text = ", ".join(f"{k}={v}" for k, v in variables.items())
            self.console.print(f"Variables: [dim]{vars_text}[/dim]")
        
        self.console.print()
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Executing template...", total=None)
                
                # Get template manager and execute template
                manager = await get_template_manager()
                
                # Execute template to context preview
                success, result = await manager.execute_template_to_context_preview(
                    template_id=template_id,
                    variables=variables,
                    agent_id=agent_id,
                    model_target=model_target
                )
                
                progress.update(task, description="Template executed!")
            
            if not success:
                self.console.print(f"[red]✗ Template execution failed: {result}[/red]")
                return ApprovalDecision.CANCEL, None
            
            preview = result
            
            # Create approval session
            session_id = f"template_approval_{datetime.utcnow().timestamp()}"
            self.current_session = ApprovalSession(
                session_id=session_id,
                preview=preview
            )
            
            # Add template metadata to session
            self.current_session.template_id = template_id
            self.current_session.template_variables = variables
            
            # Check for auto-approval
            if auto_approve and self.user_preferences.should_auto_approve(preview):
                self.console.print("[green]✓ Template context auto-approved[/green]")
                self.current_session.decision = ApprovalDecision.APPROVE
                self.current_session.decision_time = datetime.utcnow()
                self.current_session.approval_speed = (datetime.utcnow() - start_time).total_seconds()
                self.user_preferences.record_approval_session(self.current_session)
                return ApprovalDecision.APPROVE, preview
            
            # Display context preview with template context
            self._display_template_preview(preview, template_id, variables, preview_format)
            
            # Get user decision
            decision, final_preview = await self._get_template_decision(preview, start_time)
            
            # Record session
            self.current_session.decision = decision
            self.current_session.decision_time = datetime.utcnow()
            self.current_session.approval_speed = (datetime.utcnow() - start_time).total_seconds()
            self.user_preferences.record_approval_session(self.current_session)
            
            return decision, final_preview
            
        except Exception as e:
            logger.error(f"Template approval failed: {e}")
            self.console.print(f"[red]Error during template approval: {e}[/red]")
            return ApprovalDecision.CANCEL, None
    
    def _display_template_preview(
        self,
        preview: ContextPreview,
        template_id: str,
        variables: Dict[str, Any],
        format: PreviewFormat
    ):
        """Display context preview with template context information."""
        
        # Template-specific header
        template_info = f"""
[bold]Template:[/bold] {template_id}
[bold]Generated Context:[/bold] {len(preview.sources)} sources, {preview.total_tokens} tokens
[bold]Estimated Cost:[/bold] ${preview.cost_estimate:.4f}
[bold]Quality Score:[/bold] {preview.quality_score:.2f}/5.0
"""
        
        if preview.metadata and preview.metadata.get('template_render_time_ms'):
            template_info += f"[bold]Render Time:[/bold] {preview.metadata['template_render_time_ms']}ms\n"
        
        self.console.print(Panel(
            template_info.strip(),
            title="[bold cyan]Template Execution Results[/bold cyan]",
            border_style="cyan"
        ))
        
        # Show variables used
        if variables:
            var_table = Table(title="Template Variables")
            var_table.add_column("Variable", style="cyan")
            var_table.add_column("Value", style="white")
            var_table.add_column("Type", style="dim")
            
            for name, value in variables.items():
                value_str = str(value)
                if len(value_str) > 50:
                    value_str = value_str[:47] + "..."
                
                var_table.add_row(
                    name,
                    value_str,
                    type(value).__name__
                )
            
            self.console.print(var_table)
            self.console.print()
        
        # Display the actual context preview
        self._display_context_preview(preview, format)
    
    async def _get_template_decision(
        self,
        preview: ContextPreview,
        start_time: datetime
    ) -> Tuple[ApprovalDecision, ContextPreview]:
        """Get user decision for template-generated context."""
        
        while True:
            self.console.print("\n[bold]Choose an action for the template-generated context:[/bold]")
            self.console.print("  [green]1. Approve[/green] - Use this context")
            self.console.print("  [cyan]2. Edit Template[/cyan] - Edit the template itself")
            self.console.print("  [blue]3. Edit Context[/blue] - Modify just this context")
            self.console.print("  [yellow]4. Re-execute[/yellow] - Re-run template with different variables")
            self.console.print("  [magenta]5. Optimize[/magenta] - Apply automatic optimizations")
            self.console.print("  [dim]6. Save as Template[/dim] - Save this context as a new template")
            self.console.print("  [red]7. Cancel[/red] - Cancel operation")
            
            choice = Prompt.ask(
                "Your choice",
                choices=["1", "2", "3", "4", "5", "6", "7"],
                default="1"
            )
            
            if choice == "1":  # Approve
                if await self._confirm_approval(preview):
                    return ApprovalDecision.APPROVE, preview
            
            elif choice == "2":  # Edit template
                self.console.print("[yellow]Template editing not yet implemented in this interface.[/yellow]")
                self.console.print("[dim]Use: autom8 templates show <template_id> to view template details[/dim]")
                
            elif choice == "3":  # Edit context
                edited_preview = await self._handle_interactive_edit(preview)
                if edited_preview:
                    preview = edited_preview
                    self._display_context_preview(preview, PreviewFormat.DETAILED)
            
            elif choice == "4":  # Re-execute with different variables
                new_variables = await self._get_new_template_variables()
                if new_variables is not None:
                    # Return special decision to indicate re-execution needed
                    return ApprovalDecision.EDIT, preview  # Reuse EDIT for re-execution
            
            elif choice == "5":  # Optimize
                optimized_preview = await self._handle_optimization(preview)
                if optimized_preview:
                    preview = optimized_preview
                    self._display_context_preview(preview, PreviewFormat.DETAILED)
            
            elif choice == "6":  # Save as template
                await self._handle_save_as_template(preview)
                
            elif choice == "7":  # Cancel
                if Confirm.ask("Are you sure you want to cancel?"):
                    return ApprovalDecision.CANCEL, preview
    
    async def _get_new_template_variables(self) -> Optional[Dict[str, Any]]:
        """Interactive input for new template variables."""
        from autom8.core.templates import get_template_manager
        
        if not hasattr(self.current_session, 'template_id'):
            return None
        
        try:
            manager = await get_template_manager()
            template = await manager.get_template(self.current_session.template_id)
            
            if not template:
                self.console.print("[red]Template not found[/red]")
                return None
            
            self.console.print(f"\n[bold]Re-execute template with new variables:[/bold]")
            
            new_variables = {}
            for var in template.variables:
                if var.required:
                    current_value = self.current_session.template_variables.get(var.name, "")
                    new_value = Prompt.ask(
                        f"{var.name} ({var.description})",
                        default=str(current_value)
                    )
                    
                    # Convert value based on type
                    if var.type.value == "number":
                        try:
                            new_variables[var.name] = float(new_value) if '.' in new_value else int(new_value)
                        except ValueError:
                            new_variables[var.name] = new_value
                    elif var.type.value == "boolean":
                        new_variables[var.name] = new_value.lower() in ("true", "yes", "1")
                    else:
                        new_variables[var.name] = new_value
            
            return new_variables
            
        except Exception as e:
            self.console.print(f"[red]Error getting new variables: {e}[/red]")
            return None
    
    async def _handle_save_as_template(self, preview: ContextPreview):
        """Handle saving current context as a new template."""
        from autom8.core.templates import get_template_manager
        
        self.console.print(f"\n[bold]Save Context as Template[/bold]")
        
        try:
            template_id = Prompt.ask("Template ID")
            title = Prompt.ask("Template title")
            description = Prompt.ask("Template description", default="")
            category = Prompt.ask("Category", default="user_generated")
            
            manager = await get_template_manager()
            
            # Check if template exists
            existing = await manager.get_template(template_id)
            if existing:
                if not Confirm.ask(f"Template '{template_id}' exists. Overwrite?"):
                    return
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Creating template from context...", total=None)
                
                from autom8.models.template import TemplateMetadata
                
                metadata = TemplateMetadata(
                    title=title,
                    description=description,
                    category=category,
                    tags=["user_generated", "from_context"]
                )
                
                template = await manager.create_template_from_context(
                    context_preview=preview,
                    template_id=template_id,
                    metadata=metadata,
                    extract_variables=True
                )
                
                progress.update(task, description="Template created!")
            
            self.console.print(f"[green]✓ Template '{template_id}' created successfully![/green]")
            self.console.print(f"[dim]Variables: {len(template.variables)}, Sources: {len(template.sources)}[/dim]")
            
        except Exception as e:
            self.console.print(f"[red]Error saving template: {e}[/red]")
    
    def _display_context_preview(self, preview: ContextPreview, format: PreviewFormat):
        """Display the context preview with Rich formatting"""
        
        # Header with key metrics
        self._display_preview_header(preview)
        
        # Show format-specific content
        if format == PreviewFormat.MINIMAL:
            self._display_minimal_preview(preview)
        elif format == PreviewFormat.COMPACT:
            self._display_compact_preview(preview)
        elif format == PreviewFormat.DETAILED:
            self._display_detailed_preview(preview)
        elif format == PreviewFormat.FULL:
            self._display_full_preview(preview)
        
        # Always show warnings and optimization suggestions
        self._display_warnings_and_optimizations(preview)
        
        # Show user guidance
        self._display_user_guidance(preview)
    
    def _display_preview_header(self, preview: ContextPreview):
        """Display header with key metrics"""
        # Create metrics table
        metrics_table = Table.grid(padding=1)
        metrics_table.add_column(style="bold cyan")
        metrics_table.add_column(style="bold white")
        
        metrics_table.add_row("Total Tokens:", f"{preview.total_tokens:,}")
        metrics_table.add_row("Sources:", str(len(preview.sources)))
        metrics_table.add_row("Estimated Cost:", f"${preview.cost_estimate:.4f}")
        metrics_table.add_row("Quality Score:", f"{preview.quality_score:.2f}")
        
        if preview.model_target:
            metrics_table.add_row("Target Model:", preview.model_target)
        
        # Status indicators
        status_items = []
        
        if preview.is_over_budget:
            status_items.append("[red]⚠ Over Budget[/red]")
        else:
            status_items.append("[green]✓ Within Budget[/green]")
        
        if preview.has_high_warnings:
            status_items.append("[red]⚠ High Warnings[/red]")
        elif preview.warnings:
            status_items.append("[yellow]⚠ Warnings[/yellow]")
        else:
            status_items.append("[green]✓ No Issues[/green]")
        
        status_text = " | ".join(status_items)
        
        # Create header panel
        header_content = Columns([
            Panel(metrics_table, title="Context Metrics", border_style="blue"),
            Panel(Text(status_text, justify="center"), title="Status", border_style="blue")
        ])
        
        self.console.print(Panel(
            header_content,
            title=f"[bold blue]Context Preview - {preview.agent_id}[/bold blue]",
            border_style="blue"
        ))
    
    def _display_minimal_preview(self, preview: ContextPreview):
        """Display minimal preview with just source count and types"""
        source_types = {}
        for source in preview.sources:
            type_name = source.type.value
            source_types[type_name] = source_types.get(type_name, 0) + 1
        
        type_summary = ", ".join([f"{count} {type}" for type, count in source_types.items()])
        
        self.console.print(f"\n[dim]Sources: {type_summary}[/dim]")
        self.console.print(f"[dim]Query: {preview.query[:100]}{'...' if len(preview.query) > 100 else ''}[/dim]")
    
    def _display_compact_preview(self, preview: ContextPreview):
        """Display compact preview with source summary"""
        table = Table(title="Context Sources Summary")
        table.add_column("Type", style="cyan")
        table.add_column("Count", style="yellow", justify="right")
        table.add_column("Tokens", style="green", justify="right")
        table.add_column("Preview", style="dim")
        
        # Group sources by type
        type_groups = {}
        for source in preview.sources:
            type_name = source.type.value
            if type_name not in type_groups:
                type_groups[type_name] = []
            type_groups[type_name].append(source)
        
        for type_name, sources in type_groups.items():
            total_tokens = sum(s.tokens for s in sources)
            preview_content = sources[0].content[:60] + "..." if len(sources[0].content) > 60 else sources[0].content
            
            table.add_row(
                type_name,
                str(len(sources)),
                str(total_tokens),
                preview_content
            )
        
        self.console.print(table)
    
    def _display_detailed_preview(self, preview: ContextPreview):
        """Display detailed preview with all sources"""
        table = Table(title="Context Sources")
        table.add_column("Index", style="dim", width=5)
        table.add_column("Type", style="cyan", width=12)
        table.add_column("Source", style="green", width=20)
        table.add_column("Priority", style="yellow", width=8, justify="right")
        table.add_column("Tokens", style="blue", width=8, justify="right")
        table.add_column("Content Preview", style="white")
        
        for i, source in enumerate(preview.sources):
            content_preview = source.content[:80] + "..." if len(source.content) > 80 else source.content
            content_preview = content_preview.replace("\n", " ")
            
            table.add_row(
                str(i),
                source.type.value,
                source.source[:18] + "..." if len(source.source) > 18 else source.source,
                str(source.priority),
                str(source.tokens),
                content_preview
            )
        
        self.console.print(table)
        
        # Show token breakdown
        breakdown_table = Table(title="Token Breakdown by Type")
        breakdown_table.add_column("Source Type", style="cyan")
        breakdown_table.add_column("Tokens", style="yellow", justify="right")
        breakdown_table.add_column("Percentage", style="green", justify="right")
        
        for type_name, tokens in preview.source_breakdown.items():
            percentage = (tokens / preview.total_tokens) * 100 if preview.total_tokens > 0 else 0
            breakdown_table.add_row(
                type_name,
                str(tokens),
                f"{percentage:.1f}%"
            )
        
        self.console.print(breakdown_table)
    
    def _display_full_preview(self, preview: ContextPreview):
        """Display full preview with complete content"""
        self.console.print(Rule("[bold]Full Context Content[/bold]"))
        
        for i, source in enumerate(preview.sources):
            # Create source header
            source_info = f"[{i}] {source.type.value.upper()}: {source.source}"
            if source.priority != 50:  # Show priority if not default
                source_info += f" (priority: {source.priority})"
            
            self.console.print(f"\n[bold cyan]{source_info}[/bold cyan]")
            self.console.print(f"[dim]Tokens: {source.tokens}[/dim]")
            
            # Show content in a panel
            content_panel = Panel(
                source.content,
                border_style="dim",
                padding=(0, 1)
            )
            self.console.print(content_panel)
        
        self.console.print(Rule())
    
    def _display_warnings_and_optimizations(self, preview: ContextPreview):
        """Display warnings and optimization suggestions"""
        if preview.warnings:
            warning_table = Table(title="⚠ Context Warnings", border_style="yellow")
            warning_table.add_column("Severity", style="yellow")
            warning_table.add_column("Type", style="cyan")
            warning_table.add_column("Message", style="white")
            warning_table.add_column("Suggestion", style="dim")
            
            for warning in preview.warnings:
                severity_text = "🔴 High" if warning.severity >= 3 else "🟡 Medium" if warning.severity >= 2 else "🔵 Low"
                suggestion_text = warning.suggestion or "N/A"
                
                warning_table.add_row(
                    severity_text,
                    warning.type.value,
                    warning.message,
                    suggestion_text[:50] + "..." if len(suggestion_text) > 50 else suggestion_text
                )
            
            self.console.print(warning_table)
        
        if preview.optimizations:
            opt_table = Table(title="💡 Optimization Suggestions", border_style="green")
            opt_table.add_column("Description", style="white")
            opt_table.add_column("Token Savings", style="green", justify="right")
            opt_table.add_column("Quality Impact", style="yellow", justify="right")
            opt_table.add_column("Action Required", style="cyan")
            
            for opt in preview.optimizations:
                quality_impact_text = f"{opt.quality_impact:.1%}"
                impact_color = "red" if opt.quality_impact > 0.3 else "yellow" if opt.quality_impact > 0.1 else "green"
                
                opt_table.add_row(
                    opt.description,
                    str(opt.estimated_savings),
                    f"[{impact_color}]{quality_impact_text}[/{impact_color}]",
                    opt.action_required[:40] + "..." if len(opt.action_required) > 40 else opt.action_required
                )
            
            self.console.print(opt_table)
    
    def _display_user_guidance(self, preview: ContextPreview):
        """Display user guidance and preferences"""
        guidance_items = []
        
        # Cost guidance
        if preview.cost_estimate > 0.01:
            guidance_items.append(f"💰 This query will cost approximately ${preview.cost_estimate:.4f}")
        
        # Token budget guidance
        if preview.is_over_budget:
            guidance_items.append("⚠ Context exceeds recommended token limit - consider optimization")
        
        # Preference suggestions
        suggested_action = self.user_preferences.get_suggested_action(preview)
        if suggested_action:
            guidance_items.append(f"💡 Based on your history, you usually {suggested_action} similar contexts")
        
        if guidance_items:
            guidance_text = "\n".join(f"  • {item}" for item in guidance_items)
            self.console.print(Panel(
                guidance_text,
                title="[bold yellow]Guidance[/bold yellow]",
                border_style="yellow"
            ))
    
    async def _get_user_decision(
        self, 
        preview: ContextPreview, 
        start_time: datetime
    ) -> Tuple[ApprovalDecision, ContextPreview]:
        """Get user decision through interactive prompt"""
        
        while True:
            self.console.print("\n[bold]Choose an action:[/bold]")
            self.console.print("  [green]1. Approve[/green] - Send context to model")
            self.console.print("  [cyan]2. Edit[/cyan] - Modify context before sending")
            self.console.print("  [blue]3. Optimize[/blue] - Apply automatic optimizations")
            self.console.print("  [yellow]4. Change View[/yellow] - Change preview format")
            self.console.print("  [dim]5. Save Draft[/dim] - Save for later review")
            self.console.print("  [red]6. Cancel[/red] - Cancel operation")
            
            # Show keyboard shortcuts
            self.console.print("\n[dim]Shortcuts: a=approve, e=edit, o=optimize, v=view, s=save, c=cancel[/dim]")
            
            choice = Prompt.ask(
                "Your choice",
                choices=["1", "2", "3", "4", "5", "6", "a", "e", "o", "v", "s", "c"],
                default="1"
            )
            
            # Map choices to actions
            choice_map = {
                "1": "approve", "a": "approve",
                "2": "edit", "e": "edit",
                "3": "optimize", "o": "optimize",
                "4": "view", "v": "view",
                "5": "save", "s": "save",
                "6": "cancel", "c": "cancel"
            }
            
            action = choice_map[choice]
            
            if action == "approve":
                # Confirm if high cost or warnings
                if await self._confirm_approval(preview):
                    return ApprovalDecision.APPROVE, preview
                # If not confirmed, continue loop
            
            elif action == "edit":
                edited_preview = await self._handle_interactive_edit(preview)
                if edited_preview:
                    preview = edited_preview
                    self._display_context_preview(preview, PreviewFormat.DETAILED)
                # Continue loop to get decision on edited preview
            
            elif action == "optimize":
                optimized_preview = await self._handle_optimization(preview)
                if optimized_preview:
                    preview = optimized_preview
                    self._display_context_preview(preview, PreviewFormat.DETAILED)
                # Continue loop to get decision on optimized preview
            
            elif action == "view":
                await self._handle_view_change(preview)
                # Continue loop
            
            elif action == "save":
                await self._handle_save_draft(preview)
                return ApprovalDecision.SAVE_DRAFT, preview
            
            elif action == "cancel":
                if Confirm.ask("Are you sure you want to cancel?"):
                    return ApprovalDecision.CANCEL, preview
                # If not confirmed, continue loop
    
    async def _confirm_approval(self, preview: ContextPreview) -> bool:
        """Confirm approval if there are concerns"""
        concerns = []
        
        if preview.cost_estimate > 0.01:
            concerns.append(f"Cost: ${preview.cost_estimate:.4f}")
        
        if preview.has_high_warnings:
            concerns.append(f"{len(preview.warnings)} high-severity warnings")
        
        if preview.is_over_budget:
            concerns.append(f"Large context ({preview.total_tokens:,} tokens)")
        
        if not concerns:
            return True
        
        self.console.print(f"\n[yellow]⚠ Please confirm - this context has:[/yellow]")
        for concern in concerns:
            self.console.print(f"  • {concern}")
        
        return Confirm.ask("\nProceed anyway?")
    
    async def _handle_interactive_edit(self, preview: ContextPreview) -> Optional[ContextPreview]:
        """Handle interactive editing of context"""
        self.console.print("\n[bold cyan]Starting Interactive Context Editor...[/bold cyan]")
        
        try:
            # Create edit session
            session = await self.inspector.create_edit_session(
                preview=preview,
                max_history=50,
                validation_level="permissive",
                auto_backup=True
            )
            
            self.current_session.edit_session_id = session.session_id
            
            # Run simplified interactive editing (similar to existing CLI edit command)
            edited_preview = await self._run_edit_session(session)
            
            if edited_preview:
                self.console.print("[green]✓ Context edited successfully[/green]")
                return edited_preview
            else:
                self.console.print("[yellow]Edit cancelled[/yellow]")
                return None
                
        except Exception as e:
            self.console.print(f"[red]Edit failed: {e}[/red]")
            return None
    
    async def _run_edit_session(self, session) -> Optional[ContextPreview]:
        """Run a simplified edit session"""
        from rich.prompt import Prompt
        
        while True:
            # Show current state
            self._display_edit_state_summary(session)
            
            # Show available commands
            self.console.print("\n[bold]Edit Commands:[/bold]")
            self.console.print("  [cyan]show[/cyan] - Show current context")
            self.console.print("  [cyan]add[/cyan] - Add new context source")
            self.console.print("  [cyan]edit <index>[/cyan] - Edit source by index")
            self.console.print("  [cyan]remove <index>[/cyan] - Remove source by index")
            self.console.print("  [cyan]done[/cyan] - Finish editing")
            self.console.print("  [cyan]cancel[/cyan] - Cancel editing")
            
            command = Prompt.ask("\nEdit command").strip().lower()
            
            if command == "done":
                return session.current_preview
            elif command == "cancel":
                return None
            elif command == "show":
                self._display_detailed_preview(session.current_preview)
            elif command == "add":
                await self._handle_add_source_quick(session)
            elif command.startswith("edit "):
                try:
                    index = int(command.split()[1])
                    await self._handle_edit_source_quick(session, index)
                except (IndexError, ValueError):
                    self.console.print("[red]Usage: edit <index>[/red]")
            elif command.startswith("remove "):
                try:
                    index = int(command.split()[1])
                    await self._handle_remove_source_quick(session, index)
                except (IndexError, ValueError):
                    self.console.print("[red]Usage: remove <index>[/red]")
            else:
                self.console.print("[red]Unknown command. Type 'done' to finish or 'cancel' to cancel.[/red]")
    
    def _display_edit_state_summary(self, session):
        """Display brief edit state summary"""
        summary = session.get_edit_summary()
        
        status_text = (
            f"Sources: {summary['source_count']} | "
            f"Tokens: {summary['total_tokens']} | "
            f"Edits: {summary['edit_count']}"
        )
        
        self.console.print(f"\n[bold blue]Edit Session Status:[/bold blue] {status_text}")
    
    async def _handle_add_source_quick(self, session):
        """Quick add source interface"""
        from autom8.models.context import ContextSourceType
        
        content = Prompt.ask("Content")
        source_type_str = Prompt.ask(
            "Source type", 
            choices=["reference", "memory", "summary"], 
            default="reference"
        )
        source_type = ContextSourceType(source_type_str)
        priority = int(Prompt.ask("Priority (0-100)", default="50"))
        
        # Create source
        from autom8.models.context import ContextSource
        new_source = ContextSource(
            type=source_type,
            content=content,
            tokens=0,  # Will be calculated
            source="user_added",
            priority=priority
        )
        
        success = await session.add_source(new_source)
        if success:
            self.console.print("[green]Source added[/green]")
        else:
            self.console.print("[red]Failed to add source[/red]")
    
    async def _handle_edit_source_quick(self, session, index):
        """Quick edit source interface"""
        if not (0 <= index < len(session.current_preview.sources)):
            self.console.print(f"[red]Invalid index: {index}[/red]")
            return
        
        source = session.current_preview.sources[index]
        self.console.print(f"Editing source {index}: {source.source}")
        
        new_content = Prompt.ask("New content", default=source.content)
        new_priority = int(Prompt.ask(f"New priority", default=str(source.priority)))
        
        success = await session.modify_source(index, {
            "content": new_content,
            "priority": new_priority
        })
        
        if success:
            self.console.print("[green]Source modified[/green]")
        else:
            self.console.print("[red]Failed to modify source[/red]")
    
    async def _handle_remove_source_quick(self, session, index):
        """Quick remove source interface"""
        if not (0 <= index < len(session.current_preview.sources)):
            self.console.print(f"[red]Invalid index: {index}[/red]")
            return
        
        source = session.current_preview.sources[index]
        if Confirm.ask(f"Remove source {index} ({source.source})?"):
            success = await session.remove_source(index)
            if success:
                self.console.print("[green]Source removed[/green]")
            else:
                self.console.print("[red]Failed to remove source[/red]")
    
    async def _handle_optimization(self, preview: ContextPreview) -> Optional[ContextPreview]:
        """Handle context optimization"""
        self.console.print("\n[bold blue]Optimizing Context...[/bold blue]")
        
        # Choose optimization profile
        profile = Prompt.ask(
            "Optimization profile",
            choices=["conservative", "balanced", "aggressive"],
            default="balanced"
        )
        
        try:
            optimized_preview, report = await self.inspector.optimize_with_edits(
                preview,
                optimization_profile=profile
            )
            
            # Display optimization results
            self._display_optimization_results(report)
            
            if Confirm.ask("Apply optimization?"):
                return optimized_preview
            else:
                return None
                
        except Exception as e:
            self.console.print(f"[red]Optimization failed: {e}[/red]")
            return None
    
    def _display_optimization_results(self, report: Dict[str, Any]):
        """Display optimization results"""
        results_table = Table(title="Optimization Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Before", style="yellow")
        results_table.add_column("After", style="green")
        results_table.add_column("Change", style="blue")
        
        results_table.add_row(
            "Tokens",
            str(report["original_tokens"]),
            str(report["optimized_tokens"]),
            f"-{report['tokens_saved']} ({report['compression_ratio']:.1%})"
        )
        
        results_table.add_row(
            "Quality",
            "100%",
            f"{report['quality_retention']:.1%}",
            f"-{100 - report['quality_retention']:.1f}%"
        )
        
        self.console.print(results_table)
        
        if report.get("strategies_applied"):
            self.console.print(f"\n[dim]Strategies applied: {', '.join(report['strategies_applied'])}[/dim]")
    
    async def _handle_view_change(self, preview: ContextPreview):
        """Handle changing the preview format"""
        current_format = self.user_preferences.preferences.get("default_format", "detailed")
        
        format_choice = Prompt.ask(
            "Preview format",
            choices=["minimal", "compact", "detailed", "full"],
            default=current_format
        )
        
        # Update preference
        self.user_preferences.preferences["default_format"] = format_choice
        self.user_preferences.save_preferences()
        
        # Redisplay with new format
        self.console.clear()
        self._display_context_preview(preview, PreviewFormat(format_choice))
    
    async def _handle_save_draft(self, preview: ContextPreview):
        """Handle saving a draft for later review"""
        filename = f"context_draft_{datetime.utcnow().timestamp()}.json"
        
        try:
            draft_data = {
                "preview": preview.dict(),
                "saved_at": datetime.utcnow().isoformat(),
                "session_id": self.current_session.session_id if self.current_session else None
            }
            
            with open(filename, 'w') as f:
                json.dump(draft_data, f, indent=2, default=str)
            
            self.console.print(f"[green]Draft saved to {filename}[/green]")
            
        except Exception as e:
            self.console.print(f"[red]Failed to save draft: {e}[/red]")
    
    def get_approval_analytics(self) -> Dict[str, Any]:
        """Get analytics about user approval patterns"""
        history = self.user_preferences.preferences.get("approval_history", [])
        
        if not history:
            return {"message": "No approval history available"}
        
        total_sessions = len(history)
        decisions = [s["decision"] for s in history if s["decision"]]
        
        decision_counts = {}
        for decision in decisions:
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        
        avg_approval_speed = sum(
            s["approval_speed"] for s in history 
            if s["approval_speed"] is not None
        ) / len([s for s in history if s["approval_speed"] is not None]) if any(s["approval_speed"] for s in history) else 0
        
        return {
            "total_sessions": total_sessions,
            "decision_breakdown": decision_counts,
            "approval_rate": decision_counts.get("approve", 0) / total_sessions if total_sessions > 0 else 0,
            "average_approval_speed_seconds": avg_approval_speed,
            "most_common_decision": max(decision_counts.items(), key=lambda x: x[1])[0] if decision_counts else None
        }