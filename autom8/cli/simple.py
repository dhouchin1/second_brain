"""
Simple CLI for testing core components without complex configuration.
"""

import asyncio
import click
from rich.console import Console
from rich.panel import Panel

console = Console()


@click.group()
def cli():
    """Simple Autom8 CLI for testing core components."""
    pass


@cli.command()
@click.argument('query')
def analyze(query: str):
    """Analyze query complexity."""
    console.print(f"[bold blue]Analyzing:[/bold blue] {query}")
    
    async def run_analysis():
        from autom8.core.complexity.analyzer import ComplexityAnalyzer
        
        analyzer = ComplexityAnalyzer()
        complexity = await analyzer.analyze(query)
        
        result = f"""
[bold]Complexity Score:[/bold] {complexity.raw_score:.3f}
[bold]Recommended Tier:[/bold] {complexity.recommended_tier.value}
[bold]Task Pattern:[/bold] {complexity.task_pattern.value}
[bold]Confidence:[/bold] {complexity.confidence:.3f}

[bold]Reasoning:[/bold]
{complexity.reasoning}
"""
        
        console.print(Panel(result.strip(), title="Complexity Analysis"))
    
    asyncio.run(run_analysis())


@cli.command()
@click.argument('query')
def inspect(query: str):
    """Inspect context for query."""
    console.print(f"[bold blue]Inspecting:[/bold blue] {query}")
    
    async def run_inspection():
        from autom8.core.context.inspector import ContextInspector
        
        inspector = ContextInspector()
        preview = await inspector.preview(query, "test-agent")
        
        result = f"""
[bold]Total Tokens:[/bold] {preview.total_tokens}
[bold]Sources:[/bold] {len(preview.sources)}
[bold]Quality Score:[/bold] {preview.quality_score:.2f}
[bold]Estimated Cost:[/bold] ${preview.cost_estimate:.4f}

[bold]Warnings:[/bold] {len(preview.warnings)}
[bold]Optimizations:[/bold] {len(preview.optimizations)}
"""
        
        console.print(Panel(result.strip(), title="Context Preview"))
    
    asyncio.run(run_inspection())


@cli.command()
@click.argument('query')
def route(query: str):
    """Route query to optimal model."""
    console.print(f"[bold blue]Routing:[/bold blue] {query}")
    
    async def run_routing():
        from autom8.core.complexity.analyzer import ComplexityAnalyzer
        from autom8.core.routing.router import ModelRouter
        
        # Analyze complexity
        analyzer = ComplexityAnalyzer()
        complexity = await analyzer.analyze(query)
        
        # Route to model
        router = ModelRouter()
        selection = await router.route(query, complexity)
        
        model = selection.primary_model
        
        result = f"""
[bold]Selected Model:[/bold] {model.display_name}
[bold]Type:[/bold] {model.model_type.value}
[bold]Capability:[/bold] {model.capability_score:.2f}
[bold]Estimated Cost:[/bold] ${selection.estimated_cost:.4f}
[bold]Estimated Latency:[/bold] {selection.estimated_latency_ms:.0f}ms

[bold]Reasoning:[/bold]
{selection.selection_reasoning}
"""
        
        console.print(Panel(result.strip(), title="Model Selection"))
    
    asyncio.run(run_routing())


if __name__ == "__main__":
    cli()
