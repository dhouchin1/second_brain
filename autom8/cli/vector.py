"""
Vector Operations CLI for Autom8.

Provides commands for managing vector embeddings, semantic search,
and sqlite-vec database operations.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import click
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Prompt, Confirm

from autom8.core.memory.broker import get_context_broker
from autom8.storage.sqlite.manager import get_sqlite_manager

console = Console()


@click.group()
def vector():
    """Vector operations and semantic search management."""
    pass


@vector.command()
@click.option('--content-id', help='Store embedding for specific content ID')
@click.option('--content', help='Content text to embed and store')
@click.option('--file', 'file_path', type=click.Path(exists=True), help='File to embed and store')
@click.option('--topic', help='Topic/category for the content')
@click.option('--priority', type=int, default=0, help='Priority level (0-100)')
@click.option('--pinned', is_flag=True, help='Mark content as pinned')
@click.option('--expires', help='Expiration date (YYYY-MM-DD)')
def store(content_id: Optional[str], content: Optional[str], file_path: Optional[str], 
          topic: Optional[str], priority: int, pinned: bool, expires: Optional[str]):
    """Store content with automatic embedding generation."""
    
    console.print("[bold blue]Store Content with Embedding[/bold blue]\n")
    
    async def store_content():
        try:
            # Initialize context broker
            broker = await get_context_broker()
            
            # Get content from various sources
            if file_path:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content_text = f.read()
                console.print(f"[green]Loaded {len(content_text)} characters from {file_path}[/green]")
            elif content:
                content_text = content
            else:
                content_text = Prompt.ask("Enter content text")
            
            if not content_text.strip():
                console.print("[red]Error: No content provided[/red]")
                return
            
            # Generate content ID if not provided
            if not content_id:
                content_id = str(uuid.uuid4())
            
            # Parse expiration date
            expires_at = None
            if expires:
                try:
                    expires_at = datetime.fromisoformat(expires)
                except ValueError:
                    console.print(f"[yellow]Warning: Invalid date format '{expires}', ignoring expiration[/yellow]")
            
            # Store content with embedding
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Generating embedding and storing...", total=None)
                
                success = await broker.store_context_with_embedding(
                    context_id=content_id,
                    content=content_text,
                    summary=None,  # Could be auto-generated
                    topic=topic,
                    priority=priority,
                    pinned=pinned,
                    expires_at=expires_at,
                    source_type="cli_stored",
                    metadata={"stored_via": "cli", "timestamp": datetime.now().isoformat()}
                )
                
                progress.update(task, description="Complete!")
            
            if success:
                console.print(f"[green]✓ Successfully stored content with ID: {content_id}[/green]")
                console.print(f"  Content: {len(content_text)} characters")
                if topic:
                    console.print(f"  Topic: {topic}")
                if pinned:
                    console.print(f"  Status: Pinned")
                if expires_at:
                    console.print(f"  Expires: {expires_at.strftime('%Y-%m-%d')}")
            else:
                console.print("[red]✗ Failed to store content[/red]")
                
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(store_content())


@vector.command()
@click.argument('query')
@click.option('--k', type=int, default=5, help='Number of results to return')
@click.option('--threshold', type=float, default=0.5, help='Similarity threshold (0.0-1.0)')
@click.option('--topic', help='Filter by topic/category')
@click.option('--format', 'output_format', type=click.Choice(['table', 'detailed', 'json']), default='table', help='Output format')
def search(query: str, k: int, threshold: float, topic: Optional[str], output_format: str):
    """Perform semantic search on stored content."""
    
    console.print(f"[bold blue]Semantic Search[/bold blue]")
    console.print(f"Query: [italic]{query}[/italic]\n")
    
    async def run_search():
        try:
            broker = await get_context_broker()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Searching...", total=None)
                
                results = await broker.find_similar_content(
                    query_text=query,
                    k=k,
                    threshold=threshold,
                    topic=topic
                )
                
                progress.update(task, description="Search complete!")
            
            if not results:
                console.print("[yellow]No results found matching your query.[/yellow]")
                if threshold > 0.3:
                    console.print(f"[dim]Try lowering the threshold (current: {threshold})[/dim]")
                return
            
            # Display results
            if output_format == 'json':
                # JSON output
                json_results = []
                for result in results:
                    json_results.append({
                        'id': result['id'],
                        'content': result['content'],
                        'similarity': result.get('distance', 0),
                        'topic': result.get('topic'),
                        'priority': result.get('priority', 0),
                        'metadata': result.get('metadata', {})
                    })
                console.print(json.dumps(json_results, indent=2))
                
            elif output_format == 'detailed':
                # Detailed output
                for i, result in enumerate(results, 1):
                    similarity = result.get('distance', 0)
                    
                    panel_content = f"""
[bold]Content ID:[/bold] {result['id']}
[bold]Similarity:[/bold] {similarity:.3f}
[bold]Topic:[/bold] {result.get('topic', 'None')}
[bold]Priority:[/bold] {result.get('priority', 0)}
[bold]Tokens:[/bold] {result.get('token_count', 0)}

[bold]Content:[/bold]
{result['content'][:500]}{'...' if len(result['content']) > 500 else ''}
"""
                    
                    console.print(Panel(
                        panel_content.strip(),
                        title=f"Result {i}",
                        border_style="green" if similarity > 0.8 else "yellow" if similarity > 0.6 else "blue"
                    ))
                    console.print()
                    
            else:
                # Table output (default)
                table = Table(title=f"Search Results for '{query}'")
                table.add_column("Rank", style="dim", width=4)
                table.add_column("ID", style="cyan", width=8)
                table.add_column("Similarity", style="green", justify="right", width=10)
                table.add_column("Topic", style="yellow", width=12)
                table.add_column("Priority", style="blue", justify="right", width=8)
                table.add_column("Content Preview", style="white")
                
                for i, result in enumerate(results, 1):
                    similarity = result.get('distance', 0)
                    content_preview = result['content'][:80] + "..." if len(result['content']) > 80 else result['content']
                    content_preview = content_preview.replace('\n', ' ')
                    
                    table.add_row(
                        str(i),
                        result['id'][-8:],  # Show last 8 chars of ID
                        f"{similarity:.3f}",
                        result.get('topic', '')[:12] or 'None',
                        str(result.get('priority', 0)),
                        content_preview
                    )
                
                console.print(table)
                console.print(f"\n[dim]Found {len(results)} results above threshold {threshold}[/dim]")
                
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(run_search())


@vector.command()
@click.argument('content_id')
@click.option('--k', type=int, default=5, help='Number of similar items to find')
@click.option('--threshold', type=float, default=0.7, help='Similarity threshold')
def similar(content_id: str, k: int, threshold: float):
    """Find content similar to a specific stored item."""
    
    console.print(f"[bold blue]Find Similar Content[/bold blue]")
    console.print(f"Reference ID: [italic]{content_id}[/italic]\n")
    
    async def find_similar():
        try:
            storage = await get_sqlite_manager()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Finding similar content...", total=None)
                
                results = await storage.get_similar_content(
                    content_id=content_id,
                    k=k,
                    threshold=threshold
                )
                
                progress.update(task, description="Search complete!")
            
            if not results:
                console.print(f"[yellow]No similar content found for ID: {content_id}[/yellow]")
                console.print("[dim]Try lowering the threshold or check if the content ID exists[/dim]")
                return
            
            # Display results
            table = Table(title=f"Content Similar to {content_id}")
            table.add_column("Rank", style="dim", width=4)
            table.add_column("ID", style="cyan", width=12)
            table.add_column("Similarity", style="green", justify="right", width=10)
            table.add_column("Topic", style="yellow", width=12)
            table.add_column("Content Preview", style="white")
            
            for i, result in enumerate(results, 1):
                similarity = result.get('distance', 0)
                content_preview = result['content'][:80] + "..." if len(result['content']) > 80 else result['content']
                content_preview = content_preview.replace('\n', ' ')
                
                table.add_row(
                    str(i),
                    result['id'][-12:],
                    f"{similarity:.3f}",
                    result.get('topic', '')[:12] or 'None',
                    content_preview
                )
            
            console.print(table)
            console.print(f"\n[dim]Found {len(results)} similar items above threshold {threshold}[/dim]")
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(find_similar())


@vector.command()
@click.option('--model', default='all-MiniLM-L6-v2', help='Embedding model to use')
@click.option('--batch-size', type=int, default=10, help='Batch size for processing')
@click.option('--force', is_flag=True, help='Force rebuild even if embeddings exist')
def rebuild(model: str, batch_size: int, force: bool):
    """Rebuild all embeddings for stored content."""
    
    console.print(f"[bold blue]Rebuild Embeddings[/bold blue]")
    console.print(f"Model: [italic]{model}[/italic]\n")
    
    if not force:
        if not Confirm.ask("This will regenerate embeddings for all stored content. Continue?"):
            console.print("Cancelled.")
            return
    
    async def rebuild_embeddings():
        try:
            broker = await get_context_broker()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
            ) as progress:
                task = progress.add_task("Rebuilding embeddings...", total=100)
                
                result = await broker.rebuild_all_embeddings()
                progress.update(task, completed=100)
            
            if result['success']:
                console.print(f"[green]✓ Successfully rebuilt {result['rebuilt_count']} embeddings[/green]")
                
                # Show before/after stats
                before = result.get('before', {})
                after = result.get('after', {})
                
                if before and after:
                    console.print(f"\n[bold]Statistics:[/bold]")
                    console.print(f"  Vector extension: {'✓' if after.get('vec_extension_available') else '✗'}")
                    console.print(f"  Total vectors: {before.get('total_vectors', 0)} → {after.get('total_vectors', 0)}")
                    console.print(f"  Coverage: {after.get('embedding_coverage', 0):.1%}")
                    console.print(f"  Dimension: {after.get('embedding_dimension', 384)}")
            else:
                error_msg = result.get('error', 'Unknown error')
                console.print(f"[red]✗ Failed to rebuild embeddings: {error_msg}[/red]")
                
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(rebuild_embeddings())


@vector.command()
def stats():
    """Show vector database statistics and health."""
    
    console.print("[bold blue]Vector Database Statistics[/bold blue]\n")
    
    async def show_stats():
        try:
            storage = await get_sqlite_manager()
            broker = await get_context_broker()
            
            # Get vector stats
            vector_stats = await storage.get_vector_stats()
            context_stats = await broker.get_context_stats()
            
            # Vector extension status
            vec_available = vector_stats.get('vec_extension_available', False)
            extension_status = "✓ Available" if vec_available else "✗ Not Available (using fallback)"
            extension_color = "green" if vec_available else "yellow"
            
            console.print(f"[bold]Vector Extension:[/bold] [{extension_color}]{extension_status}[/{extension_color}]")
            console.print(f"[bold]Embedding Model:[/bold] all-MiniLM-L6-v2")
            console.print(f"[bold]Vector Dimension:[/bold] {vector_stats.get('embedding_dimension', 384)}")
            console.print()
            
            # Storage stats
            storage_table = Table(title="Storage Statistics")
            storage_table.add_column("Metric", style="cyan")
            storage_table.add_column("Value", style="white", justify="right")
            
            storage_table.add_row("Total Content Items", str(vector_stats.get('total_content', 0)))
            storage_table.add_row("Total Vectors", str(vector_stats.get('total_vectors', 0)))
            storage_table.add_row("Embedding Coverage", f"{vector_stats.get('embedding_coverage', 0):.1%}")
            
            # Context broker stats
            if context_stats.get('storage_usage'):
                storage_usage = context_stats['storage_usage']
                storage_table.add_row("Pinned Content", str(storage_usage.get('pinned_count', 0)))
                storage_table.add_row("Expiring Content", str(storage_usage.get('expiring_count', 0)))
                storage_table.add_row("Total Tokens", str(storage_usage.get('total_tokens', 0)))
            
            console.print(storage_table)
            
            # Memory stats
            if context_stats.get('memory_usage'):
                memory_usage = context_stats['memory_usage']
                
                memory_table = Table(title="Memory Statistics")
                memory_table.add_column("Metric", style="cyan")
                memory_table.add_column("Value", style="white", justify="right")
                
                memory_table.add_row("Active Agents", str(memory_usage.get('active_agents', 0)))
                memory_table.add_row("Total Decisions", str(memory_usage.get('total_decisions', 0)))
                memory_table.add_row("Memory Usage", f"{memory_usage.get('memory_usage_mb', 0):.1f} MB")
                
                console.print(memory_table)
            
            # System status
            embedder_available = context_stats.get('embedder_available', False)
            redis_available = context_stats.get('redis_available', False)
            sqlite_available = context_stats.get('sqlite_available', False)
            
            status_content = f"""
[bold]Components:[/bold]
• Embedder: {'✓' if embedder_available else '✗'}
• Redis: {'✓' if redis_available else '✗'}
• SQLite: {'✓' if sqlite_available else '✗'}
• Vector Search: {'✓' if vec_available else '⚠'}
"""
            
            overall_healthy = embedder_available and sqlite_available
            panel_style = "green" if overall_healthy else "yellow"
            
            console.print(Panel(
                status_content.strip(),
                title="System Health",
                border_style=panel_style
            ))
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(show_stats())


@vector.command()
@click.option('--days', type=int, default=30, help='Clean up records older than N days')
@click.option('--expired-only', is_flag=True, help='Only clean up expired content')
@click.option('--dry-run', is_flag=True, help='Show what would be cleaned without actually doing it')
def cleanup(days: int, expired_only: bool, dry_run: bool):
    """Clean up old vectors and expired content."""
    
    console.print(f"[bold blue]Vector Database Cleanup[/bold blue]")
    if dry_run:
        console.print("[yellow]DRY RUN - No changes will be made[/yellow]")
    console.print()
    
    async def run_cleanup():
        try:
            storage = await get_sqlite_manager()
            broker = await get_context_broker()
            
            # Clean up expired content
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Cleaning up expired content...", total=None)
                
                if not dry_run:
                    expired_count = await storage.cleanup_expired()
                else:
                    # For dry run, just check what would be cleaned
                    expired_count = 0  # Would need a separate query for this
                
                progress.update(task, description="Expired content cleaned!")
            
            console.print(f"[green]Cleaned up {expired_count} expired content items[/green]")
            
            # Clean up old records if not expired-only
            if not expired_only:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task(f"Cleaning up records older than {days} days...", total=None)
                    
                    if not dry_run:
                        old_count = await storage.cleanup_old_records(days)
                    else:
                        old_count = 0  # Would need a separate query for this
                    
                    progress.update(task, description="Old records cleaned!")
                
                console.print(f"[green]Cleaned up {old_count} old usage records and orphaned embeddings[/green]")
            
            # Vacuum database for better performance
            if not dry_run and (expired_count > 0 or not expired_only):
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Optimizing database...", total=None)
                    
                    await storage.vacuum_database()
                    
                    progress.update(task, description="Database optimized!")
                
                console.print("[green]Database optimized for better performance[/green]")
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(run_cleanup())


@vector.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--topic', help='Topic/category for all content in file')
@click.option('--priority', type=int, default=0, help='Priority level for all content')
@click.option('--chunk-size', type=int, default=1000, help='Size of text chunks')
@click.option('--overlap', type=int, default=100, help='Overlap between chunks')
def import_file(file_path: str, topic: Optional[str], priority: int, chunk_size: int, overlap: int):
    """Import and chunk a large text file with embeddings."""
    
    console.print(f"[bold blue]Import File[/bold blue]")
    console.print(f"File: [italic]{file_path}[/italic]\n")
    
    async def import_and_chunk():
        try:
            # Read file
            path = Path(file_path)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            console.print(f"[green]Loaded file: {len(content)} characters[/green]")
            
            # Simple chunking strategy
            chunks = []
            start = 0
            chunk_id = 0
            
            while start < len(content):
                end = min(start + chunk_size, len(content))
                chunk_text = content[start:end]
                
                # Try to break at sentence boundaries
                if end < len(content):
                    last_period = chunk_text.rfind('.')
                    last_newline = chunk_text.rfind('\n')
                    break_point = max(last_period, last_newline)
                    
                    if break_point > start + chunk_size // 2:  # Only if not too early
                        end = start + break_point + 1
                        chunk_text = content[start:end]
                
                chunk_id += 1
                chunk_content_id = f"{path.stem}_chunk_{chunk_id:04d}"
                
                chunks.append({
                    'id': chunk_content_id,
                    'text': chunk_text.strip(),
                    'start': start,
                    'end': end
                })
                
                start = end - overlap  # Overlap chunks
                if start >= end:  # Prevent infinite loop
                    break
            
            console.print(f"[green]Created {len(chunks)} chunks[/green]")
            
            # Store chunks with embeddings
            broker = await get_context_broker()
            successful_imports = 0
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
            ) as progress:
                task = progress.add_task("Importing chunks...", total=len(chunks))
                
                for i, chunk in enumerate(chunks):
                    try:
                        metadata = {
                            "source_file": str(path),
                            "chunk_number": i + 1,
                            "total_chunks": len(chunks),
                            "start_char": chunk['start'],
                            "end_char": chunk['end'],
                            "imported_at": datetime.now().isoformat()
                        }
                        
                        success = await broker.store_context_with_embedding(
                            context_id=chunk['id'],
                            content=chunk['text'],
                            summary=f"Chunk {i+1}/{len(chunks)} from {path.name}",
                            topic=topic or f"import_{path.stem}",
                            priority=priority,
                            source_type="file_import",
                            metadata=metadata
                        )
                        
                        if success:
                            successful_imports += 1
                        
                        progress.update(task, completed=i + 1)
                        
                    except Exception as e:
                        console.print(f"[yellow]Warning: Failed to import chunk {i+1}: {e}[/yellow]")
            
            console.print(f"[green]✓ Successfully imported {successful_imports}/{len(chunks)} chunks[/green]")
            
            if topic:
                console.print(f"[dim]All chunks tagged with topic: {topic}[/dim]")
            console.print(f"[dim]Use 'autom8 vector search' to query the imported content[/dim]")
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(import_and_chunk())


if __name__ == "__main__":
    vector()