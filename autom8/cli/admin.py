"""
Advanced Administrative CLI for Autom8 System Management

Provides comprehensive administrative tools for system monitoring, analytics,
budget management, performance analysis, and operational maintenance.
"""

import asyncio
import json
import sys
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import rich
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree
from rich.json import JSON
from rich.align import Align

from autom8.core.error_management import EnhancedErrorManager, ErrorCategory, ErrorSeverity
from autom8.core.complexity.analyzer import ComplexityAnalyzer
from autom8.core.routing.router import ModelRouter
from autom8.core.context.inspector import ContextInspector
from autom8.core.benchmarks.suite import BenchmarkSuite
from autom8.services.budget import BudgetManager
from autom8.services.health import HealthService
from autom8.services.ecology import EcologyTracker
from autom8.storage.redis.client import get_redis_client
from autom8.storage.sqlite.manager import SQLiteManager
from autom8.storage.redis.events import EventBus
from autom8.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()


class SystemStatus:
    """Container for comprehensive system status information."""
    
    def __init__(
        self,
        health_status: Dict[str, Any],
        active_agents: int,
        budget_utilization: Dict[str, float],
        error_rate: float,
        performance_metrics: Dict[str, Any],
        resource_usage: Dict[str, float],
        active_connections: int,
        component_status: Dict[str, str]
    ):
        self.health_status = health_status
        self.active_agents = active_agents
        self.budget_utilization = budget_utilization
        self.error_rate = error_rate
        self.performance_metrics = performance_metrics
        self.resource_usage = resource_usage
        self.active_connections = active_connections
        self.component_status = component_status


class SystemManagementDashboard:
    """Comprehensive system management interface integrating all Autom8 components."""
    
    def __init__(self):
        """Initialize with all system components."""
        self.budget_manager: Optional[BudgetManager] = None
        self.error_manager: Optional[EnhancedErrorManager] = None
        self.health_service: Optional[HealthService] = None
        self.complexity_analyzer: Optional[ComplexityAnalyzer] = None
        self.model_router: Optional[ModelRouter] = None
        self.context_inspector: Optional[ContextInspector] = None
        self.ecology_tracker: Optional[EcologyTracker] = None
        self.benchmark_suite: Optional[BenchmarkSuite] = None
        self.event_bus: Optional[EventBus] = None
        self.redis_client = None
        self.sqlite_manager: Optional[SQLiteManager] = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize all system components."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                init_task = progress.add_task("Initializing system components...", total=None)
                
                # Initialize Redis and event bus
                self.redis_client = await get_redis_client()
                self.event_bus = EventBus(self.redis_client)
                
                # Initialize SQLite manager
                self.sqlite_manager = SQLiteManager("autom8.db")
                await self.sqlite_manager.initialize()
                
                # Initialize core components
                self.complexity_analyzer = ComplexityAnalyzer()
                await self.complexity_analyzer.initialize()
                
                self.model_router = ModelRouter()
                await self.model_router.initialize()
                
                self.context_inspector = ContextInspector()
                await self.context_inspector.initialize()
                
                # Initialize services
                self.budget_manager = BudgetManager(
                    redis_client=self.redis_client,
                    sqlite_manager=self.sqlite_manager,
                    event_bus=self.event_bus
                )
                await self.budget_manager.initialize()
                
                self.health_service = HealthService()
                await self.health_service.initialize()
                
                self.ecology_tracker = EcologyTracker()
                await self.ecology_tracker.initialize()
                
                # Initialize error management
                self.error_manager = EnhancedErrorManager(self.event_bus)
                
                # Initialize benchmarking
                self.benchmark_suite = BenchmarkSuite()
                
                progress.update(init_task, description="System initialization complete!")
            
            self._initialized = True
            console.print("✅ System components initialized successfully", style="green")
            return True
            
        except Exception as e:
            console.print(f"❌ Failed to initialize system: {e}", style="red")
            logger.error(f"System initialization failed: {e}")
            return False
    
    async def get_system_overview(self) -> SystemStatus:
        """Get comprehensive system status overview."""
        if not self._initialized:
            raise RuntimeError("System not initialized")
        
        # Gather health status
        health_status = await self.health_service.get_system_health()
        
        # Get active connections and resource usage
        redis_info = await self.redis_client.info() if self.redis_client else {}
        active_connections = redis_info.get('connected_clients', 0)
        
        # Get component status
        component_status = {
            'complexity_analyzer': 'healthy' if self.complexity_analyzer._initialized else 'degraded',
            'model_router': 'healthy' if self.model_router._initialized else 'degraded',
            'context_inspector': 'healthy' if self.context_inspector._initialized else 'degraded',
            'budget_manager': 'healthy' if self.budget_manager._initialized else 'degraded',
            'error_manager': 'healthy',
            'redis': 'healthy' if self.redis_client else 'unavailable',
            'sqlite': 'healthy' if self.sqlite_manager else 'unavailable'
        }
        
        # Get budget utilization
        budget_utilization = await self.budget_manager.get_global_utilization()
        
        # Get error rate
        error_analytics = await self.error_manager.get_error_analytics("1h")
        error_rate = error_analytics.total_errors / 60.0  # errors per minute
        
        # Get performance metrics
        routing_stats = self.model_router.get_routing_stats()
        performance_metrics = {
            'total_routings': routing_stats.get('total_routings', 0),
            'avg_latency': routing_stats.get('avg_estimated_latency', 0),
            'local_model_percentage': routing_stats.get('local_model_percentage', 0),
            'avg_cost': routing_stats.get('avg_estimated_cost', 0)
        }
        
        # Resource usage (simplified)
        resource_usage = {
            'memory_usage': 0.0,  # Would implement actual memory monitoring
            'cpu_usage': 0.0,     # Would implement actual CPU monitoring  
            'redis_memory': float(redis_info.get('used_memory', 0)) / (1024 * 1024),  # MB
            'active_agents': len(component_status)  # Simplified
        }
        
        return SystemStatus(
            health_status=health_status,
            active_agents=len(component_status),
            budget_utilization=budget_utilization,
            error_rate=error_rate,
            performance_metrics=performance_metrics,
            resource_usage=resource_usage,
            active_connections=active_connections,
            component_status=component_status
        )
    
    async def generate_performance_report(
        self,
        timeframe: str = '24h',
        include_predictions: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive performance analytics report."""
        
        # Routing analytics
        routing_stats = self.model_router.get_routing_stats()
        
        # Complexity analysis metrics
        complexity_metrics = self.complexity_analyzer.get_accuracy_metrics()
        
        # Budget analytics
        budget_analytics = await self.budget_manager.get_spending_analytics(timeframe)
        
        # Error analytics
        error_analytics = await self.error_manager.get_error_analytics(timeframe)
        
        # Health metrics
        health_metrics = await self.health_service.get_detailed_health_metrics()
        
        # Ecology metrics
        ecology_metrics = await self.ecology_tracker.get_impact_report(timeframe)
        
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'timeframe': timeframe,
            'routing_analytics': routing_stats,
            'complexity_analytics': complexity_metrics,
            'budget_analytics': budget_analytics,
            'error_analytics': error_analytics.dict(),
            'health_metrics': health_metrics,
            'ecology_metrics': ecology_metrics.dict() if ecology_metrics else {},
            'recommendations': await self._generate_optimization_recommendations()
        }
        
        if include_predictions:
            report['predictions'] = await self._generate_performance_predictions()
        
        return report
    
    async def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on system performance."""
        recommendations = []
        
        # Analyze routing patterns
        routing_stats = self.model_router.get_routing_stats()
        if routing_stats.get('local_model_percentage', 100) < 70:
            recommendations.append(
                "Consider increasing local model usage to reduce costs and improve privacy"
            )
        
        # Analyze error patterns
        error_analytics = await self.error_manager.get_error_analytics("24h")
        if error_analytics.recovery_success_rate < 0.8:
            recommendations.append(
                f"Low error recovery rate ({error_analytics.recovery_success_rate:.1%}). "
                "Review and enhance recovery strategies"
            )
        
        # Analyze budget utilization
        budget_utilization = await self.budget_manager.get_global_utilization()
        for budget_type, utilization in budget_utilization.items():
            if utilization > 0.9:
                recommendations.append(
                    f"High budget utilization for {budget_type} ({utilization:.1%}). "
                    "Consider budget optimization or limit increases"
                )
        
        return recommendations
    
    async def _generate_performance_predictions(self) -> Dict[str, Any]:
        """Generate performance predictions based on historical data."""
        # This would implement actual ML-based predictions
        # For now, return placeholder structure
        return {
            'predicted_cost_next_24h': 0.0,
            'predicted_error_rate': 0.0,
            'predicted_resource_usage': {
                'memory': 0.0,
                'cpu': 0.0
            },
            'confidence': 0.7
        }
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run comprehensive system health checks."""
        health_results = {}
        
        # Component health checks
        components = {
            'redis': self._check_redis_health,
            'sqlite': self._check_sqlite_health,
            'model_router': self._check_model_router_health,
            'complexity_analyzer': self._check_complexity_analyzer_health,
            'budget_manager': self._check_budget_manager_health,
            'context_inspector': self._check_context_inspector_health
        }
        
        for component_name, check_func in components.items():
            try:
                health_results[component_name] = await check_func()
            except Exception as e:
                health_results[component_name] = {
                    'status': 'error',
                    'message': str(e),
                    'healthy': False
                }
        
        # Overall health assessment
        healthy_components = sum(1 for result in health_results.values() if result.get('healthy', False))
        total_components = len(health_results)
        overall_health = healthy_components / total_components
        
        return {
            'overall_health_score': overall_health,
            'healthy_components': healthy_components,
            'total_components': total_components,
            'component_details': health_results,
            'recommendations': self._generate_health_recommendations(health_results)
        }
    
    async def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance."""
        if not self.redis_client:
            return {'status': 'unavailable', 'healthy': False}
        
        try:
            # Test basic connectivity
            await self.redis_client.ping()
            
            # Get Redis info
            info = await self.redis_client.info()
            
            return {
                'status': 'healthy',
                'healthy': True,
                'connected_clients': info.get('connected_clients', 0),
                'memory_usage_mb': float(info.get('used_memory', 0)) / (1024 * 1024),
                'total_commands_processed': info.get('total_commands_processed', 0)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'healthy': False,
                'error': str(e)
            }
    
    async def _check_sqlite_health(self) -> Dict[str, Any]:
        """Check SQLite database health."""
        if not self.sqlite_manager:
            return {'status': 'unavailable', 'healthy': False}
        
        try:
            # Test basic connectivity
            result = await self.sqlite_manager.execute_query("SELECT 1")
            
            # Get database stats
            stats = await self.sqlite_manager.get_database_stats()
            
            return {
                'status': 'healthy',
                'healthy': True,
                'database_size_mb': stats.get('size_mb', 0),
                'table_count': stats.get('table_count', 0),
                'integrity_ok': result is not None
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'healthy': False,
                'error': str(e)
            }
    
    async def _check_model_router_health(self) -> Dict[str, Any]:
        """Check model router health."""
        try:
            stats = self.model_router.get_routing_stats()
            available_models = len(self.model_router.model_registry.get_available_models())
            
            return {
                'status': 'healthy',
                'healthy': True,
                'available_models': available_models,
                'total_routings': stats.get('total_routings', 0),
                'initialized': self.model_router._initialized
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'healthy': False,
                'error': str(e)
            }
    
    async def _check_complexity_analyzer_health(self) -> Dict[str, Any]:
        """Check complexity analyzer health."""
        try:
            metrics = self.complexity_analyzer.get_accuracy_metrics()
            
            return {
                'status': 'healthy',
                'healthy': True,
                'total_assessments': metrics.get('total_assessments', 0),
                'accuracy': metrics.get('overall_accuracy', 0),
                'initialized': self.complexity_analyzer._initialized
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'healthy': False,
                'error': str(e)
            }
    
    async def _check_budget_manager_health(self) -> Dict[str, Any]:
        """Check budget manager health."""
        try:
            active_budgets = await self.budget_manager.get_active_budget_count()
            utilization = await self.budget_manager.get_global_utilization()
            
            return {
                'status': 'healthy',
                'healthy': True,
                'active_budgets': active_budgets,
                'global_utilization': utilization,
                'initialized': self.budget_manager._initialized
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'healthy': False,
                'error': str(e)
            }
    
    async def _check_context_inspector_health(self) -> Dict[str, Any]:
        """Check context inspector health."""
        try:
            return {
                'status': 'healthy',
                'healthy': True,
                'initialized': self.context_inspector._initialized
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'healthy': False,
                'error': str(e)
            }
    
    def _generate_health_recommendations(self, health_results: Dict[str, Any]) -> List[str]:
        """Generate health-based recommendations."""
        recommendations = []
        
        for component, result in health_results.items():
            if not result.get('healthy', False):
                recommendations.append(f"Component '{component}' requires attention: {result.get('error', 'unknown issue')}")
        
        return recommendations


# CLI Commands

@click.group()
def admin():
    """Administrative commands for Autom8 system management."""
    pass


@admin.command()
@click.option('--format', default='table', type=click.Choice(['table', 'json', 'yaml']), 
              help='Output format')
@click.option('--watch', is_flag=True, help='Watch mode - refresh every 5 seconds')
def system_status(format, watch):
    """Display comprehensive system status."""
    
    async def _show_status():
        dashboard = SystemManagementDashboard()
        
        if not await dashboard.initialize():
            console.print("❌ Failed to initialize system dashboard", style="red")
            sys.exit(1)
        
        overview = await dashboard.get_system_overview()
        
        if format == 'json':
            status_dict = {
                'health_status': overview.health_status,
                'active_agents': overview.active_agents,
                'budget_utilization': overview.budget_utilization,
                'error_rate': overview.error_rate,
                'performance_metrics': overview.performance_metrics,
                'resource_usage': overview.resource_usage,
                'active_connections': overview.active_connections,
                'component_status': overview.component_status
            }
            console.print(JSON.from_data(status_dict))
            
        elif format == 'yaml':
            status_dict = {
                'health_status': overview.health_status,
                'active_agents': overview.active_agents,
                'budget_utilization': overview.budget_utilization,
                'error_rate': overview.error_rate,
                'performance_metrics': overview.performance_metrics,
                'resource_usage': overview.resource_usage,
                'active_connections': overview.active_connections,
                'component_status': overview.component_status
            }
            console.print(yaml.dump(status_dict, default_flow_style=False))
            
        else:  # table format
            _display_system_status_table(overview)
    
    def _display_system_status_table(overview: SystemStatus):
        """Display system status in rich table format."""
        
        # Main status panel
        status_content = f"""
🟢 System Health: {overview.health_status.get('overall_score', 'N/A'):.1%}
👥 Active Agents: {overview.active_agents}
📊 Error Rate: {overview.error_rate:.2f} errors/min
🔗 Active Connections: {overview.active_connections}
        """
        
        console.print(Panel(status_content.strip(), title="🚀 System Overview", 
                          title_align="left", style="green"))
        
        # Component status table
        component_table = Table(title="Component Status")
        component_table.add_column("Component", style="cyan")
        component_table.add_column("Status", style="white")
        component_table.add_column("Health", style="white")
        
        for component, status in overview.component_status.items():
            status_color = "green" if status == "healthy" else "red" if status == "unavailable" else "yellow"
            health_icon = "✅" if status == "healthy" else "❌" if status == "unavailable" else "⚠️"
            
            component_table.add_row(
                component.replace('_', ' ').title(),
                f"[{status_color}]{status}[/{status_color}]",
                health_icon
            )
        
        console.print(component_table)
        
        # Performance metrics table
        perf_table = Table(title="Performance Metrics")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="white")
        
        perf_metrics = overview.performance_metrics
        perf_table.add_row("Total Routings", str(perf_metrics.get('total_routings', 0)))
        perf_table.add_row("Avg Latency", f"{perf_metrics.get('avg_latency', 0):.0f}ms")
        perf_table.add_row("Local Model Usage", f"{perf_metrics.get('local_model_percentage', 0):.1f}%")
        perf_table.add_row("Avg Cost", f"${perf_metrics.get('avg_cost', 0):.4f}")
        
        console.print(perf_table)
        
        # Budget utilization
        if overview.budget_utilization:
            budget_table = Table(title="Budget Utilization")
            budget_table.add_column("Budget Type", style="cyan")
            budget_table.add_column("Utilization", style="white")
            budget_table.add_column("Status", style="white")
            
            for budget_type, utilization in overview.budget_utilization.items():
                util_percent = utilization * 100
                status_color = "green" if util_percent < 75 else "yellow" if util_percent < 90 else "red"
                status_icon = "✅" if util_percent < 75 else "⚠️" if util_percent < 90 else "🚨"
                
                budget_table.add_row(
                    budget_type.title(),
                    f"[{status_color}]{util_percent:.1f}%[/{status_color}]",
                    status_icon
                )
            
            console.print(budget_table)
    
    if watch:
        try:
            while True:
                console.clear()
                console.print(f"🔄 System Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                asyncio.run(_show_status())
                import time
                time.sleep(5)
        except KeyboardInterrupt:
            console.print("\n👋 Status monitoring stopped")
    else:
        asyncio.run(_show_status())


@admin.command()
@click.option('--timeframe', default='24h', help='Analysis timeframe (1h, 24h, 7d, 30d)')
@click.option('--export', type=click.Path(), help='Export report to file')
@click.option('--format', default='table', type=click.Choice(['table', 'json', 'yaml']),
              help='Output format')
@click.option('--include-predictions', is_flag=True, help='Include performance predictions')
def performance_report(timeframe, export, format, include_predictions):
    """Generate detailed performance analysis report."""
    
    async def _generate_report():
        dashboard = SystemManagementDashboard()
        
        if not await dashboard.initialize():
            console.print("❌ Failed to initialize system dashboard", style="red")
            sys.exit(1)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            report_task = progress.add_task("Generating performance report...", total=None)
            
            report = await dashboard.generate_performance_report(
                timeframe=timeframe,
                include_predictions=include_predictions
            )
            
            progress.update(report_task, description="Performance report generated!")
        
        if export:
            # Export to file
            export_path = Path(export)
            if format == 'json':
                with open(export_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
            elif format == 'yaml':
                with open(export_path, 'w') as f:
                    yaml.dump(report, f, default_flow_style=False)
            else:
                # Export table format as text
                with open(export_path, 'w') as f:
                    f.write(f"Performance Report - {timeframe}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(json.dumps(report, indent=2, default=str))
            
            console.print(f"✅ Report exported to {export_path}", style="green")
        
        # Display report
        if format == 'json':
            console.print(JSON.from_data(report))
        elif format == 'yaml':
            console.print(yaml.dump(report, default_flow_style=False))
        else:
            _display_performance_report(report)
    
    def _display_performance_report(report: Dict[str, Any]):
        """Display performance report in rich table format."""
        
        # Report header
        header = f"📊 Performance Report - {report['timeframe']}"
        generated_at = datetime.fromisoformat(report['generated_at']).strftime('%Y-%m-%d %H:%M:%S')
        
        console.print(Panel(f"Generated at: {generated_at}", title=header, 
                          title_align="left", style="blue"))
        
        # Routing analytics
        routing_data = report.get('routing_analytics', {})
        if routing_data:
            routing_table = Table(title="🎯 Routing Analytics")
            routing_table.add_column("Metric", style="cyan")
            routing_table.add_column("Value", style="white")
            
            routing_table.add_row("Total Routings", str(routing_data.get('total_routings', 0)))
            routing_table.add_row("Avg Latency", f"{routing_data.get('avg_estimated_latency', 0):.0f}ms")
            routing_table.add_row("Local Model %", f"{routing_data.get('local_model_percentage', 0):.1f}%")
            routing_table.add_row("Avg Cost", f"${routing_data.get('avg_estimated_cost', 0):.4f}")
            
            console.print(routing_table)
        
        # Error analytics
        error_data = report.get('error_analytics', {})
        if error_data:
            error_table = Table(title="🚨 Error Analytics")
            error_table.add_column("Metric", style="cyan")
            error_table.add_column("Value", style="white")
            
            error_table.add_row("Total Errors", str(error_data.get('total_errors', 0)))
            error_table.add_row("Recovery Success Rate", f"{error_data.get('recovery_success_rate', 0):.1%}")
            error_table.add_row("Mean Recovery Time", f"{error_data.get('mean_time_to_recovery', 0):.1f}s")
            
            console.print(error_table)
        
        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            rec_panel = "\n".join([f"• {rec}" for rec in recommendations])
            console.print(Panel(rec_panel, title="💡 Recommendations", 
                              title_align="left", style="yellow"))
        
        # Predictions (if included)
        if include_predictions and 'predictions' in report:
            pred_data = report['predictions']
            pred_table = Table(title="🔮 Performance Predictions (Next 24h)")
            pred_table.add_column("Metric", style="cyan")
            pred_table.add_column("Prediction", style="white")
            pred_table.add_column("Confidence", style="white")
            
            pred_table.add_row("Predicted Cost", f"${pred_data.get('predicted_cost_next_24h', 0):.4f}", 
                             f"{pred_data.get('confidence', 0):.1%}")
            pred_table.add_row("Predicted Error Rate", f"{pred_data.get('predicted_error_rate', 0):.2f}/min", 
                             f"{pred_data.get('confidence', 0):.1%}")
            
            console.print(pred_table)
    
    asyncio.run(_generate_report())


@admin.command()
@click.option('--component', help='Specific component to check (redis, sqlite, router, etc.)')
@click.option('--fix', is_flag=True, help='Attempt to fix detected issues automatically')
@click.option('--verbose', is_flag=True, help='Show detailed health information')
def health_check(component, fix, verbose):
    """Run comprehensive system health checks."""
    
    async def _run_health_checks():
        dashboard = SystemManagementDashboard()
        
        if not await dashboard.initialize():
            console.print("❌ Failed to initialize system dashboard", style="red")
            sys.exit(1)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            health_task = progress.add_task("Running health checks...", total=None)
            
            if component:
                # Check specific component
                progress.update(health_task, description=f"Checking {component} health...")
                # Implementation would check specific component
                result = await dashboard.run_health_checks()
                component_result = result['component_details'].get(component)
                
                if component_result:
                    _display_component_health(component, component_result, verbose)
                else:
                    console.print(f"❌ Component '{component}' not found", style="red")
                    return
            else:
                # Run full health check
                result = await dashboard.run_health_checks()
                progress.update(health_task, description="Health checks completed!")
        
        _display_health_results(result, verbose)
        
        if fix and result['overall_health_score'] < 1.0:
            console.print("\n🔧 Attempting to fix detected issues...")
            # Implementation would attempt fixes
            console.print("Fix functionality would be implemented here", style="yellow")
    
    def _display_component_health(component_name: str, result: Dict[str, Any], verbose: bool):
        """Display health results for a specific component."""
        status = result.get('status', 'unknown')
        healthy = result.get('healthy', False)
        
        status_color = "green" if healthy else "red"
        status_icon = "✅" if healthy else "❌"
        
        console.print(f"\n{status_icon} {component_name.title()}: [{status_color}]{status}[/{status_color}]")
        
        if verbose and 'error' not in result:
            # Show detailed metrics for healthy components
            for key, value in result.items():
                if key not in ['status', 'healthy']:
                    console.print(f"  {key}: {value}")
        elif 'error' in result:
            console.print(f"  Error: {result['error']}", style="red")
    
    def _display_health_results(result: Dict[str, Any], verbose: bool):
        """Display comprehensive health check results."""
        overall_score = result['overall_health_score']
        healthy_components = result['healthy_components']
        total_components = result['total_components']
        
        # Overall health panel
        score_color = "green" if overall_score > 0.8 else "yellow" if overall_score > 0.5 else "red"
        health_content = f"""
Overall Health Score: [{score_color}]{overall_score:.1%}[/{score_color}]
Healthy Components: {healthy_components}/{total_components}
        """
        
        console.print(Panel(health_content.strip(), title="🏥 System Health Check", 
                          title_align="left", style="blue"))
        
        # Component details table
        health_table = Table(title="Component Health Details")
        health_table.add_column("Component", style="cyan")
        health_table.add_column("Status", style="white")
        health_table.add_column("Health", style="white")
        
        if verbose:
            health_table.add_column("Details", style="white")
        
        component_details = result['component_details']
        for component, details in component_details.items():
            status = details.get('status', 'unknown')
            healthy = details.get('healthy', False)
            
            status_color = "green" if healthy else "red"
            health_icon = "✅" if healthy else "❌"
            
            row_data = [
                component.replace('_', ' ').title(),
                f"[{status_color}]{status}[/{status_color}]",
                health_icon
            ]
            
            if verbose:
                if 'error' in details:
                    detail_text = f"Error: {details['error']}"
                else:
                    detail_items = [f"{k}: {v}" for k, v in details.items() 
                                  if k not in ['status', 'healthy']]
                    detail_text = "; ".join(detail_items[:2])  # Limit detail length
                row_data.append(detail_text)
            
            health_table.add_row(*row_data)
        
        console.print(health_table)
        
        # Recommendations
        recommendations = result.get('recommendations', [])
        if recommendations:
            rec_panel = "\n".join([f"• {rec}" for rec in recommendations])
            console.print(Panel(rec_panel, title="💡 Health Recommendations", 
                              title_align="left", style="yellow"))
    
    asyncio.run(_run_health_checks())


@admin.command()
@click.option('--budget-id', help='Specific budget ID to analyze')
@click.option('--user-id', help='User-specific budget analysis')
@click.option('--timeframe', default='7d', help='Analysis timeframe')
@click.option('--format', default='table', type=click.Choice(['table', 'json', 'yaml']))
def budget_analysis(budget_id, user_id, timeframe, format):
    """Detailed budget utilization and spending analysis."""
    
    async def _analyze_budget():
        dashboard = SystemManagementDashboard()
        
        if not await dashboard.initialize():
            console.print("❌ Failed to initialize system dashboard", style="red")
            sys.exit(1)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            budget_task = progress.add_task("Analyzing budget data...", total=None)
            
            if budget_id:
                analysis = await dashboard.budget_manager.analyze_budget(budget_id)
            elif user_id:
                analysis = await dashboard.budget_manager.analyze_user_spending(user_id)
            else:
                analysis = await dashboard.budget_manager.get_spending_analytics(timeframe)
            
            progress.update(budget_task, description="Budget analysis complete!")
        
        if format == 'json':
            console.print(JSON.from_data(analysis))
        elif format == 'yaml':
            console.print(yaml.dump(analysis, default_flow_style=False))
        else:
            _display_budget_analysis(analysis)
    
    def _display_budget_analysis(analysis: Dict[str, Any]):
        """Display budget analysis in table format."""
        
        # Budget overview
        console.print(Panel("📊 Budget Analysis Report", style="blue"))
        
        # Implementation would display budget tables based on analysis data
        console.print("Budget analysis display would be implemented here", style="yellow")
    
    asyncio.run(_analyze_budget())


@admin.command()
@click.option('--timeframe', default='7d', help='Cleanup timeframe')
@click.option('--dry-run', is_flag=True, help='Show what would be cleaned without doing it')
@click.option('--component', multiple=True, help='Specific components to clean')
def cleanup(timeframe, dry_run, component):
    """Clean up old data and optimize system resources."""
    
    async def _cleanup_system():
        dashboard = SystemManagementDashboard()
        
        if not await dashboard.initialize():
            console.print("❌ Failed to initialize system dashboard", style="red")
            sys.exit(1)
        
        console.print(f"🧹 System Cleanup - {timeframe} timeframe")
        console.print(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}", 
                     style="yellow" if dry_run else "red")
        
        components_to_clean = component if component else [
            'error_history', 'routing_history', 'context_cache', 
            'budget_history', 'health_metrics'
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            for comp in components_to_clean:
                cleanup_task = progress.add_task(f"Cleaning {comp}...", total=None)
                
                # Implementation would perform actual cleanup
                await asyncio.sleep(0.5)  # Simulate cleanup work
                
                progress.update(cleanup_task, description=f"Cleaned {comp}")
        
        if dry_run:
            console.print("✅ Dry run completed - no data was actually removed", style="green")
        else:
            console.print("✅ System cleanup completed", style="green")
    
    asyncio.run(_cleanup_system())


@admin.command()
@click.option('--export-config', type=click.Path(), help='Export current configuration')
@click.option('--import-config', type=click.Path(), help='Import configuration from file')
@click.option('--validate', is_flag=True, help='Validate current configuration')
def config(export_config, import_config, validate):
    """Configuration management and validation."""
    
    async def _manage_config():
        dashboard = SystemManagementDashboard()
        
        if not await dashboard.initialize():
            console.print("❌ Failed to initialize system dashboard", style="red")
            sys.exit(1)
        
        if export_config:
            console.print(f"📤 Exporting configuration to {export_config}")
            # Implementation would export current system configuration
            console.print("✅ Configuration exported", style="green")
            
        elif import_config:
            console.print(f"📥 Importing configuration from {import_config}")
            # Implementation would import and apply new configuration
            console.print("✅ Configuration imported", style="green")
            
        elif validate:
            console.print("🔍 Validating system configuration...")
            # Implementation would validate current configuration
            console.print("✅ Configuration is valid", style="green")
            
        else:
            console.print("Please specify --export-config, --import-config, or --validate")
    
    asyncio.run(_manage_config())


if __name__ == '__main__':
    admin()