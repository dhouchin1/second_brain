"""
Real-time Monitoring Dashboard

Provides comprehensive real-time monitoring dashboard with live metrics,
tracing visualization, SLO status, and system health insights.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn

from autom8.monitoring.logger import get_observability_logger
from autom8.monitoring.metrics import get_metrics_collector, SLOStatus
from autom8.monitoring.tracing import get_tracing_manager

logger = get_observability_logger(__name__)


class MonitoringDashboard:
    """
    Real-time monitoring dashboard providing comprehensive system observability.
    
    Features:
    - Live metrics visualization
    - SLO status monitoring
    - Distributed tracing insights
    - Performance analytics
    - Health status overview
    """
    
    def __init__(self):
        self.console = Console()
        self.metrics_collector = get_metrics_collector()
        self.tracing_manager = get_tracing_manager()
        self._running = False
        self._dashboard_task: Optional[asyncio.Task] = None
    
    def create_system_status_panel(self) -> Panel:
        """Create system status overview panel."""
        
        health = self.metrics_collector.get_health_status()
        
        status_text = Text()
        
        # Overall status
        status_color = "green" if health["overall_status"] == "healthy" else "yellow" if health["overall_status"] == "degraded" else "red"
        status_text.append("🟢 " if health["overall_status"] == "healthy" else "🟡 " if health["overall_status"] == "degraded" else "🔴 ")
        status_text.append(f"System Status: {health['overall_status'].upper()}\n", style=f"bold {status_color}")
        
        # SLO summary
        status_text.append(f"📊 SLO Status: ", style="bold")
        status_text.append(f"{health['healthy_slos']}/{health['total_slos']} Healthy", style="green")
        
        if health["warning_slos"] > 0:
            status_text.append(f", {health['warning_slos']} Warning", style="yellow")
        
        if health["violated_slos"] > 0:
            status_text.append(f", {health['violated_slos']} Violated", style="red")
        
        status_text.append(f"\n⏰ Last Updated: {datetime.utcnow().strftime('%H:%M:%S')}")
        
        return Panel(
            status_text,
            title="🏥 System Health",
            border_style="green" if health["overall_status"] == "healthy" else "yellow" if health["overall_status"] == "degraded" else "red"
        )
    
    def create_metrics_table(self) -> Table:
        """Create real-time metrics table."""
        
        table = Table(title="📊 Real-time Metrics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Current", style="white", width=15)
        table.add_column("Avg (5m)", style="blue", width=15)
        table.add_column("P95 (5m)", style="yellow", width=15)
        table.add_column("Trend", style="green", width=10)
        
        key_metrics = [
            "requests_total",
            "response_time_ms", 
            "model_inference_time_ms",
            "active_connections",
            "cache_hit_rate"
        ]
        
        for metric_name in key_metrics:
            summary = self.metrics_collector.get_metric_summary(metric_name, duration_minutes=5)
            
            if not summary or summary.get("count", 0) == 0:
                table.add_row(metric_name, "No data", "No data", "No data", "➡️")
                continue
            
            current = summary.get("latest", 0)
            average = summary.get("average", 0)
            p95 = summary.get("p95", 0)
            
            # Format values based on metric type
            if "time" in metric_name:
                current_str = f"{current:.0f}ms"
                avg_str = f"{average:.0f}ms"
                p95_str = f"{p95:.0f}ms"
            elif "rate" in metric_name:
                current_str = f"{current:.1f}%"
                avg_str = f"{average:.1f}%"
                p95_str = "N/A"
            elif metric_name == "requests_total":
                rate = summary.get("rate_per_minute", 0)
                current_str = f"{current:.0f}"
                avg_str = f"{rate:.1f}/min"
                p95_str = "N/A"
            else:
                current_str = f"{current:.0f}"
                avg_str = f"{average:.1f}"
                p95_str = f"{p95:.0f}" if p95 != 0 else "N/A"
            
            # Simple trend indicator (would need historical data for real trends)
            trend = "📈" if current > average else "📉" if current < average * 0.9 else "➡️"
            
            table.add_row(
                metric_name.replace("_", " ").title(),
                current_str,
                avg_str,
                p95_str,
                trend
            )
        
        return table
    
    def create_slo_status_table(self) -> Table:
        """Create SLO status monitoring table."""
        
        table = Table(title="🎯 SLO Status", show_header=True, header_style="bold blue")
        table.add_column("SLO", style="cyan", width=20)
        table.add_column("Target", style="white", width=10)
        table.add_column("Current", style="white", width=12)
        table.add_column("Status", style="white", width=10)
        table.add_column("Health", style="white", width=15)
        
        slo_results = self.metrics_collector.evaluate_all_slos()
        
        for slo_name, (status, percentage) in slo_results.items():
            target = self.metrics_collector.slo_monitor.targets[slo_name]
            
            # Status color and icon
            if status == SLOStatus.HEALTHY:
                status_text = "🟢 Healthy"
                status_style = "green"
            elif status == SLOStatus.WARNING:
                status_text = "🟡 Warning"
                status_style = "yellow"
            elif status == SLOStatus.VIOLATED:
                status_text = "🔴 Violated"
                status_style = "red"
            else:
                status_text = "⚪ Unknown"
                status_style = "dim"
            
            # Health bar
            health_percentage = min(100, max(0, percentage))
            health_bar_length = int((health_percentage / 100) * 10)
            health_bar = "█" * health_bar_length + "░" * (10 - health_bar_length)
            
            table.add_row(
                slo_name.replace("_", " ").title(),
                f"{target.target_percentage:.1f}%",
                f"{percentage:.2f}%",
                status_text,
                health_bar,
                style=status_style if status != SLOStatus.HEALTHY else None
            )
        
        return table
    
    def create_tracing_insights_panel(self) -> Panel:
        """Create distributed tracing insights panel."""
        
        trace_stats = self.tracing_manager.get_trace_statistics(duration_minutes=15)
        
        insights_text = Text()
        insights_text.append("🔍 Distributed Tracing Insights (15m window)\n\n", style="bold cyan")
        
        if trace_stats["total_traces"] == 0:
            insights_text.append("No traces recorded in the last 15 minutes.", style="dim")
        else:
            insights_text.append(f"📈 Total Traces: {trace_stats['total_traces']}\n")
            insights_text.append(f"⚡ Avg Duration: {trace_stats['average_duration_ms']:.0f}ms\n")
            insights_text.append(f"📊 P95 Duration: {trace_stats['p95_duration_ms']:.0f}ms\n")
            
            error_rate = trace_stats['error_rate']
            error_color = "green" if error_rate < 1 else "yellow" if error_rate < 5 else "red"
            insights_text.append(f"❌ Error Rate: {error_rate:.1f}%\n", style=error_color)
            
            # Top operations
            if trace_stats["operations"]:
                insights_text.append("\n🎯 Top Operations:\n", style="bold")
                for op, stats in list(trace_stats["operations"].items())[:3]:
                    insights_text.append(f"  • {op}: {stats['count']} traces")
                    if stats['errors'] > 0:
                        insights_text.append(f" ({stats['errors']} errors)", style="red")
                    insights_text.append(f" - {stats['avg_duration']:.0f}ms avg\n")
        
        return Panel(
            insights_text,
            title="🔍 Distributed Tracing",
            border_style="cyan"
        )
    
    def create_performance_trends_panel(self) -> Panel:
        """Create performance trends analysis panel."""
        
        trends_text = Text()
        trends_text.append("📈 Performance Trends & Alerts\n\n", style="bold yellow")
        
        # Check for performance issues
        response_time_summary = self.metrics_collector.get_metric_summary("response_time_ms", duration_minutes=5)
        
        if response_time_summary.get("count", 0) > 0:
            p95_response_time = response_time_summary.get("p95", 0)
            avg_response_time = response_time_summary.get("average", 0)
            
            if p95_response_time > 2000:
                trends_text.append("⚠️  High P95 response time detected\n", style="red")
                trends_text.append(f"   Current: {p95_response_time:.0f}ms (threshold: 2000ms)\n")
            
            if avg_response_time > 1000:
                trends_text.append("⚠️  High average response time\n", style="yellow")
                trends_text.append(f"   Current: {avg_response_time:.0f}ms\n")
            
            # Model inference performance
            model_time_summary = self.metrics_collector.get_metric_summary("model_inference_time_ms", duration_minutes=5)
            if model_time_summary.get("count", 0) > 0:
                model_p95 = model_time_summary.get("p95", 0)
                if model_p95 > 5000:
                    trends_text.append("⚠️  Slow model inference detected\n", style="red")
                    trends_text.append(f"   P95: {model_p95:.0f}ms (threshold: 5000ms)\n")
            
            # Cache performance
            cache_summary = self.metrics_collector.get_metric_summary("cache_hit_rate", duration_minutes=5)
            if cache_summary.get("count", 0) > 0:
                cache_rate = cache_summary.get("average", 0)
                if cache_rate < 70:
                    trends_text.append("⚠️  Low cache hit rate\n", style="yellow")
                    trends_text.append(f"   Current: {cache_rate:.1f}% (target: >70%)\n")
        
        if len(trends_text.plain) == len("📈 Performance Trends & Alerts\n\n"):
            trends_text.append("✅ All performance metrics within normal ranges", style="green")
        
        return Panel(
            trends_text,
            title="📈 Performance Analysis",
            border_style="yellow"
        )
    
    def create_dashboard_layout(self) -> Layout:
        """Create the complete dashboard layout."""
        
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(name="metrics"),
            Layout(name="slos")
        )
        
        layout["right"].split_column(
            Layout(name="tracing"),
            Layout(name="performance")
        )
        
        # Populate layout
        layout["header"].update(self.create_system_status_panel())
        layout["metrics"].update(self.create_metrics_table())
        layout["slos"].update(self.create_slo_status_table())
        layout["tracing"].update(self.create_tracing_insights_panel())
        layout["performance"].update(self.create_performance_trends_panel())
        
        # Footer
        footer_text = Text()
        footer_text.append("🔧 Autom8 Monitoring Dashboard", style="bold cyan")
        footer_text.append(" | ")
        footer_text.append("Real-time System Observability", style="dim")
        footer_text.append(" | ")
        footer_text.append(f"Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", style="dim")
        
        layout["footer"].update(Panel(footer_text, style="dim"))
        
        return layout
    
    async def start_live_dashboard(self, refresh_interval: float = 2.0):
        """Start the live monitoring dashboard."""
        
        logger.info("Starting live monitoring dashboard", refresh_interval=refresh_interval)
        
        self._running = True
        
        with Live(self.create_dashboard_layout(), refresh_per_second=1/refresh_interval) as live:
            while self._running:
                try:
                    # Update dashboard
                    live.update(self.create_dashboard_layout())
                    await asyncio.sleep(refresh_interval)
                    
                except KeyboardInterrupt:
                    logger.info("Dashboard stopped by user")
                    break
                    
                except Exception as e:
                    logger.error("Error updating dashboard", error=e)
                    await asyncio.sleep(refresh_interval)
        
        self._running = False
    
    def stop_dashboard(self):
        """Stop the live dashboard."""
        self._running = False
        logger.info("Stopping monitoring dashboard")
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get dashboard data as dictionary for API consumption."""
        
        health = self.metrics_collector.get_health_status()
        trace_stats = self.tracing_manager.get_trace_statistics(duration_minutes=15)
        slo_results = self.metrics_collector.evaluate_all_slos()
        
        key_metrics = {}
        for metric_name in ["requests_total", "response_time_ms", "model_inference_time_ms", "active_connections", "cache_hit_rate"]:
            key_metrics[metric_name] = self.metrics_collector.get_metric_summary(metric_name, duration_minutes=5)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system_health": health,
            "key_metrics": key_metrics,
            "slo_status": {name: {"status": status.value, "percentage": pct} for name, (status, pct) in slo_results.items()},
            "tracing_insights": trace_stats,
            "active_traces": len(self.tracing_manager.active_traces),
            "completed_traces": len(self.tracing_manager.completed_traces)
        }
    
    def create_summary_report(self, duration_hours: int = 24) -> str:
        """Create a comprehensive monitoring summary report."""
        
        health = self.metrics_collector.get_health_status()
        trace_stats = self.tracing_manager.get_trace_statistics(duration_minutes=duration_hours * 60)
        
        report = f"""
🔧 AUTOM8 MONITORING SUMMARY REPORT
Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
Time Window: {duration_hours} hours

🏥 SYSTEM HEALTH:
- Overall Status: {health['overall_status'].upper()}
- SLO Compliance: {health['healthy_slos']}/{health['total_slos']} targets meeting SLO
- Warnings: {health['warning_slos']} SLOs in warning state
- Violations: {health['violated_slos']} SLOs violated

📊 KEY METRICS:
"""
        
        key_metrics = ["requests_total", "response_time_ms", "model_inference_time_ms"]
        for metric_name in key_metrics:
            summary = self.metrics_collector.get_metric_summary(metric_name, duration_minutes=duration_hours * 60)
            if summary.get("count", 0) > 0:
                report += f"- {metric_name.replace('_', ' ').title()}: "
                if "time" in metric_name:
                    report += f"avg {summary['average']:.0f}ms, P95 {summary.get('p95', 0):.0f}ms\n"
                else:
                    report += f"{summary['latest']:.0f} current, {summary.get('rate_per_minute', 0):.1f}/min rate\n"
        
        report += f"""
🔍 DISTRIBUTED TRACING:
- Total Traces: {trace_stats['total_traces']}
- Average Duration: {trace_stats['average_duration_ms']:.0f}ms
- P95 Duration: {trace_stats['p95_duration_ms']:.0f}ms
- Error Rate: {trace_stats['error_rate']:.1f}%

🎯 SLO STATUS:
"""
        
        slo_results = self.metrics_collector.evaluate_all_slos()
        for slo_name, (status, percentage) in slo_results.items():
            report += f"- {slo_name.replace('_', ' ').title()}: {status.value.upper()} ({percentage:.2f}%)\n"
        
        return report


# Global dashboard instance
_monitoring_dashboard: Optional[MonitoringDashboard] = None


def get_monitoring_dashboard() -> MonitoringDashboard:
    """Get or create the global monitoring dashboard."""
    global _monitoring_dashboard
    if _monitoring_dashboard is None:
        _monitoring_dashboard = MonitoringDashboard()
    return _monitoring_dashboard