"""
FastAPI router for monitoring endpoints.
Provides health checks, metrics, and monitoring dashboard endpoints.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from services.monitoring_service import MonitoringService, get_monitoring_service
from config import settings


# Initialize templates
templates = Jinja2Templates(directory="templates")

# Create router
router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/health")
async def health_check(
    monitoring: MonitoringService = Depends(get_monitoring_service)
) -> JSONResponse:
    """
    Basic liveness check - returns 200 if application is running.
    This endpoint should always respond quickly and not depend on external services.
    """
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "second-brain",
            "version": "3.0.0"
        }
    )


@router.get("/ready")
async def readiness_check(
    monitoring: MonitoringService = Depends(get_monitoring_service)
) -> JSONResponse:
    """
    Readiness check - returns 200 if application is ready to serve traffic.
    Checks all critical dependencies and returns detailed status.
    """
    try:
        health_status = await monitoring.get_health_status()
        
        # Determine HTTP status code based on overall health
        if health_status["status"] == "healthy":
            status_code = 200
        elif health_status["status"] == "degraded":
            status_code = 200  # Still ready, but with warnings
        else:
            status_code = 503  # Service unavailable
        
        return JSONResponse(
            status_code=status_code,
            content=health_status
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        )


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics(
    monitoring: MonitoringService = Depends(get_monitoring_service)
) -> str:
    """
    Export metrics in Prometheus format for external monitoring systems.
    """
    try:
        return monitoring.export_prometheus_metrics()
    except Exception as e:
        # Return basic error metric in Prometheus format
        return f"""# HELP second_brain_metrics_error Error generating metrics
# TYPE second_brain_metrics_error gauge
second_brain_metrics_error{{error="{str(e).replace('"', '\\"')}"}} 1
"""


@router.get("/metrics/json")
async def metrics_json(
    monitoring: MonitoringService = Depends(get_monitoring_service)
) -> Dict[str, Any]:
    """
    Get metrics in JSON format for dashboard consumption.
    """
    try:
        return monitoring.get_metrics_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics: {str(e)}")


@router.get("/status")
async def detailed_status(
    monitoring: MonitoringService = Depends(get_monitoring_service)
) -> Dict[str, Any]:
    """
    Get comprehensive application status including health checks and metrics.
    """
    try:
        # Run health checks and get metrics in parallel
        health_status, metrics_summary = await asyncio.gather(
            monitoring.get_health_status(),
            asyncio.create_task(asyncio.to_thread(monitoring.get_metrics_summary))
        )
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "health": health_status,
            "metrics": metrics_summary,
            "alerts": monitoring.check_alert_conditions()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving status: {str(e)}")


@router.get("/alerts")
async def active_alerts(
    monitoring: MonitoringService = Depends(get_monitoring_service)
) -> Dict[str, Any]:
    """
    Get currently active alerts and alert conditions.
    """
    try:
        alerts = monitoring.check_alert_conditions()
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "active_alerts": alerts,
            "alert_count": len(alerts),
            "thresholds": monitoring.alert_thresholds
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving alerts: {str(e)}")


@router.get("/security/events")
async def security_events(
    monitoring: MonitoringService = Depends(get_monitoring_service),
    limit: int = 50
) -> Dict[str, Any]:
    """
    Get recent security events and monitoring data.
    """
    try:
        security_events = list(monitoring.metrics_collector.security_events)
        # Return most recent events first
        security_events.reverse()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "events": security_events[:limit],
            "total_events": len(security_events),
            "daily_counters": dict(monitoring.metrics_collector.daily_counters)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving security events: {str(e)}")


@router.get("/system")
async def system_metrics(
    monitoring: MonitoringService = Depends(get_monitoring_service)
) -> Dict[str, Any]:
    """
    Get detailed system resource metrics.
    """
    try:
        system_metrics = monitoring.system_monitor.get_system_metrics()
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system": system_metrics.__dict__
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving system metrics: {str(e)}")


@router.get("/performance")
async def performance_metrics(
    monitoring: MonitoringService = Depends(get_monitoring_service)
) -> Dict[str, Any]:
    """
    Get detailed performance metrics and endpoint statistics.
    """
    try:
        collector = monitoring.metrics_collector
        
        # Calculate endpoint statistics
        endpoint_stats = {}
        for endpoint, stats in collector.endpoint_stats.items():
            if stats["count"] > 0:
                endpoint_stats[endpoint] = {
                    "request_count": stats["count"],
                    "total_time_seconds": stats["total_time"],
                    "average_time_ms": (stats["total_time"] / stats["count"]) * 1000,
                    "error_count": stats["errors"],
                    "error_rate": (stats["errors"] / stats["count"]) * 100 if stats["count"] > 0 else 0
                }
        
        # Get recent error counts
        error_summary = dict(collector.error_counts)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint_statistics": endpoint_stats,
            "error_summary": error_summary,
            "total_requests": sum(stats["count"] for stats in collector.endpoint_stats.values()),
            "total_errors": sum(collector.error_counts.values())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving performance metrics: {str(e)}")


@router.get("/dashboard")
async def monitoring_dashboard(
    request: Request,
    monitoring: MonitoringService = Depends(get_monitoring_service)
):
    """
    Serve the monitoring dashboard HTML page.
    """
    try:
        # Get initial data for the dashboard
        status_data = await detailed_status(monitoring)
        
        return templates.TemplateResponse(
            "monitoring/system-dashboard.html",
            {
                "request": request,
                "initial_data": status_data,
                "refresh_interval": 5000  # 5 seconds
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dashboard: {str(e)}")


@router.get("/dashboard/health")
async def health_dashboard(
    request: Request,
    monitoring: MonitoringService = Depends(get_monitoring_service)
):
    """
    Serve the health monitoring dashboard HTML page.
    """
    try:
        health_data = await monitoring.get_health_status()
        
        return templates.TemplateResponse(
            "monitoring/health-dashboard.html",
            {
                "request": request,
                "health_data": health_data,
                "refresh_interval": 10000  # 10 seconds
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading health dashboard: {str(e)}")


@router.get("/dashboard/security")
async def security_dashboard(
    request: Request,
    monitoring: MonitoringService = Depends(get_monitoring_service)
):
    """
    Serve the security monitoring dashboard HTML page.
    """
    try:
        security_data = await security_events(monitoring)
        alerts_data = await active_alerts(monitoring)
        
        return templates.TemplateResponse(
            "monitoring/security-dashboard.html",
            {
                "request": request,
                "security_data": security_data,
                "alerts_data": alerts_data,
                "refresh_interval": 15000  # 15 seconds
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading security dashboard: {str(e)}")


# WebSocket endpoint for real-time metrics streaming
@router.websocket("/ws/metrics")
async def metrics_websocket(
    websocket,
    monitoring: MonitoringService = Depends(get_monitoring_service)
):
    """
    WebSocket endpoint for real-time metrics streaming to dashboards.
    """
    await websocket.accept()
    
    try:
        while True:
            # Get current metrics and status
            try:
                metrics = monitoring.get_metrics_summary()
                health = await monitoring.get_health_status()
                alerts = monitoring.check_alert_conditions()
                
                data = {
                    "type": "metrics_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "metrics": metrics,
                    "health": health,
                    "alerts": alerts
                }
                
                await websocket.send_json(data)
                
            except Exception as e:
                error_data = {
                    "type": "error",
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(e)
                }
                await websocket.send_json(error_data)
            
            # Wait 5 seconds before next update
            await asyncio.sleep(5)
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# Add route for configuration
@router.get("/config")
async def monitoring_config() -> Dict[str, Any]:
    """
    Get monitoring configuration and settings.
    """
    return {
        "monitoring_enabled": True,
        "metrics_retention_points": 1000,
        "health_check_timeout": 5.0,
        "alert_thresholds": {
            "error_rate_percent": 5.0,
            "response_time_p95_ms": 2000.0,
            "memory_percent": 80.0,
            "disk_usage_percent": 90.0,
            "cpu_percent": 90.0
        },
        "dashboard_refresh_intervals": {
            "system": 5000,
            "health": 10000,
            "security": 15000
        }
    }