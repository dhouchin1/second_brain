"""
Health Monitor Service

Monitors the health and availability of models, services, and system components.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from autom8.integrations.ollama import get_ollama_client
from autom8.storage.sqlite.manager import get_sqlite_manager
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class ComponentHealth:
    """Represents the health status of a system component."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_healthy = False
        self.last_check = datetime.utcnow()
        self.response_time_ms = 0
        self.error_message: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
        
    def update(self, is_healthy: bool, response_time_ms: float = 0, error: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Update health status."""
        self.is_healthy = is_healthy
        self.last_check = datetime.utcnow()
        self.response_time_ms = response_time_ms
        self.error_message = error
        if metadata:
            self.metadata.update(metadata)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'is_healthy': self.is_healthy,
            'last_check': self.last_check.isoformat(),
            'response_time_ms': self.response_time_ms,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


class HealthMonitor:
    """
    Monitors the health and availability of system components.
    
    Tracks Ollama models, database connections, and other critical services.
    """
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval  # seconds
        self.components: Dict[str, ComponentHealth] = {}
        self._monitor_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        # Initialize component health trackers
        self.components['ollama'] = ComponentHealth('ollama')
        self.components['sqlite'] = ComponentHealth('sqlite')
        
    async def start(self):
        """Start the health monitoring service."""
        if self._is_running:
            return
        
        self._is_running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        
        # Do initial health check
        await self.check_all_components()
        
        logger.info("Health monitor started")
    
    async def stop(self):
        """Stop the health monitoring service."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health monitor stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._is_running:
            try:
                await self.check_all_components()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(min(self.check_interval, 30))  # Shorter retry interval on error
    
    async def check_all_components(self):
        """Check health of all components."""
        logger.debug("Checking health of all components")
        
        # Check components in parallel
        tasks = [
            self.check_ollama_health(),
            self.check_sqlite_health(),
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def check_ollama_health(self):
        """Check Ollama service health."""
        start_time = time.time()
        
        try:
            ollama_client = await get_ollama_client()
            health_info = await ollama_client.health_check()
            
            response_time = (time.time() - start_time) * 1000
            is_healthy = health_info['service_available'] and len(health_info['errors']) == 0
            
            # Update component health
            self.components['ollama'].update(
                is_healthy=is_healthy,
                response_time_ms=response_time,
                error='; '.join(health_info['errors']) if health_info['errors'] else None,
                metadata={
                    'models_count': health_info['models_count'],
                    'models': health_info['models'],
                    'version': health_info.get('version'),
                    'service_available': health_info['service_available']
                }
            )
            
            # Update individual model health
            for model_info in health_info['models']:
                model_name = model_info['name']
                component_name = f"model:{model_name}"
                
                if component_name not in self.components:
                    self.components[component_name] = ComponentHealth(component_name)
                
                self.components[component_name].update(
                    is_healthy=is_healthy,  # Model is healthy if Ollama is healthy
                    response_time_ms=response_time,
                    metadata=model_info
                )
            
            if is_healthy:
                logger.debug(f"Ollama health check passed: {health_info['models_count']} models available")
            else:
                logger.warning(f"Ollama health check failed: {health_info['errors']}")
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.components['ollama'].update(
                is_healthy=False,
                response_time_ms=response_time,
                error=str(e)
            )
            logger.error(f"Ollama health check error: {e}")
    
    async def check_sqlite_health(self):
        """Check SQLite database health."""
        start_time = time.time()
        
        try:
            sqlite_manager = await get_sqlite_manager()
            
            # Simple health check - try to query the database
            stats = await sqlite_manager.get_usage_stats()
            
            response_time = (time.time() - start_time) * 1000
            is_healthy = True  # If we got stats, database is working
            
            self.components['sqlite'].update(
                is_healthy=is_healthy,
                response_time_ms=response_time,
                metadata={
                    'stats': stats,
                    'database_path': str(sqlite_manager.db_path)
                }
            )
            
            logger.debug("SQLite health check passed")
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.components['sqlite'].update(
                is_healthy=False,
                response_time_ms=response_time,
                error=str(e)
            )
            logger.error(f"SQLite health check error: {e}")
    
    async def check_component_health(self, component_name: str) -> ComponentHealth:
        """Check health of a specific component."""
        if component_name == 'ollama':
            await self.check_ollama_health()
        elif component_name == 'sqlite':
            await self.check_sqlite_health()
        elif component_name.startswith('model:'):
            # Model health is checked as part of Ollama health
            await self.check_ollama_health()
        
        return self.components.get(component_name, ComponentHealth(component_name))
    
    def get_component_health(self, component_name: str) -> Optional[ComponentHealth]:
        """Get cached health status of a component."""
        return self.components.get(component_name)
    
    def get_all_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all components."""
        return {name: component.to_dict() for name, component in self.components.items()}
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        total_components = len(self.components)
        healthy_components = sum(1 for c in self.components.values() if c.is_healthy)
        
        # Categorize components
        core_components = ['ollama', 'sqlite']
        core_healthy = sum(1 for name in core_components if self.components.get(name, {}).is_healthy)
        
        model_components = [name for name in self.components.keys() if name.startswith('model:')]
        models_healthy = sum(1 for name in model_components if self.components[name].is_healthy)
        
        # Determine overall health
        overall_healthy = core_healthy == len(core_components)
        
        # Calculate average response time for healthy components
        healthy_response_times = [
            c.response_time_ms for c in self.components.values() 
            if c.is_healthy and c.response_time_ms > 0
        ]
        avg_response_time = sum(healthy_response_times) / len(healthy_response_times) if healthy_response_times else 0
        
        return {
            'overall_healthy': overall_healthy,
            'total_components': total_components,
            'healthy_components': healthy_components,
            'health_percentage': (healthy_components / total_components * 100) if total_components > 0 else 0,
            'core_services': {
                'total': len(core_components),
                'healthy': core_healthy,
                'all_healthy': core_healthy == len(core_components)
            },
            'models': {
                'total': len(model_components),
                'healthy': models_healthy,
                'available_models': [
                    name.replace('model:', '') for name in model_components 
                    if self.components[name].is_healthy
                ]
            },
            'performance': {
                'avg_response_time_ms': avg_response_time
            },
            'last_check': max(
                (c.last_check for c in self.components.values()),
                default=datetime.utcnow()
            ).isoformat(),
            'issues': [
                {
                    'component': name,
                    'error': component.error_message
                }
                for name, component in self.components.items()
                if not component.is_healthy and component.error_message
            ]
        }
    
    def get_model_availability(self) -> Dict[str, bool]:
        """Get availability status of all models."""
        model_availability = {}
        
        for name, component in self.components.items():
            if name.startswith('model:'):
                model_name = name.replace('model:', '')
                model_availability[model_name] = component.is_healthy
        
        return model_availability
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        component_name = f"model:{model_name}"
        component = self.components.get(component_name)
        return component.is_healthy if component else False
    
    def get_health_alerts(self, severity: str = 'warning') -> List[Dict[str, Any]]:
        """
        Get health alerts for unhealthy components.
        
        Args:
            severity: Filter by severity level ('info', 'warning', 'error')
        """
        alerts = []
        
        for name, component in self.components.items():
            if not component.is_healthy:
                # Determine severity
                if name in ['ollama', 'sqlite']:
                    alert_severity = 'error'  # Core services are critical
                elif name.startswith('model:'):
                    alert_severity = 'warning'  # Individual models are less critical
                else:
                    alert_severity = 'info'
                
                # Filter by requested severity
                severity_levels = {'info': 1, 'warning': 2, 'error': 3}
                if severity_levels.get(alert_severity, 0) >= severity_levels.get(severity, 0):
                    alerts.append({
                        'component': name,
                        'severity': alert_severity,
                        'message': component.error_message or f"{name} is unhealthy",
                        'last_check': component.last_check.isoformat(),
                        'response_time_ms': component.response_time_ms
                    })
        
        return alerts
    
    async def repair_component(self, component_name: str) -> bool:
        """
        Attempt to repair a failed component.
        
        Args:
            component_name: Name of component to repair
            
        Returns:
            True if repair was successful
        """
        logger.info(f"Attempting to repair component: {component_name}")
        
        try:
            if component_name == 'ollama':
                # For Ollama, we can't really "repair" it, but we can re-check
                await self.check_ollama_health()
                return self.components['ollama'].is_healthy
            
            elif component_name == 'sqlite':
                # For SQLite, try to reinitialize
                sqlite_manager = await get_sqlite_manager()
                await sqlite_manager.initialize()
                await self.check_sqlite_health()
                return self.components['sqlite'].is_healthy
            
            elif component_name.startswith('model:'):
                # For models, check if they're still available in Ollama
                await self.check_ollama_health()
                return self.components.get(component_name, ComponentHealth(component_name)).is_healthy
            
            else:
                logger.warning(f"Don't know how to repair component: {component_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to repair component {component_name}: {e}")
            return False


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


async def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    global _health_monitor
    
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    
    return _health_monitor


async def start_health_monitoring():
    """Start global health monitoring."""
    monitor = await get_health_monitor()
    await monitor.start()


async def stop_health_monitoring():
    """Stop global health monitoring."""
    global _health_monitor
    
    if _health_monitor:
        await _health_monitor.stop()
        _health_monitor = None
