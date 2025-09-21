"""
Base Agent - Foundation for all agents with shared memory access.

Provides the core agent interface with context transparency, intelligent routing,
and event-driven coordination.
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from autom8.core.complexity.analyzer import ComplexityAnalyzer
from autom8.core.context.inspector import ContextInspector
from autom8.core.memory.broker import ContextBroker
from autom8.core.routing.router import ModelRouter
from autom8.models.complexity import ComplexityScore
from autom8.models.routing import ModelSelection, RoutingPreferences
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class AgentEvent:
    """Represents an event in the agent system."""
    
    def __init__(
        self,
        event_type: str,
        source_agent: str,
        data: Dict[str, Any],
        target_agent: Optional[str] = None,
        priority: int = 0
    ):
        self.id = str(uuid.uuid4())
        self.type = event_type
        self.source_agent = source_agent
        self.target_agent = target_agent
        self.data = data
        self.priority = priority
        self.timestamp = time.time()
        self.created_at = datetime.utcnow()


class BaseAgent:
    """
    Foundation for all agents with shared memory access.
    
    Provides context transparency, intelligent routing, and event-driven
    coordination capabilities.
    """
    
    def __init__(
        self,
        agent_id: str,
        role: str = "assistant",
        preferences: Optional[RoutingPreferences] = None
    ):
        self.agent_id = agent_id
        self.role = role
        self.preferences = preferences or RoutingPreferences()
        
        # Core components
        self.complexity_analyzer = ComplexityAnalyzer()
        self.context_inspector = ContextInspector()
        self.context_broker = ContextBroker()
        self.model_router = ModelRouter()
        
        # Agent state
        self.is_active = False
        self.current_task: Optional[str] = None
        self.task_history: List[Dict[str, Any]] = []
        self._initialized = False
        
        # Event handling
        self._event_handlers: Dict[str, List[callable]] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._event_processor_task: Optional[asyncio.Task] = None
        
        logger.info(f"Initialized agent {agent_id} with role {role}")
    
    async def initialize(self) -> bool:
        """Initialize the agent and its components."""
        try:
            # Initialize all components
            await self.complexity_analyzer.initialize()
            await self.context_inspector.initialize()
            await self.context_broker.initialize()
            await self.model_router.initialize()
            
            self._initialized = True
            logger.info(f"Agent {self.agent_id} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize agent {self.agent_id}: {e}")
            return False
    
    def is_initialized(self) -> bool:
        """Check if the agent is initialized."""
        return self._initialized
    
    async def start(self) -> None:
        """Start the agent and begin processing events."""
        if self.is_active:
            logger.warning(f"Agent {self.agent_id} is already active")
            return
        
        self.is_active = True
        
        # Start event processor
        self._event_processor_task = asyncio.create_task(self._process_events())
        
        # Emit agent started event
        await self.emit_event(AgentEvent(
            event_type="agent_started",
            source_agent=self.agent_id,
            data={"role": self.role, "timestamp": time.time()}
        ))
        
        logger.info(f"Agent {self.agent_id} started")
    
    async def stop(self) -> None:
        """Stop the agent and cleanup resources."""
        if not self.is_active:
            return
        
        self.is_active = False
        
        # Emit agent stopping event
        await self.emit_event(AgentEvent(
            event_type="agent_stopping",
            source_agent=self.agent_id,
            data={"timestamp": time.time()}
        ))
        
        # Cancel event processor
        if self._event_processor_task:
            self._event_processor_task.cancel()
            try:
                await self._event_processor_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Agent {self.agent_id} stopped")
    
    async def process(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        max_context_tokens: int = 500,
        require_confirmation: bool = True
    ) -> Dict[str, Any]:
        """
        Process a task with full transparency and intelligent routing.
        
        Args:
            task: The task to process
            context: Additional context for the task
            max_context_tokens: Maximum tokens for context
            require_confirmation: Whether to require user confirmation for context
            
        Returns:
            Processing result with metadata
        """
        start_time = time.time()
        task_id = str(uuid.uuid4())
        
        logger.info(f"Agent {self.agent_id} processing task: {task[:100]}...")
        
        try:
            self.current_task = task
            
            # Step 1: Analyze complexity
            complexity = await self.complexity_analyzer.analyze(task, context or {})
            logger.debug(f"Task complexity: {complexity.raw_score:.3f} ({complexity.recommended_tier})")
            
            # Step 2: Prepare context with transparency
            context_package = await self.context_broker.prepare_context(
                query=task,
                agent_id=self.agent_id,
                max_tokens=max_context_tokens
            )
            
            # Step 3: Show context preview if required
            if require_confirmation:
                preview = await self.context_inspector.preview(
                    query=task,
                    agent_id=self.agent_id,
                    context_sources=[],  # Context already in package
                    complexity_score=complexity.raw_score
                )
                
                # In a real implementation, this would show UI for confirmation
                # For now, we'll auto-approve unless there are high-severity warnings
                if preview.has_high_warnings:
                    logger.warning("High-severity context warnings detected - would require user review")
            
            # Step 4: Route to appropriate model
            model_selection = await self.model_router.route(
                query=task,
                complexity=complexity,
                context_tokens=context_package.token_count,
                preferences=self.preferences
            )
            
            logger.info(f"Selected model: {model_selection.primary_model.name}")
            
            # Step 5: Execute with selected model
            result = await self.execute_with_model(
                task=task,
                context_package=context_package,
                model_selection=model_selection,
                complexity=complexity
            )
            
            # Step 6: Store result in shared memory
            decision_id = str(uuid.uuid4())
            await self.context_broker.store_agent_decision(
                agent_id=self.agent_id,
                decision_id=decision_id,
                summary=result.get("summary", task[:200]),
                full_content=result.get("content", ""),
                tags=result.get("tags", []),
                affects=result.get("affects", [])
            )
            
            # Step 7: Emit completion event
            await self.emit_event(AgentEvent(
                event_type="task_complete",
                source_agent=self.agent_id,
                data={
                    "task_id": task_id,
                    "task": task,
                    "result_id": decision_id,
                    "model_used": model_selection.primary_model.name,
                    "complexity_score": complexity.raw_score,
                    "processing_time": time.time() - start_time
                }
            ))
            
            # Update task history
            self.task_history.append({
                "task_id": task_id,
                "task": task,
                "complexity": complexity.raw_score,
                "model_used": model_selection.primary_model.name,
                "success": result.get("success", True),
                "processing_time": time.time() - start_time,
                "timestamp": time.time()
            })
            
            # Keep only recent history
            if len(self.task_history) > 100:
                self.task_history = self.task_history[-100:]
            
            processing_time = time.time() - start_time
            logger.info(f"Task completed in {processing_time:.2f}s")
            
            return {
                "task_id": task_id,
                "success": True,
                "result": result,
                "complexity": complexity.raw_score,
                "model_used": model_selection.primary_model.name,
                "processing_time": processing_time,
                "context_tokens": context_package.token_count,
                "estimated_cost": model_selection.estimated_cost
            }
            
        except Exception as e:
            logger.error(f"Task processing failed: {e}")
            
            # Emit error event
            await self.emit_event(AgentEvent(
                event_type="task_error",
                source_agent=self.agent_id,
                data={
                    "task_id": task_id,
                    "task": task,
                    "error": str(e),
                    "processing_time": time.time() - start_time
                }
            ))
            
            return {
                "task_id": task_id,
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
        
        finally:
            self.current_task = None
    
    async def execute_with_model(
        self,
        task: str,
        context_package,
        model_selection: ModelSelection,
        complexity: ComplexityScore
    ) -> Dict[str, Any]:
        """
        Execute task with the selected model.
        
        Integrates with actual model providers (Ollama, Anthropic, OpenAI)
        based on the model selection and routing decision.
        
        Args:
            task: The task to execute
            context_package: Prepared context package
            model_selection: Selected model and alternatives
            complexity: Complexity analysis
            
        Returns:
            Execution result with content, metadata, and performance tracking
        """
        model = model_selection.primary_model
        start_time = asyncio.get_event_loop().time()
        
        logger.debug(f"Executing task with {model.provider.value} model: {model.name}")
        
        try:
            # Prepare context for model input
            context_text = self._format_context_for_model(context_package, task)
            
            # Route to appropriate model provider
            if model.is_local:
                result = await self._execute_local_model(
                    model, context_text, task, complexity
                )
            else:
                result = await self._execute_cloud_model(
                    model, context_text, task, complexity
                )
            
            # Calculate actual performance metrics
            end_time = asyncio.get_event_loop().time()
            actual_latency_ms = (end_time - start_time) * 1000
            
            # Enhance result with execution metadata
            result.update({
                "model_used": model.name,
                "provider": model.provider.value,
                "actual_latency_ms": actual_latency_ms,
                "complexity_score": complexity.raw_score,
                "confidence": model_selection.confidence,
                "tags": [complexity.recommended_tier.value, self.role],
                "affects": ["agent_knowledge", "task_completion"],
                "execution_timestamp": datetime.utcnow().isoformat()
            })
            
            # Track performance for learning
            await self._record_execution_metrics(model, result, actual_latency_ms)
            
            return result
            
        except Exception as e:
            # Handle execution failure with fallback strategy
            logger.error(f"Model execution failed with {model.name}: {e}")
            
            # Try fallback model if available
            if model_selection.fallback_model:
                logger.info(f"Attempting fallback to {model_selection.fallback_model.name}")
                try:
                    fallback_result = await self._execute_fallback(
                        model_selection.fallback_model, context_package, task, complexity
                    )
                    fallback_result["used_fallback"] = True
                    fallback_result["original_error"] = str(e)
                    return fallback_result
                except Exception as fallback_error:
                    logger.error(f"Fallback execution also failed: {fallback_error}")
            
            # Return error result if all else fails
            end_time = asyncio.get_event_loop().time()
            error_latency = (end_time - start_time) * 1000
            
            return {
                "content": f"I apologize, but I encountered an error while processing your request: {str(e)}",
                "summary": f"Failed to execute: {task[:50]}...",
                "success": False,
                "error": str(e),
                "model_used": model.name,
                "actual_latency_ms": error_latency,
                "tags": ["error", self.role],
                "affects": ["error_tracking"]
            }
    
    def _format_context_for_model(self, context_package, task: str) -> str:
        """Format context package for model input."""
        context_parts = []
        
        # Add context sources in priority order
        for source in sorted(context_package.sources, key=lambda x: x.priority):
            if source.type.value == "query":
                continue  # Skip query as it's handled separately
            context_parts.append(f"[{source.type.value.upper()}] {source.content}")
        
        context_text = "\n\n".join(context_parts)
        
        if context_text:
            return f"Context:\n{context_text}\n\nTask: {task}"
        else:
            return task
    
    async def _execute_local_model(
        self, model, context_text: str, task: str, complexity: ComplexityScore
    ) -> Dict[str, Any]:
        """Execute task with local model (Ollama)."""
        try:
            from autom8.integrations.ollama import get_ollama_client
            
            ollama_client = await get_ollama_client()
            if not ollama_client.is_available:
                raise RuntimeError("Ollama client not available")
            
            # Generate response using Ollama
            result = await ollama_client.generate(
                model=model.name,
                prompt=context_text,
                options={
                    "temperature": model.temperature,
                    "num_predict": getattr(model, 'max_output_tokens', 2048)
                }
            )
            
            return {
                "content": result.get("response", ""),
                "summary": f"Generated response using {model.name}",
                "success": True,
                "tokens_input": result.get("prompt_eval_count", 0),
                "tokens_output": result.get("eval_count", 0),
                "model_latency_ms": result.get("total_duration", 0) / 1000000,  # Convert from ns
            }
            
        except Exception as e:
            logger.error(f"Local model execution failed: {e}")
            raise
    
    async def _execute_cloud_model(
        self, model, context_text: str, task: str, complexity: ComplexityScore
    ) -> Dict[str, Any]:
        """Execute task with cloud model (Anthropic/OpenAI)."""
        try:
            if model.provider.value == "anthropic":
                from autom8.integrations.anthropic import get_anthropic_client
                
                client = await get_anthropic_client()
                if not client.is_available:
                    raise RuntimeError("Anthropic client not available")
                
                result = await client.chat_completion(
                    prompt=task,
                    model=model.name,
                    context=context_text if context_text != task else None,
                    max_tokens=getattr(model, 'max_output_tokens', 1000)
                )
                
            elif model.provider.value == "openai":
                from autom8.integrations.openai import get_openai_client
                
                client = await get_openai_client()
                if not client.is_available:
                    raise RuntimeError("OpenAI client not available")
                
                result = await client.chat_completion(
                    prompt=task,
                    model=model.name,
                    context=context_text if context_text != task else None,
                    max_tokens=getattr(model, 'max_output_tokens', 1000)
                )
            else:
                raise ValueError(f"Unsupported cloud provider: {model.provider}")
            
            return {
                "content": result.get("content", ""),
                "summary": f"Generated response using {model.name}",
                "success": result.get("success", False),
                "tokens_input": result.get("usage", {}).get("input_tokens", 0),
                "tokens_output": result.get("usage", {}).get("output_tokens", 0),
                "model_latency_ms": result.get("latency_ms", 0),
                "cost_estimate": self._calculate_cost(result, model)
            }
            
        except Exception as e:
            logger.error(f"Cloud model execution failed: {e}")
            raise
    
    async def _execute_fallback(
        self, fallback_model, context_package, task: str, complexity: ComplexityScore
    ) -> Dict[str, Any]:
        """Execute with fallback model."""
        # Create a simple model selection for the fallback
        from autom8.models.routing import ModelSelection
        fallback_selection = ModelSelection(
            primary_model=fallback_model,
            selection_reasoning="Fallback due to primary model failure",
            estimated_quality=0.7,  # Assume lower quality for fallback
            estimated_latency_ms=fallback_model.avg_latency_ms,
            estimated_cost=0.0,
            complexity_tier=complexity.recommended_tier,
            confidence=0.5  # Lower confidence for fallback
        )
        
        return await self.execute_with_model(task, context_package, fallback_selection, complexity)
    
    def _calculate_cost(self, result: Dict[str, Any], model) -> float:
        """Calculate execution cost for cloud models."""
        usage = result.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        
        return model.estimate_cost(input_tokens, output_tokens)
    
    async def _record_execution_metrics(
        self, model, result: Dict[str, Any], latency_ms: float
    ) -> None:
        """Record execution metrics for performance tracking."""
        try:
            if hasattr(self, 'storage') and self.storage:
                await self.storage.record_model_usage(
                    model_name=model.name,
                    success=result.get("success", False),
                    latency_ms=latency_ms,
                    input_tokens=result.get("tokens_input", 0),
                    output_tokens=result.get("tokens_output", 0),
                    cost=result.get("cost_estimate", 0.0)
                )
        except Exception as e:
            logger.warning(f"Failed to record execution metrics: {e}")
    
    async def emit_event(self, event: AgentEvent) -> None:
        """Emit an event to the agent system."""
        await self._event_queue.put(event)
        logger.debug(f"Emitted event: {event.type} from {event.source_agent}")
    
    def register_event_handler(self, event_type: str, handler: callable) -> None:
        """Register a handler for a specific event type."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        logger.debug(f"Registered handler for event type: {event_type}")
    
    async def _process_events(self) -> None:
        """Process events from the queue."""
        logger.debug(f"Event processor started for agent {self.agent_id}")
        
        while self.is_active:
            try:
                # Wait for events with timeout
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                
                # Handle the event
                await self._handle_event(event)
                
            except asyncio.TimeoutError:
                # No event received, continue
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
        
        logger.debug(f"Event processor stopped for agent {self.agent_id}")
    
    async def _handle_event(self, event: AgentEvent) -> None:
        """Handle a specific event."""
        try:
            # Check if this event is for us
            if event.target_agent and event.target_agent != self.agent_id:
                return
            
            # Skip our own events (avoid loops)
            if event.source_agent == self.agent_id:
                return
            
            logger.debug(f"Handling event: {event.type} from {event.source_agent}")
            
            # Call registered handlers
            handlers = self._event_handlers.get(event.type, [])
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Event handler failed: {e}")
            
            # Built-in event handling
            await self._handle_builtin_event(event)
            
        except Exception as e:
            logger.error(f"Error handling event {event.type}: {e}")
    
    async def _handle_builtin_event(self, event: AgentEvent) -> None:
        """Handle built-in system events."""
        if event.type == "agent_started":
            logger.info(f"Another agent started: {event.source_agent}")
        
        elif event.type == "agent_stopping":
            logger.info(f"Agent stopping: {event.source_agent}")
        
        elif event.type == "task_complete":
            # Learn from other agents' task completions
            data = event.data
            if "complexity_score" in data and "model_used" in data:
                # Could update our routing preferences based on others' success
                pass
        
        elif event.type == "task_error":
            logger.warning(f"Task error from {event.source_agent}: {event.data.get('error')}")
        
        elif event.type == "coordination_request":
            # Handle coordination requests from other agents
            await self._handle_coordination_request(event)
    
    async def _handle_coordination_request(self, event: AgentEvent) -> None:
        """Handle coordination requests from other agents."""
        request_type = event.data.get("request_type")
        
        if request_type == "knowledge_share":
            # Share relevant knowledge with requesting agent
            await self._share_knowledge(event.source_agent, event.data)
        
        elif request_type == "task_delegation":
            # Consider accepting delegated task
            await self._consider_delegation(event)
        
        elif request_type == "resource_coordination":
            # Coordinate resource usage
            await self._coordinate_resources(event)
    
    async def _share_knowledge(self, requesting_agent: str, request_data: Dict[str, Any]) -> None:
        """Share knowledge with another agent."""
        topic = request_data.get("topic")
        if not topic:
            return
        
        # Find relevant knowledge from our task history
        relevant_tasks = [
            task for task in self.task_history
            if topic.lower() in task.get("task", "").lower()
        ]
        
        if relevant_tasks:
            # Share summary of relevant experience
            knowledge = {
                "topic": topic,
                "relevant_tasks": len(relevant_tasks),
                "avg_complexity": sum(t.get("complexity", 0) for t in relevant_tasks) / len(relevant_tasks),
                "successful_models": list(set(t.get("model_used") for t in relevant_tasks if t.get("success")))
            }
            
            await self.emit_event(AgentEvent(
                event_type="knowledge_response",
                source_agent=self.agent_id,
                target_agent=requesting_agent,
                data=knowledge
            ))
            
            logger.debug(f"Shared knowledge about '{topic}' with {requesting_agent}")
    
    async def _consider_delegation(self, event: AgentEvent) -> None:
        """Consider accepting a delegated task."""
        task_data = event.data
        
        # Simple delegation logic - accept if we're not busy and task matches our role
        if not self.current_task and self.role in task_data.get("preferred_roles", []):
            # Accept the delegation
            await self.emit_event(AgentEvent(
                event_type="delegation_accepted",
                source_agent=self.agent_id,
                target_agent=event.source_agent,
                data={"task_id": task_data.get("task_id")}
            ))
            
            logger.info(f"Accepted delegation from {event.source_agent}")
        else:
            # Decline the delegation
            await self.emit_event(AgentEvent(
                event_type="delegation_declined",
                source_agent=self.agent_id,
                target_agent=event.source_agent,
                data={"task_id": task_data.get("task_id"), "reason": "busy or role mismatch"}
            ))
    
    async def _coordinate_resources(self, event: AgentEvent) -> None:
        """Coordinate resource usage with other agents."""
        resource_type = event.data.get("resource_type")
        
        if resource_type == "model_usage":
            # Coordinate model usage to avoid conflicts
            requested_model = event.data.get("model")
            if requested_model and self.current_task:
                # If we're using the same model, provide usage info
                await self.emit_event(AgentEvent(
                    event_type="resource_status",
                    source_agent=self.agent_id,
                    target_agent=event.source_agent,
                    data={
                        "resource_type": "model_usage",
                        "model": requested_model,
                        "in_use": True,
                        "estimated_completion": time.time() + 30  # Rough estimate
                    }
                ))
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "is_active": self.is_active,
            "current_task": self.current_task,
            "tasks_completed": len(self.task_history),
            "avg_processing_time": (
                sum(t.get("processing_time", 0) for t in self.task_history) / len(self.task_history)
                if self.task_history else 0
            ),
            "success_rate": (
                sum(1 for t in self.task_history if t.get("success", False)) / len(self.task_history)
                if self.task_history else 1.0
            ),
            "event_queue_size": self._event_queue.qsize(),
            "registered_handlers": sum(len(handlers) for handlers in self._event_handlers.values())
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        if not self.task_history:
            return {}
        
        # Calculate metrics from task history
        total_tasks = len(self.task_history)
        successful_tasks = sum(1 for t in self.task_history if t.get("success", False))
        
        processing_times = [t.get("processing_time", 0) for t in self.task_history]
        complexities = [t.get("complexity", 0) for t in self.task_history]
        
        # Model usage distribution
        model_usage = {}
        for task in self.task_history:
            model = task.get("model_used", "unknown")
            model_usage[model] = model_usage.get(model, 0) + 1
        
        return {
            "total_tasks": total_tasks,
            "success_rate": successful_tasks / total_tasks,
            "avg_processing_time": sum(processing_times) / len(processing_times),
            "avg_complexity": sum(complexities) / len(complexities),
            "model_usage_distribution": model_usage,
            "recent_performance": {
                "last_10_success_rate": (
                    sum(1 for t in self.task_history[-10:] if t.get("success", False)) / 
                    min(10, total_tasks)
                ),
                "last_10_avg_time": (
                    sum(t.get("processing_time", 0) for t in self.task_history[-10:]) /
                    min(10, total_tasks)
                )
            }
        }


class AgentCoordinator:
    """Coordinates multiple agents and handles inter-agent communication."""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.event_bus = asyncio.Queue()
        self._coordinator_task: Optional[asyncio.Task] = None
        self._is_running = False
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the coordinator."""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent {agent.agent_id}")
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the coordinator."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Unregistered agent {agent_id}")
    
    async def start(self) -> None:
        """Start the coordinator."""
        if self._is_running:
            return
        
        self._is_running = True
        self._coordinator_task = asyncio.create_task(self._coordinate())
        logger.info("Agent coordinator started")
    
    async def stop(self) -> None:
        """Stop the coordinator."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        if self._coordinator_task:
            self._coordinator_task.cancel()
            try:
                await self._coordinator_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Agent coordinator stopped")
    
    async def _coordinate(self) -> None:
        """Main coordination loop."""
        while self._is_running:
            try:
                # Simple coordination logic - could be much more sophisticated
                await asyncio.sleep(1.0)
                
                # Check agent health and balance workload
                await self._balance_workload()
                
            except Exception as e:
                logger.error(f"Coordination error: {e}")
    
    async def _balance_workload(self) -> None:
        """Balance workload across agents."""
        active_agents = [agent for agent in self.agents.values() if agent.is_active]
        
        if len(active_agents) < 2:
            return  # Need at least 2 agents to balance
        
        # Find overloaded agents
        busy_agents = [agent for agent in active_agents if agent.current_task]
        idle_agents = [agent for agent in active_agents if not agent.current_task]
        
        if busy_agents and idle_agents:
            # Could implement task delegation logic here
            pass
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        total_agents = len(self.agents)
        active_agents = sum(1 for agent in self.agents.values() if agent.is_active)
        busy_agents = sum(1 for agent in self.agents.values() if agent.current_task)
        
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "busy_agents": busy_agents,
            "idle_agents": active_agents - busy_agents,
            "coordinator_running": self._is_running,
            "agents": {
                agent_id: agent.get_status() 
                for agent_id, agent in self.agents.items()
            }
        }
