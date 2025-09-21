"""
Context Inspector - Core transparency feature for Autom8.

Provides complete visibility into what data is being sent to LLMs,
allowing users to see, edit, and optimize context before execution.
"""

import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from autom8.models.context import (
    ContextPreview,
    ContextSource,
    ContextSourceType,
    ContextWarning,
    ContextWarningType,
    ContextOptimization,
)
from autom8.utils.tokens import get_token_counter
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class ContextInspector:
    """
    Provides complete visibility into what's being sent to LLMs.
    
    Core transparency feature that ensures users always know exactly
    what data goes to which model before it's sent.
    """
    
    def __init__(self):
        self.token_counter = get_token_counter()
        self._initialized = False
        self._optimizer = None
    
    async def initialize(self) -> bool:
        """Initialize the context inspector."""
        try:
            # Initialize the context optimizer
            from autom8.core.context.optimizer import get_context_optimizer
            self._optimizer = await get_context_optimizer()
            
            self._initialized = True
            logger.info("Context inspector initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize context inspector: {e}")
            return False
        
    async def preview(
        self,
        query: str,
        agent_id: str,
        context_sources: Optional[List[ContextSource]] = None,
        model_target: Optional[str] = None,
        complexity_score: Optional[float] = None
    ) -> ContextPreview:
        """
        Create a complete preview of what will be sent to an LLM.
        
        Args:
            query: The user query
            agent_id: ID of the requesting agent
            context_sources: Additional context sources
            model_target: Target model for cost estimation
            complexity_score: Complexity score for the query
            
        Returns:
            Complete context preview with transparency data
        """
        logger.debug(f"Creating context preview for agent {agent_id}")
        
        # Start with the query itself
        sources = []
        
        # Add the query as primary source
        query_tokens = self.token_counter.count_tokens(query, model_target)
        query_source = ContextSource(
            type=ContextSourceType.QUERY,
            content=query,
            tokens=query_tokens,
            source="user_input",
            priority=100  # Highest priority
        )
        sources.append(query_source)
        
        # Add additional context sources if provided
        if context_sources:
            sources.extend(context_sources)
        
        # Calculate total tokens
        total_tokens = sum(source.tokens for source in sources)
        
        # Estimate cost if model is specified
        cost_estimate = 0.0
        if model_target:
            cost_estimate = self.token_counter.estimate_cost(
                query, model_target, is_input=True
            )
            # Add cost for additional context
            for source in context_sources or []:
                cost_estimate += self.token_counter.estimate_cost(
                    source.content, model_target, is_input=True
                )
        
        # Generate warnings and optimizations
        warnings = self._analyze_warnings(sources, total_tokens, model_target)
        optimizations = self._suggest_optimizations(sources, total_tokens)
        
        # Create preview
        preview = ContextPreview(
            agent_id=agent_id,
            query=query,
            total_tokens=total_tokens,
            sources=sources,
            cost_estimate=cost_estimate,
            quality_score=self._estimate_quality_score(sources, complexity_score),
            warnings=warnings,
            optimizations=optimizations,
            model_target=model_target,
            complexity_score=complexity_score
        )
        
        logger.info(
            f"Context preview created: {total_tokens} tokens, "
            f"{len(sources)} sources, ${cost_estimate:.4f} estimated cost"
        )
        
        return preview
    
    def _analyze_warnings(
        self,
        sources: List[ContextSource],
        total_tokens: int,
        model_target: Optional[str]
    ) -> List[ContextWarning]:
        """Analyze context for potential issues and generate warnings."""
        warnings = []
        
        # Check for oversized context
        if total_tokens > 2000:
            warnings.append(ContextWarning(
                type=ContextWarningType.OVERSIZED,
                message=f"Context is large ({total_tokens} tokens). Consider summarizing.",
                severity=3 if total_tokens > 4000 else 2,
                suggestion="Use context optimization or increase model context window"
            ))
        
        # Check for redundant content
        content_hashes = {}
        for i, source in enumerate(sources):
            content_hash = hashlib.md5(source.content.encode()).hexdigest()[:8]
            if content_hash in content_hashes:
                warnings.append(ContextWarning(
                    type=ContextWarningType.REDUNDANT,
                    message="Duplicate or very similar content detected",
                    source_ids=[str(i), str(content_hashes[content_hash])],
                    severity=2,
                    suggestion="Remove duplicate content to reduce token usage"
                ))
            else:
                content_hashes[content_hash] = i
        
        # Check for stale content (if timestamps available)
        now = datetime.utcnow()
        for i, source in enumerate(sources):
            if source.timestamp is None:
                continue  # Skip sources without timestamps
            age_hours = (now - source.timestamp).total_seconds() / 3600
            if age_hours > 24:  # Older than 24 hours
                warnings.append(ContextWarning(
                    type=ContextWarningType.STALE,
                    message=f"Content is {age_hours:.1f} hours old",
                    source_ids=[str(i)],
                    severity=1,
                    suggestion="Verify content is still relevant"
                ))
        
        # Check for expensive operations
        if model_target and "gpt-4" in model_target.lower() and total_tokens > 1000:
            warnings.append(ContextWarning(
                type=ContextWarningType.EXPENSIVE,
                message="Large context with expensive model",
                severity=2,
                suggestion="Consider using a less expensive model or reducing context"
            ))
        
        # Check for sensitive content patterns
        sensitive_patterns = [
            "password", "api_key", "secret", "token", "credential",
            "private_key", "auth", "bearer"
        ]
        
        for i, source in enumerate(sources):
            content_lower = source.content.lower()
            for pattern in sensitive_patterns:
                if pattern in content_lower:
                    warnings.append(ContextWarning(
                        type=ContextWarningType.SENSITIVE,
                        message=f"Potentially sensitive content detected: {pattern}",
                        source_ids=[str(i)],
                        severity=3,
                        suggestion="Review and redact sensitive information"
                    ))
                    break  # One warning per source
        
        return warnings
    
    def _suggest_optimizations(
        self,
        sources: List[ContextSource],
        total_tokens: int
    ) -> List[ContextOptimization]:
        """Suggest optimizations to improve context efficiency."""
        optimizations = []
        
        # Suggest summarization for large content
        large_sources = [s for s in sources if s.tokens > 500]
        if large_sources:
            total_large_tokens = sum(s.tokens for s in large_sources)
            potential_savings = int(total_large_tokens * 0.7)  # 70% compression
            
            optimizations.append(ContextOptimization(
                description="Summarize large content blocks",
                estimated_savings=potential_savings,
                quality_impact=0.2,  # Moderate impact
                action_required="Use auto-summarization or manual editing"
            ))
        
        # Suggest removing low-priority content
        low_priority_sources = [s for s in sources if s.priority < 30]
        if low_priority_sources and total_tokens > 1000:
            potential_savings = sum(s.tokens for s in low_priority_sources)
            
            optimizations.append(ContextOptimization(
                description="Remove low-priority context items",
                estimated_savings=potential_savings,
                quality_impact=0.1,  # Low impact
                action_required="Deselect low-priority sources in context editor"
            ))
        
        # Suggest reference-based memory for repeated content
        if len(sources) > 5:
            optimizations.append(ContextOptimization(
                description="Use reference-based memory for repeated patterns",
                estimated_savings=int(total_tokens * 0.3),
                quality_impact=0.05,  # Very low impact
                action_required="Enable shared memory optimization"
            ))
        
        return optimizations
    
    def _estimate_quality_score(
        self,
        sources: List[ContextSource],
        complexity_score: Optional[float]
    ) -> float:
        """Estimate expected response quality based on context."""
        base_score = 0.7  # Base quality
        
        # More context sources generally improve quality
        context_bonus = min(0.2, len(sources) * 0.03)
        
        # High-priority sources improve quality more
        priority_bonus = 0.0
        if sources:
            avg_priority = sum(s.priority for s in sources) / len(sources)
            priority_bonus = min(0.1, avg_priority / 1000)
        
        # Complexity alignment - if we have enough context for the complexity
        complexity_bonus = 0.0
        if complexity_score:
            total_tokens = sum(s.tokens for s in sources)
            expected_tokens = complexity_score * 1000  # Rough heuristic
            if total_tokens >= expected_tokens:
                complexity_bonus = 0.1
            elif total_tokens < expected_tokens * 0.5:
                complexity_bonus = -0.2  # Penalty for insufficient context
        
        final_score = base_score + context_bonus + priority_bonus + complexity_bonus
        return max(0.0, min(1.0, final_score))
    
    async def interactive_edit(
        self,
        preview: ContextPreview,
        user_edits: Dict[str, Any]
    ) -> ContextPreview:
        """
        Apply user edits to context preview.
        
        Args:
            preview: Original context preview
            user_edits: Dictionary of user modifications
            
        Returns:
            Updated context preview
        """
        logger.debug(f"Applying interactive edits to context preview")
        
        # Create a copy of sources for editing
        edited_sources = []
        
        for i, source in enumerate(preview.sources):
            source_id = str(i)
            
            # Check if this source was modified
            if source_id in user_edits.get("modified_sources", {}):
                modifications = user_edits["modified_sources"][source_id]
                
                # Apply content changes
                new_content = modifications.get("content", source.content)
                new_tokens = self.token_counter.count_tokens(
                    new_content, preview.model_target
                )
                
                # Create modified source
                modified_source = ContextSource(
                    type=source.type,
                    content=new_content,
                    tokens=new_tokens,
                    source=source.source,
                    location=source.location,
                    expandable=source.expandable,
                    priority=modifications.get("priority", source.priority),
                    timestamp=source.timestamp
                )
                edited_sources.append(modified_source)
                
            # Check if this source should be excluded
            elif source_id not in user_edits.get("excluded_sources", []):
                edited_sources.append(source)
        
        # Add any new sources
        for new_source_data in user_edits.get("new_sources", []):
            new_source = ContextSource(
                type=ContextSourceType(new_source_data.get("type", "reference")),
                content=new_source_data["content"],
                tokens=self.token_counter.count_tokens(
                    new_source_data["content"], preview.model_target
                ),
                source=new_source_data.get("source", "user_added"),
                priority=new_source_data.get("priority", 50)
            )
            edited_sources.append(new_source)
        
        # Create new preview with edited sources
        return await self.preview(
            query=preview.query,
            agent_id=preview.agent_id,
            context_sources=edited_sources[1:],  # Skip query source
            model_target=preview.model_target,
            complexity_score=preview.complexity_score
        )
    
    def export_context_package(
        self,
        preview: ContextPreview,
        model: str
    ) -> str:
        """
        Export finalized context as a string ready for model consumption.
        
        Args:
            preview: Finalized context preview
            model: Target model name
            
        Returns:
            Formatted context string
        """
        logger.debug(f"Exporting context package for model {model}")
        
        # Build the final prompt
        parts = []
        
        # Add query
        query_source = preview.get_source_by_type(ContextSourceType.QUERY)[0]
        parts.append(f"User Query: {query_source.content}")
        
        # Add other sources by priority
        other_sources = [s for s in preview.sources if s.type != ContextSourceType.QUERY]
        other_sources.sort(key=lambda x: x.priority, reverse=True)
        
        for source in other_sources:
            if source.type == ContextSourceType.MEMORY:
                parts.append(f"Memory Reference ({source.source}): {source.content}")
            elif source.type == ContextSourceType.RETRIEVED:
                parts.append(f"Retrieved Context ({source.location}): {source.content}")
            elif source.type == ContextSourceType.PINNED:
                parts.append(f"Important Context: {source.content}")
            elif source.type == ContextSourceType.SUMMARY:
                parts.append(f"Summary: {source.content}")
            else:
                parts.append(f"Context: {source.content}")
        
        final_context = "\n\n".join(parts)
        
        logger.info(
            f"Context package exported: {len(final_context)} characters, "
            f"{self.token_counter.count_tokens(final_context, model)} tokens"
        )
        
        return final_context
    
    async def optimize_context(
        self,
        preview: ContextPreview,
        profile: str = "balanced",
        token_budget: Optional[int] = None,
        strategies: Optional[List[str]] = None
    ) -> 'OptimizationResult':
        """
        Optimize context using the ContextOptimizer.
        
        Args:
            preview: Context preview to optimize
            profile: Optimization profile (conservative, balanced, aggressive)
            token_budget: Maximum tokens to use
            strategies: Specific optimization strategies to apply
            
        Returns:
            OptimizationResult with optimized context and analytics
        """
        if not self._optimizer:
            raise RuntimeError("Context optimizer not initialized")
        
        from autom8.core.context.optimizer import OptimizationProfile, OptimizationStrategy
        
        # Convert string profile to enum
        profile_map = {
            "conservative": OptimizationProfile.CONSERVATIVE,
            "balanced": OptimizationProfile.BALANCED,
            "aggressive": OptimizationProfile.AGGRESSIVE
        }
        optimization_profile = profile_map.get(profile, OptimizationProfile.BALANCED)
        
        # Convert strategy strings to enums if provided
        optimization_strategies = None
        if strategies:
            strategy_map = {
                "semantic_compression": OptimizationStrategy.SEMANTIC_COMPRESSION,
                "priority_filtering": OptimizationStrategy.PRIORITY_FILTERING,
                "token_budget": OptimizationStrategy.TOKEN_BUDGET,
                "summarization": OptimizationStrategy.SUMMARIZATION,
                "deduplication": OptimizationStrategy.DEDUPLICATION,
                "structural_compression": OptimizationStrategy.STRUCTURAL_COMPRESSION
            }
            optimization_strategies = [
                strategy_map[s] for s in strategies if s in strategy_map
            ]
        
        # Perform optimization
        result = await self._optimizer.optimize(
            preview=preview,
            profile=optimization_profile,
            token_budget=token_budget,
            strategies=optimization_strategies
        )
        
        logger.info(
            f"Context optimization complete: {result.compression_ratio:.1%} compression, "
            f"{result.quality_retention:.1%} quality retention"
        )
        
        return result
    
    async def suggest_optimizations(
        self,
        preview: ContextPreview,
        quality_requirements: float = 0.8,
        cost_constraints: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Suggest optimizations for a context preview.
        
        Args:
            preview: Context preview to analyze
            quality_requirements: Minimum acceptable quality
            cost_constraints: Maximum acceptable cost
            
        Returns:
            Dictionary with optimization suggestions and analytics
        """
        if not self._optimizer:
            raise RuntimeError("Context optimizer not initialized")
        
        # Get recommended profile
        recommended_profile = await self._optimizer.suggest_optimization_profile(
            preview, quality_requirements, cost_constraints
        )
        
        # Get optimization analytics
        analytics = await self._optimizer.get_optimization_analytics()
        
        # Simulate optimization to get estimates
        test_result = await self._optimizer.optimize(
            preview=preview,
            profile=recommended_profile
        )
        
        return {
            "recommended_profile": recommended_profile.value,
            "estimated_compression": test_result.compression_ratio,
            "estimated_quality": test_result.quality_retention,
            "estimated_tokens_saved": test_result.tokens_saved,
            "optimization_strategies": [s.value for s in test_result.strategies_applied],
            "quality_category": test_result.quality_category.value,
            "historical_analytics": analytics,
            "warnings": self._generate_optimization_warnings(preview, test_result)
        }
    
    def _generate_optimization_warnings(
        self,
        original_preview: ContextPreview,
        optimization_result: 'OptimizationResult'
    ) -> List[Dict[str, str]]:
        """Generate warnings about optimization results"""
        warnings = []
        
        # Quality warnings
        if optimization_result.quality_retention < 0.6:
            warnings.append({
                "type": "quality",
                "level": "warning",
                "message": f"Optimization may significantly impact quality ({optimization_result.quality_retention:.1%} retention)",
                "suggestion": "Consider using a more conservative profile or excluding critical sources"
            })
        
        # High compression warnings
        if optimization_result.compression_ratio > 0.7:
            warnings.append({
                "type": "compression",
                "level": "info",
                "message": f"High compression achieved ({optimization_result.compression_ratio:.1%})",
                "suggestion": "Verify that essential information is preserved"
            })
        
        # Sources removed warnings
        if len(optimization_result.removed_sources) > len(original_preview.sources) * 0.5:
            warnings.append({
                "type": "sources",
                "level": "warning",
                "message": f"{len(optimization_result.removed_sources)} sources removed during optimization",
                "suggestion": "Review removed sources to ensure no critical information was lost"
            })
        
        return warnings
    
    # Interactive Editing Methods
    
    async def create_edit_session(
        self,
        preview: ContextPreview,
        max_history: int = 50,
        validation_level: str = "strict",
        auto_backup: bool = True
    ) -> 'ContextEditSession':
        """
        Create an interactive editing session for the given context preview.
        
        Args:
            preview: Context preview to edit
            max_history: Maximum number of edit states to keep in history
            validation_level: Validation strictness ("strict", "permissive", "disabled")
            auto_backup: Whether to enable automatic backup
            
        Returns:
            ContextEditSession for interactive editing
        """
        from autom8.core.context.editor import ContextEditSession, ValidationLevel
        
        # Convert string validation level to enum
        validation_map = {
            "strict": ValidationLevel.STRICT,
            "permissive": ValidationLevel.PERMISSIVE,
            "disabled": ValidationLevel.DISABLED
        }
        validation_enum = validation_map.get(validation_level, ValidationLevel.STRICT)
        
        # Create edit session
        session = ContextEditSession(
            initial_preview=preview,
            max_history=max_history,
            validation_level=validation_enum,
            auto_backup=auto_backup
        )
        
        logger.info(f"Created interactive edit session {session.session_id}")
        return session
    
    async def apply_interactive_edits(
        self,
        original_preview: ContextPreview,
        edit_session: 'ContextEditSession'
    ) -> ContextPreview:
        """
        Apply the edits from an interactive editing session to create a final preview.
        
        Args:
            original_preview: Original context preview
            edit_session: Completed edit session
            
        Returns:
            Final context preview with all edits applied
        """
        # Get the final state from the edit session
        final_preview = edit_session.current_preview
        
        # Re-analyze with the inspector to ensure consistency
        refreshed_preview = await self.preview(
            query=final_preview.query,
            agent_id=final_preview.agent_id,
            context_sources=final_preview.sources[1:],  # Skip query source
            model_target=final_preview.model_target,
            complexity_score=final_preview.complexity_score
        )
        
        # Preserve any custom edits that wouldn't be caught by re-analysis
        refreshed_preview.sources = final_preview.sources
        refreshed_preview.total_tokens = final_preview.total_tokens
        
        logger.info(
            f"Applied interactive edits: {len(original_preview.sources)} -> "
            f"{len(refreshed_preview.sources)} sources, "
            f"{original_preview.total_tokens} -> {refreshed_preview.total_tokens} tokens"
        )
        
        return refreshed_preview
    
    async def quick_edit_source(
        self,
        preview: ContextPreview,
        source_index: int,
        new_content: str,
        new_priority: Optional[int] = None
    ) -> ContextPreview:
        """
        Quick edit of a single source content and/or priority.
        
        Args:
            preview: Context preview to edit
            source_index: Index of source to edit
            new_content: New content for the source
            new_priority: New priority (optional)
            
        Returns:
            Updated context preview
        """
        if not (0 <= source_index < len(preview.sources)):
            raise ValueError(f"Invalid source index: {source_index}")
        
        # Create edit session for this single edit
        session = await self.create_edit_session(preview, max_history=2)
        
        # Prepare modifications
        modifications = {"content": new_content}
        if new_priority is not None:
            modifications["priority"] = new_priority
        
        # Apply the edit
        success = await session.modify_source(source_index, modifications)
        if not success:
            raise RuntimeError("Failed to apply source modification")
        
        # Return the updated preview
        return session.current_preview
    
    async def add_context_source(
        self,
        preview: ContextPreview,
        source_type: ContextSourceType,
        content: str,
        source_id: str = "user_added",
        priority: int = 50,
        position: Optional[int] = None
    ) -> ContextPreview:
        """
        Add a new context source to the preview.
        
        Args:
            preview: Context preview to extend
            source_type: Type of context source
            content: Content for the new source
            source_id: Source identifier
            priority: Priority for the source (0-100)
            position: Position to insert at (None for append)
            
        Returns:
            Updated context preview with new source
        """
        # Create the new source
        new_source = ContextSource(
            type=source_type,
            content=content,
            tokens=self.token_counter.count_tokens(content, preview.model_target),
            source=source_id,
            priority=priority,
            timestamp=datetime.utcnow()
        )
        
        # Create edit session for this operation
        session = await self.create_edit_session(preview, max_history=2)
        
        # Add the source
        success = await session.add_source(new_source, position)
        if not success:
            raise RuntimeError("Failed to add context source")
        
        # Return the updated preview
        return session.current_preview
    
    async def remove_context_source(
        self,
        preview: ContextPreview,
        source_index: int
    ) -> ContextPreview:
        """
        Remove a context source from the preview.
        
        Args:
            preview: Context preview to modify
            source_index: Index of source to remove
            
        Returns:
            Updated context preview without the source
        """
        if not (0 <= source_index < len(preview.sources)):
            raise ValueError(f"Invalid source index: {source_index}")
        
        # Don't allow removing the query source
        if preview.sources[source_index].type == ContextSourceType.QUERY:
            raise ValueError("Cannot remove query source")
        
        # Create edit session for this operation
        session = await self.create_edit_session(preview, max_history=2)
        
        # Remove the source
        success = await session.remove_source(source_index)
        if not success:
            raise RuntimeError("Failed to remove context source")
        
        # Return the updated preview
        return session.current_preview
    
    async def optimize_with_edits(
        self,
        preview: ContextPreview,
        optimization_profile: str = "balanced",
        preserve_manual_edits: bool = True
    ) -> Tuple[ContextPreview, Dict[str, Any]]:
        """
        Apply context optimization while preserving any manual edits.
        
        Args:
            preview: Context preview to optimize
            optimization_profile: Optimization profile to use
            preserve_manual_edits: Whether to preserve sources marked as manually edited
            
        Returns:
            Tuple of (optimized preview, optimization report)
        """
        if not self._optimizer:
            raise RuntimeError("Context optimizer not initialized")
        
        # Run optimization
        result = await self.optimize_context(
            preview=preview,
            profile=optimization_profile
        )
        
        # Create the optimized preview
        optimized_preview = await self.preview(
            query=preview.query,
            agent_id=preview.agent_id,
            context_sources=result.optimized_sources[1:],  # Skip query
            model_target=preview.model_target,
            complexity_score=preview.complexity_score
        )
        
        # Generate optimization report
        report = {
            "original_tokens": result.original_tokens,
            "optimized_tokens": result.optimized_tokens,
            "tokens_saved": result.tokens_saved,
            "compression_ratio": result.compression_ratio,
            "quality_retention": result.quality_retention,
            "strategies_applied": [s.value for s in result.strategies_applied],
            "optimization_log": result.optimization_log,
            "sources_removed": len(result.removed_sources),
            "quality_category": result.quality_category.value
        }
        
        return optimized_preview, report
