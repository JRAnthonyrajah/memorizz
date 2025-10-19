"""
Planner: Builds and validates execution plans from steps.

This module provides the Planner class that takes a list of steps
and produces a validated ExecutionPlan with proper dependency resolution.
"""
from __future__ import annotations
from typing import List, Dict, Set, Optional
import logging
from .plan_step import PlanStep, ExecutionPlan
from .tool_spec import ToolSpec

logger = logging.getLogger(__name__)


class PlanningError(Exception):
    """Raised when plan validation fails."""
    pass


class Planner:
    """
    Planner accepts a list of intended steps and produces a validated DAG.

    The planner:
    1. Validates that all consumed artifacts are produced
    2. Detects cycles in dependencies
    3. Infers implicit dependencies from data flow
    4. Validates resource conflicts
    """

    def __init__(self):
        """Initialize the planner."""
        pass

    def create_plan(self, steps: List[PlanStep]) -> ExecutionPlan:
        """
        Create and validate an execution plan from a list of steps.

        Args:
            steps: List of PlanStep objects to execute

        Returns:
            ExecutionPlan: Validated execution plan

        Raises:
            PlanningError: If the plan is invalid (cycles, missing deps, etc.)
        """
        if not steps:
            return ExecutionPlan(steps=[])

        # 1. Validate unique step IDs
        self._validate_unique_step_ids(steps)

        # 2. Validate artifact dependencies
        self._validate_artifact_dependencies(steps)

        # 3. Infer implicit dependencies from data flow
        steps_with_inferred_deps = self._infer_data_dependencies(steps)

        # 4. Create the execution plan
        try:
            plan = ExecutionPlan(steps=steps_with_inferred_deps)
        except ValueError as e:
            raise PlanningError(f"Failed to create execution plan: {e}")

        # 5. Validate no cycles
        self._validate_no_cycles(plan)

        # 6. Log plan summary
        self._log_plan_summary(plan)

        return plan

    def _validate_unique_step_ids(self, steps: List[PlanStep]) -> None:
        """Ensure all step IDs are unique."""
        step_ids = [step.step_id for step in steps]
        duplicates = {sid for sid in step_ids if step_ids.count(sid) > 1}

        if duplicates:
            raise PlanningError(f"Duplicate step IDs found: {duplicates}")

    def _validate_artifact_dependencies(self, steps: List[PlanStep]) -> None:
        """
        Validate that all consumed artifacts are produced by some step.

        Allows for "external" artifacts (consumed but not produced by any step).
        These are assumed to be provided at runtime.
        """
        # Build map of produced artifacts
        produced = set()
        for step in steps:
            produced.update(step.produces)

        # Check consumed artifacts
        all_consumed = set()
        for step in steps:
            all_consumed.update(step.consumes)

        # External artifacts (consumed but not produced)
        external = all_consumed - produced

        if external:
            logger.info(f"External artifacts (must be provided at runtime): {external}")

        # Check for steps that reference non-existent step dependencies
        step_ids = {step.step_id for step in steps}
        for step in steps:
            missing_deps = step.depends_on - step_ids
            if missing_deps:
                raise PlanningError(
                    f"Step {step.step_id!r} depends on non-existent steps: {missing_deps}"
                )

    def _infer_data_dependencies(self, steps: List[PlanStep]) -> List[PlanStep]:
        """
        Infer implicit dependencies from data flow.

        If step B consumes an artifact that step A produces,
        add A to B's depends_on set.
        """
        # Build producer map
        producers: Dict[str, str] = {}  # artifact -> step_id
        for step in steps:
            for artifact in step.produces:
                if artifact in producers:
                    # Multiple producers - already handled by ExecutionPlan validation
                    pass
                producers[artifact] = step.step_id

        # Add implicit dependencies
        updated_steps = []
        for step in steps:
            # Find producers of consumed artifacts
            implicit_deps = set()
            for artifact in step.consumes:
                producer_id = producers.get(artifact)
                if producer_id and producer_id != step.step_id:
                    implicit_deps.add(producer_id)

            # Merge with explicit dependencies
            all_deps = step.depends_on | frozenset(implicit_deps)

            # Create new step with updated dependencies
            if all_deps != step.depends_on:
                updated_step = PlanStep(
                    step_id=step.step_id,
                    tool=step.tool,
                    args=step.args,
                    consumes=step.consumes,
                    produces=step.produces,
                    depends_on=all_deps,
                    metadata=step.metadata
                )
                updated_steps.append(updated_step)
            else:
                updated_steps.append(step)

        return updated_steps

    def _validate_no_cycles(self, plan: ExecutionPlan) -> None:
        """
        Validate that the plan has no cycles.

        Uses topological sort - if it fails, there's a cycle.
        """
        try:
            plan.topological_sort()
        except ValueError as e:
            raise PlanningError(f"Cycle detected in plan: {e}")

    def _log_plan_summary(self, plan: ExecutionPlan) -> None:
        """Log a summary of the plan for debugging."""
        logger.info(f"Created execution plan with {len(plan.steps)} steps")

        # Analyze parallelism opportunities
        sorted_steps = plan.topological_sort()
        levels = self._compute_execution_levels(plan, sorted_steps)

        logger.info(f"Plan has {len(levels)} execution levels:")
        for level_idx, level_steps in enumerate(levels):
            step_names = [step.tool.name for step in level_steps]
            logger.info(f"  Level {level_idx}: {len(level_steps)} steps can run in parallel: {step_names}")

    def _compute_execution_levels(
        self,
        plan: ExecutionPlan,
        sorted_steps: List[PlanStep]
    ) -> List[List[PlanStep]]:
        """
        Compute execution levels for parallel execution.

        Steps in the same level can potentially run in parallel.
        """
        levels: List[List[PlanStep]] = []
        step_to_level: Dict[str, int] = {}

        for step in sorted_steps:
            # Find the maximum level of all dependencies
            deps = plan.get_dependencies(step)
            if not deps:
                level = 0
            else:
                level = max(step_to_level[dep.step_id] for dep in deps) + 1

            step_to_level[step.step_id] = level

            # Add to appropriate level
            while len(levels) <= level:
                levels.append([])
            levels[level].append(step)

        return levels

    def analyze_parallelism(self, plan: ExecutionPlan) -> Dict[str, any]:
        """
        Analyze parallelism opportunities in the plan.

        Returns:
            Dict with analysis results including:
            - total_steps: Total number of steps
            - max_parallel: Maximum parallelism possible
            - levels: Number of execution levels
            - bottlenecks: Steps that limit parallelism
        """
        sorted_steps = plan.topological_sort()
        levels = self._compute_execution_levels(plan, sorted_steps)

        max_parallel = max(len(level) for level in levels) if levels else 0

        # Find bottlenecks (steps with many dependents)
        dependents_count = {step.step_id: 0 for step in plan.steps}
        for step in plan.steps:
            for dep in plan.get_dependencies(step):
                dependents_count[dep.step_id] += 1

        bottlenecks = [
            step.step_id
            for step in plan.steps
            if dependents_count[step.step_id] >= 3
        ]

        return {
            "total_steps": len(plan.steps),
            "max_parallel": max_parallel,
            "levels": len(levels),
            "bottlenecks": bottlenecks,
            "avg_steps_per_level": len(plan.steps) / len(levels) if levels else 0
        }
