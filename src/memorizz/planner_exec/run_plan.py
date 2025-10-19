"""
High-level entry point for plan execution.

This module provides the main `run_plan()` function that orchestrates
the complete planning and execution workflow.
"""
from __future__ import annotations
import asyncio
import logging
from typing import List, Dict, Any, Optional
from .plan_step import PlanStep, ExecutionPlan
from .planner import Planner
from .executor import Executor, ExecutionError

logger = logging.getLogger(__name__)


async def run_plan(
    steps: List[PlanStep],
    initial_artifacts: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    default_timeout: Optional[float] = 300.0,
    enable_tracing: bool = True,
    validate_only: bool = False
) -> Dict[str, Any]:
    """
    High-level entry point to create and execute a plan.

    This function:
    1. Creates a validated ExecutionPlan from steps using the Planner
    2. Executes the plan using the Executor with proper concurrency control
    3. Returns the produced artifacts

    Args:
        steps: List of PlanStep objects to execute
        initial_artifacts: Initial artifacts to provide (external dependencies)
        max_retries: Maximum number of retries for idempotent tools
        default_timeout: Default timeout in seconds for tool execution
        enable_tracing: Whether to enable detailed execution tracing
        validate_only: If True, only validate the plan without executing it

    Returns:
        Dict mapping artifact names to their values

    Raises:
        PlanningError: If plan validation fails
        ExecutionError: If plan execution fails

    Example:
        >>> from planner_exec import run_plan, PlanStep, ToolSpec, tool_spec
        >>>
        >>> @tool_spec(writes={"result"}, idempotent=True)
        >>> def compute(x: int) -> int:
        ...     return x * 2
        >>>
        >>> steps = [
        ...     PlanStep(
        ...         step_id="step1",
        ...         tool=compute,
        ...         args={"x": 5},
        ...         produces={"result"}
        ...     )
        ... ]
        >>>
        >>> artifacts = asyncio.run(run_plan(steps))
        >>> print(artifacts["result"])  # 10
    """
    logger.info(f"Planning execution of {len(steps)} steps")

    # Step 1: Create and validate the plan
    planner = Planner()
    plan = planner.create_plan(steps)

    # Analyze and log parallelism opportunities
    analysis = planner.analyze_parallelism(plan)
    logger.info(
        f"Plan analysis: {analysis['total_steps']} steps, "
        f"{analysis['levels']} execution levels, "
        f"max {analysis['max_parallel']} parallel steps"
    )

    if analysis['bottlenecks']:
        logger.warning(f"Bottleneck steps (>=3 dependents): {analysis['bottlenecks']}")

    # If validation only, return empty artifacts
    if validate_only:
        logger.info("Validation complete (validate_only=True)")
        return {}

    # Step 2: Execute the plan
    executor = Executor(
        max_retries=max_retries,
        default_timeout=default_timeout,
        enable_tracing=enable_tracing
    )

    try:
        artifacts = await executor.execute_plan(plan, initial_artifacts)
        logger.info(f"Plan execution successful. Produced {len(artifacts)} artifacts.")

        # Log execution trace summary
        if enable_tracing:
            trace = executor.get_trace()
            _log_trace_summary(trace)

        return artifacts

    except ExecutionError as e:
        logger.error(f"Plan execution failed: {e}")

        # Log partial results and trace for debugging
        if enable_tracing:
            trace = executor.get_trace()
            _log_trace_summary(trace, error=True)

        raise


def run_plan_sync(
    steps: List[PlanStep],
    initial_artifacts: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Synchronous wrapper for run_plan().

    Useful when calling from synchronous code.

    Args:
        steps: List of PlanStep objects to execute
        initial_artifacts: Initial artifacts to provide
        **kwargs: Additional arguments passed to run_plan()

    Returns:
        Dict mapping artifact names to their values

    Example:
        >>> artifacts = run_plan_sync(steps)
    """
    return asyncio.run(run_plan(steps, initial_artifacts, **kwargs))


def _log_trace_summary(trace: List[Dict[str, Any]], error: bool = False) -> None:
    """Log a summary of the execution trace."""
    if not trace:
        return

    total_steps = len({entry['step_id'] for entry in trace})
    total_attempts = len(trace)
    retries = sum(1 for entry in trace if entry.get('retry', False))
    failures = sum(1 for entry in trace if not entry['success'])

    # Calculate total execution time
    durations = [entry['duration'] for entry in trace if entry.get('duration')]
    total_time = sum(durations) if durations else 0
    avg_time = total_time / len(durations) if durations else 0

    level = logging.ERROR if error else logging.INFO
    logger.log(
        level,
        f"Execution trace summary: {total_steps} steps, {total_attempts} attempts, "
        f"{retries} retries, {failures} failures, "
        f"total time {total_time:.2f}s, avg {avg_time:.2f}s/step"
    )

    # Log slowest steps
    if durations:
        sorted_by_duration = sorted(
            [entry for entry in trace if entry.get('duration')],
            key=lambda e: e['duration'],
            reverse=True
        )
        logger.debug("Slowest steps:")
        for entry in sorted_by_duration[:5]:
            logger.debug(
                f"  {entry['step_id']} ({entry['tool']}): {entry['duration']:.2f}s"
            )


async def create_and_validate_plan(steps: List[PlanStep]) -> ExecutionPlan:
    """
    Create and validate a plan without executing it.

    Useful for pre-flight validation.

    Args:
        steps: List of PlanStep objects

    Returns:
        Validated ExecutionPlan

    Example:
        >>> plan = asyncio.run(create_and_validate_plan(steps))
        >>> print(f"Plan has {len(plan.steps)} steps")
    """
    planner = Planner()
    return planner.create_plan(steps)


def create_and_validate_plan_sync(steps: List[PlanStep]) -> ExecutionPlan:
    """
    Synchronous wrapper for create_and_validate_plan().

    Args:
        steps: List of PlanStep objects

    Returns:
        Validated ExecutionPlan
    """
    return asyncio.run(create_and_validate_plan(steps))
