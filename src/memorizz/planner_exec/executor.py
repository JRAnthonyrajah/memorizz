"""
Executor: Runs execution plans with async concurrency and proper isolation.

This module provides the Executor class that executes plans with:
- Async concurrency (runs independent steps in parallel)
- Per-resource RW-locks (ensures isolation)
- Per-tool semaphores (respects concurrency caps)
- Timeouts and retries (for idempotent tools)
- Structured tracing and error handling
"""
from __future__ import annotations
import asyncio
import logging
import time
from typing import Any, Dict, Set, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from .plan_step import PlanStep, ExecutionPlan
from .tool_spec import ToolSpec

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a step execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RETRY = "retry"


@dataclass
class StepResult:
    """Result of executing a single step."""
    step_id: str
    status: StepStatus
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    attempts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def success(self) -> bool:
        """Check if the step completed successfully."""
        return self.status == StepStatus.COMPLETED


class ResourceLockManager:
    """
    Manages read-write locks for resources.

    Multiple readers can hold a lock simultaneously, but writers are exclusive.
    """

    def __init__(self):
        self._locks: Dict[str, asyncio.Lock] = {}
        self._reader_counts: Dict[str, int] = {}
        self._writer_active: Dict[str, bool] = {}
        self._condition: Dict[str, asyncio.Condition] = {}

    async def acquire_read(self, resource: str) -> None:
        """Acquire a read lock on a resource."""
        logger.debug(f"[LOCK_MGR] Requesting READ lock on '{resource}'")

        if resource not in self._condition:
            logger.debug(f"[LOCK_MGR] Initializing lock structures for '{resource}'")
            self._condition[resource] = asyncio.Condition()
            self._reader_counts[resource] = 0
            self._writer_active[resource] = False

        async with self._condition[resource]:
            # Wait while a writer is active
            if self._writer_active[resource]:
                logger.debug(f"[LOCK_MGR] Waiting for writer to release '{resource}'")
            while self._writer_active[resource]:
                await self._condition[resource].wait()

            self._reader_counts[resource] += 1
            logger.debug(f"[LOCK_MGR] Acquired READ lock on '{resource}' (reader_count={self._reader_counts[resource]})")

    async def release_read(self, resource: str) -> None:
        """Release a read lock on a resource."""
        logger.debug(f"[LOCK_MGR] Releasing READ lock on '{resource}'")
        async with self._condition[resource]:
            self._reader_counts[resource] -= 1
            logger.debug(f"[LOCK_MGR] Released READ lock on '{resource}' (reader_count={self._reader_counts[resource]})")
            if self._reader_counts[resource] == 0:
                # Notify waiting writers
                logger.debug(f"[LOCK_MGR] No more readers on '{resource}', notifying waiters")
                self._condition[resource].notify_all()

    async def acquire_write(self, resource: str) -> None:
        """Acquire a write lock on a resource (exclusive)."""
        logger.debug(f"[LOCK_MGR] Requesting WRITE lock on '{resource}'")

        if resource not in self._condition:
            logger.debug(f"[LOCK_MGR] Initializing lock structures for '{resource}'")
            self._condition[resource] = asyncio.Condition()
            self._reader_counts[resource] = 0
            self._writer_active[resource] = False

        async with self._condition[resource]:
            # Wait while any readers or writers are active
            if self._reader_counts[resource] > 0 or self._writer_active[resource]:
                logger.debug(
                    f"[LOCK_MGR] Waiting for lock on '{resource}' - "
                    f"readers={self._reader_counts[resource]}, writer_active={self._writer_active[resource]}"
                )
            while self._reader_counts[resource] > 0 or self._writer_active[resource]:
                await self._condition[resource].wait()

            self._writer_active[resource] = True
            logger.debug(f"[LOCK_MGR] Acquired WRITE lock on '{resource}'")

    async def release_write(self, resource: str) -> None:
        """Release a write lock on a resource."""
        logger.debug(f"[LOCK_MGR] Releasing WRITE lock on '{resource}'")
        async with self._condition[resource]:
            self._writer_active[resource] = False
            logger.debug(f"[LOCK_MGR] Released WRITE lock on '{resource}', notifying waiters")
            # Notify all waiting readers and writers
            self._condition[resource].notify_all()


class Executor:
    """
    Executes execution plans with async concurrency and proper resource isolation.

    Features:
    - Runs independent steps in parallel
    - Enforces resource RW-locks
    - Respects per-tool concurrency caps
    - Handles timeouts and retries for idempotent tools
    - Provides structured execution traces
    """

    def __init__(
        self,
        max_retries: int = 3,
        default_timeout: Optional[float] = 300.0,  # 5 minutes default
        enable_tracing: bool = True
    ):
        """
        Initialize the executor.

        Args:
            max_retries: Maximum number of retries for idempotent tools
            default_timeout: Default timeout in seconds (if tool doesn't specify)
            enable_tracing: Whether to enable detailed execution tracing
        """
        self.max_retries = max_retries
        self.default_timeout = default_timeout
        self.enable_tracing = enable_tracing

        # Resource management
        self._resource_locks = ResourceLockManager()
        self._tool_semaphores: Dict[str, asyncio.Semaphore] = {}

        # Execution state
        self._results: Dict[str, StepResult] = {}
        self._artifacts: Dict[str, Any] = {}
        self._trace: List[Dict[str, Any]] = []

    async def execute_plan(
        self,
        plan: ExecutionPlan,
        initial_artifacts: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute an execution plan asynchronously.

        Args:
            plan: The ExecutionPlan to execute
            initial_artifacts: Initial artifacts to provide (external dependencies)

        Returns:
            Dict mapping artifact names to their values

        Raises:
            ExecutionError: If any non-retriable step fails
        """
        logger.info(f"Starting execution of plan with {len(plan.steps)} steps")

        # Initialize artifacts with external dependencies
        if initial_artifacts:
            self._artifacts.update(initial_artifacts)

        # Initialize results
        self._results = {
            step.step_id: StepResult(step_id=step.step_id, status=StepStatus.PENDING)
            for step in plan.steps
        }

        # Create semaphores for tools with concurrency caps
        self._create_tool_semaphores(plan)

        # Execute the plan
        try:
            await self._execute_plan_concurrent(plan)
        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            raise ExecutionError(f"Plan execution failed: {e}") from e

        # Check for failures
        failed_steps = [
            step_id for step_id, result in self._results.items()
            if not result.success
        ]

        if failed_steps:
            error_details = {
                step_id: str(self._results[step_id].error)
                for step_id in failed_steps
            }
            raise ExecutionError(f"Steps failed: {error_details}")

        logger.info(f"Plan execution completed successfully")
        return self._artifacts.copy()

    def _create_tool_semaphores(self, plan: ExecutionPlan) -> None:
        """Create semaphores for tools with concurrency caps."""
        for step in plan.steps:
            tool = step.tool
            if tool.concurrency_cap and tool.name not in self._tool_semaphores:
                self._tool_semaphores[tool.name] = asyncio.Semaphore(tool.concurrency_cap)
                logger.debug(f"Created semaphore for tool {tool.name} with cap {tool.concurrency_cap}")

    async def _execute_plan_concurrent(self, plan: ExecutionPlan) -> None:
        """
        Execute plan with maximum concurrency while respecting dependencies.

        Uses a wave-based approach: execute all ready steps concurrently,
        then move to the next wave.
        """
        completed: Set[str] = set()
        pending = {step.step_id for step in plan.steps}

        while pending:
            # Find steps that are ready to run (all dependencies completed)
            ready = []
            for step_id in pending:
                step = plan.get_step(step_id)
                deps = plan.get_dependencies(step)
                dep_ids = {d.step_id for d in deps}

                if dep_ids.issubset(completed):
                    ready.append(step)

            if not ready:
                # No steps ready but some pending - this shouldn't happen with valid plan
                pending_steps = [plan.get_step(sid) for sid in pending]
                raise ExecutionError(
                    f"Deadlock detected: {len(pending)} steps pending but none ready. "
                    f"Pending: {pending}"
                )

            logger.debug(f"Executing wave with {len(ready)} steps: {[s.step_id for s in ready]}")

            # Execute all ready steps concurrently
            tasks = [self._execute_step(step, plan) for step in ready]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Mark completed and update pending
            for step in ready:
                completed.add(step.step_id)
                pending.remove(step.step_id)

            logger.debug(f"Wave completed. {len(completed)} total completed, {len(pending)} pending")

    async def _execute_step(self, step: PlanStep, plan: ExecutionPlan) -> None:
        """
        Execute a single step with retries, timeouts, and resource locks.

        Args:
            step: The step to execute
            plan: The execution plan (for context)
        """
        logger.info(f"[EXEC] Starting execution of step {step.step_id} (tool: {step.tool.name})")

        result = self._results[step.step_id]
        result.status = StepStatus.RUNNING
        result.start_time = time.time()

        tool = step.tool
        max_attempts = self.max_retries + 1 if tool.idempotent else 1

        logger.debug(f"[EXEC] Step {step.step_id} will attempt up to {max_attempts} times")

        for attempt in range(1, max_attempts + 1):
            result.attempts = attempt
            logger.debug(f"[EXEC] Step {step.step_id} attempt {attempt}/{max_attempts}")

            try:
                # Acquire resource locks
                await self._acquire_step_locks(step)

                try:
                    # Acquire tool semaphore if exists
                    semaphore = self._tool_semaphores.get(tool.name)
                    if semaphore:
                        async with semaphore:
                            output = await self._execute_tool_with_timeout(step)
                    else:
                        output = await self._execute_tool_with_timeout(step)

                    # Store output in artifacts
                    for artifact_name in step.produces:
                        self._artifacts[artifact_name] = output

                    result.result = output
                    result.status = StepStatus.COMPLETED
                    result.end_time = time.time()

                    self._trace_step(step, result, attempt, success=True)
                    logger.info(
                        f"Step {step.step_id} completed successfully in {result.duration:.2f}s "
                        f"(attempt {attempt}/{max_attempts})"
                    )
                    return

                finally:
                    # Always release locks
                    await self._release_step_locks(step)

            except asyncio.TimeoutError as e:
                result.error = e
                result.status = StepStatus.TIMEOUT
                logger.warning(
                    f"Step {step.step_id} timed out (attempt {attempt}/{max_attempts})"
                )

                if attempt < max_attempts:
                    result.status = StepStatus.RETRY
                    self._trace_step(step, result, attempt, success=False, retry=True)
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    result.end_time = time.time()
                    self._trace_step(step, result, attempt, success=False)
                    raise ExecutionError(f"Step {step.step_id} timed out after {attempt} attempts")

            except Exception as e:
                result.error = e
                result.status = StepStatus.FAILED
                logger.error(
                    f"Step {step.step_id} failed: {e} (attempt {attempt}/{max_attempts})"
                )

                if attempt < max_attempts and tool.idempotent:
                    result.status = StepStatus.RETRY
                    self._trace_step(step, result, attempt, success=False, retry=True)
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    result.end_time = time.time()
                    self._trace_step(step, result, attempt, success=False)
                    raise ExecutionError(
                        f"Step {step.step_id} failed after {attempt} attempts: {e}"
                    ) from e

    async def _execute_tool_with_timeout(self, step: PlanStep) -> Any:
        """Execute the tool function with timeout."""
        tool = step.tool
        timeout = tool.timeout or self.default_timeout

        # Prepare arguments (resolve from artifacts)
        args = self._resolve_arguments(step)

        # Log tool execution mode
        logger.info(f"Executing tool {tool.name}: is_async={tool.is_async}, timeout={timeout}s")

        # Execute the tool (async or sync)
        if tool.is_async:
            logger.debug(f"Tool {tool.name} using async execution path")
            coro = tool.func(**args)
        else:
            logger.debug(f"Tool {tool.name} using ThreadPoolExecutor (SYNC) path")
            # Wrap sync function in executor
            loop = asyncio.get_event_loop()
            coro = loop.run_in_executor(None, lambda: tool.func(**args))

        # Apply timeout
        if timeout:
            return await asyncio.wait_for(coro, timeout=timeout)
        else:
            return await coro

    def _resolve_arguments(self, step: PlanStep) -> Dict[str, Any]:
        """
        Resolve step arguments by replacing artifact references with actual values.

        Arguments can reference artifacts using the syntax: {"$artifact": "artifact_name"}
        """
        resolved = {}
        for key, value in step.args.items():
            if isinstance(value, dict) and "$artifact" in value:
                artifact_name = value["$artifact"]
                if artifact_name not in self._artifacts:
                    raise ExecutionError(
                        f"Step {step.step_id} requires artifact {artifact_name!r} "
                        f"which is not available"
                    )
                resolved[key] = self._artifacts[artifact_name]
            else:
                resolved[key] = value
        return resolved

    async def _acquire_step_locks(self, step: PlanStep) -> None:
        """Acquire all necessary resource locks for a step."""
        logger.info(
            f"[LOCK] Step {step.step_id} attempting to acquire locks - "
            f"reads={list(step.tool.resources.reads)}, writes={list(step.tool.resources.writes)}"
        )

        # Acquire read locks
        for resource in step.tool.resources.reads:
            logger.debug(f"[LOCK] Acquiring READ lock on resource: {resource}")
            await self._resource_locks.acquire_read(resource)
            logger.debug(f"[LOCK] Acquired READ lock on resource: {resource}")

        # Acquire write locks
        for resource in step.tool.resources.writes:
            logger.debug(f"[LOCK] Acquiring WRITE lock on resource: {resource}")
            await self._resource_locks.acquire_write(resource)
            logger.debug(f"[LOCK] Acquired WRITE lock on resource: {resource}")

        logger.info(f"[LOCK] Step {step.step_id} successfully acquired all locks")

    async def _release_step_locks(self, step: PlanStep) -> None:
        """Release all resource locks for a step."""
        logger.info(f"[LOCK] Step {step.step_id} releasing locks")

        # Release read locks
        for resource in step.tool.resources.reads:
            logger.debug(f"[LOCK] Releasing READ lock on resource: {resource}")
            await self._resource_locks.release_read(resource)
            logger.debug(f"[LOCK] Released READ lock on resource: {resource}")

        # Release write locks
        for resource in step.tool.resources.writes:
            logger.debug(f"[LOCK] Releasing WRITE lock on resource: {resource}")
            await self._resource_locks.release_write(resource)
            logger.debug(f"[LOCK] Released WRITE lock on resource: {resource}")

        logger.info(f"[LOCK] Step {step.step_id} released all locks")

    def _trace_step(
        self,
        step: PlanStep,
        result: StepResult,
        attempt: int,
        success: bool,
        retry: bool = False
    ) -> None:
        """Record step execution in trace."""
        if not self.enable_tracing:
            return

        trace_entry = {
            "step_id": step.step_id,
            "tool": step.tool.name,
            "attempt": attempt,
            "status": result.status.value,
            "success": success,
            "retry": retry,
            "duration": result.duration,
            "timestamp": datetime.now().isoformat(),
        }

        if result.error:
            trace_entry["error"] = str(result.error)

        self._trace.append(trace_entry)

    def get_trace(self) -> List[Dict[str, Any]]:
        """Get the execution trace."""
        return self._trace.copy()

    def get_results(self) -> Dict[str, StepResult]:
        """Get all step results."""
        return self._results.copy()


class ExecutionError(Exception):
    """Raised when plan execution fails."""
    pass
