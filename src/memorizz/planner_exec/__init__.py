"""
planner_exec: Declarative task planning and execution with concurrency control.

This package provides a complete system for executing tasks with:
- Declarative tool specifications with side-effect declarations
- Automatic dependency resolution and DAG planning
- Concurrent execution with proper resource isolation
- Retry logic, timeouts, and structured tracing

Core Components:
- ToolSpec: Declarative contract for each tool
- PlanStep: A step referencing a tool with dependencies
- Planner: Builds validated DAGs from steps
- Executor: Runs plans with async concurrency and locks
- run_plan(): High-level entry point

Example:
    >>> from memorizz.planner_exec import tool_spec, PlanStep, run_plan, LatencyClass
    >>> import asyncio
    >>>
    >>> @tool_spec(
    ...     reads={"input_file"},
    ...     writes={"database:users"},
    ...     idempotent=True,
    ...     latency_class=LatencyClass.MEDIUM,
    ...     timeout=30.0
    ... )
    >>> def load_users(filename: str) -> list:
    ...     '''Load users from file into database.'''
    ...     # implementation
    ...     return ["user1", "user2"]
    >>>
    >>> @tool_spec(
    ...     reads={"database:users"},
    ...     writes={"cache:user_count"},
    ...     idempotent=True
    ... )
    >>> def count_users() -> int:
    ...     '''Count users in database.'''
    ...     return 2
    >>>
    >>> steps = [
    ...     PlanStep(
    ...         step_id="load",
    ...         tool=load_users,
    ...         args={"filename": "users.txt"},
    ...         produces={"user_list"}
    ...     ),
    ...     PlanStep(
    ...         step_id="count",
    ...         tool=count_users,
    ...         args={},
    ...         produces={"user_count"},
    ...         depends_on={"load"}  # Explicit dependency
    ...     )
    ... ]
    >>>
    >>> artifacts = asyncio.run(run_plan(steps))
    >>> print(artifacts["user_count"])  # 2
"""

from .tool_spec import (
    ToolSpec,
    ResourceSpec,
    LatencyClass,
    tool_spec,
)

from .plan_step import (
    PlanStep,
    ExecutionPlan,
)

from .planner import (
    Planner,
    PlanningError,
)

from .executor import (
    Executor,
    ExecutionError,
    StepResult,
    StepStatus,
)

from .run_plan import (
    run_plan,
    run_plan_sync,
    create_and_validate_plan,
    create_and_validate_plan_sync,
)

__all__ = [
    # Tool specification
    "ToolSpec",
    "ResourceSpec",
    "LatencyClass",
    "tool_spec",
    # Plan steps
    "PlanStep",
    "ExecutionPlan",
    # Planning
    "Planner",
    "PlanningError",
    # Execution
    "Executor",
    "ExecutionError",
    "StepResult",
    "StepStatus",
    # High-level API
    "run_plan",
    "run_plan_sync",
    "create_and_validate_plan",
    "create_and_validate_plan_sync",
]

__version__ = "1.0.0"
