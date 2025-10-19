"""
ToolSpec: Declarative contract for each tool.

This module defines the specification for tools including their side-effects,
concurrency constraints, and execution characteristics.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Set, FrozenSet
from enum import Enum
import inspect


class LatencyClass(Enum):
    """Expected latency class for a tool."""
    FAST = "fast"           # < 100ms (e.g., simple calculations, cache lookups)
    MEDIUM = "medium"       # 100ms - 1s (e.g., DB queries, simple API calls)
    SLOW = "slow"          # 1s - 10s (e.g., complex queries, external APIs)
    VERY_SLOW = "very_slow" # > 10s (e.g., LLM calls, heavy computations)


@dataclass(frozen=True)
class ResourceSpec:
    """
    Specification of resources that a tool reads or writes.

    Resources are identified by string names (e.g., "database:users", "file:/tmp/data.txt").
    This allows the executor to manage concurrent access with proper locking.
    """
    reads: FrozenSet[str] = field(default_factory=frozenset)
    writes: FrozenSet[str] = field(default_factory=frozenset)

    def __post_init__(self):
        """Validate that reads and writes are frozensets."""
        if not isinstance(self.reads, frozenset):
            object.__setattr__(self, 'reads', frozenset(self.reads))
        if not isinstance(self.writes, frozenset):
            object.__setattr__(self, 'writes', frozenset(self.writes))

    @property
    def all_resources(self) -> FrozenSet[str]:
        """Return all resources (both reads and writes)."""
        return self.reads | self.writes

    def conflicts_with(self, other: ResourceSpec) -> bool:
        """
        Check if this resource spec conflicts with another.

        Conflicts occur when:
        - One writes to a resource the other reads
        - Both write to the same resource

        Multiple concurrent reads are allowed.
        """
        # Write-Write conflict
        if self.writes & other.writes:
            return True
        # Write-Read conflict
        if self.writes & other.reads:
            return True
        if self.reads & other.writes:
            return True
        return False


@dataclass
class ToolSpec:
    """
    Declarative specification for a tool.

    Attributes:
        name: Unique identifier for the tool
        func: The callable function (can be sync or async)
        resources: ResourceSpec declaring what this tool reads/writes
        idempotent: Whether the tool can safely be retried on failure
        concurrency_cap: Maximum number of concurrent executions (None = unlimited)
        latency_class: Expected latency class for planning purposes
        timeout: Maximum execution time in seconds (None = no timeout)
        description: Human-readable description of what the tool does
        is_async: Whether the function is async (auto-detected if None)
    """
    name: str
    func: Callable[..., Any]
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    idempotent: bool = False
    concurrency_cap: Optional[int] = None
    latency_class: LatencyClass = LatencyClass.MEDIUM
    timeout: Optional[float] = None
    description: str = ""
    is_async: Optional[bool] = None

    def __post_init__(self):
        """Validate and auto-detect async status."""
        # Auto-detect async if not specified
        if self.is_async is None:
            self.is_async = inspect.iscoroutinefunction(self.func)

        # Validate concurrency_cap
        if self.concurrency_cap is not None and self.concurrency_cap < 1:
            raise ValueError(f"concurrency_cap must be >= 1, got {self.concurrency_cap}")

        # Validate timeout
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {self.timeout}")

        # Set default description if not provided
        if not self.description and self.func.__doc__:
            self.description = self.func.__doc__.strip().split('\n')[0]

    def can_run_concurrently_with(self, other: ToolSpec) -> bool:
        """
        Check if this tool can run concurrently with another tool.

        Returns False if there are resource conflicts.
        """
        return not self.resources.conflicts_with(other.resources)

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any],
        *,
        reads: Optional[Set[str]] = None,
        writes: Optional[Set[str]] = None,
        idempotent: bool = False,
        concurrency_cap: Optional[int] = None,
        latency_class: LatencyClass = LatencyClass.MEDIUM,
        timeout: Optional[float] = None,
        description: str = "",
    ) -> ToolSpec:
        """
        Create a ToolSpec from a function with keyword arguments.

        Example:
            @ToolSpec.from_function(
                reads={"database:users"},
                writes={"cache:user_data"},
                idempotent=True,
                latency_class=LatencyClass.FAST
            )
            def get_user(user_id: str) -> dict:
                ...
        """
        resources = ResourceSpec(
            reads=frozenset(reads or set()),
            writes=frozenset(writes or set())
        )

        return cls(
            name=func.__name__,
            func=func,
            resources=resources,
            idempotent=idempotent,
            concurrency_cap=concurrency_cap,
            latency_class=latency_class,
            timeout=timeout,
            description=description or (func.__doc__.strip().split('\n')[0] if func.__doc__ else "")
        )

    def __repr__(self) -> str:
        return (
            f"ToolSpec(name={self.name!r}, "
            f"reads={set(self.resources.reads)}, "
            f"writes={set(self.resources.writes)}, "
            f"idempotent={self.idempotent}, "
            f"async={self.is_async})"
        )


def tool_spec(
    *,
    reads: Optional[Set[str]] = None,
    writes: Optional[Set[str]] = None,
    idempotent: bool = False,
    concurrency_cap: Optional[int] = None,
    latency_class: LatencyClass = LatencyClass.MEDIUM,
    timeout: Optional[float] = None,
    description: str = "",
) -> Callable[[Callable], ToolSpec]:
    """
    Decorator to create a ToolSpec from a function.

    Example:
        @tool_spec(
            reads={"database:users"},
            writes={"cache:user_data"},
            idempotent=True,
            latency_class=LatencyClass.FAST,
            timeout=5.0
        )
        def get_user(user_id: str) -> dict:
            '''Get user from database and cache result.'''
            # implementation
            pass
    """
    def decorator(func: Callable) -> ToolSpec:
        return ToolSpec.from_function(
            func,
            reads=reads,
            writes=writes,
            idempotent=idempotent,
            concurrency_cap=concurrency_cap,
            latency_class=latency_class,
            timeout=timeout,
            description=description
        )
    return decorator
