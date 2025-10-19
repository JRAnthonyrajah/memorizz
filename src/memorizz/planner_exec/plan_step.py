"""
PlanStep: A step in an execution plan with dependencies.

This module defines steps that reference tools, their arguments,
and data-flow dependencies.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Set, Optional, List, FrozenSet
from .tool_spec import ToolSpec


@dataclass
class PlanStep:
    """
    A single step in an execution plan.

    Attributes:
        step_id: Unique identifier for this step
        tool: The ToolSpec to execute
        args: Arguments to pass to the tool
        consumes: Set of artifact names this step consumes (depends on)
        produces: Set of artifact names this step produces
        depends_on: Set of step_ids that must complete before this step
        metadata: Optional metadata for debugging/logging
    """
    step_id: str
    tool: ToolSpec
    args: Dict[str, Any] = field(default_factory=dict)
    consumes: FrozenSet[str] = field(default_factory=frozenset)
    produces: FrozenSet[str] = field(default_factory=frozenset)
    depends_on: FrozenSet[str] = field(default_factory=frozenset)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Convert mutable sets to frozensets."""
        if not isinstance(self.consumes, frozenset):
            object.__setattr__(self, 'consumes', frozenset(self.consumes))
        if not isinstance(self.produces, frozenset):
            object.__setattr__(self, 'produces', frozenset(self.produces))
        if not isinstance(self.depends_on, frozenset):
            object.__setattr__(self, 'depends_on', frozenset(self.depends_on))

    def has_data_dependency_on(self, other: PlanStep) -> bool:
        """
        Check if this step has a data dependency on another step.

        A data dependency exists if this step consumes something
        the other step produces.
        """
        return bool(self.consumes & other.produces)

    def has_resource_conflict_with(self, other: PlanStep) -> bool:
        """
        Check if this step has a resource conflict with another step.

        Delegates to the tool's resource specification.
        """
        return not self.tool.can_run_concurrently_with(other.tool)

    def can_run_concurrently_with(self, other: PlanStep) -> bool:
        """
        Check if this step can run concurrently with another step.

        Steps can run concurrently if:
        1. No data dependencies between them
        2. No resource conflicts
        3. Neither depends (directly or transitively) on the other
        """
        # Check for data dependencies
        if self.has_data_dependency_on(other) or other.has_data_dependency_on(self):
            return False

        # Check for resource conflicts
        if self.has_resource_conflict_with(other):
            return False

        return True

    def __repr__(self) -> str:
        return (
            f"PlanStep(id={self.step_id!r}, "
            f"tool={self.tool.name!r}, "
            f"consumes={set(self.consumes)}, "
            f"produces={set(self.produces)}, "
            f"depends_on={set(self.depends_on)})"
        )

    def __hash__(self) -> int:
        """Make PlanStep hashable for use in sets/dicts."""
        return hash(self.step_id)

    def __eq__(self, other: Any) -> bool:
        """Compare PlanSteps by step_id."""
        if not isinstance(other, PlanStep):
            return NotImplemented
        return self.step_id == other.step_id


@dataclass
class ExecutionPlan:
    """
    A complete execution plan with validated dependencies.

    Attributes:
        steps: List of all steps in the plan
        artifacts: Map of artifact names to the steps that produce them
        step_map: Map of step_ids to steps for quick lookup
    """
    steps: List[PlanStep]
    artifacts: Dict[str, str] = field(default_factory=dict)  # artifact_name -> step_id
    step_map: Dict[str, PlanStep] = field(default_factory=dict)

    def __post_init__(self):
        """Build internal indexes."""
        self.step_map = {step.step_id: step for step in self.steps}

        # Build artifacts map
        for step in self.steps:
            for artifact in step.produces:
                if artifact in self.artifacts:
                    raise ValueError(
                        f"Artifact {artifact!r} produced by multiple steps: "
                        f"{self.artifacts[artifact]!r} and {step.step_id!r}"
                    )
                self.artifacts[artifact] = step.step_id

    def get_step(self, step_id: str) -> Optional[PlanStep]:
        """Get a step by ID."""
        return self.step_map.get(step_id)

    def get_dependencies(self, step: PlanStep) -> Set[PlanStep]:
        """
        Get all direct dependencies for a step.

        Returns steps that must complete before the given step can run.
        """
        deps = set()

        # Add explicit dependencies
        for dep_id in step.depends_on:
            dep_step = self.get_step(dep_id)
            if dep_step:
                deps.add(dep_step)

        # Add implicit data dependencies (steps producing consumed artifacts)
        for artifact in step.consumes:
            producer_id = self.artifacts.get(artifact)
            if producer_id:
                producer = self.get_step(producer_id)
                if producer:
                    deps.add(producer)

        return deps

    def get_transitive_dependencies(self, step: PlanStep) -> Set[PlanStep]:
        """
        Get all transitive dependencies for a step (full ancestor tree).
        """
        visited = set()
        to_visit = {step}

        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            visited.add(current)

            deps = self.get_dependencies(current)
            to_visit.update(deps)

        # Remove the step itself from its dependencies
        visited.discard(step)
        return visited

    def topological_sort(self) -> List[PlanStep]:
        """
        Return steps in topological order (dependencies before dependents).

        Raises ValueError if there are cycles.
        """
        # Kahn's algorithm
        in_degree = {step.step_id: 0 for step in self.steps}

        # Calculate in-degrees
        for step in self.steps:
            for dep in self.get_dependencies(step):
                in_degree[step.step_id] += 1

        # Start with steps that have no dependencies
        queue = [step for step in self.steps if in_degree[step.step_id] == 0]
        result = []

        while queue:
            # Process step with no remaining dependencies
            current = queue.pop(0)
            result.append(current)

            # Find steps that depend on current
            for step in self.steps:
                if current in self.get_dependencies(step):
                    in_degree[step.step_id] -= 1
                    if in_degree[step.step_id] == 0:
                        queue.append(step)

        if len(result) != len(self.steps):
            raise ValueError("Cycle detected in execution plan")

        return result

    def __repr__(self) -> str:
        return f"ExecutionPlan(steps={len(self.steps)}, artifacts={len(self.artifacts)})"
