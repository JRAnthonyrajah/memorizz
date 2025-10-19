# Integrating planner_exec with MemAgent

This guide shows how to integrate the new concurrent task execution system with the existing `MemAgent` class.

## Overview

The `planner_exec` system provides:

1. **ToolSpec**: Declarative contracts for tools with side-effect declarations
2. **PlanStep**: Steps with data-flow dependencies
3. **Planner**: Automatic DAG construction and validation
4. **Executor**: Concurrent execution with resource locks and retries
5. **run_plan()**: High-level orchestration

## Current MemAgent Issues

The current `MemAgent._handle_tool_calls()` (line 1313) executes tools **serially**:

```python
for call in tool_calls:
    # Execute each tool one at a time - no concurrency!
    result, error_message, workflow_outcome = self._execute_single_tool(...)
```

**Problems:**
- All tools block each other, even when independent
- No resource conflict detection
- No retry logic
- No timeout handling
- No support for async tools

## Integration Approach

There are two integration approaches:

### Option 1: Drop-in Replacement (Minimal Changes)

Replace `_handle_tool_calls()` with a planner-aware version that automatically detects dependencies and runs tools concurrently where safe.

### Option 2: Explicit Planning Mode (Recommended)

Add a new "planning mode" to MemAgent where the LLM generates a plan, and the executor runs it with proper concurrency control.

## Implementation: Option 2 (Explicit Planning)

### Step 1: Add ToolSpec Metadata to Toolbox

Enhance the `Toolbox` to store `ToolSpec` declarations:

```python
# In toolbox.py
from ..planner_exec import ToolSpec, ResourceSpec, LatencyClass

class Toolbox:
    def register_tool(
        self,
        func: Callable,
        reads: Optional[Set[str]] = None,
        writes: Optional[Set[str]] = None,
        idempotent: bool = False,
        concurrency_cap: Optional[int] = None,
        latency_class: LatencyClass = LatencyClass.MEDIUM,
        timeout: Optional[float] = None,
        **metadata
    ):
        """Register a tool with its specification."""
        # Create ToolSpec
        tool_spec = ToolSpec(
            name=func.__name__,
            func=func,
            resources=ResourceSpec(
                reads=frozenset(reads or set()),
                writes=frozenset(writes or set())
            ),
            idempotent=idempotent,
            concurrency_cap=concurrency_cap,
            latency_class=latency_class,
            timeout=timeout
        )

        # Store both the spec and function
        self.tools[func.__name__] = {
            "spec": tool_spec,
            "function": func,
            "metadata": metadata
        }
```

### Step 2: Extend MemAgent with Plan Execution

Add plan execution capability to `MemAgent`:

```python
# In memagent.py
from .planner_exec import (
    PlanStep,
    run_plan,
    Planner,
    ExecutionError
)
import asyncio

class MemAgent:
    # ... existing code ...

    async def _execute_plan_mode(
        self,
        tool_calls: List,
        messages: List[Dict],
        query: str,
        memory_id: str,
        tool_metas: List[Dict]
    ) -> List[Dict]:
        """
        Execute tool calls using the planner/executor system.

        This enables concurrent execution with proper dependency tracking.
        """
        # Convert tool calls to PlanSteps
        steps = []
        for idx, call in enumerate(tool_calls):
            name = call.name
            args = json.loads(call.arguments)

            # Get tool spec from toolbox
            tool_entry = self.tools.get_tool_spec(name)
            if not tool_entry:
                # Fallback to serial execution for unknown tools
                logger.warning(f"Tool {name} has no spec, falling back to serial")
                return self._handle_tool_calls(tool_calls, messages, query, memory_id, tool_metas)

            tool_spec = tool_entry["spec"]

            # Create plan step
            step = PlanStep(
                step_id=f"step_{idx}_{name}",
                tool=tool_spec,
                args=args,
                produces={f"result_{idx}"}
            )
            steps.append(step)

        # Execute plan concurrently
        try:
            artifacts = await run_plan(steps, max_retries=3, default_timeout=60.0)

            # Convert results back to messages format
            for idx, call in enumerate(tool_calls):
                result = artifacts.get(f"result_{idx}", "No result")

                messages.append({
                    "type": "function_call",
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": call.arguments,
                })

                messages.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": str(result),
                })

            return messages

        except ExecutionError as e:
            logger.error(f"Plan execution failed: {e}")
            # Add error to messages
            for call in tool_calls:
                messages.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": f"Error: {e}",
                })
            return messages

    def _execute_main_loop(self, messages: List[Dict], query: str, memory_id: str, conversation_id: str) -> str:
        """Enhanced main loop with plan execution support."""
        # ... existing code up to tool_calls ...

        if tool_calls:
            # Check if planning mode is enabled
            if getattr(self, 'enable_planning', False):
                # Use async plan execution
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Already in async context
                    messages = asyncio.create_task(
                        self._execute_plan_mode(tool_calls, messages, query, memory_id, tool_metas)
                    )
                else:
                    # Not in async context, run in new loop
                    messages = loop.run_until_complete(
                        self._execute_plan_mode(tool_calls, messages, query, memory_id, tool_metas)
                    )
            else:
                # Use traditional serial execution
                messages = self._handle_tool_calls(tool_calls, messages, query, memory_id, tool_metas)

            continue

        # ... rest of existing code ...
```

### Step 3: Enable Planning Mode in Constructor

```python
class MemAgent:
    def __init__(
        self,
        # ... existing parameters ...
        enable_planning: bool = False,  # NEW: Enable concurrent plan execution
        **kwargs
    ):
        # ... existing initialization ...

        self.enable_planning = enable_planning

        # If planning mode is enabled, ensure toolbox has specs
        if enable_planning and isinstance(self.tools, Toolbox):
            self._validate_tool_specs()
```

## Usage Example

```python
from memorizz import MemAgent
from memorizz.long_term_memory.procedural.toolbox.toolbox import Toolbox
from memorizz.planner_exec import LatencyClass

# Create toolbox with specs
toolbox = Toolbox()

# Register tools with side-effect declarations
@toolbox.register_tool(
    reads={"database:users"},
    writes={"cache:user_data"},
    idempotent=True,
    latency_class=LatencyClass.MEDIUM,
    timeout=30.0
)
def get_user_data(user_id: str) -> dict:
    """Fetch user data from database and cache it."""
    # implementation
    pass

@toolbox.register_tool(
    reads={"api:weather"},
    idempotent=True,
    concurrency_cap=10,  # Limit concurrent API calls
    latency_class=LatencyClass.SLOW
)
async def get_weather(city: str) -> dict:  # Async tool!
    """Fetch weather data (async)."""
    # implementation
    pass

# Create agent with planning enabled
agent = MemAgent(
    tools=toolbox,
    enable_planning=True,  # Enable concurrent execution!
    instruction="You are a helpful assistant"
)

# Use agent normally - tool calls will be executed concurrently where safe!
response = agent.run("Get user data for user123 and weather for London")
```

## Automatic Dependency Detection

The planner automatically detects dependencies based on:

1. **Data flow**: If tool B consumes what tool A produces, B depends on A
2. **Resource conflicts**: Tools that write to the same resource must run serially
3. **Read-write conflicts**: Writers block readers and vice versa

Example:

```python
# Tool declarations
@tool_spec(writes={"database:users"})
def create_user(name: str): ...

@tool_spec(reads={"database:users"})
def list_users(): ...

@tool_spec(writes={"database:users"})
def update_user(id: str): ...

# Execution plan
steps = [
    PlanStep("create", create_user, args={"name": "Alice"}),
    PlanStep("list", list_users, args={}),      # BLOCKED: waits for create
    PlanStep("update", update_user, args={"id": "1"})  # BLOCKED: waits for list
]

# Planner detects:
# - create writes to database:users
# - list reads from database:users -> depends on create
# - update writes to database:users -> conflicts with list
# Result: All run serially in order
```

But independent tools run in parallel:

```python
@tool_spec(writes={"cache:weather"})
async def get_weather(city: str): ...

@tool_spec(writes={"cache:news"})
async def get_news(topic: str): ...

steps = [
    PlanStep("weather", get_weather, args={"city": "London"}),
    PlanStep("news", get_news, args={"topic": "tech"})
]

# Planner detects:
# - No resource conflicts (different caches)
# - No data dependencies
# Result: Both run in parallel!
```

## Benefits

1. **Automatic Concurrency**: Tools run in parallel where safe
2. **Correctness**: Resource locks prevent race conditions
3. **Async Support**: Mix sync and async tools seamlessly
4. **Retry Logic**: Automatic retries for idempotent operations
5. **Timeouts**: Prevent hung operations
6. **Observability**: Structured execution traces

## Migration Path

1. **Phase 1**: Add `ToolSpec` support to `Toolbox` (backward compatible)
2. **Phase 2**: Add `enable_planning` flag to `MemAgent` (opt-in)
3. **Phase 3**: Update example tools with specs
4. **Phase 4**: Make planning mode the default

## Testing

```python
import asyncio
from memorizz.planner_exec import run_plan, PlanStep, tool_spec

# Test: Parallel execution
@tool_spec(idempotent=True)
async def task_a():
    await asyncio.sleep(1)
    return "A"

@tool_spec(idempotent=True)
async def task_b():
    await asyncio.sleep(1)
    return "B"

steps = [
    PlanStep("a", task_a, produces={"result_a"}),
    PlanStep("b", task_b, produces={"result_b"})
]

# Should complete in ~1s (parallel), not ~2s (serial)
start = time.time()
result = asyncio.run(run_plan(steps))
duration = time.time() - start

assert duration < 1.5, f"Expected <1.5s, got {duration}s"
assert result == {"result_a": "A", "result_b": "B"}
```

## Advanced: LLM-Generated Plans

For even more control, teach the LLM to generate explicit plans:

```python
system_prompt = """
When executing multiple tools, describe your execution plan using this format:

PLAN:
1. Step A: call tool_x(args) -> produces artifact_name
2. Step B: call tool_y(args) -> depends on Step A
3. Step C: call tool_z(args) -> can run in parallel with Step B

Then I will execute the plan optimally with concurrency.
"""

# Parse LLM output to create PlanSteps
# Execute with run_plan()
```

This gives the LLM explicit control over parallelism while ensuring correctness through the executor's resource locks.
