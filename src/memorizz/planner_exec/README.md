# planner_exec: Declarative Task Planning & Concurrent Execution

A complete system for executing tasks with automatic concurrency, dependency resolution, and resource isolation.

## Features

- **Declarative Tool Specifications**: Define what each tool reads/writes
- **Automatic Dependency Resolution**: Build validated DAGs from data-flow
- **Concurrent Execution**: Run independent tasks in parallel
- **Resource Isolation**: Per-resource RW-locks prevent conflicts
- **Async Support**: Mix sync and async tools seamlessly
- **Retry Logic**: Automatic retries for idempotent operations
- **Timeout Handling**: Prevent hung operations
- **Structured Tracing**: Detailed execution logs and metrics

## Quick Start

```python
from memorizz.planner_exec import tool_spec, PlanStep, run_plan, LatencyClass
import asyncio

# Step 1: Define tools with specifications
@tool_spec(
    reads={"database:users"},
    writes={"cache:user_data"},
    idempotent=True,
    latency_class=LatencyClass.MEDIUM,
    timeout=30.0
)
def fetch_users(limit: int) -> list:
    """Fetch users from database."""
    return [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

@tool_spec(
    reads={"cache:user_data"},
    writes={"output_file"},
    idempotent=True
)
async def export_users(users: list, filename: str) -> str:
    """Export users to file (async)."""
    return f"Exported {len(users)} users to {filename}"

# Step 2: Create execution plan
steps = [
    PlanStep(
        step_id="fetch",
        tool=fetch_users,
        args={"limit": 100},
        produces={"users"}
    ),
    PlanStep(
        step_id="export",
        tool=export_users,
        args={
            "users": {"$artifact": "users"},  # Reference previous step's output
            "filename": "users.json"
        },
        consumes={"users"},
        produces={"result"}
    )
]

# Step 3: Execute plan
artifacts = asyncio.run(run_plan(steps))
print(artifacts["result"])  # "Exported 2 users to users.json"
```

## Core Concepts

### ToolSpec: Declarative Tool Contract

Each tool declares:
- **Resources it reads** (e.g., `{"database:users", "api:weather"}`)
- **Resources it writes** (e.g., `{"cache:results", "output_file"}`)
- **Idempotency**: Whether it's safe to retry on failure
- **Concurrency cap**: Maximum concurrent executions
- **Latency class**: Expected execution time
- **Timeout**: Maximum execution time

```python
@tool_spec(
    reads={"api:weather"},
    writes={"cache:weather"},
    idempotent=True,
    concurrency_cap=10,      # Max 10 concurrent calls
    latency_class=LatencyClass.SLOW,
    timeout=5.0
)
async def get_weather(city: str) -> dict:
    """Fetch weather data from API."""
    # Async tools are supported!
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.weather/{city}") as resp:
            return await resp.json()
```

### PlanStep: A Step in the Execution Plan

```python
PlanStep(
    step_id="unique_id",          # Unique identifier
    tool=my_tool_spec,             # ToolSpec to execute
    args={"param": "value"},       # Arguments to pass
    consumes={"input_artifact"},   # Data dependencies
    produces={"output_artifact"},  # What this step produces
    depends_on={"other_step_id"}   # Explicit dependencies (optional)
)
```

**Artifact References**: Use `{"$artifact": "name"}` to reference outputs from previous steps:

```python
args={"data": {"$artifact": "previous_step_output"}}
```

### Planner: Build & Validate DAGs

The planner:
1. Validates all dependencies exist
2. Infers implicit dependencies from data flow
3. Detects cycles and raises errors
4. Analyzes parallelism opportunities

```python
from memorizz.planner_exec import Planner

planner = Planner()
plan = planner.create_plan(steps)

# Analyze plan
analysis = planner.analyze_parallelism(plan)
print(f"Max parallel steps: {analysis['max_parallel']}")
print(f"Execution levels: {analysis['levels']}")
```

### Executor: Run Plans with Concurrency

The executor:
- Runs independent steps in parallel
- Enforces RW-locks per resource
- Respects per-tool concurrency caps
- Retries idempotent tools on failure
- Handles timeouts
- Provides execution traces

```python
from memorizz.planner_exec import Executor

executor = Executor(
    max_retries=3,           # Retry idempotent tools up to 3 times
    default_timeout=300.0,   # 5 minute default timeout
    enable_tracing=True      # Enable execution traces
)

artifacts = await executor.execute_plan(plan, initial_artifacts={"input": "data"})

# Get execution trace
trace = executor.get_trace()
for entry in trace:
    print(f"{entry['step_id']}: {entry['status']} in {entry['duration']:.2f}s")
```

## Resource Management

### Read-Write Locks

The executor uses RW-locks to ensure correctness:

- **Multiple readers**: Can access the same resource concurrently
- **Exclusive writers**: Writers get exclusive access
- **Read-write conflicts**: Readers block writers and vice versa

Example:

```python
# These can run in parallel (both reading)
@tool_spec(reads={"database:users"})
def count_users(): ...

@tool_spec(reads={"database:users"})
def list_users(): ...

# This must wait for all readers (exclusive write)
@tool_spec(writes={"database:users"})
def delete_user(): ...
```

### Concurrency Caps

Limit concurrent executions per tool:

```python
@tool_spec(
    reads={"api:external"},
    concurrency_cap=5  # Max 5 concurrent API calls
)
async def call_api(endpoint: str):
    """Call external API with rate limiting."""
    ...

# Create 20 steps calling this API
steps = [
    PlanStep(f"call_{i}", call_api, args={"endpoint": f"/data/{i}"})
    for i in range(20)
]

# Executor will run max 5 at a time, respecting the cap
```

## Dependency Detection

The planner automatically infers dependencies from:

1. **Data flow**: Steps consuming artifacts produced by other steps
2. **Resource conflicts**: Tools reading/writing the same resources
3. **Explicit dependencies**: Manually declared in `depends_on`

Example:

```python
steps = [
    PlanStep(
        "load_data",
        load_from_db,
        produces={"raw_data"}
    ),
    PlanStep(
        "transform",
        transform_data,
        args={"data": {"$artifact": "raw_data"}},  # Data dependency!
        consumes={"raw_data"},
        produces={"clean_data"}
    ),
    PlanStep(
        "save",
        save_to_file,
        args={"data": {"$artifact": "clean_data"}},
        consumes={"clean_data"}
    )
]

# Planner infers: load_data -> transform -> save
```

## Error Handling & Retries

### Automatic Retries for Idempotent Tools

```python
@tool_spec(
    idempotent=True,  # Safe to retry
    timeout=10.0
)
def fetch_data(url: str):
    """Fetch data from URL (may fail temporarily)."""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# Will retry up to max_retries times on failure
artifacts = await run_plan(steps, max_retries=3)
```

### Timeout Handling

```python
@tool_spec(timeout=5.0)  # Max 5 seconds
async def slow_operation():
    await asyncio.sleep(10)  # Will timeout!

# Raises ExecutionError after timeout
```

## Parallel Execution Patterns

### Pattern 1: Fan-out / Fan-in

```python
# Fan-out: Fetch data from multiple sources in parallel
sources = ["api1", "api2", "api3"]
fetch_steps = [
    PlanStep(f"fetch_{src}", fetch_from_source, args={"source": src}, produces={f"data_{src}"})
    for src in sources
]

# Fan-in: Combine results
combine_step = PlanStep(
    "combine",
    combine_results,
    args={f"data_{src}": {"$artifact": f"data_{src}"} for src in sources},
    consumes={f"data_{src}" for src in sources},
    produces={"combined"}
)

steps = fetch_steps + [combine_step]

# fetch_* run in parallel, combine waits for all
```

### Pattern 2: Pipeline

```python
steps = [
    PlanStep("extract", extract_data, produces={"raw"}),
    PlanStep("transform", transform_data, consumes={"raw"}, produces={"clean"}),
    PlanStep("load", load_data, consumes={"clean"}, produces={"result"})
]

# Runs sequentially: extract -> transform -> load
```

### Pattern 3: Diamond Dependency

```python
steps = [
    PlanStep("fetch", fetch_config, produces={"config"}),

    # These two can run in parallel after fetch
    PlanStep("analyze_users", analyze_users, consumes={"config"}, produces={"user_stats"}),
    PlanStep("analyze_products", analyze_products, consumes={"config"}, produces={"product_stats"}),

    # Final step waits for both
    PlanStep("report", generate_report,
             consumes={"user_stats", "product_stats"},
             produces={"final_report"})
]

# Execution: fetch -> (analyze_users || analyze_products) -> report
```

## Sync vs Async Tools

Mix sync and async tools freely:

```python
# Sync tool
@tool_spec(reads={"file:data.txt"})
def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()

# Async tool
@tool_spec(reads={"api:data"})
async def fetch_api(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()

# Both work in the same plan!
steps = [
    PlanStep("read", read_file, args={"path": "data.txt"}),
    PlanStep("fetch", fetch_api, args={"url": "https://api.com/data"})
]
```

The executor automatically wraps sync tools in `run_in_executor()`.

## Validation & Analysis

Validate plans before execution:

```python
from memorizz.planner_exec import create_and_validate_plan_sync

# Validate without executing
plan = create_and_validate_plan_sync(steps)

# Analyze parallelism
planner = Planner()
analysis = planner.analyze_parallelism(plan)

print(f"Total steps: {analysis['total_steps']}")
print(f"Max parallel: {analysis['max_parallel']}")
print(f"Bottlenecks: {analysis['bottlenecks']}")
```

## Observability

### Execution Traces

```python
executor = Executor(enable_tracing=True)
artifacts = await executor.execute_plan(plan)

# Get detailed trace
trace = executor.get_trace()

for entry in trace:
    print(f"""
    Step: {entry['step_id']}
    Tool: {entry['tool']}
    Status: {entry['status']}
    Duration: {entry['duration']:.2f}s
    Attempt: {entry['attempt']}
    """)
```

### Step Results

```python
results = executor.get_results()

for step_id, result in results.items():
    print(f"{step_id}:")
    print(f"  Status: {result.status.value}")
    print(f"  Duration: {result.duration:.2f}s")
    print(f"  Attempts: {result.attempts}")
    if result.error:
        print(f"  Error: {result.error}")
```

## API Reference

See individual module documentation:
- `tool_spec.py`: Tool specifications and resource management
- `plan_step.py`: Execution plan data structures
- `planner.py`: Plan creation and validation
- `executor.py`: Concurrent plan execution
- `run_plan.py`: High-level execution API

## Examples

See `examples.py` for complete working examples:
- ETL pipeline with sequential dependencies
- Parallel data fetching from multiple sources
- Diamond dependency pattern
- Retry logic for unreliable operations

## Integration with MemAgent

See `INTEGRATION.md` for detailed guide on integrating with the `MemAgent` class to enable concurrent tool execution in AI agents.
