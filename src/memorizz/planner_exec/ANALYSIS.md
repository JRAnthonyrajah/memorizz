# MemAgent Analysis & planner_exec Solution

## Executive Summary

The current `MemAgent` implementation executes tool calls **serially** without any concurrency control or dependency tracking. The new `planner_exec` system provides a complete solution for concurrent task execution with proper resource isolation and automatic dependency resolution.

## Current MemAgent Issues

### Location: `memagent.py:1313-1445`

**Critical Problems:**

1. **Serial Execution** (`_handle_tool_calls:1313`)
   ```python
   for call in tool_calls:
       # Execute each tool one at a time - NO CONCURRENCY!
       result, error_message, workflow_outcome = self._execute_single_tool(...)
   ```
   - All tools block each other, even when completely independent
   - No parallelism opportunities are exploited

2. **No Resource Tracking** (`_execute_single_tool:1387`)
   ```python
   result = fn(**args)  # Direct synchronous call
   ```
   - No tracking of what resources each tool reads/writes
   - Race conditions possible if tools conflict
   - No isolation guarantees

3. **No Async Support**
   - All tools must be synchronous
   - Cannot leverage async I/O for network calls, DB queries, etc.

4. **No Retry Logic**
   - Single execution attempt for all tools
   - Transient failures cause complete workflow failure
   - No idempotency tracking

5. **No Timeout Protection**
   - Tools can hang indefinitely
   - No way to enforce SLAs or prevent resource exhaustion

## planner_exec Solution

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         User Request                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   List of PlanSteps    │  ← Declarative intent
         └────────────┬───────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │       Planner          │  ← Validates & builds DAG
         │  - Infer dependencies  │
         │  - Detect cycles       │
         │  - Analyze parallelism │
         └────────────┬───────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │    ExecutionPlan       │  ← Validated DAG
         └────────────┬───────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │      Executor          │  ← Concurrent execution
         │  - RW locks per resource│
         │  - Semaphores per tool │
         │  - Timeout handling    │
         │  - Retry logic         │
         │  - Async support       │
         └────────────┬───────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │      Artifacts         │  ← Results dict
         └────────────────────────┘
```

### Core Components

#### 1. ToolSpec (tool_spec.py)

Declarative contract for each tool:

```python
@tool_spec(
    reads={"database:users"},        # Resources read
    writes={"cache:user_data"},      # Resources written
    idempotent=True,                 # Safe to retry
    concurrency_cap=10,              # Max concurrent executions
    latency_class=LatencyClass.MEDIUM,  # Expected latency
    timeout=30.0                     # Max execution time
)
async def fetch_users(limit: int) -> list:
    """Fetch users from database (async tool)."""
    ...
```

**Key Features:**
- Declares side-effects (reads/writes)
- Supports both sync and async functions
- Enables automatic conflict detection
- Allows per-tool concurrency limits

#### 2. PlanStep (plan_step.py)

A step in the execution plan:

```python
PlanStep(
    step_id="fetch_users",
    tool=fetch_users,                    # ToolSpec reference
    args={"limit": 100},                 # Arguments
    consumes={"config"},                 # Data dependencies
    produces={"user_list"},              # Output artifacts
    depends_on={"load_config"}           # Explicit dependencies
)
```

**Data Flow:**
- `consumes`: Artifacts this step needs
- `produces`: Artifacts this step creates
- Automatic dependency inference from data flow

#### 3. Planner (planner.py)

Validates and builds execution DAG:

```python
planner = Planner()
plan = planner.create_plan(steps)

# Automatically:
# 1. Infers implicit dependencies from data flow
# 2. Validates no cycles exist
# 3. Checks all dependencies are satisfied
# 4. Analyzes parallelism opportunities
```

**Validation:**
- ✓ Unique step IDs
- ✓ All consumed artifacts are produced
- ✓ No circular dependencies
- ✓ Valid dependency references

#### 4. Executor (executor.py)

Executes plans with concurrency:

```python
executor = Executor(
    max_retries=3,
    default_timeout=300.0,
    enable_tracing=True
)

artifacts = await executor.execute_plan(plan)
```

**Concurrency Control:**
- **RW Locks**: Multiple readers OR single writer per resource
- **Semaphores**: Per-tool concurrency caps
- **Wave Execution**: Maximal parallelism per dependency level
- **Async Support**: Mix sync and async tools

**Reliability:**
- Automatic retries for idempotent tools
- Timeout enforcement
- Structured error handling
- Execution traces

#### 5. run_plan() (run_plan.py)

High-level orchestration:

```python
artifacts = await run_plan(
    steps,
    initial_artifacts={"input": "data"},
    max_retries=3,
    default_timeout=300.0
)
```

## How It Solves MemAgent Issues

### Problem 1: Serial Execution → **Concurrent Execution**

**Before:**
```python
for call in tool_calls:
    result = execute_single_tool(call)  # Serial
```

**After:**
```python
# Executor automatically runs independent steps in parallel
steps = [
    PlanStep("weather", get_weather, args={"city": "London"}),
    PlanStep("news", get_news, args={"topic": "tech"}),
    PlanStep("stocks", get_stocks, args={"symbol": "AAPL"})
]

# All three run concurrently if no resource conflicts!
artifacts = await run_plan(steps)
```

### Problem 2: No Resource Tracking → **RW Locks**

**Before:**
```python
# No tracking - race conditions possible
fn1()  # writes to database:users
fn2()  # reads from database:users - may see partial writes!
```

**After:**
```python
@tool_spec(writes={"database:users"})
def create_user(): ...

@tool_spec(reads={"database:users"})
def list_users(): ...

# Executor enforces: list_users waits for create_user to finish
# No race conditions - guaranteed isolation!
```

### Problem 3: No Async Support → **Full Async Support**

**Before:**
```python
# All tools must be sync
def fetch_api(url):
    return requests.get(url).json()  # Blocks thread
```

**After:**
```python
@tool_spec(idempotent=True, concurrency_cap=10)
async def fetch_api(url: str) -> dict:
    # True async I/O - non-blocking
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()

# Executor handles both sync and async seamlessly
```

### Problem 4: No Retry Logic → **Automatic Retries**

**Before:**
```python
result = fn(**args)  # Single attempt - fail fast
```

**After:**
```python
@tool_spec(idempotent=True)  # Mark as safe to retry
def fetch_data(url):
    # May fail transiently
    return requests.get(url).json()

# Executor retries up to max_retries times with exponential backoff
artifacts = await run_plan(steps, max_retries=3)
```

### Problem 5: No Timeouts → **Timeout Protection**

**Before:**
```python
result = fn(**args)  # Can hang forever
```

**After:**
```python
@tool_spec(timeout=10.0)  # Max 10 seconds
def slow_operation():
    # Will be terminated if exceeds timeout
    ...

# Executor enforces timeout and raises ExecutionError
```

## Dependency Detection Examples

### Example 1: Data Flow Dependencies

```python
steps = [
    PlanStep("fetch", fetch_data, produces={"raw_data"}),
    PlanStep("clean", clean_data,
             args={"data": {"$artifact": "raw_data"}},
             consumes={"raw_data"},
             produces={"clean_data"}),
    PlanStep("save", save_data,
             args={"data": {"$artifact": "clean_data"}},
             consumes={"clean_data"})
]

# Planner infers: fetch → clean → save (serial execution)
```

### Example 2: Resource Conflicts

```python
@tool_spec(writes={"database:users"})
def create_users(): ...

@tool_spec(writes={"database:users"})  # Same resource!
def delete_users(): ...

# Executor enforces: These CANNOT run concurrently
# Must run serially to prevent conflicts
```

### Example 3: No Conflicts (Parallel)

```python
@tool_spec(writes={"cache:weather"})
async def get_weather(): ...

@tool_spec(writes={"cache:news"})  # Different resource
async def get_news(): ...

@tool_spec(writes={"cache:stocks"})  # Different resource
async def get_stocks(): ...

# Executor runs ALL THREE in parallel!
# No data dependencies, no resource conflicts
```

### Example 4: Diamond Pattern

```python
steps = [
    PlanStep("config", load_config, produces={"cfg"}),

    # These two can run in parallel after config loads
    PlanStep("users", analyze_users,
             consumes={"cfg"}, produces={"user_stats"}),
    PlanStep("products", analyze_products,
             consumes={"cfg"}, produces={"product_stats"}),

    # Final step waits for both
    PlanStep("report", generate_report,
             consumes={"user_stats", "product_stats"},
             produces={"report"})
]

# Execution: config → (users || products) → report
# Level 0: config
# Level 1: users, products (parallel)
# Level 2: report
```

## Benefits Summary

| Feature | Current MemAgent | planner_exec |
|---------|-----------------|--------------|
| Concurrency | ❌ Serial only | ✅ Automatic parallelism |
| Resource Safety | ❌ No tracking | ✅ RW locks |
| Async Support | ❌ Sync only | ✅ Mix sync/async |
| Retry Logic | ❌ None | ✅ Automatic for idempotent |
| Timeouts | ❌ None | ✅ Per-tool timeouts |
| Dependency Tracking | ❌ None | ✅ Automatic inference |
| Cycle Detection | ❌ None | ✅ Validation |
| Observability | ⚠️ Basic logs | ✅ Structured traces |
| Error Handling | ⚠️ Basic | ✅ Comprehensive |

## Integration Path

See `INTEGRATION.md` for detailed integration guide with `MemAgent`.

**Quick Overview:**

1. **Add ToolSpec to Toolbox**: Store side-effect declarations
2. **Add Planning Mode**: `enable_planning=True` in MemAgent
3. **Convert tool_calls to PlanSteps**: Map LLM tool calls to steps
4. **Execute with run_plan()**: Replace serial loop with executor

**Backward Compatible**: Existing code continues to work, planning is opt-in.

## Files Delivered

```
src/memorizz/planner_exec/
├── __init__.py           # Package exports
├── tool_spec.py          # ToolSpec, ResourceSpec, decorators
├── plan_step.py          # PlanStep, ExecutionPlan, DAG structures
├── planner.py            # Planner, validation, analysis
├── executor.py           # Executor, RW locks, async execution
├── run_plan.py           # High-level run_plan() API
├── examples.py           # Complete working examples
├── README.md             # Full documentation
├── INTEGRATION.md        # MemAgent integration guide
└── ANALYSIS.md           # This file
```

## Correctness Guarantees

The system ensures correctness through:

1. **Isolation**: RW locks prevent concurrent conflicting access
2. **Ordering**: DAG ensures dependencies run before dependents
3. **Atomicity**: Each step executes fully or fails (no partial state)
4. **Idempotency**: Retry-safe operations marked explicitly
5. **Validation**: Pre-execution checks prevent invalid plans

**Even with all async tools, correctness is enforced by the scheduler, not by await alone.**
