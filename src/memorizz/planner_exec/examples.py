"""
Examples demonstrating the planner_exec system.

This module contains practical examples showing how to use the task
planning and execution system with various scenarios.
"""
import asyncio
import time
from typing import List, Dict, Any

from .tool_spec import ToolSpec, tool_spec, LatencyClass
from .plan_step import PlanStep
from .run_plan import run_plan, run_plan_sync


# =============================================================================
# Example 1: Basic sequential workflow with data dependencies
# =============================================================================

@tool_spec(
    reads={"input_file"},
    writes={"database:raw_data"},
    idempotent=True,
    latency_class=LatencyClass.SLOW,
    timeout=30.0
)
def extract_data(filename: str) -> Dict[str, Any]:
    """Extract data from a file."""
    print(f"Extracting data from {filename}...")
    time.sleep(0.1)  # Simulate I/O
    return {"records": [{"id": 1, "value": 100}, {"id": 2, "value": 200}]}


@tool_spec(
    reads={"database:raw_data"},
    writes={"database:processed_data"},
    idempotent=True,
    latency_class=LatencyClass.MEDIUM
)
def transform_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform raw data."""
    print("Transforming data...")
    time.sleep(0.05)
    records = data["records"]
    return {"records": [{"id": r["id"], "value": r["value"] * 2} for r in records]}


@tool_spec(
    reads={"database:processed_data"},
    writes={"output_file"},
    idempotent=True,
    latency_class=LatencyClass.MEDIUM
)
def load_data(data: Dict[str, Any], output_path: str) -> str:
    """Load processed data to output."""
    print(f"Loading data to {output_path}...")
    time.sleep(0.05)
    return f"Loaded {len(data['records'])} records to {output_path}"


def example_etl_pipeline():
    """Example: Classic ETL pipeline with sequential dependencies."""
    print("\n=== Example 1: ETL Pipeline ===")

    steps = [
        PlanStep(
            step_id="extract",
            tool=extract_data,
            args={"filename": "input.csv"},
            produces={"raw_data"}
        ),
        PlanStep(
            step_id="transform",
            tool=transform_data,
            args={"data": {"$artifact": "raw_data"}},
            consumes={"raw_data"},
            produces={"processed_data"}
        ),
        PlanStep(
            step_id="load",
            tool=load_data,
            args={
                "data": {"$artifact": "processed_data"},
                "output_path": "output.csv"
            },
            consumes={"processed_data"},
            produces={"result"}
        )
    ]

    result = run_plan_sync(steps)
    print(f"Result: {result['result']}")


# =============================================================================
# Example 2: Parallel execution with no dependencies
# =============================================================================

@tool_spec(
    reads={"api:weather"},
    idempotent=True,
    latency_class=LatencyClass.SLOW,
    concurrency_cap=5  # Limit concurrent API calls
)
async def fetch_weather(city: str) -> Dict[str, Any]:
    """Fetch weather data for a city (async)."""
    print(f"Fetching weather for {city}...")
    await asyncio.sleep(0.2)  # Simulate API call
    return {"city": city, "temp": 20 + hash(city) % 15}


@tool_spec(
    reads={"api:news"},
    idempotent=True,
    latency_class=LatencyClass.SLOW,
    concurrency_cap=5
)
async def fetch_news(topic: str) -> List[str]:
    """Fetch news for a topic (async)."""
    print(f"Fetching news for {topic}...")
    await asyncio.sleep(0.15)
    return [f"News article about {topic} #{i}" for i in range(3)]


async def example_parallel_fetching():
    """Example: Fetch data from multiple sources in parallel."""
    print("\n=== Example 2: Parallel Data Fetching ===")

    cities = ["London", "Paris", "Tokyo", "New York"]
    topics = ["technology", "sports", "science"]

    steps = []

    # Add weather fetching steps (can run in parallel)
    for city in cities:
        steps.append(
            PlanStep(
                step_id=f"weather_{city}",
                tool=fetch_weather,
                args={"city": city},
                produces={f"weather_{city}"}
            )
        )

    # Add news fetching steps (can run in parallel)
    for topic in topics:
        steps.append(
            PlanStep(
                step_id=f"news_{topic}",
                tool=fetch_news,
                args={"topic": topic},
                produces={f"news_{topic}"}
            )
        )

    start = time.time()
    result = await run_plan(steps)
    duration = time.time() - start

    print(f"\nFetched {len(result)} items in {duration:.2f}s")
    print(f"Expected ~0.2s with parallelism (vs ~{0.2 * len(steps):.1f}s sequential)")


# =============================================================================
# Example 3: Diamond dependency pattern with resource conflicts
# =============================================================================

@tool_spec(
    reads=set(),
    writes={"database:config"},
    idempotent=True
)
def load_config() -> Dict[str, Any]:
    """Load configuration."""
    print("Loading config...")
    time.sleep(0.05)
    return {"threshold": 100, "batch_size": 10}


@tool_spec(
    reads={"database:config", "database:users"},
    writes={"cache:user_stats"},
    idempotent=False  # Not idempotent - has side effects
)
def compute_user_stats(config: Dict[str, Any]) -> Dict[str, int]:
    """Compute user statistics."""
    print("Computing user stats...")
    time.sleep(0.1)
    return {"total_users": 1000, "active_users": 750}


@tool_spec(
    reads={"database:config", "database:products"},
    writes={"cache:product_stats"},
    idempotent=False
)
def compute_product_stats(config: Dict[str, Any]) -> Dict[str, int]:
    """Compute product statistics."""
    print("Computing product stats...")
    time.sleep(0.1)
    return {"total_products": 500, "in_stock": 400}


@tool_spec(
    reads={"cache:user_stats", "cache:product_stats"},
    writes={"output_file"},
    idempotent=True
)
def generate_report(
    user_stats: Dict[str, int],
    product_stats: Dict[str, int]
) -> str:
    """Generate final report combining all stats."""
    print("Generating report...")
    time.sleep(0.05)
    return (
        f"Report: {user_stats['active_users']} active users, "
        f"{product_stats['in_stock']} products in stock"
    )


def example_diamond_dependency():
    """
    Example: Diamond dependency pattern.

           load_config
            /        \
    user_stats    product_stats
            \        /
          generate_report

    user_stats and product_stats can run in parallel after load_config.
    """
    print("\n=== Example 3: Diamond Dependency ===")

    steps = [
        PlanStep(
            step_id="config",
            tool=load_config,
            args={},
            produces={"config"}
        ),
        PlanStep(
            step_id="user_stats",
            tool=compute_user_stats,
            args={"config": {"$artifact": "config"}},
            consumes={"config"},
            produces={"user_stats"}
        ),
        PlanStep(
            step_id="product_stats",
            tool=compute_product_stats,
            args={"config": {"$artifact": "config"}},
            consumes={"config"},
            produces={"product_stats"}
        ),
        PlanStep(
            step_id="report",
            tool=generate_report,
            args={
                "user_stats": {"$artifact": "user_stats"},
                "product_stats": {"$artifact": "product_stats"}
            },
            consumes={"user_stats", "product_stats"},
            produces={"final_report"}
        )
    ]

    start = time.time()
    result = run_plan_sync(steps)
    duration = time.time() - start

    print(f"\n{result['final_report']}")
    print(f"Completed in {duration:.2f}s")
    print("Note: user_stats and product_stats ran in parallel after config loaded")


# =============================================================================
# Example 4: Retry logic for idempotent operations
# =============================================================================

# Global counter to simulate intermittent failures
_attempt_counter = 0


@tool_spec(
    reads={"api:external"},
    writes={"cache:api_result"},
    idempotent=True,  # Safe to retry
    latency_class=LatencyClass.VERY_SLOW,
    timeout=5.0
)
def unreliable_api_call(endpoint: str) -> Dict[str, Any]:
    """Simulate an unreliable API that sometimes fails."""
    global _attempt_counter
    _attempt_counter += 1

    print(f"Calling API {endpoint} (attempt {_attempt_counter})...")

    # Fail on first 2 attempts, succeed on 3rd
    if _attempt_counter < 3:
        raise Exception(f"API temporarily unavailable (attempt {_attempt_counter})")

    return {"status": "success", "data": "API response"}


def example_retry_logic():
    """Example: Automatic retry for idempotent operations."""
    print("\n=== Example 4: Retry Logic ===")

    global _attempt_counter
    _attempt_counter = 0  # Reset counter

    steps = [
        PlanStep(
            step_id="api_call",
            tool=unreliable_api_call,
            args={"endpoint": "/data"},
            produces={"api_result"}
        )
    ]

    try:
        result = run_plan_sync(steps, max_retries=3)
        print(f"Success after retries: {result['api_result']}")
    except Exception as e:
        print(f"Failed even with retries: {e}")


# =============================================================================
# Main: Run all examples
# =============================================================================

def run_all_examples():
    """Run all examples."""
    example_etl_pipeline()
    asyncio.run(example_parallel_fetching())
    example_diamond_dependency()
    example_retry_logic()


if __name__ == "__main__":
    run_all_examples()
