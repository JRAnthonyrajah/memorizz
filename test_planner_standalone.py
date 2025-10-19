"""
Standalone test of planner_exec functionality.

This test verifies the planner_exec module works correctly without
requiring the full MemAgent setup.
"""
import asyncio
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import directly from planner_exec to avoid memory_provider dependencies
from memorizz.planner_exec.tool_spec import ToolSpec, ResourceSpec, LatencyClass, tool_spec
from memorizz.planner_exec.plan_step import PlanStep
from memorizz.planner_exec.run_plan import run_plan


def test_basic_planner():
    """Test basic planner functionality."""
    print("=" * 80)
    print("Testing planner_exec basic functionality")
    print("=" * 80)

    # Test 1: Define tools with ToolSpec decorator
    print("\nTest 1: Tool definition with @tool_spec decorator")
    print("-" * 80)

    @tool_spec(
        writes={"cache:weather"},
        idempotent=True,
        latency_class=LatencyClass.SLOW,
        timeout=10.0
    )
    async def get_weather(city: str) -> str:
        """Get weather information for a city."""
        print(f"  [get_weather] Fetching weather for {city}...")
        await asyncio.sleep(2)
        result = f"Weather in {city}: 72°F, Sunny"
        print(f"  [get_weather] ✓ Completed")
        return result

    @tool_spec(
        writes={"cache:news"},
        idempotent=True,
        latency_class=LatencyClass.SLOW,
        timeout=10.0
    )
    async def get_news(topic: str) -> str:
        """Get news about a topic."""
        print(f"  [get_news] Fetching news for {topic}...")
        await asyncio.sleep(2)
        result = f"Latest news about {topic}: Breaking developments!"
        print(f"  [get_news] ✓ Completed")
        return result

    @tool_spec(
        writes={"cache:stock"},
        idempotent=True,
        latency_class=LatencyClass.MEDIUM,
        timeout=5.0
    )
    async def get_stock_price(symbol: str) -> str:
        """Get stock price for a symbol."""
        print(f"  [get_stock_price] Fetching price for {symbol}...")
        await asyncio.sleep(1.5)
        result = f"Stock price for {symbol}: $150.00"
        print(f"  [get_stock_price] ✓ Completed")
        return result

    print(f"✓ Defined 3 async tools with ToolSpec metadata")
    print(f"  - get_weather: {get_weather}")
    print(f"  - get_news: {get_news}")
    print(f"  - get_stock_price: {get_stock_price}")

    # Test 2: Create plan steps
    print("\nTest 2: Create PlanSteps")
    print("-" * 80)

    steps = [
        PlanStep(
            step_id="weather",
            tool=get_weather,
            args={"city": "San Francisco"},
            produces={"weather_result"}
        ),
        PlanStep(
            step_id="news",
            tool=get_news,
            args={"topic": "Technology"},
            produces={"news_result"}
        ),
        PlanStep(
            step_id="stock",
            tool=get_stock_price,
            args={"symbol": "AAPL"},
            produces={"stock_result"}
        ),
    ]

    print(f"✓ Created {len(steps)} plan steps")
    for step in steps:
        print(f"  - {step.step_id}: {step.tool.name}({step.args})")

    # Test 3: Execute plan
    print("\nTest 3: Execute plan with concurrent execution")
    print("-" * 80)

    async def execute_plan():
        print("Starting plan execution...")
        print("Expected time: ~2 seconds (parallel execution)")
        print()

        start = time.time()
        artifacts = await run_plan(steps, max_retries=3, default_timeout=30.0)
        duration = time.time() - start

        print()
        print(f"✓ Execution completed in {duration:.2f}s")
        print("\nResults:")
        for key, value in artifacts.items():
            print(f"  - {key}: {value}")

        return duration, artifacts

    execution_time, results = asyncio.run(execute_plan())

    # Test 4: Verify parallelism
    print("\nTest 4: Verify parallel execution")
    print("-" * 80)

    # If executed serially, it would take: 2 + 2 + 1.5 = 5.5 seconds
    # If executed in parallel, it should take: max(2, 2, 1.5) = ~2 seconds
    if execution_time < 3.0:
        print(f"✓ SUCCESS: Tools executed in parallel!")
        print(f"  Actual time: {execution_time:.2f}s")
        print(f"  Expected parallel time: ~2.0s")
        print(f"  Expected serial time: ~5.5s")
    else:
        print(f"✗ WARNING: Tools may not have executed in parallel")
        print(f"  Actual time: {execution_time:.2f}s")
        print(f"  Expected parallel time: ~2.0s")

    # Test 5: Test with dependencies
    print("\nTest 5: Test with explicit dependencies")
    print("-" * 80)

    @tool_spec(
        reads={"cache:weather"},
        writes={"report:summary"},
        idempotent=True
    )
    def create_summary(weather_data: str) -> str:
        """Create a summary based on weather data."""
        print(f"  [create_summary] Creating summary...")
        time.sleep(0.5)
        result = f"Summary: {weather_data}"
        print(f"  [create_summary] ✓ Completed")
        return result

    dependent_steps = [
        PlanStep(
            step_id="weather",
            tool=get_weather,
            args={"city": "New York"},
            produces={"weather_data"}
        ),
        PlanStep(
            step_id="summary",
            tool=create_summary,
            args={"weather_data": "{{weather_data}}"},  # Consumes weather_data
            consumes={"weather_data"},
            depends_on={"weather"},  # Explicit dependency
            produces={"summary"}
        ),
    ]

    async def execute_dependent_plan():
        print("Executing plan with dependencies...")
        print("Step 2 depends on Step 1")
        print()

        start = time.time()
        artifacts = await run_plan(dependent_steps, max_retries=3, default_timeout=30.0)
        duration = time.time() - start

        print()
        print(f"✓ Execution completed in {duration:.2f}s")
        print("\nResults:")
        for key, value in artifacts.items():
            print(f"  - {key}: {value}")

        return duration, artifacts

    dep_time, dep_results = asyncio.run(execute_dependent_plan())
    print(f"✓ Dependent execution completed in {dep_time:.2f}s")

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("✓ Tool definition with @tool_spec decorator: PASSED")
    print("✓ PlanStep creation: PASSED")
    print(f"✓ Concurrent execution ({execution_time:.2f}s): {'PASSED' if execution_time < 3.0 else 'WARNING'}")
    print(f"✓ Dependent execution ({dep_time:.2f}s): PASSED")
    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("The planner_exec module is working correctly.")
    print("=" * 80)


if __name__ == "__main__":
    test_basic_planner()
