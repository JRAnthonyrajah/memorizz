"""
Usage Examples: Streaming vs Non-Streaming Agent Responses

This file shows how to toggle between streaming and non-streaming modes
in your MemAgent applications.
"""

import asyncio
from src.memorizz.memagent import MemAgent
from src.memorizz.llms.openai import OpenAI


# Example 1: Basic Non-Streaming (Default)
def example_non_streaming():
    """Traditional blocking response - gets the complete answer at once."""
    agent = MemAgent(
        model=OpenAI(model="gpt-4o-mini"),
        instruction="You are a helpful assistant."
    )

    # This returns the complete response as a string
    response = agent.run("What is Python?")
    print(response)


# Example 2: Basic Streaming
async def example_streaming():
    """Streaming response - get chunks as they're generated."""
    agent = MemAgent(
        model=OpenAI(model="gpt-4o-mini"),
        instruction="You are a helpful assistant."
    )

    # This yields chunks as they arrive from the LLM
    async for chunk in agent.run_streaming("What is Python?"):
        print(chunk, end="", flush=True)
    print()  # New line at the end


# Example 3: Conditional Streaming Based on User Preference
async def example_conditional_streaming(use_streaming: bool = False):
    """Toggle streaming on/off based on a parameter."""
    agent = MemAgent(
        model=OpenAI(model="gpt-4o-mini"),
        instruction="You are a helpful assistant."
    )

    query = "Explain async programming in Python."

    if use_streaming:
        # Stream the response
        print("Streaming mode enabled:")
        async for chunk in agent.run_streaming(query):
            print(chunk, end="", flush=True)
        print()
    else:
        # Get complete response
        print("Regular mode enabled:")
        response = agent.run(query)
        print(response)


# Example 4: Collecting Streamed Chunks
async def example_collect_streamed_chunks():
    """Collect all chunks from streaming for further processing."""
    agent = MemAgent(
        model=OpenAI(model="gpt-4o-mini"),
        instruction="You are a helpful assistant."
    )

    chunks = []
    async for chunk in agent.run_streaming("List 3 benefits of streaming."):
        chunks.append(chunk)
        print(chunk, end="", flush=True)

    print(f"\n\nTotal chunks received: {len(chunks)}")
    print(f"Complete response: {''.join(chunks)}")


# Example 5: Real-Time UI Updates (Simulated)
async def example_realtime_ui_updates():
    """Simulate updating a UI in real-time with streamed chunks."""
    agent = MemAgent(
        model=OpenAI(model="gpt-4o-mini"),
        instruction="You are a helpful assistant."
    )

    print("Simulating real-time UI updates:")
    print("-" * 50)

    ui_buffer = ""
    async for chunk in agent.run_streaming("Write a haiku about AI."):
        ui_buffer += chunk
        # Simulate updating a UI element (e.g., a text box)
        print(f"\rCurrent UI: {ui_buffer}", end="", flush=True)
        await asyncio.sleep(0.01)  # Simulate processing time

    print(f"\n\nFinal UI: {ui_buffer}")


# Example 6: Wrapper Function to Choose Mode Dynamically
async def get_agent_response(query: str, streaming: bool = False):
    """
    Universal function that returns responses in either mode.

    Args:
        query: The user's question
        streaming: If True, returns an async generator; if False, returns a string
    """
    agent = MemAgent(
        model=OpenAI(model="gpt-4o-mini"),
        instruction="You are a helpful assistant."
    )

    if streaming:
        # Return the async generator for streaming
        return agent.run_streaming(query)
    else:
        # Return the complete response
        return agent.run(query)


# Example 7: Using the Wrapper
async def example_using_wrapper():
    """Demonstrate using the wrapper function."""
    print("Non-streaming response:")
    response = await get_agent_response("What is 2+2?", streaming=False)
    print(response)

    print("\nStreaming response:")
    stream = await get_agent_response("What is 3+3?", streaming=True)
    async for chunk in stream:
        print(chunk, end="", flush=True)
    print()


# Run examples
if __name__ == "__main__":
    print("=" * 60)
    print("STREAMING TOGGLE EXAMPLES")
    print("=" * 60)

    # Example 1: Non-streaming
    print("\n1. Non-Streaming Example:")
    example_non_streaming()

    # Example 2: Streaming
    print("\n2. Streaming Example:")
    asyncio.run(example_streaming())

    # Example 3: Conditional (non-streaming)
    print("\n3a. Conditional Example (non-streaming):")
    asyncio.run(example_conditional_streaming(use_streaming=False))

    print("\n3b. Conditional Example (streaming):")
    asyncio.run(example_conditional_streaming(use_streaming=True))

    # Example 4: Collecting chunks
    print("\n4. Collecting Chunks Example:")
    asyncio.run(example_collect_streamed_chunks())

    # Example 5: Real-time updates
    print("\n5. Real-Time UI Updates Example:")
    asyncio.run(example_realtime_ui_updates())

    # Example 7: Wrapper usage
    print("\n6. Wrapper Function Example:")
    asyncio.run(example_using_wrapper())

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
