"""
Test file to verify streaming functionality in MemAgent.

This script demonstrates:
1. Regular (non-streaming) agent responses
2. Streaming agent responses
3. Toggling between streaming and non-streaming modes
"""

import asyncio
import os
from src.memorizz.memagent import MemAgent
from src.memorizz.llms.openai import OpenAI


async def test_streaming():
    """Test streaming functionality."""
    # Initialize the OpenAI LLM provider
    llm = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"  # Using mini for faster/cheaper testing
    )

    # Create a MemAgent instance
    agent = MemAgent(
        model=llm,
        instruction="You are a helpful assistant."
    )

    test_query = "Write a short poem about coding in Python."

    print("=" * 80)
    print("TEST 1: Regular (Non-Streaming) Response")
    print("=" * 80)

    # Test regular response
    response = agent.run(test_query)
    print(f"\nComplete Response:\n{response}\n")

    print("=" * 80)
    print("TEST 2: Streaming Response")
    print("=" * 80)

    # Test streaming response
    print("\nStreaming Response:")
    full_response = ""
    async for chunk in agent.run_streaming(test_query):
        print(chunk, end="", flush=True)
        full_response += chunk

    print(f"\n\nFull Streamed Response:\n{full_response}\n")

    print("=" * 80)
    print("TEST 3: Another Streaming Response (Different Query)")
    print("=" * 80)

    # Test with a different query
    query2 = "Explain what async/await means in Python in 2 sentences."
    print(f"\nQuery: {query2}")
    print("\nStreaming Response:")
    async for chunk in agent.run_streaming(query2):
        print(chunk, end="", flush=True)

    print("\n")
    print("=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)


async def test_streaming_direct_llm():
    """Test streaming at the LLM level directly."""
    print("=" * 80)
    print("TEST 4: Direct LLM Streaming (Lower Level)")
    print("=" * 80)

    llm = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )

    prompt = "Count from 1 to 5, one number per line."
    print(f"\nPrompt: {prompt}")
    print("\nStreaming Response:")

    async for chunk in llm.async_stream_text(prompt, instructions="You are a helpful assistant."):
        print(chunk, end="", flush=True)

    print("\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STREAMING FUNCTIONALITY TEST SUITE")
    print("=" * 80 + "\n")

    # Run all tests
    asyncio.run(test_streaming_direct_llm())
    asyncio.run(test_streaming())

    print("\n✓ All streaming tests completed!\n")
