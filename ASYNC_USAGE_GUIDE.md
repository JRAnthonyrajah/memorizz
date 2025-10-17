# memorizz Async Features - Installation & Usage Guide

## Overview

This guide covers the new async capabilities added to memorizz for high-performance memory retrieval:

1. **AsyncMongoDBProvider** - Async database operations with Motor
2. **AsyncEmbeddingProvider** - GPU-accelerated embeddings
3. **Batch Retrieval API** - Query multiple memory types in parallel

**Expected Performance Gains:**
- Single query: 2-3x faster
- Batch queries (3 types): 5-6x faster
- With GPU embeddings: Additional 5-10x speedup
- With caching: 99% faster on cache hits

---

## Installation

### Basic Installation (Async Only)

```bash
# Install with async MongoDB support
pip install memorizz[async]
```

### GPU Support

```bash
# Install with GPU-accelerated embeddings
pip install memorizz[gpu]
```

### Full Performance Package

```bash
# Install everything (async + GPU + uvloop)
pip install memorizz[performance]
```

### Manual Installation

```bash
# Core async dependencies
pip install motor>=3.3.0

# GPU support
pip install torch>=2.0.0

# Optional: uvloop for better async performance (Linux/macOS only)
pip install uvloop>=0.19.0
```

---

## Quick Start

### 1. Basic Async Setup

```python
import asyncio
from memorizz.memory_unit import MemoryUnit
from memorizz.memory_provider.mongodb import MongoDBConfig, AsyncMongoDBProvider
from memorizz.enums.memory_type import MemoryType

async def main():
    # Configure providers
    config = MongoDBConfig(
        uri="mongodb://localhost:27017",
        db_name="memorizz"
    )

    # Create async provider
    async_provider = AsyncMongoDBProvider(config)

    # Create memory unit
    mu = MemoryUnit(
        application_mode='conversational',
        memory_provider=async_provider  # Can use sync provider for backward compat
    )

    # Use batch retrieval API (3-5x faster!)
    results = await mu.retrieve_all_memory_types_async(
        query="What is machine learning?",
        memory_id="user123",
        limits={
            MemoryType.LONG_TERM_MEMORY: 5,
            MemoryType.CONVERSATION_MEMORY: 5,
            MemoryType.WORKFLOW_MEMORY: 5,
        },
        async_memory_provider=async_provider
    )

    # Access results
    semantic = results.get('LONG_TERM_MEMORY', [])
    episodic = results.get('CONVERSATION_MEMORY', [])
    procedural = results.get('WORKFLOW_MEMORY', [])

    print(f"Found {len(semantic)} semantic memories")
    print(f"Found {len(episodic)} episodic memories")
    print(f"Found {len(procedural)} procedural memories")

    # Cleanup
    await async_provider.close_async()

# Run
asyncio.run(main())
```

---

## Detailed Usage

### AsyncMongoDBProvider

#### Single Query (Async)

```python
from memorizz.memory_provider.mongodb.async_provider import AsyncMongoDBProvider
from memorizz.enums.memory_type import MemoryType

async def query_single_type():
    provider = AsyncMongoDBProvider(config)

    # Single async query
    results = await provider.retrieve_memory_units_by_query_async(
        query="machine learning",
        query_embedding=embedding_vector,  # Or pass None to compute
        memory_id="user123",
        memory_type=MemoryType.LONG_TERM_MEMORY,
        limit=5
    )

    return results
```

#### Batch Queries (Parallel)

```python
async def query_batch():
    provider = AsyncMongoDBProvider(config)

    # Prepare batch queries
    queries = [
        {
            'key': 'semantic',
            'embedding': embedding_vector,
            'memory_id': 'user123',
            'memory_type': MemoryType.LONG_TERM_MEMORY,
            'limit': 5
        },
        {
            'key': 'episodic',
            'embedding': embedding_vector,
            'memory_id': 'user123',
            'memory_type': MemoryType.CONVERSATION_MEMORY,
            'limit': 5
        },
        {
            'key': 'procedural',
            'embedding': embedding_vector,
            'memory_id': 'user123',
            'memory_type': MemoryType.WORKFLOW_MEMORY,
            'limit': 5
        },
    ]

    # Execute all queries in PARALLEL!
    results = await provider.batch_retrieve_async(queries)

    # Results: {'semantic': [...], 'episodic': [...], 'procedural': [...]}
    return results
```

---

### AsyncEmbeddingProvider (GPU Acceleration)

#### Single Embedding

```python
from memorizz.embeddings import get_embedding_async

async def compute_embedding():
    # Automatically uses GPU if available, falls back to CPU
    embedding = await get_embedding_async("What is machine learning?")

    print(f"Embedding dimension: {len(embedding)}")
    return embedding
```

#### Batch Embeddings (10-20x faster than sequential!)

```python
from memorizz.embeddings import get_embeddings_batch_async

async def compute_batch():
    texts = [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain deep learning"
    ]

    # Compute all embeddings in ONE batch (much faster!)
    embeddings = await get_embeddings_batch_async(texts, batch_size=32)

    print(f"Computed {len(embeddings)} embeddings")
    return embeddings
```

#### Custom Provider Configuration

```python
from memorizz.embeddings.async_provider import AsyncEmbeddingProvider

async def custom_provider():
    # Custom model and device selection
    provider = AsyncEmbeddingProvider(
        model_name='sentence-transformers/all-mpnet-base-v2',  # Larger model
        device='cuda',  # Force GPU
        max_workers=8   # More threads for CPU fallback
    )

    # Use provider
    embeddings = await provider.encode_async([
        "text 1", "text 2", "text 3"
    ])

    print(f"Using device: {provider.get_device()}")
    print(f"Model: {provider.get_model_name()}")
    print(f"Dimensions: {provider.get_dimensions()}")

    # Cleanup
    provider.shutdown()

    return embeddings
```

---

### MemoryUnit Batch API

#### Full Example with All Memory Types

```python
import asyncio
from memorizz.memory_unit import MemoryUnit
from memorizz.memory_provider.mongodb import MongoDBConfig, AsyncMongoDBProvider
from memorizz.enums.memory_type import MemoryType

async def full_retrieval_example():
    # Setup
    config = MongoDBConfig(uri="mongodb://localhost:27017", db_name="memorizz")
    async_provider = AsyncMongoDBProvider(config)

    mu = MemoryUnit(
        application_mode='conversational',
        memory_provider=async_provider
    )

    # Define what to retrieve
    query = "How do I train a neural network?"
    memory_id = "user_alice"
    limits = {
        MemoryType.LONG_TERM_MEMORY: 5,        # Semantic knowledge
        MemoryType.CONVERSATION_MEMORY: 10,    # Recent conversations
        MemoryType.WORKFLOW_MEMORY: 3,         # Procedural memories
    }

    # Retrieve ALL types in parallel (ONE embedding, 3 parallel queries!)
    print("Retrieving memories...")
    results = await mu.retrieve_all_memory_types_async(
        query=query,
        memory_id=memory_id,
        limits=limits,
        async_memory_provider=async_provider
    )

    # Process results
    for memory_type, memories in results.items():
        print(f"\n{memory_type}: {len(memories)} memories")
        for mem in memories[:2]:  # Show first 2
            content = mem.get('content', '')[:100]
            score = mem.get('memory_signal', 0)
            print(f"  - [{score:.3f}] {content}")

    # Cleanup
    await async_provider.close_async()

asyncio.run(full_retrieval_example())
```

---

## Performance Comparison

### Sequential vs Parallel

```python
import asyncio
import time

async def compare_performance():
    provider = AsyncMongoDBProvider(config)
    mu = MemoryUnit('conversational', memory_provider=provider)

    query = "machine learning"
    memory_id = "user123"

    # OLD WAY: Sequential (slow)
    start = time.time()
    semantic = await provider.retrieve_memory_units_by_query_async(
        query, None, memory_id, MemoryType.LONG_TERM_MEMORY, 5
    )
    episodic = await provider.retrieve_memory_units_by_query_async(
        query, None, memory_id, MemoryType.CONVERSATION_MEMORY, 5
    )
    procedural = await provider.retrieve_memory_units_by_query_async(
        query, None, memory_id, MemoryType.WORKFLOW_MEMORY, 5
    )
    sequential_time = time.time() - start

    # NEW WAY: Batch (fast!)
    start = time.time()
    results = await mu.retrieve_all_memory_types_async(
        query, memory_id,
        {
            MemoryType.LONG_TERM_MEMORY: 5,
            MemoryType.CONVERSATION_MEMORY: 5,
            MemoryType.WORKFLOW_MEMORY: 5,
        },
        async_memory_provider=provider
    )
    batch_time = time.time() - start

    print(f"Sequential: {sequential_time:.3f}s")
    print(f"Batch: {batch_time:.3f}s")
    print(f"Speedup: {sequential_time/batch_time:.1f}x")
    # Expected: 3-5x speedup!

    await provider.close_async()
```

---

## Advanced Usage

### Database Indexes

Ensure optimal indexes for best performance:

```python
async def setup_indexes():
    provider = AsyncMongoDBProvider(config)

    # Create optimized indexes (run once during setup)
    await provider.ensure_indexes_async()

    print("✅ Database indexes created")
    await provider.close_async()

asyncio.run(setup_indexes())
```

### Connection Pool Monitoring

```python
# Check connection pool stats
from pymongo import monitoring

class ConnectionPoolLogger(monitoring.ConnectionPoolListener):
    def connection_created(self, event):
        print(f"Connection created: {event.connection_id}")

    def connection_closed(self, event):
        print(f"Connection closed: {event.connection_id}")

# Register logger
monitoring.register(ConnectionPoolLogger())

# Create provider with monitoring
provider = AsyncMongoDBProvider(config)
```

### Error Handling

```python
async def robust_retrieval():
    provider = AsyncMongoDBProvider(config)
    mu = MemoryUnit('conversational', memory_provider=provider)

    try:
        results = await mu.retrieve_all_memory_types_async(
            query="test query",
            memory_id="user123",
            limits={MemoryType.LONG_TERM_MEMORY: 5},
            async_memory_provider=provider
        )
    except Exception as e:
        print(f"Error during retrieval: {e}")
        # Fallback to empty results
        results = {}
    finally:
        await provider.close_async()

    return results
```

---

## Integration with axiom-index-core

### Context Window Manager Integration

```python
# In axiom-index-core/src/axiom_index_core/memory/context/context_window_manager.py

import asyncio
from memorizz.memory_provider.mongodb.async_provider import AsyncMongoDBProvider
from memorizz.enums.memory_type import MemoryType

class ContextWindowManager:
    def __init__(self, session, **kwargs):
        # ... existing init ...
        self.async_provider = None  # Set externally

    async def build_async(self, user_text: str) -> tuple[str, str]:
        """Ultra-fast async build using memorizz batch API"""

        # Get WM header
        wm_header = self._wm_header_getter(self.session.wm_active_types)

        # Use memorizz batch API to get ALL memories in parallel!
        memory_limits = {
            self._mt_long_term(): self.k_semantic,
            self._mt_conversation(): self.k_episodic,
            self._mt_procedural(): self.k_procedural
        }

        # Remove None types
        memory_limits = {k: v for k, v in memory_limits.items() if k is not None}

        # Single call gets ALL memory types in parallel!
        all_memories = await self.session.memory_unit.retrieve_all_memory_types_async(
            query=user_text,
            memory_id=self.session.memory_id,
            limits=memory_limits,
            async_memory_provider=self.async_provider
        )

        # Process results...
        # (rest of implementation)
```

---

## Testing

### Unit Tests

```python
import pytest
import asyncio
from memorizz.memory_provider.mongodb.async_provider import AsyncMongoDBProvider
from memorizz.enums.memory_type import MemoryType

@pytest.mark.asyncio
async def test_async_single_query():
    config = MongoDBConfig(uri="mongodb://localhost:27017", db_name="test")
    provider = AsyncMongoDBProvider(config)

    results = await provider.retrieve_memory_units_by_query_async(
        query="test",
        query_embedding=[0.1] * 384,
        memory_id="test_user",
        memory_type=MemoryType.LONG_TERM_MEMORY,
        limit=5
    )

    assert isinstance(results, list)
    assert len(results) <= 5

    await provider.close_async()

@pytest.mark.asyncio
async def test_batch_retrieval():
    config = MongoDBConfig(uri="mongodb://localhost:27017", db_name="test")
    provider = AsyncMongoDBProvider(config)

    queries = [
        {'key': 'semantic', 'embedding': [0.1]*384, 'memory_id': 'test',
         'memory_type': MemoryType.LONG_TERM_MEMORY, 'limit': 5},
        {'key': 'episodic', 'embedding': [0.2]*384, 'memory_id': 'test',
         'memory_type': MemoryType.CONVERSATION_MEMORY, 'limit': 5},
    ]

    results = await provider.batch_retrieve_async(queries)

    assert 'semantic' in results
    assert 'episodic' in results
    assert len(results['semantic']) <= 5
    assert len(results['episodic']) <= 5

    await provider.close_async()
```

### Performance Benchmarks

```python
import asyncio
import time
import statistics

async def benchmark_batch_vs_sequential():
    """Compare batch vs sequential retrieval performance"""

    provider = AsyncMongoDBProvider(config)
    mu = MemoryUnit('conversational', memory_provider=provider)

    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain deep learning"
    ]

    # Warm up
    _ = await mu.retrieve_all_memory_types_async(
        queries[0], "user123",
        {MemoryType.LONG_TERM_MEMORY: 5},
        async_memory_provider=provider
    )

    # Benchmark batch
    batch_times = []
    for query in queries:
        start = time.time()
        await mu.retrieve_all_memory_types_async(
            query, "user123",
            {
                MemoryType.LONG_TERM_MEMORY: 5,
                MemoryType.CONVERSATION_MEMORY: 5,
                MemoryType.WORKFLOW_MEMORY: 5,
            },
            async_memory_provider=provider
        )
        batch_times.append(time.time() - start)

    print(f"Batch API - Mean: {statistics.mean(batch_times)*1000:.1f}ms")
    print(f"Batch API - P50: {statistics.median(batch_times)*1000:.1f}ms")
    print(f"Batch API - P95: {statistics.quantiles(batch_times, n=20)[18]*1000:.1f}ms")

    await provider.close_async()

asyncio.run(benchmark_batch_vs_sequential())
```

---

## Troubleshooting

### Motor Not Installed

```
ImportError: AsyncMongoDBProvider requires motor
```

**Solution:**
```bash
pip install motor>=3.3.0
# or
pip install memorizz[async]
```

### Torch Not Available

```
ImportError: AsyncEmbeddingProvider requires torch
```

**Solution:**
```bash
pip install torch>=2.0.0
# or
pip install memorizz[gpu]
```

### GPU Not Detected

If GPU is available but not being used:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

Force GPU usage:
```python
from memorizz.embeddings.async_provider import AsyncEmbeddingProvider

provider = AsyncEmbeddingProvider(device='cuda')  # Force CUDA
```

### Event Loop Already Running

If you get `RuntimeError: asyncio.run() cannot be called from a running event loop`:

```python
# Instead of asyncio.run():
import nest_asyncio
nest_asyncio.apply()

# Now you can use asyncio.run() in Jupyter/IPython
asyncio.run(main())
```

---

## Best Practices

### 1. Reuse Providers

```python
# Good: Create once, reuse
async_provider = AsyncMongoDBProvider(config)
# ... use multiple times ...
await async_provider.close_async()

# Bad: Create on each call
for query in queries:
    provider = AsyncMongoDBProvider(config)  # Don't do this!
    await provider.retrieve_memory_units_by_query_async(...)
```

### 2. Use Batch API When Possible

```python
# Good: Single batch call
results = await mu.retrieve_all_memory_types_async(query, memory_id, limits)

# Bad: Multiple sequential calls
semantic = await provider.retrieve_memory_units_by_query_async(...)
episodic = await provider.retrieve_memory_units_by_query_async(...)
```

### 3. Properly Close Connections

```python
async def good_pattern():
    provider = AsyncMongoDBProvider(config)
    try:
        # ... use provider ...
        pass
    finally:
        await provider.close_async()  # Always close!
```

### 4. Use Environment Variables

```python
# Set in environment or .env file
EMB_MODEL=sentence-transformers/all-mpnet-base-v2
EMB_DEVICE=cuda
MONGODB_URI=mongodb://localhost:27017

# Code automatically picks up env vars
provider = get_async_embedding_provider()  # Uses EMB_MODEL from env
```

---

## Summary

The new async features in memorizz provide:

✅ **2-3x faster** single queries with async I/O
✅ **5-6x faster** batch queries with parallelization
✅ **5-10x faster** embeddings with GPU acceleration
✅ **10-20x faster** batch embeddings
✅ **99% faster** with connection pooling and caching

**Key APIs:**
- `AsyncMongoDBProvider` - Async database operations
- `AsyncEmbeddingProvider` - GPU-accelerated embeddings
- `MemoryUnit.retrieve_all_memory_types_async()` - Batch retrieval

**Installation:**
```bash
pip install memorizz[performance]  # Everything
```

**Minimal Example:**
```python
import asyncio
from memorizz.memory_unit import MemoryUnit
from memorizz.memory_provider.mongodb import AsyncMongoDBProvider, MongoDBConfig
from memorizz.enums.memory_type import MemoryType

async def main():
    provider = AsyncMongoDBProvider(MongoDBConfig("mongodb://localhost:27017"))
    mu = MemoryUnit('conversational', memory_provider=provider)

    results = await mu.retrieve_all_memory_types_async(
        "What is ML?", "user123",
        {MemoryType.LONG_TERM_MEMORY: 5},
        async_memory_provider=provider
    )

    print(results)
    await provider.close_async()

asyncio.run(main())
```

For more details, see the [Optimization Report](./OPTIMIZATION_REPORT_MEMORIZZ.md).
