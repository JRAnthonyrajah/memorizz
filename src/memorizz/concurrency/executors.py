# src/memorizz/concurrency/executors.py
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import asyncio
from typing import Callable, Any

# One shared pool for all “I/O-ish” blocking functions (Mongo, file, light CPU)
# Tune max_workers to your node; 8–16 is a good start.
_SHARED_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="memz-io")

# Global semaphore to bound fan-out in app-level tasks
# e.g., limit # of simultaneous embeddings or rerank batches
_IO_CONCURRENCY = asyncio.Semaphore(16)

async def run_io(func: Callable[..., Any], *args, **kwargs) -> Any:
    async with _IO_CONCURRENCY:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_SHARED_POOL, lambda: func(*args, **kwargs))

def shared_pool() -> ThreadPoolExecutor:
    return _SHARED_POOL

def io_semaphore() -> asyncio.Semaphore:
    return _IO_CONCURRENCY
