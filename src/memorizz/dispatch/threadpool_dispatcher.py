# memorizz/src/memorizz/dispatch/threadpool_dispatcher.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Iterable
from concurrent.futures import ThreadPoolExecutor
from .task_dispatcher import TaskDispatcher
from ..embeddings import get_embedding

# A small shared pool avoids per-call executor churn.
_SHARED_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="memorizz-bg")

class ThreadPoolDispatcher(TaskDispatcher):
    def __init__(self, pool: ThreadPoolExecutor = _SHARED_POOL):
        self.pool = pool

    def enqueue_memory_write(self, doc: Dict[str, Any], mem_type: Optional[str] = None) -> None:
        # import lazily to avoid circulars
        from ..memory_provider import MemoryProvider
        def _do():
            MemoryProvider.store(doc, mem_type=mem_type)
        self.pool.submit(_do)

    def enqueue_memory_bulk_write(self, docs: Iterable[Dict[str, Any]], mem_type: Optional[str] = None) -> None:
        from ..memory_provider import MemoryProvider
        def _do():
            MemoryProvider.store_many(list(docs), mem_type=mem_type)
        self.pool.submit(_do)

    def enqueue_embedding_compute(self, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        def _do():
            vec = get_embedding(content)
            # You may want to write it somewhere, keyed by meta["id"], etc.
            # No-op by default.
            return vec
        self.pool.submit(_do)

    def compute_embedding_blocking(self, content: str, meta: Optional[Dict[str, Any]] = None) -> List[float]:
        return get_embedding(content)

    def enqueue_importance_compute(self, items: List[Dict[str, Any]], query: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> None:
        # Leave unimplemented here; Celery impl will handle.
        pass

    def enqueue_metric(self, event: Dict[str, Any]) -> None:
        # Replace with your metrics sink if desired.
        pass
