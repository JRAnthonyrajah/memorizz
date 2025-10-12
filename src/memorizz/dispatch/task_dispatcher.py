# memorizz/src/memorizz/dispatch/task_dispatcher.py
from __future__ import annotations
from typing import Protocol, Any, Dict, List, Optional, Iterable

class TaskDispatcher(Protocol):
    """
    Abstract interface for offloading heavy/background work.
    Implementations may use Celery, thread pools, or be a no-op for tests.
    """

    # ---- Memory writes / upserts ------------------------------------------------
    def enqueue_memory_write(self, doc: Dict[str, Any], mem_type: Optional[str] = None) -> None: ...
    def enqueue_memory_bulk_write(self, docs: Iterable[Dict[str, Any]], mem_type: Optional[str] = None) -> None: ...

    # ---- Embeddings --------------------------------------------------------------
    def enqueue_embedding_compute(self, content: str, meta: Optional[Dict[str, Any]] = None) -> None: ...
    def compute_embedding_blocking(self, content: str, meta: Optional[Dict[str, Any]] = None) -> List[float]: ...
    # The blocking variant above is only for transitional cases where you must return an embedding now.
    # Prefer enqueuing and storing async where feasible.

    # ---- Importance / scoring / rerank ------------------------------------------
    def enqueue_importance_compute(self, items: List[Dict[str, Any]], query: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> None: ...

    # ---- Telemetry / metrics -----------------------------------------------------
    def enqueue_metric(self, event: Dict[str, Any]) -> None: ...

class NoOpDispatcher:
    """Default safe fallback that does nothing (or direct calls)."""

    def enqueue_memory_write(self, doc: Dict[str, Any], mem_type: Optional[str] = None) -> None:
        pass

    def enqueue_memory_bulk_write(self, docs, mem_type: Optional[str] = None) -> None:
        pass

    def enqueue_embedding_compute(self, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        pass

    def compute_embedding_blocking(self, content: str, meta: Optional[Dict[str, Any]] = None) -> List[float]:
        # Import locally to avoid circulars.
        from ..embeddings import get_embedding
        return get_embedding(content)

    def enqueue_importance_compute(self, items, query: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> None:
        pass

    def enqueue_metric(self, event: Dict[str, Any]) -> None:
        pass

# ---- Global accessor & setter ----------------------------------------------------
# This lets host apps (e.g., axiom_index_core) inject their own dispatcher.
_GLOBAL_DISPATCHER: TaskDispatcher = NoOpDispatcher()

def set_global_dispatcher(dispatcher: TaskDispatcher) -> None:
    global _GLOBAL_DISPATCHER
    _GLOBAL_DISPATCHER = dispatcher

def get_global_dispatcher() -> TaskDispatcher:
    return _GLOBAL_DISPATCHER
