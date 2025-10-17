# memory_unit.py
from __future__ import annotations
import asyncio, math, re
import concurrent.futures

# from memorizz.src.memorizz import memory_unit

from .conversational_memory_unit import ConversationMemoryUnit
from ..memory_provider import MemoryProvider
from ..enums.memory_type import MemoryType
from ..enums.application_mode import ApplicationMode, ApplicationModeConfig
from ..embeddings import get_embedding
from ..dispatch.task_dispatcher import get_global_dispatcher
from ..concurrency.executors import run_io


from typing import TYPE_CHECKING, Dict, Any, List, Optional, Iterable, Tuple
import time
import numpy as np
import pprint
import logging
from ..llms.llm_provider import LLMProvider

# Use lazy initialization for OpenAI
def get_openai_llm():
    from ..llms.openai import OpenAI
    return OpenAI()



_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
dispatcher = get_global_dispatcher()

def _parse_float(s: str, default: float = 0.5) -> float:
    m = _FLOAT_RE.search(str(s))
    if not m:
        return default
    try:
        x = float(m.group(0))
        return 0.0 if x < 0 else 1.0 if x > 1 else x
    except Exception:
        return default

# simple in-proc cache for importance: {(mu_id, model, schema_ver): (score, ts)}
_IMPORTANCE_CACHE: Dict[Tuple[str, str, int], Tuple[float, float]] = {}
_IMPORTANCE_CACHE_TTL = 24 * 3600
_IMPORTANCE_SCHEMA_VERSION = 1

def _cache_key(mu_id: str, model: str) -> Tuple[str, str, int]:
    return (mu_id, model, _IMPORTANCE_SCHEMA_VERSION)

def _importance_prompt(mu_text: str, query: str) -> str:
    return (
        "You are scoring the *importance* of a memory item for future recall.\n"
        "Return only a float in [0,1]. No words.\n\n"
        f"MEMORY:\n{(mu_text or '').strip()[:1200]}\n\nQUERY:\n{(query or '').strip()[:400]}"
    )

def _now() -> float:
    return time.time()

def _cache_get(mu_id: str, model: str) -> Optional[float]:
    k = _cache_key(mu_id, model)
    v = _IMPORTANCE_CACHE.get(k)
    if not v:
        return None
    score, ts = v
    if (_now() - ts) > _IMPORTANCE_CACHE_TTL:
        _IMPORTANCE_CACHE.pop(k, None)
        return None
    return score

def _cache_put(mu_id: str, model: str, score: float) -> None:
    _IMPORTANCE_CACHE[_cache_key(mu_id, model)] = (float(score), _now())

def _get_mu_id(mu: Any, fallback: str) -> str:
    # try common id fields; adjust if your provider uses a different key
    return str(mu.get("_id") or mu.get("id") or fallback)


class MemoryUnit:
    def __init__(self, application_mode: str, memory_provider: MemoryProvider = None, llm_provider: Optional[LLMProvider] = None):
        # Validate and set the application mode
        if isinstance(application_mode, str):
            self.application_mode = ApplicationModeConfig.validate_mode(application_mode)
        else:
            self.application_mode = application_mode
            
        self.memory_provider = memory_provider
        self.query_embedding = None
        if llm_provider:
            self.llm_provider = llm_provider
        else:
            self.llm_provider = get_openai_llm()
        # Get the memory types for this application mode
        self.active_memory_types = ApplicationModeConfig.get_memory_types(self.application_mode)

    def generate_memory_unit(self, content: dict):
        """
        Generate the memory unit based on the application mode.
        The memory unit type is determined by the content and active memory types.
        """

        # Generate the embedding of the memory unit
        try:
            # Transitional: compute now if your MU constructor requires an embedding field synchronously.
            content["embedding"] = dispatcher.compute_embedding_blocking(content["content"], meta={"where":"memory_unit.generate"})
        except Exception:
            # Fail-soft: do not block the read path if embedding fails.
            content["embedding"] = None
            dispatcher.enqueue_embedding_compute(
                content["content"], meta={"where": "memory_unit.generate", "mu_meta": {"type": content.get("type")}}
            )

        # Determine the appropriate memory unit type based on content and active memory types
        if MemoryType.CONVERSATION_MEMORY in self.active_memory_types and "role" in content:
            return self._generate_conversational_memory_unit(content)
        elif MemoryType.WORKFLOW_MEMORY in self.active_memory_types:
            return self._generate_workflow_memory_unit(content)
        elif MemoryType.LONG_TERM_MEMORY in self.active_memory_types:
            return self._generate_knowledge_base_unit(content)
        else:
            # Default to conversational if available, otherwise use the first active memory type
            if MemoryType.CONVERSATION_MEMORY in self.active_memory_types:
                return self._generate_conversational_memory_unit(content)
            else:
                raise ValueError(f"No suitable memory unit type for application mode: {self.application_mode.value}")

    def _generate_conversational_memory_unit(self, content: dict) -> ConversationMemoryUnit:
        """
        Generate the conversational memory unit.
        
        Parameters:
            content (dict): The content of the memory unit.

        Returns:
            ConversationMemoryUnit: The conversational memory unit.
        """
        memory_unit = ConversationMemoryUnit(
            role=content["role"],
            content=content["content"],
            timestamp=content["timestamp"],
            conversation_id=content["conversation_id"],
            memory_id=content["memory_id"],
            embedding=content["embedding"]
        )

        # Save the memory unit to the memory provider
        self._save_memory_unit(memory_unit, MemoryType.CONVERSATION_MEMORY)

        return memory_unit

    def _generate_workflow_memory_unit(self, content: dict):
        """
        Generate a workflow memory unit.
        
        Parameters:
            content (dict): The content of the memory unit.
            
        Returns:
            dict: The workflow memory unit.
        """
        workflow_component = {
            "content": content["content"],
            "timestamp": content.get("timestamp", time.time()),
            "memory_id": content["memory_id"],
            "embedding": content["embedding"],
            "component_type": "workflow",
            "workflow_step": content.get("workflow_step", "unknown"),
            "task_id": content.get("task_id"),
        }
        
        # Save the memory unit to the memory provider
        self._save_memory_unit(workflow_component, MemoryType.WORKFLOW_MEMORY)
        
        return workflow_component

    def _generate_knowledge_base_unit(self, content: dict):
        """
        Generate a knowledge base (long-term memory) component.
        
        Parameters:
            content (dict): The content of the memory unit.
            
        Returns:
            dict: The knowledge base memory unit.
        """
        knowledge_component = {
            "content": content["content"],
            "timestamp": content.get("timestamp", time.time()),
            "memory_id": content["memory_id"],
            "embedding": content["embedding"],
            "component_type": "knowledge",
            "category": content.get("category", "general"),
            "importance": content.get("importance", 0.5),
        }
        
        # Save the memory unit to the memory provider
        self._save_memory_unit(knowledge_component, MemoryType.LONG_TERM_MEMORY)
        
        return knowledge_component
    
    def _save_memory_unit(self, memory_unit: any, memory_type: MemoryType = None):
        """
        Save the memory unit to the memory provider.
        
        Parameters:
            memory_unit: The memory unit to save
            memory_type: Specific memory type to save to (optional)
        """

        # Remove the score(vector similarity score calculated by the vector search of the memory provider) from the memory unit if it exists
        if isinstance(memory_unit, dict) and "score" in memory_unit:
            memory_unit.pop("score", None)

        # Convert Pydantic model to dictionary if needed
        if hasattr(memory_unit, 'model_dump'):
            memory_unit_dict = memory_unit.model_dump()
        elif hasattr(memory_unit, 'dict'):
            memory_unit_dict = memory_unit.dict()
        else:
            # If it's already a dictionary, use it as is
            memory_unit_dict = memory_unit

        # If memory_type is not specified, determine from the component or use conversation as default
        if memory_type is None:
            if MemoryType.CONVERSATION_MEMORY in self.active_memory_types:
                memory_type = MemoryType.CONVERSATION_MEMORY
            else:
                # Use the first available memory type from active types
                memory_type = self.active_memory_types[0] if self.active_memory_types else MemoryType.CONVERSATION_MEMORY

        # Validate that the memory type is active for this application mode
        if memory_type not in self.active_memory_types:
            logger = logging.getLogger(__name__)
            logger.warning(f"Memory type {memory_type.value} not active for application mode {self.application_mode.value}")

        logger = logging.getLogger(__name__)
        
        logger.info(f"Queueing memory unit of type {memory_type.value} for background store")
        logger.debug(f"Memory component data: {memory_unit_dict}")
        # Offload the write; do not block the read path
        get_global_dispatcher().enqueue_memory_write(memory_unit_dict, memory_type)
        # Maintain API shape: return the best available identifier without waiting
        stored_id = memory_unit_dict.get("_id") or memory_unit_dict.get("id")
        
        logger.info(f"Stored memory unit with ID: {stored_id}")
        return stored_id

    def retrieve_memory_units_by_memory_id(self, memory_id: str, memory_type: MemoryType):
        """
        Retrieve the memory units by memory id.

        Parameters:
            memory_id (str): The id of the memory to retrieve the memory units for.
            memory_type (MemoryType): The type of the memory to retrieve the memory units for.

        Returns:
            List[MemoryUnit]: The memory units.
        """
        if memory_type == MemoryType.CONVERSATION_MEMORY:
            return self.memory_provider.retrieve_conversation_history_ordered_by_timestamp(memory_id)

        elif memory_type == MemoryType.WORKFLOW_MEMORY:
            return self.memory_provider.retrieve_workflow_history_ordered_by_timestamp(memory_id)
        else:
            raise ValueError(f"Invalid memory type: {memory_type}")

    def retrieve_memory_units_by_conversation_id(self, conversation_id: str):
        pass

    def retrieve_memory_units_by_query(self, query: str, memory_id: str, memory_type: MemoryType, limit: int = 5):
        """
        Retrieve the memory units by query.

        Parameters:
            query (str): The query to use for retrieval.
            memory_id (str): The id of the memory to retrieve the memory units for.
            memory_type (MemoryType): The type of the memory to retrieve the memory units for.
            limit (int): The limit of the memory units to return.

        Returns:
            List[MemoryUnit]: The memory units.
        """

        # Create the query embedding here so that it is not created for each memory unit
        dispatcher = get_global_dispatcher()
        try:
            self.query_embedding = dispatcher.compute_embedding_blocking(
                query, meta={"where": "memory_unit.retrieve.query"}
            )
        except Exception:
            self.query_embedding = None
            dispatcher.enqueue_embedding_compute(query, meta={"where": "memory_unit.retrieve.query"})
            # Fallback: a neutral later step will handle missing embedding gracefully.

        # Get the memory units by query
        memory_units = self.memory_provider.retrieve_memory_units_by_query(query, self.query_embedding, memory_id, memory_type, limit)

        # Get the surronding conversation ids from each of the memory units
        # Handle cases where conversation_id might be missing or _id is used instead
        surrounding_conversation_ids = []
        for memory_unit in memory_units:
            surrounding_conversation_ids.append(memory_unit["_id"])

        # Before returning the memory units, we need to update the memory signals within the memory units
        # 1) Update fields that do not require I/O or LLMs
        for mu in memory_units:
            self.update_memory_signals_within_memory_unit(mu, memory_type, surrounding_conversation_ids)

        # 2) Importance via background task to avoid blocking the read path.
        #    Use neutral default now; background reranker can upsert later.
        dispatcher = get_global_dispatcher()
        try:
            dispatcher.enqueue_importance_compute(
                items=[mu if isinstance(mu, dict) else getattr(mu, "model_dump", lambda: mu)() for mu in memory_units],
                query=query,
                meta={"where": "memory_unit.retrieve.importance_batch", "memory_id": memory_id if "memory_id" in locals() else None}
            )
        except Exception:
            # Best-effort only; continue without blocking
            pass

        for mu in memory_units:
            # If no precomputed importance present, set neutral default.
            if mu.get("importance") is None:
                mu["importance"] = 0.5


        # 3) Now compute the final memory_signal using the pre-filled 'importance'
        for mu in memory_units:
            mu["memory_signal"] = self.calculate_memory_signal(mu, query)


        # Sort the memory units by the memory signal
        memory_units.sort(key=lambda x: x["memory_signal"], reverse=True)

        # Return the memory units
        return memory_units
    

    async def retrieve_memory_units_by_query_async(
        self,
        query: str,
        memory_id: str,
        memory_type: MemoryType,
        limit: int = 5,
    ):
        """
        Async version of retrieval:
        - Offloads sync provider calls with asyncio.to_thread
        - Avoids any LLM fan-out or blocking work (importance is enqueued)
        """
        dispatcher = get_global_dispatcher()

        # Query embedding (fail-soft; offload long work)
        try:
            query_embedding = dispatcher.compute_embedding_blocking(
                query, meta={"where": "memory_unit.retrieve.query"}
            )
        except Exception:
            query_embedding = None
            dispatcher.enqueue_embedding_compute(query, meta={"where": "memory_unit.retrieve.query"})

        # Provider retrieval may be synchronous; run it in a worker thread
        memory_units = await run_io(
            self.memory_provider.retrieve_memory_units_by_query,
            query,
            query_embedding,
            memory_id,
            memory_type,
            limit,
        )

        # Prepare surrounding conversation ids
        surrounding_conversation_ids = [mu["_id"] for mu in memory_units]

        # Update cheap, local signals now (no I/O or LLMs)
        for mu in memory_units:
            self.update_memory_signals_within_memory_unit(mu, memory_type, surrounding_conversation_ids)

        # Enqueue importance in the background; set neutral default now
        try:
            dispatcher.enqueue_importance_compute(
                items=[mu for mu in memory_units],
                query=query,
                meta={"where": "memory_unit.retrieve.importance_batch", "memory_id": memory_id},
            )
        except Exception:
            pass
        for mu in memory_units:
            if mu.get("importance") is None:
                mu["importance"] = 0.5

        return memory_units



    def update_memory_signals_within_memory_unit(self, memory_unit: any, memory_type: MemoryType, surrounding_conversation_ids: list[str]):
        """
        Update the memory signal within the memory unit.

        Parameters:
            memory_unit (dict): The memory unit to update the memory signal within.
            memory_type (MemoryType): The type of the memory to update the memory signal within.
            surrounding_conversation_ids (list[str]): The list of surrounding conversation ids.
        """

        # Update the recall_recency field (how recently the memory unit was recalled), this is the current timestamp
        memory_unit["recall_recency"] = time.time()

        if memory_type == MemoryType.CONVERSATION_MEMORY:
            # Update the importance field with a list of calling ID and surronding conversation ID's
            memory_unit["associated_conversation_ids"] = surrounding_conversation_ids

        # Save the memory unit to the memory provider
        self._save_memory_unit(memory_unit)

    def calculate_memory_signal(self, memory_unit: any, query: str):
        """
        Calculate the memory signal within the memory unit.

        Parameters:
            memory_unit (any): The memory unit to calculate the memory signal within.
            query (str): The query to use for calculation.

        Returns:
            float: The memory signal between 0 and 1.
        """
        # Detect the gap between the current timestamp and the recall_recency field
        recency = time.time() - memory_unit["recall_recency"]

        # Get the number of associated memory ids (this is used to calcualte the importance of the memory unit)
        number_of_associated_conversation_ids = len(memory_unit["associated_conversation_ids"])

        # If the score exists, use it as the relevance score (this is the vector similarity score calculated by the vector search of the memory provider)
        if "score" in memory_unit:
            relevance = memory_unit["score"]
        else:
            # Calculate the relevance of the memory unit which is a vector score between the memory unit and the query
            relevance = self.calculate_relevance(query, memory_unit)

        # Calulate importance of the memory unit
        # Importance is expected to be precomputed upstream.
        importance = memory_unit.get("importance")
        if importance is None:
            # Do NOT call the LLM here; that would serialize N requests.
            importance = 0.5  # neutral default
            logging.getLogger(__name__).warning(
                "importance missing; using default=0.5 for mu_id=%s",
                memory_unit.get("_id") or memory_unit.get("id"),
            )

        # Calculate the normalized memory signal
        memory_signal = recency * number_of_associated_conversation_ids * relevance * importance

        # Normalize the memory signal between 0 and 1
        memory_signal = memory_signal / 100

        # Return the memory signal
        return memory_signal

    def calculate_relevance(self, query: str, memory_unit: any) -> float:
        """
        Calculate the relevance of the query with the memory unit.

        Parameters:
            query (str): The query to use for calculation.
            memory_unit (any): The memory unit to calculate the relevance within.

        Returns:
            float: The relevance between 0 and 1.
        """
        # Get embedding of the query
        dispatcher = get_global_dispatcher()
        if self.query_embedding is None:
            try:
                self.query_embedding = dispatcher.compute_embedding_blocking(
                    query, meta={"where": "memory_unit.calculate_relevance.query"}
                )
            except Exception:
                self.query_embedding = None
                dispatcher.enqueue_embedding_compute(query, meta={"where": "memory_unit.calculate_relevance.query"})

        # Get embedding of the memory unit if it is not already embedded
        if "embedding" not in memory_unit or memory_unit["embedding"] is None:
            try:
                memory_unit_embedding = dispatcher.compute_embedding_blocking(
                    memory_unit["content"], meta={"where": "memory_unit.calculate_relevance.mu"}
                )
            except Exception:
                memory_unit_embedding = None
                dispatcher.enqueue_embedding_compute(
                    memory_unit["content"],
                    meta={"where": "memory_unit.calculate_relevance.mu", "target_id": memory_unit.get("_id") or memory_unit.get("id")}
                )
        else:
            memory_unit_embedding = memory_unit["embedding"]

        # Calculate the cosine similarity between the query embedding and the memory unit embedding
        if self.query_embedding is None or memory_unit_embedding is None:
            # Neutral fallback to avoid blocking / exceptions
            relevance = 0.5
        else:
            relevance = self.cosine_similarity(self.query_embedding, memory_unit_embedding)


        # Return the relevance
        return relevance
        

    # We might not need this as the memory compoennt should have a score from retrieval
    def cosine_similarity(self, query_embedding: list[float], memory_unit_embedding: list[float]) -> float:
        """
        Calculate the cosine similarity between two embeddings.

        Parameters:
            query_embedding (list[float]): The query embedding.
            memory_unit_embedding (list[float]): The memory unit embedding.

        Returns:
            float: The cosine similarity between the two embeddings.
        """
        # Calculate the dot product of the two embeddings
        dot_product = np.dot(query_embedding, memory_unit_embedding)

        # Calculate the magnitude of the two embeddings
        magnitude_query_embedding = np.linalg.norm(query_embedding)
        magnitude_memory_unit_embedding = np.linalg.norm(memory_unit_embedding)

        # Calculate the cosine similarity
        cosine_similarity = dot_product / (magnitude_query_embedding * magnitude_memory_unit_embedding)

        # Return the cosine similarity
        return cosine_similarity


    def calculate_importance(self, memory_unit_content: str, query: str) -> float:
        """
        Calculate the importance of the memory unit.
        Using an LLM to calculate the importance of the memory unit.

        Parameters:
            memory_unit_content (str): The content of the memory unit to calculate the importance within.
            query (str): The query to use for calculation.

        Returns:
            float: The importance between 0 and 1.
        """
   

        importance_prompt = f"""
        Calculate the importance of the following memory unit:
        {memory_unit_content}
        in relation to the following query and rate the likely poignancy of the memory unit:
        {query}
        Return the importance of the memory unit as a number between 0 and 1.
        """

        # Get the importance of the memory unit
        importance = self.llm_provider.generate_text(importance_prompt, instructions="Return the importance of the memory unit as a number between 0 and 1. No other text or comments, just the number. For example: 0.5")

        # Return the importance
        return float(importance)

    # --- Add inside class MemoryUnit (method definitions) ---

    async def _async_importance_batch(
        self,
        memory_units: List[dict],
        query: str,
        *,
        max_concurrency: int = 5,
        instructions: str = "Return only a number in [0,1]."
    ) -> List[float]:
        """
        Compute LLM-based importance for many memory units in parallel with bounded concurrency.
        Uses provider async methods if available; else threadpooled sync fallback.
        Applies a simple in-proc cache keyed by (mu_id, model, schema_version).
        """
        logger = logging.getLogger(__name__)
        model = getattr(self.llm_provider, "model", "unknown")

        # Build todo list and prefill cached results
        scores: Dict[int, float] = {}
        todo: List[Tuple[int, dict, str, str]] = []  # (idx, mu, mu_id, prompt)

        for idx, mu in enumerate(memory_units):
            mu_id = _get_mu_id(mu, f"idx:{idx}")
            cached = _cache_get(mu_id, model)
            if cached is not None:
                scores[idx] = cached
                continue
            prompt = _importance_prompt(mu.get("content", ""), query)
            todo.append((idx, mu, mu_id, prompt))

        # Nothing to do?
        if not todo:
            return [scores.get(i, 0.5) for i in range(len(memory_units))]

        # ---- define helpers FIRST (so no reference-before-definition) ----
        async def _helper_async_batch() -> None:
            outs = await self.llm_provider.async_generate_batch(  # type: ignore[attr-defined]
                (p for (_, _, _, p) in todo),
                instructions=instructions,
            )
            for (idx, _, mu_id, _), out in zip(todo, outs):
                score = _parse_float(out)
                scores[idx] = score
                _cache_put(mu_id, model, score)

        async def _helper_async_fanout() -> None:
            sem = asyncio.Semaphore(max_concurrency)
            async def one(idx: int, mu_id: str, prompt: str):
                async with sem:
                    out = await self.llm_provider.async_generate_text(  # type: ignore[attr-defined]
                        prompt, instructions=instructions
                    )
                    score = _parse_float(out)
                    scores[idx] = score
                    _cache_put(mu_id, model, score)
            await asyncio.gather(*(one(idx, mu_id, prompt) for (idx, _, mu_id, prompt) in todo))

        def _helper_sync_threadpool() -> None:
            def one(args):
                idx, _, mu_id, prompt = args
                out = self.llm_provider.generate_text(prompt, instructions=instructions)
                score = _parse_float(out)
                scores[idx] = score
                _cache_put(mu_id, model, score)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as ex:
                list(ex.map(one, todo))

        # ---- decide branch AFTER helpers exist ----
        has_async_batch = hasattr(self.llm_provider, "async_generate_batch")
        has_async_single = hasattr(self.llm_provider, "async_generate_text")

        if has_async_batch:
            logger.info("[importance] using async_generate_batch; todo=%d", len(todo))
            await _helper_async_batch()
        elif has_async_single:
            logger.info("[importance] using async fanout; todo=%d max_conc=%d", len(todo), max_concurrency)
            await _helper_async_fanout()
        else:
            logger.info("[importance] using sync threadpool; todo=%d max_workers=%d", len(todo), max_concurrency)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _helper_sync_threadpool)

        # Return in original order
        return [scores.get(i, 0.5) for i in range(len(memory_units))]

    async def retrieve_all_memory_types_async(
        self,
        query: str,
        memory_id: str,
        limits: Dict[MemoryType, int],
        async_memory_provider = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve ALL memory types in PARALLEL with a SINGLE embedding computation.

        This is the key optimization for high-performance context building:
        1. Compute query embedding ONCE
        2. Query all memory types in PARALLEL
        3. Post-process results concurrently

        Parameters:
        -----------
        query : str
            User query text
        memory_id : str
            Memory ID to filter by
        limits : Dict[MemoryType, int]
            Dict mapping MemoryType to result limit
        async_memory_provider : AsyncMongoDBProvider, optional
            Async provider to use. If None, uses sync provider with executor.

        Returns:
        --------
        Dict[str, List[Dict[str, Any]]]
            Dict mapping memory type names to result lists

        Example:
        --------
        ```python
        from memorizz.memory_provider.mongodb.async_provider import AsyncMongoDBProvider

        async_provider = AsyncMongoDBProvider(config)
        mu = MemoryUnit('conversational', memory_provider=provider)

        results = await mu.retrieve_all_memory_types_async(
            query="What is machine learning?",
            memory_id="user123",
            limits={
                MemoryType.LONG_TERM_MEMORY: 5,
                MemoryType.CONVERSATION_MEMORY: 5,
                MemoryType.PROCEDURAL_MEMORY: 5,
            },
            async_memory_provider=async_provider
        )

        semantic = results.get('LONG_TERM_MEMORY', [])
        episodic = results.get('CONVERSATION_MEMORY', [])
        ```
        """

        # 1. Compute embedding ONCE for all queries
        from ..embeddings import get_embedding_async
        try:
            query_embedding = await get_embedding_async(query)
        except ImportError:
            # Fallback to sync if async not available
            from ..embeddings import get_embedding
            query_embedding = get_embedding(query)

        # 2. Build batch query specifications
        batch_queries = []
        for mem_type, limit in limits.items():
            if mem_type is None or limit <= 0:
                continue

            batch_queries.append({
                'key': mem_type.value,
                'query': query,
                'embedding': query_embedding,  # Reuse same embedding!
                'memory_id': memory_id,
                'memory_type': mem_type,
                'limit': limit
            })

        # 3. Execute ALL queries in PARALLEL
        if async_memory_provider:
            # Use async provider's batch API
            results = await async_memory_provider.batch_retrieve_async(batch_queries)
        else:
            # Fallback: run sync provider in parallel executors
            loop = asyncio.get_event_loop()
            tasks = []

            for q in batch_queries:
                task = loop.run_in_executor(
                    None,  # Default executor
                    self.memory_provider.retrieve_memory_units_by_query,
                    q['query'],
                    q['embedding'],
                    q['memory_id'],
                    q['memory_type'],
                    q['limit']
                )
                tasks.append(task)

            batch_results = await asyncio.gather(*tasks)
            results = {
                batch_queries[i]['key']: batch_results[i]
                for i in range(len(batch_queries))
            }

        # 4. Post-process: Add memory signals in PARALLEL
        async def add_memory_signals(
            key: str,
            mems: List[Dict]
        ) -> tuple:
            """Calculate memory signals for a list of memories"""
            for mu in mems:
                mu["memory_signal"] = self.calculate_memory_signal(mu, query)

            # Sort by memory signal
            mems.sort(key=lambda x: x["memory_signal"], reverse=True)

            return key, mems

        # Run signal calculation in parallel
        signal_tasks = [
            add_memory_signals(k, v)
            for k, v in results.items()
        ]
        processed = await asyncio.gather(*signal_tasks)

        return dict(processed)



    def _run_coro_in_loop(self, coro):
        """DEPRECATED: do not use. Retrieval is async; importance is offloaded."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        return fut.result()