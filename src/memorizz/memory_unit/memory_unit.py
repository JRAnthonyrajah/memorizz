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
        content["embedding"] = get_embedding(content["content"])

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
        
        logger.info(f"Storing memory unit of type {memory_type.value} in memory provider")
        logger.debug(f"Memory component data: {memory_unit_dict}")
        stored_id = self.memory_provider.store(memory_unit_dict, memory_type)
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
        self.query_embedding = get_embedding(query)

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

        # 2) Batch-compute LLM importance once for all units (bounded concurrency; cached)
        t0 = time.perf_counter()
        importance_scores = self._run_coro_in_loop(
            self._async_importance_batch(memory_units, query, max_concurrency=5)
        )
        dt_ms = (time.perf_counter() - t0) * 1000
        logging.getLogger(__name__).info(
            "[importance.batch] n=%d took=%.1f ms", len(importance_scores), dt_ms
        )
        for mu, s in zip(memory_units, importance_scores):
            mu["importance"] = float(s)


        # 3) Now compute the final memory_signal using the pre-filled 'importance'
        for mu in memory_units:
            mu["memory_signal"] = self.calculate_memory_signal(mu, query)


        # Sort the memory units by the memory signal
        memory_units.sort(key=lambda x: x["memory_signal"], reverse=True)

        # Return the memory units
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
        if self.query_embedding is None:
            self.query_embedding = get_embedding(query)

        # Get embedding of the memory unit if it is not already embedded
        if "embedding" not in memory_unit or memory_unit["embedding"] is None:
            memory_unit_embedding = get_embedding(memory_unit["content"])
        else:
            memory_unit_embedding = memory_unit["embedding"]

        # Calculate the cosine similarity between the query embedding and the memory unit embedding
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
            return [scores[i] if i in scores else 0.5 for i in range(len(memory_units))]

        # Choose the best available path
        has_async_batch = hasattr(self.llm_provider, "async_generate_batch")
        has_async_single = hasattr(self.llm_provider, "async_generate_text")
        logger = logging.getLogger(__name__)

        if has_async_batch:
            logger.info("[importance] using async_generate_batch; todo=%d", len(todo))
            await _run_async_batch()
        elif has_async_single:
            logger.info("[importance] using async fanout; todo=%d max_conc=%d", len(todo), max_concurrency)
            await _run_async_fanout()
        else:
            logger.info("[importance] using sync threadpool; todo=%d max_workers=%d", len(todo), max_concurrency)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _run_sync_threadpool)


        async def _run_async_batch() -> None:
            # Provider-level batch if implemented
            outs = await self.llm_provider.async_generate_batch(  # type: ignore[attr-defined]
                (p for (_, _, _, p) in todo),
                instructions=instructions,
            )
            for (idx, _, mu_id, _), out in zip(todo, outs):
                score = _parse_float(out)
                scores[idx] = score
                _cache_put(mu_id, model, score)

        async def _run_async_fanout() -> None:
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

        def _run_sync_threadpool() -> None:
            def one(args):
                idx, _, mu_id, prompt = args
                out = self.llm_provider.generate_text(prompt, instructions=instructions)
                score = _parse_float(out)
                scores[idx] = score
                _cache_put(mu_id, model, score)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as ex:
                list(ex.map(one, todo))

        if has_async_batch:
            await _run_async_batch()
        elif has_async_single:
            await _run_async_fanout()
        else:
            # we’re in an async method but provider is sync → run in a thread to not block loop
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _run_sync_threadpool)

        # Return in original order
        return [scores.get(i, 0.5) for i in range(len(memory_units))]

    def _run_coro_in_loop(self, coro):
        """
        Bridge for calling an async coroutine from sync code.
        Uses current loop if running; else creates a temp loop.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        return fut.result()