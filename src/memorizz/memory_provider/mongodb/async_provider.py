"""
Async MongoDB Provider using Motor for high-performance async I/O.

This module provides AsyncMongoDBProvider, an async version of MongoDBProvider
that uses Motor (async MongoDB driver) for non-blocking database operations.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from motor.motor_asyncio import AsyncIOMotorClient

from .provider import MongoDBProvider, MongoDBConfig
from ...enums.memory_type import MemoryType

logger = logging.getLogger(__name__)


class AsyncMongoDBProvider(MongoDBProvider):
    """
    Async version of MongoDB provider for non-blocking database operations.

    Uses Motor (async MongoDB driver) for true async I/O, enabling:
    - Parallel query execution
    - Non-blocking database operations
    - Better performance under concurrent load
    - Batch query support

    Example:
        ```python
        config = MongoDBConfig(uri="mongodb://localhost:27017", db_name="memorizz")
        provider = AsyncMongoDBProvider(config)

        # Single query
        results = await provider.retrieve_memory_units_by_query_async(
            query="What is machine learning?",
            query_embedding=embedding,
            memory_id="user123",
            memory_type=MemoryType.LONG_TERM_MEMORY,
            limit=5
        )

        # Batch query (multiple types in parallel)
        batch_results = await provider.batch_retrieve_async([
            {'key': 'semantic', 'embedding': emb, 'memory_id': 'user123',
             'memory_type': MemoryType.LONG_TERM_MEMORY, 'limit': 5},
            {'key': 'episodic', 'embedding': emb, 'memory_id': 'user123',
             'memory_type': MemoryType.CONVERSATION_MEMORY, 'limit': 5},
        ])
        ```
    """

    def __init__(self, config: MongoDBConfig):
        """
        Initialize async MongoDB provider with both sync and async clients.

        Parameters:
        -----------
        config : MongoDBConfig
            Configuration containing MongoDB URI and database name
        """
        # Call parent init for sync client (backward compatibility)
        super().__init__(config)

        # Create async client with connection pooling
        self.async_client = AsyncIOMotorClient(
            config.uri,
            maxPoolSize=50,           # Max concurrent connections
            minPoolSize=10,           # Min pooled connections
            maxIdleTimeMS=60000,      # 1 minute idle timeout
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            socketTimeoutMS=20000,
            retryWrites=True,
            retryReads=True,
            appname='memorizz-async'
        )

        # Async database reference
        self.async_db = self.async_client[config.db_name]

        # Async collection references
        self.async_persona_collection = self.async_db[MemoryType.PERSONAS.value]
        self.async_toolbox_collection = self.async_db[MemoryType.TOOLBOX.value]
        self.async_short_term_memory_collection = self.async_db[MemoryType.SHORT_TERM_MEMORY.value]
        self.async_long_term_memory_collection = self.async_db[MemoryType.LONG_TERM_MEMORY.value]
        self.async_conversation_memory_collection = self.async_db[MemoryType.CONVERSATION_MEMORY.value]
        self.async_workflow_memory_collection = self.async_db[MemoryType.WORKFLOW_MEMORY.value]
        self.async_memagent_collection = self.async_db[MemoryType.MEMAGENT.value]
        self.async_shared_memory_collection = self.async_db[MemoryType.SHARED_MEMORY.value]
        self.async_summaries_collection = self.async_db[MemoryType.SUMMARIES.value]
        self.async_semantic_cache_collection = self.async_db[MemoryType.SEMANTIC_CACHE.value]

        logger.info("AsyncMongoDBProvider initialized with connection pooling")

    def _get_async_collection(self, memory_type: MemoryType):
        """Get async collection for a memory type"""
        collection_map = {
            MemoryType.PERSONAS: self.async_persona_collection,
            MemoryType.TOOLBOX: self.async_toolbox_collection,
            MemoryType.SHORT_TERM_MEMORY: self.async_short_term_memory_collection,
            MemoryType.LONG_TERM_MEMORY: self.async_long_term_memory_collection,
            MemoryType.CONVERSATION_MEMORY: self.async_conversation_memory_collection,
            MemoryType.WORKFLOW_MEMORY: self.async_workflow_memory_collection,
            MemoryType.MEMAGENT: self.async_memagent_collection,
            MemoryType.SHARED_MEMORY: self.async_shared_memory_collection,
            MemoryType.SUMMARIES: self.async_summaries_collection,
            MemoryType.SEMANTIC_CACHE: self.async_semantic_cache_collection,
        }
        return collection_map.get(memory_type)

    async def retrieve_memory_units_by_query_async(
        self,
        query: str = None,
        query_embedding: List[float] = None,
        memory_id: str = None,
        memory_type: MemoryType = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Async vector similarity search using Motor.

        Non-blocking I/O for concurrent queries. This method can be called
        multiple times in parallel using asyncio.gather().

        Parameters:
        -----------
        query : str, optional
            Query text (used if embedding not provided)
        query_embedding : list[float], optional
            Pre-computed query embedding vector
        memory_id : str
            Memory ID to filter by
        memory_type : MemoryType
            Type of memory to query
        limit : int
            Maximum number of results to return

        Returns:
        --------
        List[Dict[str, Any]]
            List of memory units with scores, sorted by relevance
        """

        if memory_type == MemoryType.CONVERSATION_MEMORY:
            return await self._get_conversation_memory_units_async(
                query, query_embedding, memory_id, limit
            )
        elif memory_type == MemoryType.WORKFLOW_MEMORY:
            return await self._get_workflow_memory_units_async(
                query, query_embedding, memory_id, limit
            )
        elif memory_type == MemoryType.SUMMARIES:
            return await self._get_summaries_memory_units_async(
                query, query_embedding, memory_id, limit
            )
        else:
            return []

    async def _get_conversation_memory_units_async(
        self,
        query: str = None,
        query_embedding: List[float] = None,
        memory_id: str = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Async conversation memory retrieval"""

        # If embedding not provided, use sync version (embedding is CPU-bound)
        if query_embedding is None and query is not None:
            try:
                from ...embeddings import get_embedding
                query_embedding = get_embedding(query)
            except Exception as e:
                logger.error(f"Failed to generate embedding for query: {e}")
                return []

        # Build vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": 100,
                    "limit": limit,
                    "filter": {"memory_id": memory_id}
                }
            },
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {"$project": {"embedding": 0}},
            {"$sort": {"score": -1, "timestamp": 1}},
        ]

        # ASYNC aggregation - non-blocking!
        cursor = self.async_conversation_memory_collection.aggregate(pipeline)
        results = await cursor.to_list(length=limit)

        return results

    async def _get_workflow_memory_units_async(
        self,
        query: str = None,
        query_embedding: List[float] = None,
        memory_id: str = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Async workflow memory retrieval"""

        if query_embedding is None and query is not None:
            try:
                from ...embeddings import get_embedding
                query_embedding = get_embedding(query)
            except Exception as e:
                logger.error(f"Failed to generate embedding for query: {e}")
                return []

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": 100,
                    "limit": limit,
                    "filter": {"memory_id": memory_id}
                }
            },
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {"$project": {"embedding": 0}},
            {"$sort": {"score": -1, "timestamp": 1}},
        ]

        cursor = self.async_workflow_memory_collection.aggregate(pipeline)
        results = await cursor.to_list(length=limit)

        return results

    async def _get_summaries_memory_units_async(
        self,
        query: str = None,
        query_embedding: List[float] = None,
        memory_id: str = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Async summaries memory retrieval"""

        if query_embedding is None and query is not None:
            try:
                from ...embeddings import get_embedding
                query_embedding = get_embedding(query)
            except Exception as e:
                logger.error(f"Failed to generate embedding for query: {e}")
                return []

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": 100,
                    "limit": limit,
                    "filter": {"memory_id": memory_id}
                }
            },
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {"$project": {"embedding": 0}},
            {"$sort": {"score": -1, "timestamp": 1}},
        ]

        cursor = self.async_summaries_collection.aggregate(pipeline)
        results = await cursor.to_list(length=limit)

        return results

    async def batch_retrieve_async(
        self,
        queries: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Execute multiple queries in PARALLEL for maximum throughput.

        This is the key optimization method that enables parallel database queries.
        All queries execute simultaneously, limited only by the slowest query.

        Parameters:
        -----------
        queries : List[Dict[str, Any]]
            List of query specifications, each containing:
            - key: Result key (e.g., 'semantic', 'episodic', 'procedural')
            - query: Query text (optional)
            - embedding: Query embedding vector
            - memory_id: Memory ID to filter by
            - memory_type: MemoryType enum
            - limit: Max results (default 5)

        Returns:
        --------
        Dict[str, List[Dict[str, Any]]]
            Dict mapping query keys to result lists

        Example:
        --------
        ```python
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
        ]

        results = await provider.batch_retrieve_async(queries)
        # results = {'semantic': [...], 'episodic': [...]}
        ```
        """

        # Create tasks for all queries
        tasks = []
        for q in queries:
            task = self.retrieve_memory_units_by_query_async(
                query=q.get('query'),
                query_embedding=q.get('embedding'),
                memory_id=q.get('memory_id'),
                memory_type=q.get('memory_type'),
                limit=q.get('limit', 5)
            )
            tasks.append(task)

        # Execute ALL queries in PARALLEL
        # return_exceptions=True ensures one failure doesn't break all queries
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build result dict, handling any exceptions
        result_dict = {}
        for i, q in enumerate(queries):
            key = q.get('key', f'query_{i}')
            if isinstance(results[i], Exception):
                logger.error(f"Query {key} failed: {results[i]}")
                result_dict[key] = []  # Empty on error
            else:
                result_dict[key] = results[i]

        return result_dict

    async def ensure_indexes_async(self):
        """
        Create optimized indexes for vector search and filtering.

        Should be called during application startup to ensure proper indexes exist.
        Creates indexes in the background to avoid blocking.
        """

        collections_info = [
            ('conversation_memory', self.async_conversation_memory_collection),
            ('long_term_memory', self.async_long_term_memory_collection),
            ('workflow_memory', self.async_workflow_memory_collection),
            ('summaries', self.async_summaries_collection),
        ]

        for name, collection in collections_info:
            try:
                # Compound index for memory_id + timestamp queries
                await collection.create_index(
                    [("memory_id", 1), ("timestamp", -1)],
                    name=f"{name}_memory_time_idx",
                    background=True
                )

                # Index for memory_type filtering
                await collection.create_index(
                    [("memory_id", 1), ("memory_type", 1)],
                    name=f"{name}_memory_type_idx",
                    background=True
                )

                # Role index for conversation memory
                if name == 'conversation_memory':
                    await collection.create_index(
                        [("memory_id", 1), ("role", 1)],
                        name=f"{name}_role_idx",
                        background=True
                    )

                # Situational tags index
                await collection.create_index(
                    [("situational_tags", 1)],
                    name=f"{name}_tags_idx",
                    background=True,
                    sparse=True  # Only index docs with tags
                )

                logger.info(f"Created indexes for {name} collection")

            except Exception as e:
                logger.warning(f"Failed to create indexes for {name}: {e}")

        logger.info("✅ Database indexes ensured")

    async def close_async(self):
        """Close async client connections properly"""
        self.async_client.close()
        logger.info("AsyncMongoDBProvider connections closed")

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.async_client.close()
        except:
            pass
