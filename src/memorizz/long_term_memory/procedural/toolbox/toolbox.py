# --- toolbox.py ---
# This refactored code adds support for any LLM provider while maintaining
# full backward compatibility with the original OpenAI implementation.

from typing import Dict, Any, List, Callable, Optional, Union, Protocol, Mapping
from ....memory_provider import MemoryProvider
from ....enums.memory_type import MemoryType
from ....embeddings import get_embedding
import inspect
from .tool_schema import ToolSchemaType
from bson import ObjectId
import logging
import json
import hashlib, time
from dataclasses import asdict


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# -------------- Step 1: Define a Generic LLM Interface ------------------
# This protocol defines the "contract" any compatible LLM provider must follow.
# It's good practice to place this in a separate file (e.g., llm_provider_protocol.py)
# but it's included here for a complete, single-file response.
from ....llms.llm_provider import LLMProvider


# ------------------ Step 2: Create the Default Fallback ------------------
# This function provides the default OpenAI client if no other LLM is specified,
# ensuring existing code continues to work without modification.

def get_openai_default() -> LLMProvider:
    """Initializes the default OpenAI client lazily."""
    from ....llms.openai import OpenAI
    return OpenAI()



EMBED_MODEL_ID = "text-embedding-3-small@1.0"  # or whatever you use

def _tool_key(func) -> str:
    mod = getattr(func, "__module__", "")
    qual = getattr(func, "__qualname__", getattr(func, "__name__", ""))
    return f"{mod}:{qual}"

def _material_for_embedding(func, docstring: str, signature: str, queries: list[str] | None) -> str:
    parts = [func.__name__, docstring, signature]
    if queries:
        parts.extend(queries)
    return " || ".join(parts)

def _content_hash(material: str, embed_model_id: str) -> str:
    h = hashlib.sha256()
    h.update(embed_model_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(material.encode("utf-8"))
    return h.hexdigest()
# ------------------ Step 3: Refactor the Toolbox Class ------------------

class Toolbox:
    """A toolbox for managing and retrieving tools using a memory provider."""
    
    def __init__(self, memory_provider: MemoryProvider, llm_provider: Optional[LLMProvider] = None):
        """
        Initialize the toolbox.

        This constructor is backward-compatible. If `llm_provider` is not specified,
        it defaults to using the OpenAI client.

        Parameters:
        -----------
        memory_provider : MemoryProvider
            The memory provider for storing and retrieving tools.
        llm_provider : LLMProvider, optional
            The LLM provider for metadata generation. Defaults to OpenAI if None.
        """
        self.memory_provider = memory_provider
        
        # If no provider is passed, create the default OpenAI client.
        if llm_provider is None:
            self.llm_provider = get_openai_default()
        else:
            self.llm_provider = llm_provider
        
        # In-memory storage of functions
        self._tools: Dict[str, Callable] = {}


    def register_tool(self, func: Optional[Callable] = None, augment: bool = False) -> Union[str, Callable]:
        """
        Register a function as a tool in the toolbox.
        """
        def decorator(f: Callable) -> str:
            t0 = time.perf_counter()
            docstring = f.__doc__ or ""
            signature = str(inspect.signature(f))
            tool_key = _tool_key(f)
            log.debug("[toolbox] registering tool: key=%s func=%s augment=%s", tool_key, getattr(f, "__name__", f), augment)

            # --- canonical parameters extraction/flattening ---
            def _canonical_param_list_from_fn(f: Callable) -> List[Dict[str, Any]]:
                schema = getattr(f, "_openai_parameters", None)
                if not isinstance(schema, dict) or schema.get("type") != "object":
                    log.debug("[toolbox] no _openai_parameters found for %s", tool_key)
                    return []
                props = schema.get("properties", {}) or {}
                required = set(schema.get("required", []) or [])
                out: List[Dict[str, Any]] = []
                for name, spec in props.items():
                    out.append({
                        "name": name,
                        "description": spec.get("description", ""),
                        "type": spec.get("type", "string"),
                        "required": name in required,
                    })
                log.debug("[toolbox] extracted %d canonical params for %s", len(out), tool_key)
                return out

            param_list = _canonical_param_list_from_fn(f)
            if param_list:
                log.trace if hasattr(log, "trace") else log.debug  # no-op if no TRACE level
                log.debug("[toolbox] params(%s)=%s", tool_key, param_list)

            # ---- material + metadata (augmentation optional) ----
            if augment:
                log.debug("[toolbox] augmenting docstring for %s", tool_key)
                augmented_docstring = self._augment_docstring(docstring)
                queries = self._generate_queries(augmented_docstring)
                material = _material_for_embedding(f, augmented_docstring, signature, queries)
                tool_data = self._get_tool_metadata(f)
                log.debug("[toolbox] augmentation produced %d queries for %s", len(queries or []), tool_key)
            else:
                queries = None
                material = _material_for_embedding(f, docstring, signature, None)
                tool_data = self._get_tool_metadata(f)

            # ---- hash / existing lookup ----
            h = _content_hash(material, EMBED_MODEL_ID)
            existing = self.memory_provider.retrieve_by_name(tool_key, memory_store_type=MemoryType.TOOLBOX)
            if existing:
                log.debug("[toolbox] existing record found for %s (_id=%s)", tool_key, existing.get("_id"))
            else:
                log.debug("[toolbox] no existing record for %s; creating new", tool_key)

            # ---- unchanged fast-path ----
            if existing and existing.get("content_hash") == h and existing.get("embedding_model_id") == EMBED_MODEL_ID:
                tool_id = str(existing["_id"])
                self._tools[tool_id] = f  # rebind callable in this process
                dt = (time.perf_counter() - t0) * 1000
                log.info("[toolbox] ✓ reused tool (no re-embed): %s (_id=%s) in %.1f ms", tool_key, tool_id, dt)
                log.debug("[toolbox] totals: in_memory=%d stored≈%s",
                        len(self._tools), getattr(self, "memory_provider", None) and "?" )
                return tool_id

            # ---- changed/new → embed & upsert ----
            e0 = time.perf_counter()
            embedding = get_embedding(material)
            e_dt = (time.perf_counter() - e0) * 1000
            emb_dim = len(embedding) if hasattr(embedding, "__len__") else "?"
            log.info("[toolbox] embedding computed for %s: dim=%s in %.1f ms", tool_key, emb_dim, e_dt)

            tool_record = {
                "_id": existing["_id"] if existing and "_id" in existing else ObjectId(),
                "name": tool_key,
                "embedding": embedding,
                "embedding_model_id": EMBED_MODEL_ID,
                "content_hash": h,
                "updated_at": time.time(),
                **tool_data.model_dump(),
                # optional diagnostics (won't affect search if your schema ignores them):
                "diagnostics": {
                    "signature": signature,
                    "param_list": param_list,
                    "augment": bool(augment),
                    "queries_count": len(queries or []),
                },
            }
            if queries is not None:
                tool_record["queries"] = queries

            if existing:
                self.memory_provider.update_by_id(tool_record["_id"], tool_record, memory_store_type=MemoryType.TOOLBOX)
                tool_id = str(tool_record["_id"])
                log.info("[toolbox] ↑ updated tool: %s (_id=%s)", tool_key, tool_id)
            else:
                self.memory_provider.store(tool_record, memory_store_type=MemoryType.TOOLBOX)
                tool_id = str(tool_record["_id"])
                log.info("[toolbox] + inserted tool: %s (_id=%s)", tool_key, tool_id)

            self._tools[tool_id] = f
            dt = (time.perf_counter() - t0) * 1000
            try:
                stored_count = len(self.list_tools())
            except Exception:
                stored_count = "?"
            log.debug("[toolbox] totals after register: in_memory=%d stored=%s (%.1f ms)", len(self._tools), stored_count, dt)
            return tool_id

        return decorator if func is None else decorator(func)



    def replace_function_by_id(self, tool_id: str, func: Callable) -> bool:
            """Replace the callable function for a tool by its ID."""
            if tool_id not in self._tools:
                log.warning(f"[toolbox.replace_function_by_id] Tool ID {tool_id} not found in in-memory tools")
                return False
            self._tools[tool_id] = func
            log.debug(f"[toolbox.replace_function_by_id] Replaced function for tool ID {tool_id}")
            return True


    def get_tool_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a single tool by its name.
        
        Parameters:
        -----------
        name : str
            The name of the tool to retrieve.
        
        Returns:
        --------
        Dict[str, Any]
            The tool data, or None if not found.
        """
        # Note: This method only retrieves metadata from the provider.
        # Use get_function_by_id to retrieve the callable function.
        return self.memory_provider.retrieve_by_name(name, memory_store_type=MemoryType.TOOLBOX)
    
    def get_tool_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Get a tool's metadata by its id.

        Parameters:
        -----------
        id : str
            The id of the tool to retrieve.
        
        Returns:
        --------
        Dict[str, Any]
            The tool data, or None if not found.
        """
        return self.memory_provider.retrieve_by_id(id, memory_store_type=MemoryType.TOOLBOX)

    def get_most_similar_tools(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most similar tools to a query using vector search.
        
        Parameters:
        -----------
        query : str
            The query to search for.
        limit : int, optional
            The maximum number of tools to return.
        
        Returns:
        --------
        List[Dict[str, Any]]
            A list of the most similar tool metadata.
        """
        return self.memory_provider.retrieve_by_query(
            query,
            memory_store_type=MemoryType.TOOLBOX,
            limit=limit
        )

    def delete_tool_by_name(self, name: str) -> bool:
        """
        Delete a tool from the toolbox by name.
        
        Parameters:
        -----------
        name : str
            The name of the tool to delete.
        
        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        tool_data = self.memory_provider.retrieve_by_name(name, memory_store_type=MemoryType.TOOLBOX)
        if tool_data and "_id" in tool_data:
            tool_id = str(tool_data["_id"])
            if tool_id in self._tools:
                del self._tools[tool_id]
        
        return self.memory_provider.delete_by_name(name, memory_store_type=MemoryType.TOOLBOX)
    
    def delete_tool_by_id(self, id: str) -> bool:
        """
        Delete a tool from the toolbox by id.

        Parameters:
        -----------
        id : str
            The id of the tool to delete.
        
        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        if id in self._tools:
            del self._tools[id]
        
        return self.memory_provider.delete_by_id(id, memory_store_type=MemoryType.TOOLBOX)
    
    def delete_all(self) -> bool:
        """
        Delete all tools in the toolbox.

        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        self._tools.clear()
        return self.memory_provider.delete_all(memory_store_type=MemoryType.TOOLBOX)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all tools in the toolbox from the memory provider.
        
        Returns:
        --------
        List[Dict[str, Any]]
            A list of all tool metadata from the memory provider.
        """
        return self.memory_provider.list_all(memory_store_type=MemoryType.TOOLBOX)

    def list_available_tools(self) -> List[Dict[str, Any]]:
        """
        List tools that have both metadata in the database AND are callable in the current session.
        
        Returns:
        --------
        List[Dict[str, Any]]
            A list of tool metadata for tools with available functions.
        """
        available_tools = []
        for tool_id in self._tools:
            meta = self.get_tool_by_id(tool_id)
            if meta:
                available_tools.append(meta)
        return available_tools
    
    def get_function_by_id(self, tool_id: str) -> Optional[Callable]:
        """
        Get the actual executable function by its tool ID.
        
        Parameters:
        -----------
        tool_id : str
            The ID of the tool whose function to retrieve.
        
        Returns:
        --------
        Optional[Callable]
            The function object, or None if not found in the current session.
        """
        return self._tools.get(tool_id)

    def update_tool_by_id(self, id: str, data: Dict[str, Any]) -> bool:
        """
        Update a tool's metadata in the memory provider by id.

        Parameters:
        -----------
        id : str
            The id of the tool to update.
        data : Dict[str, Any]
            The data to update the tool with.
        
        Returns:
        --------
        bool
            True if the update was successful, False otherwise.
        """
        return self.memory_provider.update_by_id(id, data, memory_store_type=MemoryType.TOOLBOX)
    
    # --- Internal methods now use the configured self.llm_provider ---
    
    def _get_tool_metadata(self, func: Callable) -> ToolSchemaType:
        """Get the metadata for a tool using the configured LLM provider."""
        return self.llm_provider.get_tool_metadata(func)
    
    def _augment_docstring(self, docstring: str) -> str:
        """Augment the docstring using the configured LLM provider."""
        return self.llm_provider.augment_docstring(docstring)
    
    def _generate_queries(self, docstring: str) -> List[str]:
        """Generate queries for the tool using the configured LLM provider."""
        return self.llm_provider.generate_queries(docstring)