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

def _tool_key(f) -> str:
    # Keep it simple and stable: the actual tool name
    return getattr(f, "__name__", "")

def _legacy_tool_key(f: Callable) -> str:
    """Previous key format you used (module:qualname)."""
    mod = getattr(f, "__module__", "")
    qn  = getattr(f, "__qualname__", "")
    return f"{mod}:{qn}".strip(":")

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
        self._by_name: dict[str, list[str]] = {}
        self._by_fqn: dict[str, str] = {}
        

    def register_tool(self, func: Optional[Callable] = None, augment: bool = False) -> Union[str, Callable]:
        def decorator(f: Callable) -> str:
            import time
            import inspect
            from bson import ObjectId
            from pymongo.errors import DuplicateKeyError

            from ....embeddings import get_embedding
            from ....enums.memory_type import MemoryType

            # -------- names / signatures / params ----------
            docstring = f.__doc__ or ""
            signature = str(inspect.signature(f))

            tool_key   = _tool_key(f)        # canonical normalized key (e.g., "read_file")
            legacy_key = _legacy_tool_key(f) # legacy identifier

            # NEW: pull metadata exported by your MCP adapter
            openai_name = getattr(f, "_openai_name", None)  # e.g., "obsidian_list_notes"
            fqn         = getattr(f, "_fqn", None)          # e.g., "fs:obsidian_list_notes"

            # -------- embedding material / metadata ----------
            if augment:
                augmented_docstring = self._augment_docstring(docstring)
                queries  = self._generate_queries(augmented_docstring)
                material = _material_for_embedding(f, augmented_docstring, signature, queries)
                tool_meta = self._get_tool_metadata(f)
            else:
                queries  = None
                material = _material_for_embedding(f, docstring, signature, None)
                tool_meta = self._get_tool_metadata(f)

            h = _content_hash(material, EMBED_MODEL_ID)

            # -------- primary lookup path: by hash + model (dedup) ----------
            existing_by_hash = self.memory_provider.retrieve_by_hash(
                h, EMBED_MODEL_ID, memory_store_type=MemoryType.TOOLBOX
            )

            # Secondary (migration support): by normalized name or legacy name
            existing_by_name = self.memory_provider.retrieve_by_name(
                tool_key, memory_store_type=MemoryType.TOOLBOX
            ) or (
                legacy_key and self.memory_provider.retrieve_by_name(
                    legacy_key, memory_store_type=MemoryType.TOOLBOX
                )
            )

            existing = existing_by_hash or existing_by_name

            # -------- embedding decision ----------
            if existing and existing.get("content_hash") == h and \
            existing.get("embedding_model_id") == EMBED_MODEL_ID and \
            isinstance(existing.get("embedding"), (list, tuple)) and existing.get("embedding"):
                embedding = existing["embedding"]
            else:
                embedding = None

            # -------- consolidate aliases / id ----------
            record_id = existing.get("_id") if existing else ObjectId()
            aliases = set((existing.get("aliases") or []) if existing else [])

            old_name = (existing.get("name") if existing else None)
            if old_name and old_name != tool_key:
                aliases.add(old_name)
            if legacy_key and legacy_key != tool_key:
                aliases.add(legacy_key)
            # NEW: include runtime-exported names
            if openai_name and openai_name != tool_key:
                aliases.add(openai_name)
            if fqn:
                aliases.add(fqn)

            # -------- build record shell ----------
            tool_record = {
                "_id": record_id,
                "name": tool_key,                 # keep canonical stable key
                "aliases": sorted(a for a in aliases if a),
                "embedding_model_id": EMBED_MODEL_ID,
                "content_hash": h,
                "updated_at": time.time(),
                # NEW: persist these for debugging / resolution
                "fqn": fqn,
                "display_name": openai_name or tool_key,
                **tool_meta.model_dump(),
            }
            if queries is not None:
                tool_record["queries"] = queries
            if embedding is not None:
                tool_record["embedding"] = embedding

            coll = self.memory_provider.collection(memory_store_type=MemoryType.TOOLBOX)

            # -------- upsert strategy ----------
            if existing:
                if "embedding" not in tool_record:
                    tool_record["embedding"] = get_embedding(material)
                # FIX: pass string id
                self.memory_provider.update_by_id(str(record_id), tool_record, memory_store_type=MemoryType.TOOLBOX)
            else:
                again = self.memory_provider.retrieve_by_hash(h, EMBED_MODEL_ID, memory_store_type=MemoryType.TOOLBOX)
                if again:
                    record_id = again["_id"]
                    tool_record["_id"] = record_id
                    tool_record["embedding"] = again.get("embedding")
                    tool_record["aliases"] = sorted(set(tool_record["aliases"]) | set(again.get("aliases", [])))
                    tool_record["name"] = tool_key
                    # FIX: pass string id
                    self.memory_provider.update_by_id(str(record_id), tool_record, memory_store_type=MemoryType.TOOLBOX)
                else:
                    if "embedding" not in tool_record:
                        tool_record["embedding"] = get_embedding(material)
                    try:
                        self.memory_provider.store(tool_record, memory_store_type=MemoryType.TOOLBOX)
                    except DuplicateKeyError:
                        winner = self.memory_provider.retrieve_by_hash(h, EMBED_MODEL_ID, memory_store_type=MemoryType.TOOLBOX) \
                            or self.memory_provider.retrieve_by_name(tool_key, memory_store_type=MemoryType.TOOLBOX)
                        if winner:
                            record_id = winner["_id"]
                            merged_aliases = sorted(set((winner.get("aliases") or [])) | set(tool_record["aliases"]))
                            patch = {
                                "name": tool_key,
                                "aliases": merged_aliases,
                                "updated_at": time.time(),
                                "fqn": fqn,
                                "display_name": openai_name or tool_key,
                            }
                            # FIX: pass string id
                            self.memory_provider.update_by_id(str(record_id), patch, memory_store_type=MemoryType.TOOLBOX)
                        else:
                            raise

            # -------- runtime registry (for actual execution) ----------
            tool_id = str(record_id)
            if not callable(f):
                raise TypeError(f"register_tool expected callable, got {type(f).__name__}")

            # Ensure maps exist
            self._tools   = getattr(self, "_tools", {})
            self._by_name = getattr(self, "_by_name", {})
            self._by_fqn  = getattr(self, "_by_fqn", {})

            self._tools[tool_id] = f

            # index by both the canonical key and the OpenAI-exported name
            for nm in {tool_key, openai_name} - {None}:
                self._by_name.setdefault(nm, []).append(tool_id)
            if fqn:
                self._by_fqn[fqn] = tool_id

            # logging (avoid reserved 'name' in logging extra)
            logger = getattr(self, "logger", None)
            if logger:
                logger.debug(
                    "[toolbox] upserted tool key=%s id=%s fqn=%s aliases=%s",
                    tool_key, tool_id, fqn, ",".join(tool_record.get("aliases", []))
                )

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