import os
import json
import openai
import logging
from typing import Callable, List, Optional, TYPE_CHECKING, Dict, Any

from .llm_provider import LLMProvider

# Suppress httpx logs to reduce noise from API requests
logging.getLogger("httpx").setLevel(logging.WARNING)

# Use TYPE_CHECKING for forward references to avoid circular imports
if TYPE_CHECKING:
    from ..long_term_memory.procedural.toolbox.tool_schema import ToolSchemaType
import inspect

class OpenAI(LLMProvider):
    """
    A class for interacting with the OpenAI API.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize the OpenAI client.

        Parameters:
        -----------
        api_key : str
            The API key for the OpenAI API.
        model : str, optional
            The model to use for the OpenAI API.
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        
    def get_config(self) -> Dict[str, Any]:
        """Returns a serializable configuration for the OpenAI provider."""
        return {
            "provider": "openai",
            "model": self.model
        }


    def get_tool_metadata(self, func: Callable) -> Dict[str, Any]:
        """
        Enrich tool docs with the LLM, but constrain parameter names/types to the
        canonical JSON-Schema (if present) and ensure arrays always include `items`.
        Returns the SAME model type as `response.output_parsed`.
        """
        from ..long_term_memory.procedural.toolbox.tool_schema import ToolSchemaType
        import inspect


        # ---------- helpers ----------
        def _patch_array_items_inplace(schema: dict) -> None:
            if not isinstance(schema, dict):
                return
            if schema.get("type") == "array" and "items" not in schema:
                schema["items"] = {"type": "string"}
            for key in ("properties", "patternProperties", "$defs", "defs"):
                obj = schema.get(key)
                if isinstance(obj, dict):
                    for v in obj.values():
                        _patch_array_items_inplace(v)
            for key in ("anyOf", "allOf", "oneOf"):
                arr = schema.get(key)
                if isinstance(arr, list):
                    for v in arr:
                        _patch_array_items_inplace(v)
            if isinstance(schema.get("items"), dict):
                _patch_array_items_inplace(schema["items"])

        def _schema_excerpt(props: Dict[str, Any], req: list[str]) -> str:
            lines, req_set = [], set(req or [])
            for name, spec in (props or {}).items():
                t = (spec.get("type") or "string") if isinstance(spec, dict) else "string"
                d = (spec.get("description") or "") if isinstance(spec, dict) else ""
                if t == "array":
                    items = spec.get("items") or {"type": "string"}
                    t = f"array<{items.get('type','object') if isinstance(items, dict) else 'object'}>"
                lines.append(f"- {name}: {t}{' [required]' if name in req_set else ''}{(': ' + d) if d else ''}")
            return "\n".join(lines) or "(none)"

        # ---------- canonical schema (if MCP attached it) ----------
        schema = getattr(func, "_openai_parameters", None)
        has_schema = isinstance(schema, dict) and schema.get("type") == "object"
        schema_props = schema.get("properties", {}) if has_schema else {}
        schema_req   = list(schema.get("required", []) or []) if has_schema else []

        # ---------- prompt ----------
        docstring = func.__doc__ or ""
        signature = str(inspect.signature(func))
        func_name = func.__name__

        system_msg = {
            "role": "system",
            "content": (
                "You are a rigorous tool metadata assistant. Follow the caller-supplied JSON-Schema exactly. "
                "Never rename parameters; never invent parameters; never omit required parameters. "
                "For arrays, use JSON-Schema semantics: arrays MUST specify an `items` schema."
            )
        }

        constraints = ""
        if has_schema:
            constraints = (
                "CANONICAL SCHEMA (use these names/types verbatim; arrays include `items`):\n"
                f"{_schema_excerpt(schema_props, schema_req)}\n\n"
                "RULES:\n"
                "1) Use parameter names exactly as above (verbatim).\n"
                "2) Do not add parameters not listed above.\n"
                "3) The 'required' list must match the canonical schema.\n"
                "4) For arrays, do NOT say 'array of X'—use JSON-Schema (`type: array` + `items`).\n"
                "5) Output must conform to ToolSchemaType."
            )

        user_msg = {
            "role": "user",
            "content": (
                f"Generate enriched metadata for `{func_name}`.\n\n"
                f"- Docstring: {docstring}\n"
                f"- Signature: {signature}\n\n"
                "Your tasks:\n"
                "• Expand the description.\n"
                "• Provide clear descriptions for each parameter (using the canonical names, if given).\n"
                "• Provide example queries (optional).\n\n"
                + constraints
            )
        }

        # ---------- LLM call ----------
        response = self.client.responses.parse(
            model=self.model,
            input=[system_msg, user_msg],
            text_format=ToolSchemaType  # your existing schema class
        )
        parsed = response.output_parsed  # usually a Pydantic model
        # Normalize to a dict for controlled edits
        if hasattr(parsed, "model_dump"):
            data = parsed.model_dump()
        else:
            # already a dict in some providers
            data = dict(parsed)

        # ---------- enforce + fix arrays ----------
        if has_schema:
            # Ensure canonical schema has items everywhere
            _patch_array_items_inplace(schema)

            # Build enriched flat parameter list (includes items for arrays)
            req_set = set(schema_req)
            # Index any LLM param descriptions by name to enrich when schema lacks one
            prov_by_name = {}
            for p in (data.get("parameters") or []):
                if isinstance(p, dict) and "name" in p:
                    prov_by_name[str(p["name"]).lower()] = p

            alias_to_canonical = {"file_path": "path", "filepath": "path",
                                "source": "source_path", "destination": "destination_path"}

            flat_params: List[Dict[str, Any]] = []
            for canon_name, spec in schema_props.items():
                if not isinstance(spec, dict):
                    spec = {}
                schema_desc = str(spec.get("description", "") or "")
                prov = prov_by_name.get(canon_name.lower())
                if prov is None:
                    # borrow description from alias if present
                    for prov_nm, rec in prov_by_name.items():
                        if alias_to_canonical.get(prov_nm) == canon_name:
                            prov = rec
                            break
                prov_desc = str(prov.get("description", "") or "") if isinstance(prov, dict) else ""
                final_desc = schema_desc if schema_desc else prov_desc

                row = {
                    "name": canon_name,
                    "description": final_desc,
                    "type": spec.get("type", "string"),
                    "required": canon_name in req_set,
                }
                if row["type"] == "array":
                    items = spec.get("items") or {"type": "string"}
                    row["items"] = items
                    row["items_type"] = (items.get("type", "object") if isinstance(items, dict) else "object")
                flat_params.append(row)

            data["parameters"] = flat_params
            data["required"]   = list(schema_req)

            # Ensure nested JSON-Schema object is the canonical one
            fn_block = data.get("function")
            if isinstance(fn_block, dict):
                fn_block = dict(fn_block)
                fn_block["parameters"] = schema
                data["function"] = fn_block

            # Optional: canonicalize example queries
            if isinstance(data.get("queries"), list):
                data["queries"] = [
                    q.replace("file_path=", "path=").replace("filepath=", "path=") if isinstance(q, str) else q
                    for q in data["queries"]
                ]
        else:
            # No canonical schema: best-effort repair on LLM's nested schema + flat list
            fn_block = data.get("function")
            if isinstance(fn_block, dict):
                params_schema = fn_block.get("parameters")
                if isinstance(params_schema, dict):
                    _patch_array_items_inplace(params_schema)
                    fn_block = dict(fn_block)
                    fn_block["parameters"] = params_schema
                    data["function"] = fn_block

            flat = []
            for p in (data.get("parameters") or []):
                if not isinstance(p, dict):
                    continue
                q = dict(p)
                if q.get("type") == "array" and "items" not in q:
                    q["items"] = {"type": "string"}
                    q["items_type"] = "string"
                flat.append(q)
            if flat:
                data["parameters"] = flat

        # ---------- return the SAME model type the parser returned ----------
        OutType = parsed.__class__  # preserves the exact pydantic model class
        return OutType(**data)

    
    def augment_docstring(self, docstring: str) -> str:
        """
        Augment the docstring with an LLM generated description.

        Parameters:
        -----------
        docstring : str
            The docstring to augment.

        Returns:
        --------
        str
        """
        response = self.client.responses.create(
            model=self.model,
            input=f"Augment the docstring {docstring} by adding more details and examples."
        )

        return response.output_text
    
    def generate_queries(self, docstring: str) -> List[str]:
        """
        Generate queries for the tool.

        Parameters:
        -----------
        docstring : str
            The docstring to generate queries for.

        Returns:
        --------
        List[str]
        """
        response = self.client.responses.create(
            model=self.model,
            input=f"Generate queries for the docstring {docstring} by adding some examples of queries that can be used to leverage the tool."
        )

        return response.output_text
    
    def generate_text(self, prompt: str, instructions: str = None) -> str:
        """
        Generate text using OpenAI's API.

        Parameters:
            prompt (str): The prompt to generate text from.
            instructions (str): The instructions to use for the generation.

        Returns:
            str: The generated text.
        """
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=prompt)
        
        return response.output_text