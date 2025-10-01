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
        Use the LLM to enrich docs, but *constrain* parameter names to the canonical
        JSON-Schema attached by the MCP adapter (func._openai_parameters), and
        ensure any array types include an 'items' subschema before returning.
        """
        from ..long_term_memory.procedural.toolbox.tool_schema import ToolSchemaType
        import inspect, json
        from typing import Any

        docstring  = func.__doc__ or ""
        signature  = str(inspect.signature(func))
        func_name  = func.__name__

        # Pull canonical schema (for prompting constraints only)
        schema = getattr(func, "_openai_parameters", None)
        schema_props, schema_req = {}, []

        # --- helper: add items={"type":"string"} wherever an array lacks items ---
        def _patch_array_items_inplace(s: Any) -> None:
            if not isinstance(s, dict):
                return
            if s.get("type") == "array" and "items" not in s:
                s["items"] = {"type": "string"}
            # descend common schema containers
            for key in ("properties", "patternProperties", "$defs", "defs"):
                obj = s.get(key)
                if isinstance(obj, dict):
                    for v in obj.values():
                        _patch_array_items_inplace(v)
            for key in ("anyOf", "allOf", "oneOf"):
                arr = s.get(key)
                if isinstance(arr, list):
                    for v in arr:
                        _patch_array_items_inplace(v)
            if isinstance(s.get("items"), dict):
                _patch_array_items_inplace(s["items"])

        if isinstance(schema, dict) and schema.get("type") == "object":
            schema_props = schema.get("properties", {}) or {}
            schema_req   = list(schema.get("required", []) or [])

        # For the prompt: concise excerpt of canonical params
        def _excerpt(props: dict) -> str:
            rows = []
            for n, spec in props.items():
                if isinstance(spec, dict):
                    t = spec.get("type") or "string"
                    d = spec.get("description") or ""
                    enum = spec.get("enum")
                    const = spec.get("const")
                    fmt = spec.get("format")
                    default = spec.get("default")
                    extra = []
                    if enum:
                        # show canonical literals exactly as the tool expects
                        extra.append("enum={" + ", ".join([repr(x) for x in enum]) + "}")
                    if const is not None:
                        extra.append(f"const={repr(const)}")
                    if fmt:
                        extra.append(f"format={fmt}")
                    if default is not None:
                        extra.append(f"default={repr(default)}")
                    extrastr = (" [" + "; ".join(extra) + "]") if extra else ""
                    req = " [required]" if n in schema_req else ""
                    desc = (": " + d) if d else ""
                    rows.append(f"- {n} ({t}){req}{extrastr}{desc}")
                else:
                    rows.append(f"- {n} (string){' [required]' if n in schema_req else ''}")
            return "\n".join(rows) or "(none)"


        disallowed_example = []
        if "path" in schema_props:
            disallowed_example = [
                {"wrong": "file_path", "right": "path"},
                {"wrong": "filepath",  "right": "path"},
            ]

        system_msg = {
            "role": "system",
            "content": (
                "You are an expert metadata assistant. Follow the caller's canonical JSON-Schema exactly. "
                "Never rename parameters; never add extra parameters; never omit required ones."
            )
        }

        constraint_block = (
            "CANONICAL PARAMETERS:\n"
            f"{_excerpt(schema_props)}\n\n"
            "RULES:\n"
            "1) Use these names exactly (verbatim). Do not invent or rename.\n"
            "2) Include all and only these parameters in the 'parameters' array.\n"
            "3) The 'required' list must match the canonical 'required'.\n"
            "4) If you were going to use any aliases, replace them with the canonical names.\n"
            "5) For any parameter with enum/const, you MUST use exactly one of the listed literals (case-sensitive).\n"
            "6) Do not invent defaults or additional properties.\n"
        )
        if disallowed_example:
            constraint_block += (
                "EXAMPLES (wrong → right):\n" +
                "\n".join([f"- {ex['wrong']} → {ex['right']}" for ex in disallowed_example]) +
                "\n"
            )
        if "sitemap" in schema_props and isinstance(schema_props["sitemap"], dict) and schema_props["sitemap"].get("enum"):
            constraint_block += (
                "EXAMPLES (wrong → right):\n"
                "- true → \"include\"\n"
                "- false → \"skip\"\n"
                "- all → \"include\"\n"
                "- Include → \"include\"\n"
                "- ONLY → \"only\"\n"
            )


        user_msg = {
            "role": "user",
            "content": (
                f"Generate enriched metadata for `{func_name}`.\n\n"
                f"- Docstring: {docstring}\n"
                f"- Signature: {signature}\n\n"
                "Your job:\n"
                "• Expand the description.\n"
                "• Provide clear descriptions for each parameter (using the canonical names).\n"
                "• Provide example queries (if helpful).\n\n"
                "Output MUST adhere to ToolSchemaType and the constraints below.\n\n"
                + constraint_block +
                "\nIMPORTANT: If the canonical schema is present, you must mirror its parameter names and 'required' exactly."
            )
        }

        # Ask the LLM using your structured parser
        response = self.client.responses.parse(
            model=self.model,
            input=[system_msg, user_msg],
            text_format=ToolSchemaType
        )

        parsed = response.output_parsed  # keep same object/type

        # --- Patch nested JSON-Schema (function.parameters) regardless of container type ---
        try:
            fn = getattr(parsed, "function", None)
            # Case A: dict-shaped
            if isinstance(fn, dict):
                params_schema = fn.get("parameters")
                if isinstance(params_schema, dict):
                    _patch_array_items_inplace(params_schema)
                    fn["parameters"] = params_schema
                    setattr(parsed, "function", fn)
            else:
                # Case B: pydantic model or similar
                # Try to get dict, patch, then set back if possible
                if hasattr(fn, "model_dump"):
                    fn_dict = fn.model_dump()
                    params_schema = fn_dict.get("parameters")
                    if isinstance(params_schema, dict):
                        _patch_array_items_inplace(params_schema)
                        fn_dict["parameters"] = params_schema
                        # If function is a pydantic model with model_construct or copy(update=...)
                        if hasattr(fn, "model_copy"):
                            fn_new = fn.model_copy(update={"parameters": params_schema})
                            setattr(parsed, "function", fn_new)
                        else:
                            # Fallback: set the field directly if present
                            try:
                                setattr(fn, "parameters", params_schema)
                            except Exception:
                                # As a last resort, replace with the dict form
                                setattr(parsed, "function", fn_dict)
                elif isinstance(fn, object) and hasattr(fn, "__dict__"):
                    # Generic object: try setattr
                    ps = getattr(fn, "parameters", None)
                    if isinstance(ps, dict):
                        _patch_array_items_inplace(ps)
                        try:
                            setattr(fn, "parameters", ps)
                        except Exception:
                            pass
        except Exception:
            # Never fail metadata generation over a patch
            pass

        # --- Optional: patch flat parameter rows if present ---
        try:
            flat = getattr(parsed, "parameters", None)
            if isinstance(flat, list):
                for p in flat:
                    if isinstance(p, dict) and p.get("type") == "array" and "items" not in p:
                        p["items"] = {"type": "string"}
                        p.setdefault("items_type", "string")
        except Exception:
            pass

        return parsed




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