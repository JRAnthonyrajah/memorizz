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
        JSON-Schema attached by the MCP adapter (func._openai_parameters).
        """
        from ..long_term_memory.procedural.toolbox.tool_schema import ToolSchemaType
        import inspect, json

        docstring  = func.__doc__ or ""
        signature  = str(inspect.signature(func))
        func_name  = func.__name__

        # Pull canonical schema from the MCP adapter (if present)
        schema = getattr(func, "_openai_parameters", None)
        schema_props  = {}
        schema_req    = []
        if isinstance(schema, dict) and schema.get("type") == "object":
            schema_props = schema.get("properties", {}) or {}
            schema_req   = list(schema.get("required", []) or [])

        # Prepare a concise schema excerpt for the prompt (names + types + desc)
        def _excerpt(props: dict) -> str:
            rows = []
            for n, spec in props.items():
                t = (spec.get("type") or "string") if isinstance(spec, dict) else "string"
                d = (spec.get("description") or "") if isinstance(spec, dict) else ""
                rows.append(f"- {n} ({t}) {('[required]' if n in schema_req else '')}"
                            f"{(': ' + d) if d else ''}")
            return "\n".join(rows) or "(none)"

        # A small alias blacklist example to steer the model
        disallowed_example = []
        if "path" in schema_props:
            disallowed_example = [
                {"wrong": "file_path", "right": "path"},
                {"wrong": "filepath",  "right": "path"},
            ]

        # ---- Prompt with hard constraints on parameter names ----
        system_msg = {
            "role": "system",
            "content": (
                "You are an expert metadata assistant. Follow the caller's canonical JSON-Schema exactly. "
                "Never rename parameters; never add extra parameters; never omit required ones."
            )
        }

        # Build a strict instruction block the model cannot miss
        constraint_block = (
            "CANONICAL PARAMETERS:\n"
            f"{_excerpt(schema_props)}\n\n"
            "RULES:\n"
            "1) Use these names exactly (verbatim). Do not invent or rename.\n"
            "2) Include all and only these parameters in the 'parameters' array.\n"
            "3) The 'required' list must match the canonical 'required'.\n"
            "4) If you were going to use any aliases, replace them with the canonical names.\n"
        )
        if disallowed_example:
            constraint_block += (
                "EXAMPLES (wrong → right):\n" +
                "\n".join([f"- {ex['wrong']} → {ex['right']}" for ex in disallowed_example]) +
                "\n"
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

        # 1) Let the LLM produce the metadata
        response = self.client.responses.parse(
            model=self.model,
            input=[system_msg, user_msg],
            text_format=ToolSchemaType
        )

        return response.output_parsed
    
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