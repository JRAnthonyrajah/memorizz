# src/memorizz/llms/grok.py
from __future__ import annotations
import os
import json
import openai
import logging
from typing import Callable, List, Optional, TYPE_CHECKING, Dict, Any, Iterable, AsyncIterator
import httpx, asyncio
from openai import AsyncOpenAI

from .llm_provider import LLMProvider

# Suppress httpx logs to reduce noise from API requests
logging.getLogger("httpx").setLevel(logging.WARNING)

# Use TYPE_CHECKING for forward references to avoid circular imports
if TYPE_CHECKING:
    from ..long_term_memory.procedural.toolbox.tool_schema import ToolSchemaType
import inspect

class Grok(LLMProvider):
    """
    Grok (xAI) provider implemented via the OpenAI Python SDK.
    We point the SDK at https://api.x.ai/v1 and authenticate with XAI_API_KEY.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "grok-2-1212",
        base_url: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        api_key : str, optional
            xAI API key; if omitted, read from env var XAI_API_KEY.
        model : str, optional
            Grok model (e.g., "grok-2-1212", "grok-2-0812", "grok-beta", "grok-vision-beta").
        base_url : str, optional
            Override the xAI endpoint (defaults to "https://api.x.ai/v1").
        """
        if api_key is None:
            api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("Missing xAI API key. Set XAI_API_KEY or pass api_key=...")

        self.model = model
        self.base_url = base_url or "https://api.x.ai/v1"

        # Use OpenAI client against xAI
        self.client = openai.OpenAI(api_key=api_key, base_url=self.base_url)

        # Set up async HTTP client
        self._httpx_async = httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(connect=3.0, read=18.0, write=10.0, pool=3.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        # True async OpenAI client pointed at xAI
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=self.base_url, http_client=self._httpx_async)

    def get_config(self) -> Dict[str, Any]:
        """Returns a serializable configuration for the Grok provider."""
        return {
            "provider": "grok",
            "model": self.model,
            "base_url": self.base_url
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

        # Ask the LLM using chat completions (xAI supports this)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg["content"]},
                {"role": "user", "content": user_msg["content"]}
            ],
            response_format={"type": "json_object"}
        )

        # Parse the JSON response
        try:
            parsed = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            parsed = {"function": {"name": func_name, "description": docstring}}

        # --- Patch nested JSON-Schema (function.parameters) regardless of container type ---
        try:
            fn = parsed.get("function", {})
            params_schema = fn.get("parameters")
            if isinstance(params_schema, dict):
                _patch_array_items_inplace(params_schema)
                fn["parameters"] = params_schema
                parsed["function"] = fn
        except Exception:
            # Never fail metadata generation over a patch
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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": f"Augment the docstring {docstring} by adding more details and examples."}
            ]
        )

        return response.choices[0].message.content

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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": f"Generate queries for the docstring {docstring} by adding some examples of queries that can be used to leverage the tool. Return as a JSON array of strings."}
            ],
            response_format={"type": "json_object"}
        )

        try:
            queries = json.loads(response.choices[0].message.content)
            if isinstance(queries, list):
                return queries
            elif isinstance(queries, dict) and "queries" in queries:
                return queries["queries"]
            else:
                # Fallback: split by newlines and clean up
                return [q.strip() for q in response.choices[0].message.content.split('\n') if q.strip()]
        except json.JSONDecodeError:
            # Fallback: split by newlines and clean up
            return [q.strip() for q in response.choices[0].message.content.split('\n') if q.strip()]

    def generate_text(self, prompt: str, instructions: str = None) -> str:
        """
        Generate text using xAI's API.

        Parameters:
            prompt (str): The prompt to generate text from.
            instructions (str): The instructions to use for the generation.

        Returns:
            str: The generated text.
        """
        messages = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        return response.choices[0].message.content

    async def async_generate_text(self, prompt: str, instructions: Optional[str] = None) -> str:
        messages = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": prompt})

        resp = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1024,
        )
        return resp.choices[0].message.content

    async def async_generate_batch(
        self, prompts: Iterable[str], instructions: Optional[str] = None, *, max_concurrency: int = 5
    ) -> List[str]:
        sem = asyncio.Semaphore(max_concurrency)
        async def one(p: str):
            async with sem:
                return await self.async_generate_text(p, instructions)
        return await asyncio.gather(*(one(p) for p in prompts))

    async def async_stream_text(self, prompt: str, instructions: Optional[str] = None) -> AsyncIterator[str]:
        """
        Stream text generation in chunks as they arrive from the LLM.

        Parameters:
            prompt: The prompt to generate text from
            instructions: Optional system instructions

        Yields:
            str: Text chunks as they are generated
        """
        messages = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": prompt})

        stream = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content

    async def aclose(self):
        await self._httpx_async.aclose()