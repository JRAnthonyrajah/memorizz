# src/memorizz/llms/grok.py
import os
import logging
import inspect
from typing import Callable, List, Optional, TYPE_CHECKING, Dict, Any

import openai  # use OpenAI SDK pointed at xAI's base_url
from .llm_provider import LLMProvider

# Reduce httpx noise (consistent with your OpenAI class)
logging.getLogger("httpx").setLevel(logging.WARNING)

if TYPE_CHECKING:
    from ..long_term_memory.procedural.toolbox.tool_schema import ToolSchemaType


class Grok(LLMProvider):
    """
    Grok (xAI) provider implemented via the OpenAI Python SDK.
    We point the SDK at https://api.x.ai/v1 and authenticate with XAI_API_KEY.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "grok-code-fast-1",
        base_url: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        api_key : str, optional
            xAI API key; if omitted, read from env var XAI_API_KEY.
        model : str, optional
            Grok model (e.g., "grok-4-0709", "grok-3", "grok-3-mini", "grok-code-fast-1").
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

    def get_config(self) -> Dict[str, Any]:
        return {"provider": "grok", "model": self.model, "base_url": self.base_url}

    def get_tool_metadata(self, func: Callable) -> Dict[str, Any]:
        # Local import to avoid circular references
        from ..long_term_memory.procedural.toolbox.tool_schema import ToolSchemaType

        docstring = func.__doc__ or ""
        signature = str(inspect.signature(func))
        func_name = func.__name__

        system_msg = {
            "role": "system",
            "content": (
                "You are an expert metadata augmentation assistant specializing in JSON schema discovery "
                "and documentation enhancement.\n\n"
                f"**IMPORTANT**: Use the function name exactly as provided (`{func_name}`) and do NOT rename it."
            ),
        }
        user_msg = {
            "role": "user",
            "content": (
                f"Generate enriched metadata for the function `{func_name}`.\n\n"
                f"- Docstring: {docstring}\n"
                f"- Signature: {signature}\n\n"
                "Enhance the metadata by:\n"
                "• Expanding the docstring into a detailed description.\n"
                "• Writing clear natural‐language descriptions for each parameter, including type, purpose, and constraints.\n"
                "• Identifying which parameters are required.\n"
                "• (Optional) Suggesting example queries or use cases.\n\n"
                "Produce a JSON object that strictly adheres to the ToolSchemaType structure."
            ),
        }

        resp = self.client.responses.parse(
            model=self.model,
            input=[system_msg, user_msg],
            text_format=ToolSchemaType,
        )
        return resp.output_parsed

    def augment_docstring(self, docstring: str) -> str:
        resp = self.client.responses.create(
            model=self.model,
            input=f"Augment the docstring {docstring} by adding more details and examples.",
        )
        return resp.output_text

    def generate_queries(self, docstring: str) -> List[str]:
        resp = self.client.responses.create(
            model=self.model,
            input=f"Generate queries for the docstring {docstring} by adding some examples of queries that can be used to leverage the tool.",
        )
        # Mirror your OpenAI class’s return type (string payload). Split upstream if needed.
        return resp.output_text

    def generate_text(self, prompt: str, instructions: str = None) -> str:
        resp = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=prompt,
        )
        return resp.output_text
