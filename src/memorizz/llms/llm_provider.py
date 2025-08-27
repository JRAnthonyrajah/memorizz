# src/memorizz/llms/llm_provider.py

from typing import Dict, Any, List, Callable, Optional, Protocol, runtime_checkable, TYPE_CHECKING

# Use TYPE_CHECKING to handle forward references for type hints
if TYPE_CHECKING:
    from ..long_term_memory.procedural.toolbox.tool_schema import ToolSchemaType

"""
A protocol in Python (introduced in PEP 544 and part of the typing module) defines a structural typing rule. 
It specifies a set of methods and properties that a class must implement, 
but it does not require inheritance.

"If it walks like a duck and quacks like a duck, it's probably a duck." 🦆

"""

@runtime_checkable
class LLMProvider(Protocol):
    """
    A generic protocol that defines the contract for any LLM provider
    to be compatible with both the OpenAI and AzureOpenAI classes.
    """    
    # --- Attributes ---
    client: Any
    """Provides direct access to the underlying API client instance (e.g., openai.OpenAI or openai.AzureOpenAI)."""

    model: str
    """Stores the specific model or deployment name as a string (e.g., "gpt-4o")."""

    # --- Methods ---
    def get_tool_metadata(self, func: Callable) -> Dict[str, Any]:
        """Creates structured metadata (a JSON schema) from a Python function."""
        ...

    def augment_docstring(self, docstring: str) -> str:
        """Uses the LLM to enhance a function's docstring with more detail."""
        ...

    def generate_queries(self, docstring: str) -> List[str]:
        """Generates a list of example user queries for a given tool."""
        ...

    def generate_text(self, prompt: str, instructions: Optional[str] = None) -> str:
        """A high-level method for simple text generation."""
        ...
        
    def get_config(self) -> Dict[str, Any]:
        """
        Returns a serializable dictionary of the provider's configuration.
        This is used for saving and reconstructing the agent.
        """
        ...