# src/memorizz/llms/llm_factory.py

from typing import Dict, Any
from .llm_provider import LLMProvider
from .openai import OpenAI
from .azure import AzureOpenAI
from .grok import Grok

_PROVIDER_CACHE = {}

def _freeze_cfg(d: Dict[str, Any]) -> tuple:
    # Turn config into a hashable key; ignore transient flags if any
    return tuple(sorted(d.items()))


def create_llm_provider(config: Dict[str, Any]) -> LLMProvider:
    """
    Factory function to create an LLM provider instance from a configuration dictionary.

    Parameters:
    -----------
    config : Dict[str, Any]
        A dictionary containing the provider name and its specific parameters.
        Example for OpenAI: {"provider": "openai", "model": "gpt-4o"}
        Example for Azure: {"provider": "azure", "deployment_name": "my-gpt4"}

    Returns:
    --------
    LLMProvider
        An instance of the specified LLM provider.
        
    Raises:
    -------
    ValueError
        If the provider specified in the config is unknown.
    """
    provider_name = config.get("provider", "openai").lower()
    key = (provider_name, _freeze_cfg({k: v for k, v in config.items() if k != "provider"}))

    if key in _PROVIDER_CACHE:
        return _PROVIDER_CACHE[key]


    if provider_name == "openai":
        openai_config = config.copy(); openai_config.pop("provider", None)
        inst = OpenAI(**openai_config)

    elif provider_name == "azure":
        azure_config = config.copy(); azure_config.pop("provider", None)
        inst = AzureOpenAI(
            azure_endpoint=azure_config.get("azure_endpoint"),
            api_version=azure_config.get("api_version"),
            deployment_name=azure_config.get("deployment_name"),
        )

    elif provider_name == "grok":
        grok_config = config.copy(); grok_config.pop("provider", None)
        inst = Grok(**grok_config)
    # To extend, add more providers here:
    # elif provider_name == "anthropic":
    #     ...
        
    else:
        raise ValueError(f"Unknown LLM provider: '{provider_name}'")
    
    _PROVIDER_CACHE[key] = inst
    return inst
