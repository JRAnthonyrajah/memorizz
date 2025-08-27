# src/memorizz/llms/llm_factory.py

from typing import Dict, Any
from .llm_provider import LLMProvider
from .openai import OpenAI
from .azure import AzureOpenAI

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
    print("CREATE_LLM_PROVIDER")
    provider_name = config.get("provider", "openai").lower()
    print(provider_name)
    if provider_name == "openai":
        # Create a copy of the config and remove the 'provider' key
        openai_config = config.copy()
        openai_config.pop("provider", None)
        return OpenAI(**openai_config)
        
    elif provider_name == "azure":
        # Create a copy of the config and remove the 'provider' key
        azure_config = config.copy()
        azure_config.pop("provider", None)
        return AzureOpenAI(
            azure_endpoint=azure_config.get("azure_endpoint"),
            api_version=azure_config.get("api_version"),
            deployment_name=azure_config.get("deployment_name"),
        )
    
    # To extend, add more providers here:
    # elif provider_name == "anthropic":
    #     ...
        
    else:
        raise ValueError(f"Unknown LLM provider: '{provider_name}'")