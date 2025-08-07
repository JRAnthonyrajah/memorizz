from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class EmbeddingProvider(Enum):
    """Enumeration of supported embedding providers."""
    OPENAI = "openai"
    OLLAMA = "ollama"
    VOYAGEAI = "voyageai"

class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the embedding provider with configuration.
        
        Parameters:
        -----------
        config : Optional[Dict[str, Any]]
            Provider-specific configuration parameters
        """
        self.config = config or {}
    
    @abstractmethod
    def get_embedding(self, text: str, **kwargs) -> List[float]:
        """
        Generate embedding for the given text.
        
        Parameters:
        -----------
        text : str
            The text to embed
        **kwargs
            Additional provider-specific parameters
            
        Returns:
        --------
        List[float]
            The embedding vector
        """
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        """
        Get the dimensionality of embeddings produced by this provider.
        
        Returns:
        --------
        int
            Number of dimensions in the embedding vector
        """
        pass
    
    @abstractmethod
    def get_default_model(self) -> str:
        """
        Get the default model name for this provider.
        
        Returns:
        --------
        str
            Default model identifier
        """
        pass

class EmbeddingManager:
    """
    Central manager for embedding providers with configuration support.
    Implements the Factory pattern for provider creation.
    """
    
    def __init__(self, provider: Union[str, EmbeddingProvider] = EmbeddingProvider.OPENAI, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the embedding manager.
        
        Parameters:
        -----------
        provider : Union[str, EmbeddingProvider]
            The embedding provider to use
        config : Optional[Dict[str, Any]]
            Configuration for the selected provider
        """
        if isinstance(provider, str):
            try:
                provider = EmbeddingProvider(provider.lower())
            except ValueError:
                raise ValueError(f"Unsupported embedding provider: {provider}")
        
        self.provider_type = provider
        self.config = config or {}
        self._provider = self._create_provider()
    
    def _create_provider(self) -> BaseEmbeddingProvider:
        """Create and return the appropriate embedding provider instance."""
        if self.provider_type == EmbeddingProvider.OPENAI:
            from .openai import OpenAIEmbeddingProvider
            return OpenAIEmbeddingProvider(self.config)
        elif self.provider_type == EmbeddingProvider.OLLAMA:
            from .ollama import OllamaEmbeddingProvider
            return OllamaEmbeddingProvider(self.config)
        elif self.provider_type == EmbeddingProvider.VOYAGEAI:
            from .voyageai import VoyageAIEmbeddingProvider
            return VoyageAIEmbeddingProvider(self.config)
        else:
            raise ValueError(f"Provider {self.provider_type} not implemented")
    
    def get_embedding(self, text: str, **kwargs) -> List[float]:
        """
        Generate embedding using the configured provider.
        
        Parameters:
        -----------
        text : str
            The text to embed
        **kwargs
            Additional provider-specific parameters
            
        Returns:
        --------
        List[float]
            The embedding vector
        """
        return self._provider.get_embedding(text, **kwargs)
    
    def get_dimensions(self) -> int:
        """Get the dimensionality of embeddings from the current provider."""
        return self._provider.get_dimensions()
    
    def get_default_model(self) -> str:
        """Get the default model for the current provider."""
        return self._provider.get_default_model()
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the current provider configuration.
        
        Returns:
        --------
        Dict[str, Any]
            Provider information including type, model, and dimensions
        """
        return {
            "provider": self.provider_type.value,
            "model": self.get_default_model(),
            "dimensions": self.get_dimensions(),
            "config": self.config
        }

# Global embedding manager instance (can be reconfigured)
_global_embedding_manager: Optional[EmbeddingManager] = None

def configure_embeddings(provider: Union[str, EmbeddingProvider] = EmbeddingProvider.OPENAI, config: Optional[Dict[str, Any]] = None) -> EmbeddingManager:
    """
    Configure the global embedding provider.
    
    Parameters:
    -----------
    provider : Union[str, EmbeddingProvider]
        The embedding provider to use globally
    config : Optional[Dict[str, Any]]
        Configuration for the selected provider
        
    Returns:
    --------
    EmbeddingManager
        The configured embedding manager
    """
    global _global_embedding_manager
    _global_embedding_manager = EmbeddingManager(provider, config)
    logger.info(f"Configured global embedding provider: {_global_embedding_manager.get_provider_info()}")
    return _global_embedding_manager

def get_embedding_manager() -> EmbeddingManager:
    """
    Get the global embedding manager, creating a default one if none exists.
    
    Returns:
    --------
    EmbeddingManager
        The global embedding manager instance
    """
    global _global_embedding_manager
    if _global_embedding_manager is None:
        logger.info("No global embedding manager configured, using default OpenAI provider")
        _global_embedding_manager = EmbeddingManager()
    return _global_embedding_manager

# Convenience functions for backward compatibility
def get_embedding(text: str, **kwargs) -> List[float]:
    """
    Generate embedding using the globally configured provider.
    Provides backward compatibility for existing code.
    
    Parameters:
    -----------
    text : str
        The text to embed
    **kwargs
        Additional provider-specific parameters
        
    Returns:
    --------
    List[float]
        The embedding vector
    """
    return get_embedding_manager().get_embedding(text, **kwargs)

def get_embedding_dimensions(model: Optional[str] = None) -> int:
    """
    Get embedding dimensions from the globally configured provider.
    Provides backward compatibility for existing code.
    
    Parameters:
    -----------
    model : Optional[str]
        Model parameter (for backward compatibility, may be ignored)
        
    Returns:
    --------
    int
        The number of dimensions in the embedding vector
    """
    return get_embedding_manager().get_dimensions() 