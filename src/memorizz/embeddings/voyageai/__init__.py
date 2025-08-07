"""
VoyageAI Embedding Provider

This package contains the VoyageAI embedding provider implementation with support for:
- Text embeddings with multiple models and configurable dimensions
- Multimodal embeddings for text and images  
- Contextualized chunk embeddings for documents
"""

from .provider import VoyageAIEmbeddingProvider

__all__ = ['VoyageAIEmbeddingProvider'] 