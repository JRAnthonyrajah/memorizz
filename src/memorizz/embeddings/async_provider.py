"""
GPU-accelerated async embedding provider for high-performance embedding computation.

This module provides AsyncEmbeddingProvider, which uses:
- GPU acceleration (CUDA/MPS) when available
- Async execution via thread pool
- Batch processing for multiple texts
- Model caching for efficiency
"""

import asyncio
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

logger = logging.getLogger(__name__)


class AsyncEmbeddingProvider:
    """
    Async embedding provider with GPU acceleration and batching support.

    Features:
    - GPU acceleration (CUDA/MPS) with automatic fallback to CPU
    - Async execution to avoid blocking event loop
    - Batch processing for efficient GPU utilization
    - Thread pool for CPU-bound work
    - Model caching (loaded once, reused)

    Example:
        ```python
        # Single text
        provider = AsyncEmbeddingProvider()
        embedding = await provider.encode_async(["What is machine learning?"])

        # Batch processing (much faster!)
        texts = ["query 1", "query 2", "query 3"]
        embeddings = await provider.encode_batch_async(texts)
        ```
    """

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        max_workers: int = 4
    ):
        """
        Initialize async embedding provider.

        Parameters:
        -----------
        model_name : str, optional
            HuggingFace model name. Defaults to env var EMB_MODEL or
            'sentence-transformers/all-MiniLM-L6-v2'
        device : str, optional
            Device to use: 'cuda', 'cpu', or 'mps'. Auto-detects if None.
        max_workers : int
            Thread pool size for CPU-bound work (default: 4)
        """

        # Import here to make it optional
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError:
            raise ImportError(
                "AsyncEmbeddingProvider requires sentence-transformers and torch. "
                "Install with: pip install 'memorizz[gpu]' or 'memorizz[performance]'"
            )

        # Get model name from env or use default
        self.model_name = model_name or os.getenv(
            'EMB_MODEL',
            'sentence-transformers/all-MiniLM-L6-v2'
        )

        # Auto-detect best device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"🚀 Using CUDA GPU for embeddings (device: {torch.cuda.get_device_name(0)})")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
                logger.info("🚀 Using Apple Silicon GPU (MPS) for embeddings")
            else:
                device = 'cpu'
                logger.info("Using CPU for embeddings (GPU not available)")
        else:
            logger.info(f"Using {device} for embeddings (manual selection)")

        self.device = device

        # Load model once (cached for all future calls)
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name).to(self.device)
        logger.info(f"✅ Model loaded on {self.device}")

        # Get embedding dimensions
        self.dimensions = self.model.get_sentence_embedding_dimension()

        # Thread pool for CPU-bound work (when not using GPU)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def encode_async(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> List[List[float]]:
        """
        Async encoding with GPU acceleration.

        Offloads computation to thread pool to avoid blocking the event loop.
        GPU work is still fast, but we use threads to keep async code responsive.

        Parameters:
        -----------
        texts : List[str]
            List of texts to encode
        batch_size : int
            Batch size for GPU processing (default: 32)
            Larger batches = better GPU utilization
        show_progress : bool
            Show progress bar (default: False)
        normalize : bool
            L2 normalize embeddings (default: True)

        Returns:
        --------
        List[List[float]]
            List of embedding vectors
        """

        if not texts:
            return []

        # Offload to thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()

        embeddings = await loop.run_in_executor(
            self.executor,
            lambda: self.model.encode(
                texts,
                device=self.device,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            ).tolist()
        )

        return embeddings

    async def encode_single_async(
        self,
        text: str,
        normalize: bool = True
    ) -> List[float]:
        """
        Encode single text (convenience method).

        Parameters:
        -----------
        text : str
            Text to encode
        normalize : bool
            L2 normalize embedding (default: True)

        Returns:
        --------
        List[float]
            Embedding vector
        """
        embeddings = await self.encode_async([text], normalize=normalize)
        return embeddings[0]

    async def encode_batch_async(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[List[float]]:
        """
        Batch encode multiple texts efficiently.

        This is much more efficient than calling encode_single_async multiple times,
        especially on GPU where batch processing is significantly faster.

        Parameters:
        -----------
        texts : List[str]
            List of texts to encode
        batch_size : int
            GPU batch size (default: 32)

        Returns:
        --------
        List[List[float]]
            List of embedding vectors
        """
        return await self.encode_async(texts, batch_size=batch_size)

    def get_dimensions(self) -> int:
        """Get embedding dimensionality"""
        return self.dimensions

    def get_model_name(self) -> str:
        """Get model name"""
        return self.model_name

    def get_device(self) -> str:
        """Get device (cuda/mps/cpu)"""
        return self.device

    def shutdown(self):
        """Shutdown thread pool"""
        self.executor.shutdown(wait=False)

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.executor.shutdown(wait=False)
        except:
            pass


# Global singleton instance
_global_async_provider: Optional[AsyncEmbeddingProvider] = None


def get_async_embedding_provider() -> AsyncEmbeddingProvider:
    """
    Get or create global async embedding provider (singleton pattern).

    Returns:
    --------
    AsyncEmbeddingProvider
        Global async embedding provider instance
    """
    global _global_async_provider

    if _global_async_provider is None:
        _global_async_provider = AsyncEmbeddingProvider(
            model_name=os.getenv('EMB_MODEL'),
            device=os.getenv('EMB_DEVICE')
        )

    return _global_async_provider


async def get_embedding_async(text: str) -> List[float]:
    """
    Async version of get_embedding() with GPU acceleration.

    This is a drop-in async replacement for get_embedding().
    Much faster on GPU, non-blocking for async code.

    Parameters:
    -----------
    text : str
        Text to embed

    Returns:
    --------
    List[float]
        Embedding vector

    Example:
    --------
    ```python
    embedding = await get_embedding_async("What is machine learning?")
    ```
    """
    provider = get_async_embedding_provider()
    return await provider.encode_single_async(text)


async def get_embeddings_batch_async(
    texts: List[str],
    batch_size: int = 32
) -> List[List[float]]:
    """
    Batch embedding computation for multiple texts.

    MUCH more efficient than calling get_embedding_async() multiple times,
    especially on GPU where batch processing gives significant speedup.

    Parameters:
    -----------
    texts : List[str]
        List of texts to embed
    batch_size : int
        GPU batch size (default: 32)

    Returns:
    --------
    List[List[float]]
        List of embedding vectors

    Example:
    --------
    ```python
    texts = ["query 1", "query 2", "query 3"]
    embeddings = await get_embeddings_batch_async(texts)
    # 10-20x faster than 3 individual calls!
    ```
    """
    provider = get_async_embedding_provider()
    return await provider.encode_batch_async(texts, batch_size=batch_size)
