# src/memorizz/embeddings/huggingface/provider.py
import logging
from typing import List, Dict, Any, Optional, Union

import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None  # We'll raise a clear error in __init__

from .. import BaseEmbeddingProvider  # provided by memorizz.embeddings package

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """
    HuggingFace (local) embedding provider using `sentence-transformers`.

    Config keys (all optional except `model` if you don't like the default):
      - model: str
          HF model id or local path. Default: all-MiniLM-L6-v2 (384 dims).
      - device: str
          "cpu", "cuda", "mps" (Apple), or None to auto-select.
      - trust_remote_code: bool
          Pass through to SentenceTransformer for custom models.
      - normalize: bool
          L2-normalize output vectors (default: True) for cosine similarity.
      - dtype: str
          "float32" (default) or "float16" (only if supported by device).
      - batch_size: int
          For future batch APIs; single-text calls ignore this.
      - revision: str
          Optional model revision/tag.
      - cache_folder: str
          Where to cache downloaded models.
    """

    # Popular models with known dimensions (for reference only).
    # We still detect dynamically in case you supply any other model.
    KNOWN_MODEL_DIMS = {
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-MiniLM-L12-v2": 384,
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": 384,
        "sentence-transformers/all-distilroberta-v1": 768,
        "intfloat/e5-small-v2": 384,
        "intfloat/e5-base-v2": 768,
        "intfloat/e5-large-v2": 1024,
        "mixedbread-ai/mxbai-embed-large-v1": 1024,
        "nomic-ai/nomic-embed-text-v1": 768,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {})
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required. Install with `pip install -U sentence-transformers`."
            )

        self.model_id: str = self.config.get("model", _DEFAULT_MODEL)
        self.normalize: bool = bool(self.config.get("normalize", True))
        self.trust_remote_code: bool = bool(self.config.get("trust_remote_code", False))
        self.revision: Optional[str] = self.config.get("revision")
        self.cache_folder: Optional[str] = self.config.get("cache_folder")
        self.batch_size: int = int(self.config.get("batch_size", 32))

        # dtype
        dtype_str = str(self.config.get("dtype", "float32")).lower()
        if dtype_str not in {"float32", "float16"}:
            logger.warning("Unsupported dtype '%s'; falling back to float32", dtype_str)
            dtype_str = "float32"
        self.dtype = np.float32 if dtype_str == "float32" else np.float16

        # device selection
        device_cfg: Optional[str] = self.config.get("device")
        self.device: str = self._select_device(device_cfg)

        # load model
        logger.info("Loading HF sentence-transformer model=%s on device=%s", self.model_id, self.device)
        self.model = SentenceTransformer(
            self.model_id,
            device=self.device,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            cache_folder=self.cache_folder,
        )

        # sentence-transformers exposes a reliable dimension getter
        try:
            self._dims = int(self.model.get_sentence_embedding_dimension())
        except Exception:
            # Fallback: try to infer with a tiny forward pass (should be rare)
            tmp = self._embed_text_local("__probe__")
            self._dims = int(len(tmp))

        # Optional: move to half precision when on CUDA and requested
        if torch is not None and self.device.startswith("cuda") and self.dtype == np.float16:
            try:
                self.model = self.model.half()  # type: ignore[attr-defined]
            except Exception:
                logger.warning("Requested float16 but model.half() not supported; continuing in float32.")
                self.dtype = np.float32

        logger.info("Initialized HuggingFace provider model=%s, dims=%d, normalize=%s",
                    self.model_id, self._dims, self.normalize)

    # ----- BaseEmbeddingProvider API -----

    def get_embedding(self, text: str, **kwargs) -> List[float]:
        """
        Compute an embedding for a single string.
        Per-call overrides:
          - normalize: bool
        """
        if not isinstance(text, str):
            text = str(text)

        normalize = bool(kwargs.get("normalize", self.normalize))
        vec = self._embed_text_local(text, normalize=normalize)
        return vec

    def get_dimensions(self) -> int:
        return int(self._dims)

    def get_default_model(self) -> str:
        return self.model_id

    # ----- Class helpers (optional, align with OpenAI example) -----

    @classmethod
    def get_available_models(cls) -> List[str]:
        # We can’t enumerate HF hub without network; return a curated list.
        return list(cls.KNOWN_MODEL_DIMS.keys())

    @classmethod
    def get_model_max_dimensions(cls, model: str) -> int:
        # For local models, “max” = actual dimension. Unknown → probe at runtime.
        return int(cls.KNOWN_MODEL_DIMS.get(model, 4096))

    # ----- Internal helpers -----

    def _select_device(self, device_cfg: Optional[str]) -> str:
        if device_cfg:
            return device_cfg
        # Auto-select
        if torch is None:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        # Apple Silicon w/ MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
        return "cpu"

    def _embed_text_local(self, text: str, normalize: Optional[bool] = None) -> List[float]:
        text = (text or "").replace("\n", " ").strip()
        if not text:
            return [0.0] * self._dims

        # sentence-transformers returns numpy array when convert_to_numpy=True
        vec = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we handle normalization to control dtype
            batch_size=1,
            show_progress_bar=False,
        )

        vec = vec.astype(self.dtype, copy=False)

        do_norm = self.normalize if normalize is None else bool(normalize)
        if do_norm:
            # L2 normalize
            denom = np.linalg.norm(vec)
            if denom > 0:
                vec = vec / denom

        # Ensure Python list[float]
        if vec.dtype != np.float32 and vec.dtype != np.float64:
            vec = vec.astype(np.float32)
        return vec.tolist()
