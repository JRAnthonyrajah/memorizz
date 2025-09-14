# src/memorizz/embeddings/huggingface/provider.py
import logging, os, re
from typing import List, Dict, Any, Optional

import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from .. import BaseEmbeddingProvider

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_META_ERR_RE = re.compile(r"meta tensor|to_empty\(\)", re.IGNORECASE)


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
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
            raise ImportError("sentence-transformers is required (pip install -U sentence-transformers).")

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

        # Select device (prefer MPS if present; otherwise CPU)
        requested_device: Optional[str] = self.config.get("device")
        self.device: str = self._select_device(requested_device)

        # Hard disable any Accelerate meta-init path that might be enabled by env
        # (these envs are harmless if not present)
        os.environ.pop("ACCELERATE_USE_FSDP", None)
        os.environ.pop("ACCELERATE_MIXED_PRECISION", None)

        # Try loading on chosen device; on meta-related failure, retry on CPU.
        self._init_on_device(self.device)

        # Determine dimensions
        try:
            self._dims = int(self.model.get_sentence_embedding_dimension())
        except Exception:
            tmp = self._embed_text_local("__probe__")
            self._dims = int(len(tmp))

        logger.info("Initialized HuggingFace provider model=%s, dims=%d, device=%s, normalize=%s",
                    self.model_id, self._dims, self.device, self.normalize)

    # ---------- BaseEmbeddingProvider API ----------

    def get_embedding(self, text: str, **kwargs) -> List[float]:
        if not isinstance(text, str):
            text = str(text)
        normalize = bool(kwargs.get("normalize", self.normalize))
        vec = self._embed_text_local(text, normalize=normalize)
        return vec

    def get_dimensions(self) -> int:
        return int(self._dims)

    def get_default_model(self) -> str:
        return self.model_id

    @classmethod
    def get_available_models(cls) -> List[str]:
        return list(cls.KNOWN_MODEL_DIMS.keys())

    @classmethod
    def get_model_max_dimensions(cls, model: str) -> int:
        return int(cls.KNOWN_MODEL_DIMS.get(model, 4096))

    # ---------- Internals ----------

    def _select_device(self, device_cfg: Optional[str]) -> str:
        if device_cfg:
            return device_cfg
        if torch is None:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        # MPS can stay as opt-in; uncomment next two lines only if you know it's stable in your env.
        # if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        #     return "mps"
        return "cpu"


    def _init_on_device(self, device: str) -> None:
        logger.info("Loading SentenceTransformer(model=%s) on CPU first (target=%s)", self.model_id, device)
        try:
            # 1) Materialize weights on CPU — guarantees real tensors, not 'meta'
            self.model = SentenceTransformer(
                self.model_id,
                device="cpu",
                trust_remote_code=self.trust_remote_code,
                revision=self.revision,
                cache_folder=self.cache_folder,
            )
            # 2) Warm-up on CPU to fully instantiate modules
            _ = self.model.encode("__warmup__", convert_to_numpy=True, show_progress_bar=False)

            # 3) Only then, move to the requested device (if not CPU)
            if device and device.lower() != "cpu":
                # SentenceTransformer proxies .to(...) to underlying torch modules
                try:
                    self.model = self.model.to(device)  # type: ignore[attr-defined]
                except Exception as e:
                    # If MPS/CUDA move fails, fall back to CPU (no meta tensors involved)
                    logger.warning("Move to device '%s' failed (%s); staying on CPU.", device, e)
                    device = "cpu"

            # 4) Optional: honor float16 only on CUDA (MPS/CPU stay float32)
            if torch is not None and device.startswith("cuda") and self.dtype == np.float16:
                try:
                    self.model = self.model.half()  # type: ignore[attr-defined]
                except Exception:
                    logger.warning("float16 requested but not supported; using float32.")
                    self.dtype = np.float32

            self.device = device
        except Exception as e:
            # If anything unexpected resembles meta-tensor flow, retry cleanly on CPU
            text = str(e)
            if _META_ERR_RE.search(text):
                logger.error("Meta-device trace detected; retrying clean on CPU. Detail: %s", text)
                self.device = "cpu"
                self.model = SentenceTransformer(
                    self.model_id,
                    device="cpu",
                    trust_remote_code=self.trust_remote_code,
                    revision=self.revision,
                    cache_folder=self.cache_folder,
                )
                _ = self.model.encode("__warmup__", convert_to_numpy=True, show_progress_bar=False)
            else:
                raise


    def _embed_text_local(self, text: str, normalize: Optional[bool] = None) -> List[float]:
        text = (text or "").replace("\n", " ").strip()
        if not text:
            return [0.0] * getattr(self, "_dims", 384)

        vec = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=False,
            batch_size=1,
            show_progress_bar=False,
        )
        vec = vec.astype(self.dtype, copy=False)

        do_norm = self.normalize if normalize is None else bool(normalize)
        if do_norm:
            denom = np.linalg.norm(vec)
            if denom > 0:
                vec = vec / denom

        if vec.dtype not in (np.float32, np.float64):
            vec = vec.astype(np.float32)
        return vec.tolist()
