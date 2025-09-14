# src/memorizz/embeddings/huggingface/provider.py
# --- MUST RUN BEFORE importing torch/transformers/sentence_transformers ---
import os
os.environ.setdefault("PYTORCH_DISABLE_META_DEVICE", "1")
os.environ.setdefault("ACCELERATE_USE_FSDP", "false")
os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "no")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
# --------------------------------------------------------------------------

import logging, re
from typing import List, Dict, Any, Optional
import numpy as np

# Torch is optional; we can run CPU-only without it
try:
    import torch
except Exception:
    torch = None

# Import BOTH the high-level wrapper and low-level modules
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from sentence_transformers.models import Transformer, Pooling
except Exception:
    Transformer = None
    Pooling = None

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

        # Ensure required deps are present (clear error if worker venv is missing packages)
        if SentenceTransformer is None or Transformer is None or Pooling is None:
            raise ImportError(
                "sentence-transformers is required (pip install -U sentence-transformers). "
                "Ensure it is installed in the SAME environment your Celery worker uses."
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

        # Select device (CUDA > CPU). Keep MPS opt-in to avoid edge cases.
        requested_device: Optional[str] = self.config.get("device")
        self.device: str = self._select_device(requested_device)

        # Hard-disable accelerate meta-init paths (harmless if absent)
        os.environ.pop("ACCELERATE_USE_FSDP", None)
        os.environ.pop("ACCELERATE_MIXED_PRECISION", None)

        # Initialize model (CPU-first; then move if needed)
        self._init_on_device(self.device)

        # Determine dimensions
        try:
            self._dims = int(self.model.get_sentence_embedding_dimension())
        except Exception:
            tmp = self._embed_text_local("__probe__")
            self._dims = int(len(tmp))

        logger.info(
            "Initialized HuggingFace provider model=%s, dims=%d, device=%s, normalize=%s",
            self.model_id, self._dims, self.device, self.normalize
        )

    # ---------- BaseEmbeddingProvider API ----------

    def get_embedding(self, text: str, **kwargs) -> List[float]:
        if not isinstance(text, str):
            text = str(text)
        normalize = bool(kwargs.get("normalize", self.normalize))
        return self._embed_text_local(text, normalize=normalize)

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
        # If you have verified MPS stability, you may enable it:
        # if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        #     return "mps"
        return "cpu"

    @staticmethod
    def _has_meta_tensors(nn_module) -> bool:
        """Inspect parameters/buffers to catch any accidental meta tensors."""
        try:
            for p in nn_module.parameters():
                if getattr(p, "is_meta", False):
                    return True
            for b in nn_module.buffers():
                if getattr(b, "is_meta", False):
                    return True
        except Exception:
            pass
        return False

    def _init_on_device(self, device: str) -> None:
        logger.info("Initializing HF embeddings (CPU-first) model=%s target=%s", self.model_id, device)

        # Steer default device away from 'meta' (PyTorch >= 2.1)
        try:
            if torch is not None and hasattr(torch, "set_default_device"):
                torch.set_default_device("cpu")
        except Exception:
            pass

        # 1) Build via low-level modules on CPU (materialized tensors; avoids accelerate/meta fastpath)
        word = Transformer(
            self.model_id,
            cache_dir=self.cache_folder,
            model_args={"trust_remote_code": self.trust_remote_code},
            device="cpu",
        )
        pool = Pooling(word.get_word_embedding_dimension())
        self.model = SentenceTransformer(modules=[word, pool], device="cpu")

        # 2) Warm-up on CPU to ensure full instantiation
        _ = self.model.encode("__warmup__", convert_to_numpy=True, show_progress_bar=False)

        # 3) Verify there are no meta tensors
        if self._has_meta_tensors(self.model):
            raise RuntimeError("Meta tensors detected after CPU build; aborting device move.")

        # 4) Move to target device if not CPU
        if device.lower() != "cpu":
            try:
                self.model = self.model.to(device)  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning("Move to device '%s' failed (%s); staying on CPU.", device, e)
                device = "cpu"

        # 5) Optional fp16 only on CUDA
        if torch is not None and device.startswith("cuda") and self.dtype == np.float16:
            try:
                self.model = self.model.half()  # type: ignore[attr-defined]
            except Exception:
                logger.warning("float16 requested but not supported; using float32.")
                self.dtype = np.float32

        # 6) Record final device
        self.device = device

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

        # Keep dtype consistent; only allow fp16 on CUDA
        if self.dtype == np.float16 and not (torch is not None and str(self.device).startswith("cuda")):
            vec = vec.astype(np.float32, copy=False)
        else:
            vec = vec.astype(self.dtype, copy=False)

        do_norm = self.normalize if normalize is None else bool(normalize)
        if do_norm:
            denom = np.linalg.norm(vec)
            if denom > 0:
                vec = vec / denom

        if vec.dtype not in (np.float32, np.float64):
            vec = vec.astype(np.float32)
        return vec.tolist()
