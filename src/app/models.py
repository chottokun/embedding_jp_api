from .config import EMBEDDING_MODELS, RERANK_MODELS
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import logging
import threading
from typing import Optional
from PIL import Image
from unittest.mock import MagicMock

# --- Multimodal Model Wrapper ---


class VisualizedBGEEmbeddingModel:
    supports_multimodal: bool = True

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        weights_path: str = "Visualized_m3.pth",
        device: str = "cuda",
    ):
        self.device = (
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.lock = threading.Lock()
        self.tokenizer_lock = threading.Lock()

        import os

        if not os.path.exists(weights_path):
            try:
                from huggingface_hub import hf_hub_download

                weights_path = hf_hub_download(
                    repo_id="BAAI/bge-visualized", filename="Visualized_m3.pth"
                )
            except Exception as e:
                logging.warning(f"Could not resolve Visualized_m3.pth from HF: {e}")

        try:
            from visual_bge.modeling import Visualized_BGE
        except ImportError:
            try:
                from .visual_bge.modeling import Visualized_BGE
            except ImportError:
                try:
                    from FlagEmbedding.visual.modeling import Visualized_BGE
                except ImportError:
                    Visualized_BGE = None

        if Visualized_BGE is None:
            raise ValueError("FlagEmbedding / visual_bge package is not installed.")

        self.model = Visualized_BGE(
            model_name_bge=model_name, model_weight=weights_path
        )
        self.model.to(self.device)
        self.model.eval()

    def encode_text(self, texts: list[str]) -> list[list[float]]:
        return self.encode_multimodal([(t, None) for t in texts])

    def encode_multimodal(
        self, items: list[tuple[Optional[str], Optional[Image.Image]]]
    ) -> list[list[float]]:
        results = []
        with self.lock:
            with torch.no_grad():
                for text, image in items:
                    vec = self.model.encode(image=image, text=text)
                    if isinstance(vec, torch.Tensor):
                        vec = vec.squeeze(0).cpu().tolist()
                    results.append(vec)
        return results


# --- Model Loader (Factory) ---

_model_cache = {}
_model_lock = threading.Lock()


def get_model(model_name: str, device: str | None = None):
    """
    Factory function to get a model instance.
    It loads real models from Hugging Face and caches them.
    """
    with _model_lock:
        if model_name in _model_cache:
            return _model_cache[model_name]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Loading model '{model_name}' on device '{device}'...")

        if model_name in {"bge-visualized-m3", "BAAI/bge-visualized-m3"}:
            model = VisualizedBGEEmbeddingModel(
                model_name="BAAI/bge-m3",
                weights_path="Visualized_m3.pth",
                device=device,
            )
        elif model_name in EMBEDDING_MODELS:
            model = SentenceTransformer(model_name, device=device)
        elif model_name in RERANK_MODELS:
            model = CrossEncoder(model_name, device=device)
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")

        # Ensure real threading locks exist on model instance
        if not hasattr(model, "lock") or isinstance(
            getattr(model, "lock", None), MagicMock
        ):
            model.lock = threading.Lock()
        if not hasattr(model, "tokenizer_lock") or isinstance(
            getattr(model, "tokenizer_lock", None), MagicMock
        ):
            model.tokenizer_lock = threading.Lock()

        _model_cache[model_name] = model
        logging.info(f"Model '{model_name}' loaded successfully.")
        return model
