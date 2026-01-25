import numpy as np
from .config import EMBEDDING_MODELS, RERANK_MODELS
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import threading

# --- Model Loader (Factory) ---

_model_cache = {}
_model_lock = threading.Lock()

def get_model(model_name: str):
    """
    Factory function to get a model instance.
    It loads real models from Hugging Face and caches them.
    """
    with _model_lock:
        if model_name in _model_cache:
            return _model_cache[model_name]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model '{model_name}' on device '{device}'...")

        if model_name in EMBEDDING_MODELS:
            model = SentenceTransformer(model_name, device=device)
            _model_cache[model_name] = model
            print(f"Model '{model_name}' loaded successfully.")
            return model

        if model_name in RERANK_MODELS:
            model = CrossEncoder(model_name, device=device)
            _model_cache[model_name] = model
            print(f"Model '{model_name}' loaded successfully.")
            return model

    raise ValueError(f"Model '{model_name}' is not supported.")
