from .health import router as health_router
from .models import router as models_router
from .embeddings import router as embeddings_router
from .rerank import router as rerank_router

__all__ = [
    "health_router",
    "models_router",
    "embeddings_router",
    "rerank_router",
]
