from abc import ABC, abstractmethod
from typing import Any, Callable, Collection, Optional
from fastapi import HTTPException
from ..schemas import EmbeddingRequest, EmbeddingResponse, RerankRequest, RerankResponse
from ..models import get_model


def get_validated_model(
    model_name: str,
    allowed_models: Collection[str],
    service_name: str,
    loader: Optional[Callable[[str], Any]] = None,
) -> Any:
    """
    Validates model_name against allowed_models and loads the model via loader/get_model.
    Raises HTTPException(400) if model is invalid or if loading raises a ValueError.
    """
    if model_name not in allowed_models:
        suffix = "s" if not service_name.endswith("s") else ""
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not found for {service_name}{suffix}.",
        )

    fetch_func = loader or get_model

    try:
        return fetch_func(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class BaseEmbeddingService(ABC):
    """Abstract Base Class for Embedding Services."""

    @abstractmethod
    async def create_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generates embeddings for the provided EmbeddingRequest."""
        pass


class BaseRerankService(ABC):
    """Abstract Base Class for Rerank Services."""

    @abstractmethod
    async def create_rerank(self, request: RerankRequest) -> RerankResponse:
        """Reranks documents for the provided RerankRequest."""
        pass
