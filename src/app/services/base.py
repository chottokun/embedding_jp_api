from abc import ABC, abstractmethod
from ..schemas import EmbeddingRequest, EmbeddingResponse, RerankRequest, RerankResponse


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
