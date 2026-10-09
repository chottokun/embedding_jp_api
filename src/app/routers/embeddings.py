import asyncio
from fastapi import APIRouter, Depends, HTTPException

from app.schemas import EmbeddingRequest, EmbeddingResponse, ErrorResponse
from app.config import INFERENCE_SEMAPHORE_TIMEOUT_SECONDS
from app.dependencies.auth import verify_api_key
from app.dependencies.services import get_embedding_service
from app.concurrency import inference_semaphore
from app.middleware.metrics import BATCH_SIZE_HISTOGRAM, PROMPT_TOKENS_COUNT
from app.models import get_model
from app.services.embedding import (
    _determine_ruri_prefix,
    _apply_prefix,
)

from app.services import BaseEmbeddingService

# Explicitly re-export to satisfy tests using this router directly if needed.
__all__ = ["router", "_determine_ruri_prefix", "_apply_prefix", "get_model"]

router = APIRouter(prefix="/v1", tags=["Embeddings"])


@router.post(
    "/embeddings",
    response_model=EmbeddingResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Create text or multimodal embeddings",
    description=(
        "Creates embedding vectors for input text or multimodal elements. "
        "Supports Matryoshka dimension reduction (`dimensions`), base64 encoding "
        "(`encoding_format`), and automated Japanese Ruri-v3 task prefixes."
    ),
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Unsupported model or invalid parameters",
        },
        401: {
            "model": ErrorResponse,
            "description": "Invalid or missing Bearer API key",
        },
        413: {
            "model": ErrorResponse,
            "description": "Payload exceeds maximum allowed size (32MB)",
        },
        429: {
            "model": ErrorResponse,
            "description": "Rate limit exceeded (Too Many Requests)",
        },
        503: {
            "model": ErrorResponse,
            "description": "Server is shutting down or inference queue timeout",
        },
    },
)
async def create_embeddings(
    request: EmbeddingRequest,
    service: BaseEmbeddingService = Depends(get_embedding_service),
):
    """
    Creates embeddings for the given input, following OpenAI's API format.
    Supports text-only and multimodal (image/composite) inputs.
    """
    batch_size = len(request.input) if isinstance(request.input, list) else 1
    BATCH_SIZE_HISTOGRAM.labels(endpoint="/v1/embeddings").observe(batch_size)

    try:
        async with asyncio.timeout(INFERENCE_SEMAPHORE_TIMEOUT_SECONDS):
            async with inference_semaphore:
                response = await service.create_embeddings(request)
    except TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Inference queue timeout. Server is under high load, please retry shortly.",
        )

    if hasattr(response, "usage") and response.usage:
        PROMPT_TOKENS_COUNT.labels(model=request.model).inc(
            response.usage.prompt_tokens
        )
    return response
