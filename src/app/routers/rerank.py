import asyncio
from fastapi import APIRouter, Depends, HTTPException

from app.schemas import RerankRequest, RerankResponse, ErrorResponse
from app.config import INFERENCE_SEMAPHORE_TIMEOUT_SECONDS
from app.dependencies.auth import verify_api_key
from app.dependencies.services import get_rerank_service
from app.concurrency import inference_semaphore
from app.middleware.metrics import BATCH_SIZE_HISTOGRAM, PROMPT_TOKENS_COUNT
from app.services import BaseRerankService

router = APIRouter(prefix="/v1", tags=["Reranking"])


@router.post(
    "/rerank",
    response_model=RerankResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(verify_api_key)],
    summary="Rerank candidate documents for a query",
    description=(
        "Reorders candidate documents by relevance score for a given query "
        "using Japanese Cross-Encoder reranking models."
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
async def create_rerank(
    request: RerankRequest,
    service: BaseRerankService = Depends(get_rerank_service),
):
    """
    Reranks a list of documents for a given query.
    """
    batch_size = len(request.documents)
    BATCH_SIZE_HISTOGRAM.labels(endpoint="/v1/rerank").observe(batch_size)

    try:
        async with asyncio.timeout(INFERENCE_SEMAPHORE_TIMEOUT_SECONDS):
            async with inference_semaphore:
                response = await service.create_rerank(request)
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
