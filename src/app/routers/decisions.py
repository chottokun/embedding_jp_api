import asyncio
from fastapi import APIRouter, Depends, HTTPException

from app.config import INFERENCE_SEMAPHORE_TIMEOUT_SECONDS
from app.concurrency import inference_semaphore
from app.dependencies.auth import verify_api_key
from app.dependencies.services import get_decisions_service
from app.middleware.metrics import BATCH_SIZE_HISTOGRAM
from app.schemas import DecisionsRequest, DecisionsResponse, ErrorResponse
from app.services import BaseDecisionsService

router = APIRouter(prefix="/v1", tags=["Decisions"])


@router.post(
    "/decisions",
    response_model=DecisionsResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Create decision answers for given input context and questions",
    description=(
        "OpenAI Decisions API compatible endpoint returning decision probabilities, "
        "choices, and scores using Logit Router."
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
            "description": "Payload exceeds maximum allowed size (10MB)",
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
async def create_decision(
    request: DecisionsRequest,
    service: BaseDecisionsService = Depends(get_decisions_service),
):
    """
    Evaluates questions against input context and produces decision responses.
    """
    batch_size = len(request.questions)
    BATCH_SIZE_HISTOGRAM.labels(endpoint="/v1/decisions").observe(batch_size)

    try:
        async with asyncio.timeout(INFERENCE_SEMAPHORE_TIMEOUT_SECONDS):
            async with inference_semaphore:
                response = await service.create_decisions(request)
    except TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Inference queue timeout. Server is under high load, please retry shortly.",
        )

    return response
