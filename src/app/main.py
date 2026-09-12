# ruff: noqa: E402
import os
from typing import Any, Optional

# Disable tokenizer parallelism to prevent "Already Borrowed" errors and deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import re
import secrets
import traceback
from contextlib import asynccontextmanager

import anyio
import httpx
from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    RerankRequest,
    RerankResponse,
    ModelList,
    ModelCard,
    ErrorResponse,
)
from .models import get_model as get_model, _model_cache
from .config import (
    EMBEDDING_MODELS,
    RERANK_MODELS,
    API_KEY,
    EMBEDDING_TEI_URL as EMBEDDING_TEI_URL,
    RERANK_TEI_URL as RERANK_TEI_URL,
)
from .services import (
    BaseEmbeddingService,
    BaseRerankService,
    EmbeddingService,
    RerankService,
)
from .services.base import get_validated_model
from .services.embedding import (
    _determine_ruri_prefix as _determine_ruri_prefix,
    _apply_prefix as _apply_prefix,
)

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")


def redact_pii(text: str) -> str:
    """
    Redacts common PII from a string.
    Currently masks email addresses.
    """
    return EMAIL_PATTERN.sub("[REDACTED]", text)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    # Initialize global HTTP client with connection pooling for TEI proxy requests
    app_instance.state.tei_client = httpx.Client(timeout=30.0)
    try:
        yield
    finally:
        app_instance.state.tei_client.close()


app = FastAPI(title="OpenAI-Compatible API", lifespan=lifespan)


@app.get("/health", tags=["Health"])
@app.get("/healthz", tags=["Health"])
async def health_check():
    """
    Liveness / readiness probe for microservice orchestrators and Docker health checks.
    """
    return {"status": "ok"}


# Authentication dependency
security = HTTPBearer(auto_error=False)


async def verify_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Security(security),
):
    if API_KEY:
        if auth is None or not secrets.compare_digest(auth.credentials, API_KEY):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API Key",
                headers={"WWW-Authenticate": "Bearer"},
            )
    return auth


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """
    Middleware that adds security headers to every response.
    """
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; frame-ancestors 'none';"
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb_str = traceback.format_exc()
    redacted_exc = redact_pii(str(exc))
    redacted_tb = redact_pii(tb_str)

    await anyio.to_thread.run_sync(
        lambda: logging.error(
            f"Unhandled exception: {redacted_exc}\n{redacted_tb}", exc_info=False
        )
    )
    response = JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; frame-ancestors 'none';"
    )
    return response


def _proxy_to_tei(tei_url: str, path: str, json_data: dict) -> Any:
    """
    Helper to send a POST request to TEI and return the JSON response.
    """
    try:
        shared_client = getattr(app.state, "tei_client", None)
        if shared_client is not None:
            response = shared_client.post(f"{tei_url}{path}", json=json_data)
        else:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(f"{tei_url}{path}", json=json_data)

        if response.status_code != 200:
            error_msg = response.text
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            raise HTTPException(
                status_code=500,
                detail=f"TEI Proxy Error ({response.status_code}): {error_msg}",
            )
        return response.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to proxy request to TEI: {str(e)}"
        )


def _get_model_or_400(model_name: str, model_type: str) -> Any:
    """
    Helper for backwards compatibility with legacy tests calling _get_model_or_400.
    """
    supported_models = EMBEDDING_MODELS if model_type == "embedding" else RERANK_MODELS
    return get_validated_model(
        model_name, supported_models, model_type, loader=get_model
    )


# Dependency Injection Providers
def get_embedding_service() -> BaseEmbeddingService:
    return EmbeddingService(proxy_to_tei_func=_proxy_to_tei, model_loader=get_model)


def get_rerank_service() -> BaseRerankService:
    return RerankService(proxy_to_tei_func=_proxy_to_tei, model_loader=get_model)


@app.get(
    "/v1/models",
    response_model=ModelList,
    tags=["Models"],
    summary="List available models",
    description="Retrieves a list of all currently supported models, including both embedding and reranking models.",
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        429: {"model": ErrorResponse, "description": "Too Many Requests"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
    },
)
async def list_models():
    """
    Lists available models by dynamically aggregating and deduplicating
    from EMBEDDING_MODELS and RERANK_MODELS.
    """
    model_ids = sorted(list(set(EMBEDDING_MODELS + RERANK_MODELS)))
    models = [ModelCard(id=model_id) for model_id in model_ids]
    return ModelList(data=models)


@app.post(
    "/v1/models/unload",
    tags=["Models"],
    summary="Unload models",
    description="Clears the model cache, releasing any loaded models from memory.",
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        429: {"model": ErrorResponse, "description": "Too Many Requests"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
    },
    dependencies=[Depends(verify_api_key)],
)
async def unload_models():
    """
    Unloads all currently cached models to free memory/VRAM.
    """
    _model_cache.clear()
    return {"status": "ok", "message": "All models unloaded"}


@app.post(
    "/v1/embeddings",
    response_model=EmbeddingResponse,
    tags=["Embeddings"],
    summary="Create embeddings",
    description="Creates embeddings for the given input text or multimodal item, following OpenAI's API format.",
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Bad Request (e.g., unsupported model, invalid input)",
        },
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        413: {"model": ErrorResponse, "description": "Payload Too Large"},
        429: {"model": ErrorResponse, "description": "Too Many Requests"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
    },
    dependencies=[Depends(verify_api_key)],
)
async def create_embeddings(
    request: EmbeddingRequest,
    service: BaseEmbeddingService = Depends(get_embedding_service),
):
    """
    Creates embeddings for the given input, following OpenAI's API format.
    Supports text-only and multimodal (image/composite) inputs.
    """
    return await service.create_embeddings(request)


@app.post(
    "/v1/rerank",
    response_model=RerankResponse,
    response_model_exclude_none=True,
    tags=["Rerank"],
    summary="Rerank documents",
    description="Reranks a list of documents for a given query to determine their relevance.",
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Bad Request (e.g., unsupported model, invalid input)",
        },
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        413: {"model": ErrorResponse, "description": "Payload Too Large"},
        429: {"model": ErrorResponse, "description": "Too Many Requests"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
    },
    dependencies=[Depends(verify_api_key)],
)
async def create_rerank(
    request: RerankRequest,
    service: BaseRerankService = Depends(get_rerank_service),
):
    """
    Reranks a list of documents for a given query.
    """
    return await service.create_rerank(request)
