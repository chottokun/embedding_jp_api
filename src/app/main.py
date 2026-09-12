# ruff: noqa: E402
import os
from typing import Any, Optional

# Disable tokenizer parallelism to prevent "Already Borrowed" errors and deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import re
import secrets
import traceback
import json
import uuid
from contextvars import ContextVar
from contextlib import asynccontextmanager

import anyio
import httpx
import time
from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from .schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    RerankRequest,
    RerankResponse,
    ModelCard,
    ModelList,
    UnloadRequest,
    UnloadResponse,
)
from .models import get_model as get_model, unload_model
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

request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        req_id = request_id_var.get()
        if req_id:
            log_data["request_id"] = req_id

        if hasattr(record, "path"):
            log_data["path"] = record.path
        if hasattr(record, "method"):
            log_data["method"] = record.method
        if hasattr(record, "status_code"):
            log_data["status_code"] = record.status_code
        if hasattr(record, "latency"):
            log_data["latency"] = record.latency

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False


# Prometheus Metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
)


def redact_pii(text: str) -> str:
    """
    Redacts common PII from a string.
    Currently masks email addresses.
    """
    return EMAIL_PATTERN.sub("[REDACTED]", text)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    # Initialize global HTTP client with connection pooling for TEI proxy requests
    app_instance.state.tei_client = httpx.AsyncClient(timeout=30.0)
    try:
        yield
    finally:
        await app_instance.state.tei_client.aclose()


app = FastAPI(title="OpenAI-Compatible API", lifespan=lifespan)


@app.get("/health", tags=["Health"])
@app.get("/healthz", tags=["Health"])
async def health_check():
    """
    Liveness probe for microservice orchestrators and Docker health checks.
    """
    return {"status": "ok"}


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """
    Readiness probe verifying model loading status and GPU availability.
    """
    import torch
    from .models import _model_cache

    return {
        "status": "ready",
        "gpu_available": torch.cuda.is_available(),
        "models_loaded": list(_model_cache.keys()),
    }


@app.get("/metrics", tags=["Metrics"])
async def metrics():
    """
    Exposes Prometheus metrics.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


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


MAX_PAYLOAD_SIZE = 32 * 1024 * 1024  # 32MB


@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next):
    """
    Middleware to collect Prometheus metrics for HTTP requests.
    """
    method = request.method
    known_endpoints = {
        "/v1/embeddings",
        "/v1/rerank",
        "/v1/models",
        "/v1/models/unload",
        "/health",
        "/healthz",
        "/ready",
        "/metrics",
        "/",
    }
    endpoint = (
        request.url.path if request.url.path in known_endpoints else "unmatched_route"
    )

    start_time = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
    except BaseException as e:
        status_code = 500
        raise e
    finally:
        latency = time.perf_counter() - start_time
        REQUEST_COUNT.labels(
            method=method, endpoint=endpoint, http_status=status_code
        ).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)

    return response


@app.middleware("http")
async def payload_size_limit_middleware(request: Request, call_next):
    """
    Rejects requests exceeding MAX_PAYLOAD_SIZE (32MB) with 413 Payload Too Large.
    """
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_PAYLOAD_SIZE:
                return JSONResponse(
                    status_code=413, content={"detail": "Payload Too Large"}
                )
        except ValueError:
            pass

    return await call_next(request)


@app.middleware("http")
async def request_logging_and_security_headers(request: Request, call_next):
    """
    Middleware that generates/propagates request ID, logs request details in JSON,
    and adds security headers to every response.
    """
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    token = request_id_var.set(req_id)

    start_time = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        raise e
    finally:
        latency = time.perf_counter() - start_time
        logger.info(
            f"{request.method} {request.url.path} - {status_code}",
            extra={
                "path": request.url.path,
                "method": request.method,
                "status_code": status_code,
                "latency": latency,
            },
        )
        request_id_var.reset(token)

    response.headers["X-Request-ID"] = req_id
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

    req_id = request_id_var.get()

    def _log_error():
        if req_id:
            request_id_var.set(req_id)
        logger.error(
            f"Unhandled exception: {redacted_exc}\n{redacted_tb}", exc_info=False
        )

    await anyio.to_thread.run_sync(_log_error)
    response = JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )
    if req_id:
        response.headers["X-Request-ID"] = req_id
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


async def _proxy_to_tei(tei_url: str, path: str, json_data: dict) -> Any:
    """
    Helper to send an async POST request to TEI and return the JSON response.
    """
    try:
        shared_client = getattr(app.state, "tei_client", None)
        if shared_client is not None:
            response = await shared_client.post(f"{tei_url}{path}", json=json_data)
        else:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{tei_url}{path}", json=json_data)

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


@app.post(
    "/v1/embeddings",
    response_model=EmbeddingResponse,
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


@app.get(
    "/v1/models",
    response_model=ModelList,
    dependencies=[Depends(verify_api_key)],
    tags=["Models"],
)
async def list_models():
    """
    Lists all available models in the OpenAI-compatible format.
    """
    all_models = set(EMBEDDING_MODELS + RERANK_MODELS)
    current_time = int(time.time())

    models = [
        ModelCard(
            id=model_id,
            created=current_time,
        )
        for model_id in sorted(list(all_models))
    ]

    return ModelList(data=models)


@app.post(
    "/v1/models/unload",
    response_model=UnloadResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["Models"],
)
async def unload_models(request: UnloadRequest):
    """
    Unloads a specific model or all models from cache, freeing memory / VRAM.
    """
    unloaded_models, remaining_memory = unload_model(request.model)
    return UnloadResponse(
        unloaded_models=unloaded_models,
        remaining_memory=remaining_memory,
    )
