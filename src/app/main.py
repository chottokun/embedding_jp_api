# ruff: noqa: E402
import os
from typing import Any, Optional

# Disable tokenizer parallelism to prevent "Already Borrowed" errors and deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import re
import secrets
import traceback
import asyncio
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
    RATE_LIMIT_PER_MINUTE,
    SHUTDOWN_DRAIN_TIMEOUT_SECONDS,
    PRELOAD_MODELS,
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
PROMPT_TOKENS_COUNT = Counter(
    "http_prompt_tokens_total",
    "Total number of prompt tokens processed",
    ["model"],
)
BATCH_SIZE_HISTOGRAM = Histogram(
    "http_request_batch_size",
    "Distribution of batch sizes (number of inputs per request)",
    ["endpoint"],
    buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
)


def redact_pii(text: str) -> str:
    """
    Redacts common PII from a string.
    Currently masks email addresses.
    """
    return EMAIL_PATTERN.sub("[REDACTED]", text)


# In-flight request tracking for graceful shutdown
active_requests: set[asyncio.Task] = set()
is_shutting_down: bool = False


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global is_shutting_down
    # Initialize global HTTP client with connection pooling for TEI proxy requests
    app_instance.state.tei_client = httpx.AsyncClient(timeout=30.0)

    # Preload configured models to eliminate cold-start latency
    if PRELOAD_MODELS:
        logging.info(f"Preloading models: {PRELOAD_MODELS}")
        for model_name in PRELOAD_MODELS:
            try:
                await anyio.to_thread.run_sync(get_model, model_name)
                logging.info(f"Preloaded model '{model_name}' successfully.")
            except Exception as e:
                logging.error(f"Failed to preload model '{model_name}': {e}")

    try:
        yield
    finally:
        is_shutting_down = True
        # Drain in-flight requests up to SHUTDOWN_DRAIN_TIMEOUT_SECONDS
        if active_requests:
            logging.info(
                f"Graceful shutdown initiated. Waiting for {len(active_requests)} in-flight request(s) "
                f"to drain (max timeout: {SHUTDOWN_DRAIN_TIMEOUT_SECONDS}s)..."
            )
            try:
                # Wait for active request tasks to finish
                await asyncio.wait(
                    active_requests,
                    timeout=SHUTDOWN_DRAIN_TIMEOUT_SECONDS,
                )
            except Exception as e:
                logging.warning(f"Error during in-flight request drain: {e}")
        await app_instance.state.tei_client.aclose()


app = FastAPI(title="OpenAI-Compatible API", lifespan=lifespan)


@app.middleware("http")
async def graceful_shutdown_drain_middleware(request: Request, call_next):
    """
    Tracks active in-flight request tasks. Rejects incoming requests with 503 Service Unavailable
    when the server is in the graceful shutdown phase.
    """
    if is_shutting_down:
        return JSONResponse(
            status_code=503,
            content={"detail": "Server is shutting down. Please retry shortly."},
        )

    current_task = asyncio.current_task()
    if current_task is not None:
        active_requests.add(current_task)
    try:
        return await call_next(request)
    finally:
        if current_task is not None:
            active_requests.discard(current_task)


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


# --- Rate Limiting (Token Bucket / Sliding Window) ---
class RateLimiter:
    def __init__(self, limit: int, window: int = 60):
        self.limit = limit
        self.window = window
        self.requests: dict[str, list[float]] = {}
        from threading import Lock

        self.lock = Lock()

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        with self.lock:
            if client_id not in self.requests:
                self.requests[client_id] = []
            # Evict timestamps older than window
            self.requests[client_id] = [
                t for t in self.requests[client_id] if now - t < self.window
            ]
            if len(self.requests[client_id]) >= self.limit:
                return False
            self.requests[client_id].append(now)
            return True

    def get_retry_after(self, client_id: str) -> int:
        now = time.time()
        with self.lock:
            if client_id not in self.requests or not self.requests[client_id]:
                return 0
            oldest = self.requests[client_id][0]
            retry_after = self.window - int(now - oldest)
            return max(1, retry_after)


rate_limiter = RateLimiter(limit=RATE_LIMIT_PER_MINUTE, window=60)
EXEMPT_RATE_LIMIT_PATHS = {"/health", "/healthz", "/ready", "/metrics"}


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """
    Applies per-minute rate limiting based on Authorization key or client host IP.
    Bypasses health and monitoring endpoints. Returns 429 Too Many Requests on breach.
    """
    if request.url.path in EXEMPT_RATE_LIMIT_PATHS:
        return await call_next(request)

    client_id = request.client.host if request.client else "unknown"
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        client_id = auth_header[7:]

    if not rate_limiter.is_allowed(client_id):
        retry_after = rate_limiter.get_retry_after(client_id)
        return JSONResponse(
            status_code=429,
            content={"detail": "Too Many Requests"},
            headers={"Retry-After": str(retry_after)},
        )

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
        if (
            shared_client is not None
            and getattr(shared_client, "is_closed", False) is not True
        ):
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
    batch_size = len(request.input) if isinstance(request.input, list) else 1
    BATCH_SIZE_HISTOGRAM.labels(endpoint="/v1/embeddings").observe(batch_size)

    response = await service.create_embeddings(request)
    if hasattr(response, "usage") and response.usage:
        PROMPT_TOKENS_COUNT.labels(model=request.model).inc(
            response.usage.prompt_tokens
        )
    return response


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
    batch_size = len(request.documents)
    BATCH_SIZE_HISTOGRAM.labels(endpoint="/v1/rerank").observe(batch_size)

    response = await service.create_rerank(request)
    if hasattr(response, "usage") and response.usage:
        PROMPT_TOKENS_COUNT.labels(model=request.model).inc(
            response.usage.prompt_tokens
        )
    return response


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
