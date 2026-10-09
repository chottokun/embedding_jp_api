# ruff: noqa: E402
import os

# Disable tokenizer parallelism to prevent "Already Borrowed" errors and deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import logging
import secrets
import traceback
from contextlib import asynccontextmanager

import anyio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import (
    API_KEY,
    API_KEYS_MAP,
    EMBEDDING_MODELS,
    EMBEDDING_TEI_URL,
    INFERENCE_SEMAPHORE_TIMEOUT_SECONDS,
    MAX_CONCURRENT_INFERENCES,
    PRELOAD_MODELS,
    RATE_LIMIT_PER_MINUTE,
    RERANK_MODELS,
    RERANK_TEI_URL,
    SHUTDOWN_DRAIN_TIMEOUT_SECONDS,
)
from .concurrency import AsyncThreadSemaphore, inference_semaphore
from .dependencies import (
    _get_model_or_400,
    _proxy_to_tei,
    get_embedding_service,
    get_rerank_service,
    security,
    verify_admin_key,
    verify_api_key,
)
from .middleware import (
    BATCH_SIZE_HISTOGRAM,
    EMAIL_PATTERN,
    INFERENCE_LATENCY,
    MAX_PAYLOAD_SIZE,
    PROMPT_TOKENS_COUNT,
    REQUEST_LATENCY,
    REQUESTS_TOTAL,
    GracefulShutdownDrainMiddleware,
    JsonFormatter,
    PayloadSizeLimitMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
    TokenBucketRateLimiter,
    active_requests,
    get_body,
    rate_limiter,
    redact_pii,
    request_id_var,
    setup_middleware,
)
from .models import get_model, unload_model
from .routers import (
    embeddings_router,
    health_router,
    models_router,
    rerank_router,
)
from .services.embedding import (
    _apply_prefix,
    _determine_ruri_prefix,
)

logger = logging.getLogger("app")

# Shutdown flag exposed for graceful shutdown tracking and backward compatibility with tests
is_shutting_down: bool = False


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global is_shutting_down
    # Initialize global HTTP client with connection pooling for TEI proxy requests
    app_instance.state.tei_client = httpx.AsyncClient(timeout=30.0)

    # Preload configured models to eliminate cold-start latency
    preload_list = getattr(app_instance, "PRELOAD_MODELS", PRELOAD_MODELS)
    try:
        import app.main as main_mod

        preload_list = getattr(main_mod, "PRELOAD_MODELS", preload_list)
    except ImportError:
        pass

    if preload_list:
        logging.info(f"Preloading models: {preload_list}")
        for model_name in preload_list:
            try:
                loader = get_model
                try:
                    import app.main as main_mod

                    loader = getattr(main_mod, "get_model", loader)
                except ImportError:
                    pass
                await anyio.to_thread.run_sync(loader, model_name)
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
                await asyncio.wait(
                    active_requests,
                    timeout=SHUTDOWN_DRAIN_TIMEOUT_SECONDS,
                )
            except Exception as e:
                logging.warning(f"Error during in-flight request drain: {e}")
        if (
            hasattr(app_instance.state, "tei_client")
            and app_instance.state.tei_client is not None
        ):
            await app_instance.state.tei_client.aclose()


app = FastAPI(
    title="Japanese Embedding & Reranking API",
    version="1.0.0",
    description=(
        "Production-grade, OpenAI-compatible REST API providing high-performance text and "
        "multimodal embeddings (Ruri-v3, Visual-BGE) and reranking (bge-reranker-v2-m3) "
        "tailored for Japanese NLP tasks."
    ),
    lifespan=lifespan,
)

# Register middlewares in correct execution order
setup_middleware(app)

# Register modular API routers
app.include_router(health_router)
app.include_router(models_router)
app.include_router(embeddings_router)
app.include_router(rerank_router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb_str = traceback.format_exc()
    redacted_exc = redact_pii(str(exc))
    redacted_tb = redact_pii(tb_str)

    req_id = request_id_var.get()

    def _log_error():
        if req_id:
            request_id_var.set(req_id)
        active_logger = logger
        try:
            import app.main as main_mod

            active_logger = getattr(main_mod, "logger", active_logger)
        except ImportError:
            pass
        active_logger.error(
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


__all__ = [
    "app",
    "lifespan",
    "is_shutting_down",
    "active_requests",
    "inference_semaphore",
    "AsyncThreadSemaphore",
    "rate_limiter",
    "TokenBucketRateLimiter",
    "MAX_PAYLOAD_SIZE",
    "API_KEY",
    "API_KEYS_MAP",
    "EMBEDDING_MODELS",
    "RERANK_MODELS",
    "EMBEDDING_TEI_URL",
    "RERANK_TEI_URL",
    "RATE_LIMIT_PER_MINUTE",
    "MAX_CONCURRENT_INFERENCES",
    "INFERENCE_SEMAPHORE_TIMEOUT_SECONDS",
    "PRELOAD_MODELS",
    "logger",
    "secrets",
    "verify_api_key",
    "verify_admin_key",
    "security",
    "_proxy_to_tei",
    "_get_model_or_400",
    "get_embedding_service",
    "get_rerank_service",
    "get_model",
    "unload_model",
    "redact_pii",
    "request_id_var",
    "get_body",
    "EMAIL_PATTERN",
    "JsonFormatter",
    "RequestLoggingMiddleware",
    "SecurityHeadersMiddleware",
    "RateLimitMiddleware",
    "GracefulShutdownDrainMiddleware",
    "PayloadSizeLimitMiddleware",
    "REQUESTS_TOTAL",
    "REQUEST_LATENCY",
    "INFERENCE_LATENCY",
    "PROMPT_TOKENS_COUNT",
    "BATCH_SIZE_HISTOGRAM",
    "_apply_prefix",
    "_determine_ruri_prefix",
]
