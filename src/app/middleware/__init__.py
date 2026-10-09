from fastapi import FastAPI

from .logging import (
    RequestLoggingMiddleware,
    JsonFormatter,
    EMAIL_PATTERN,
    redact_pii,
    request_id_var,
    get_body,
)
from .security import SecurityHeadersMiddleware
from .rate_limit import (
    TokenBucketRateLimiter,
    RateLimitMiddleware,
    rate_limiter,
)
from .metrics import (
    MetricsMiddleware,
    REQUESTS_TOTAL,
    REQUEST_LATENCY,
    INFERENCE_LATENCY,
    PROMPT_TOKENS_COUNT,
    BATCH_SIZE_HISTOGRAM,
)
from .graceful_shutdown import (
    GracefulShutdownDrainMiddleware,
    active_requests,
    shutdown_state,
)
from .payload_limit import (
    PayloadSizeLimitMiddleware,
    MAX_PAYLOAD_SIZE,
)


def setup_middleware(app: FastAPI) -> None:
    """
    Registers all required middlewares to the FastAPI app.
    Middlewares are executed in reverse order of addition.
    """
    # The last added middleware will be the first one to execute.

    # 1. Graceful Shutdown (run first to reject requests if shutting down)
    app.add_middleware(GracefulShutdownDrainMiddleware)

    # 2. Payload limit
    app.add_middleware(PayloadSizeLimitMiddleware)

    # 3. Request Logging (logs the request details, assigns request ID)
    app.add_middleware(RequestLoggingMiddleware)

    # 4. Security Headers (adds security headers to response)
    app.add_middleware(SecurityHeadersMiddleware)

    # 5. Metrics (records prometheus metrics)
    app.add_middleware(MetricsMiddleware)

    # 6. Rate Limit (enforces rate limits based on tokens/IPs)
    app.add_middleware(RateLimitMiddleware)


__all__ = [
    "setup_middleware",
    # logging.py
    "RequestLoggingMiddleware",
    "JsonFormatter",
    "EMAIL_PATTERN",
    "redact_pii",
    "request_id_var",
    "get_body",
    # security.py
    "SecurityHeadersMiddleware",
    # rate_limit.py
    "TokenBucketRateLimiter",
    "RateLimitMiddleware",
    "rate_limiter",
    # metrics.py
    "MetricsMiddleware",
    "REQUESTS_TOTAL",
    "REQUEST_LATENCY",
    "INFERENCE_LATENCY",
    "PROMPT_TOKENS_COUNT",
    "BATCH_SIZE_HISTOGRAM",
    # graceful_shutdown.py
    "GracefulShutdownDrainMiddleware",
    "active_requests",
    "shutdown_state",
    # payload_limit.py
    "PayloadSizeLimitMiddleware",
    "MAX_PAYLOAD_SIZE",
]
