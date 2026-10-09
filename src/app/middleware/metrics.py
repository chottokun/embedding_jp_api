import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from prometheus_client import Counter, Histogram

REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
)
INFERENCE_LATENCY = Histogram(
    "inference_duration_seconds",
    "Inference latency in seconds",
    ["endpoint"],
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


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
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
            request.url.path
            if request.url.path in known_endpoints
            else "unmatched_route"
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
            REQUESTS_TOTAL.labels(
                method=method, endpoint=endpoint, http_status=status_code
            ).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)

        return response
