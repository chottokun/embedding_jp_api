import time
from typing import Optional
from threading import Lock

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from ..config import RATE_LIMIT_PER_MINUTE, API_KEYS_MAP


class RateLimiter:
    def __init__(self, default_limit: int, window: int = 60):
        self.default_limit = default_limit
        self.window = window
        self.requests: dict[str, list[float]] = {}
        self.lock = Lock()

    def is_allowed(self, client_id: str, limit: Optional[int] = None) -> bool:
        max_allowed = limit if limit is not None else self.default_limit
        now = time.time()
        with self.lock:
            if client_id not in self.requests:
                self.requests[client_id] = []
            # Evict timestamps older than window
            self.requests[client_id] = [
                t for t in self.requests[client_id] if now - t < self.window
            ]
            if len(self.requests[client_id]) >= max_allowed:
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


# Alias for backward compatibility
TokenBucketRateLimiter = RateLimiter

rate_limiter = RateLimiter(default_limit=RATE_LIMIT_PER_MINUTE, window=60)
EXEMPT_RATE_LIMIT_PATHS = {"/health", "/healthz", "/ready", "/metrics"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        if request.url.path in EXEMPT_RATE_LIMIT_PATHS:
            return await call_next(request)

        active_keys_map = API_KEYS_MAP
        try:
            import app.main as main_mod

            active_keys_map = getattr(main_mod, "API_KEYS_MAP", active_keys_map)
        except ImportError:
            pass

        client_id = request.client.host if request.client else "unknown"
        client_limit = None
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            client_id = token
            if token in active_keys_map:
                client_limit = active_keys_map[token]

        if not rate_limiter.is_allowed(client_id, limit=client_limit):
            retry_after = rate_limiter.get_retry_after(client_id)
            return JSONResponse(
                status_code=429,
                content={"detail": "Too Many Requests"},
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)
