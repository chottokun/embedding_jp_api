from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

# Requirements dictate 10MB
MAX_PAYLOAD_SIZE = 10 * 1024 * 1024  # 10MB


class PayloadSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
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
