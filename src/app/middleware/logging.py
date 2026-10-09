import json
import logging
import re
import time
import uuid
from contextvars import ContextVar

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

# 1. request_id_var, EMAIL_PATTERN, redact_pii
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")


def redact_pii(text: str) -> str:
    """
    Redacts common PII from a string.
    Currently masks email addresses.
    """
    return EMAIL_PATTERN.sub("[REDACTED]", text)


# 2. JsonFormatter
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
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)


# 3. get_body(request: Request) -> bytes
async def get_body(request: Request) -> bytes:
    """
    Retrieves the request body. If the body is already consumed,
    this safely reads the cached body without hanging.
    """
    body = await request.body()

    async def receive():
        return {"type": "http.request", "body": body}

    request._receive = receive
    return body


# 4. RequestLoggingMiddleware
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
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
            active_logger = logger
            try:
                import app.main as main_mod

                active_logger = getattr(main_mod, "logger", active_logger)
            except ImportError:
                pass

            active_logger.info(
                f"{request.method} {request.url.path} - {status_code}",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "status_code": status_code,
                    "latency": latency,
                },
            )
            request_id_var.reset(token)

        # Optional: Add X-Request-ID header to response
        # SecurityHeadersMiddleware may also be adding headers, but we can do request ID here
        response.headers["X-Request-ID"] = req_id

        return response
