import asyncio
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

# In-flight request tracking for graceful shutdown
active_requests: set[asyncio.Task] = set()
is_shutting_down: bool = False


class GracefulShutdownState:
    is_shutting_down: bool = False


shutdown_state = GracefulShutdownState()


class GracefulShutdownDrainMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        # Check both local is_shutting_down and app.main.is_shutting_down for test compatibility
        shutting_down = is_shutting_down
        try:
            import app.main as main_mod

            shutting_down = shutting_down or getattr(
                main_mod, "is_shutting_down", False
            )
        except ImportError:
            pass

        if shutting_down:
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
