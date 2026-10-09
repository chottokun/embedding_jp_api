import threading
import anyio

from app.config import MAX_CONCURRENT_INFERENCES


class AsyncThreadSemaphore:
    def __init__(self, initial_value: int):
        self._val = initial_value
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(initial_value)

    async def __aenter__(self):
        # Non-blocking check first, or offload to thread to avoid event loop contention
        acquired = self._sem.acquire(blocking=False)
        if not acquired:
            await anyio.to_thread.run_sync(self._sem.acquire)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._sem.release()


inference_semaphore = AsyncThreadSemaphore(MAX_CONCURRENT_INFERENCES)
