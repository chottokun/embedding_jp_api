import asyncio
import pytest
from app.concurrency import AsyncThreadSemaphore


@pytest.mark.anyio
async def test_async_thread_semaphore_initialization():
    sem = AsyncThreadSemaphore(2)
    assert sem._val == 2


@pytest.mark.anyio
async def test_async_thread_semaphore_non_blocking_acquire():
    sem = AsyncThreadSemaphore(2)

    async with sem:
        # Acquired once non-blocking
        pass

    # Should be released and re-acquirable
    async with sem:
        pass


@pytest.mark.anyio
async def test_async_thread_semaphore_blocking_acquire_and_release():
    sem = AsyncThreadSemaphore(1)
    results = []

    async def worker(worker_id: int, delay: float):
        async with sem:
            results.append(f"start-{worker_id}")
            await asyncio.sleep(delay)
            results.append(f"end-{worker_id}")

    # Launch two workers concurrently
    t1 = asyncio.create_task(worker(1, 0.05))
    t2 = asyncio.create_task(worker(2, 0.01))

    await asyncio.gather(t1, t2)

    # Worker 1 should start and end before Worker 2 can enter
    assert results == ["start-1", "end-1", "start-2", "end-2"]


@pytest.mark.anyio
async def test_async_thread_semaphore_exception_release():
    sem = AsyncThreadSemaphore(1)

    with pytest.raises(ValueError, match="test error"):
        async with sem:
            raise ValueError("test error")

    # Should be released so another task can enter without blocking
    acquired = False
    async with sem:
        acquired = True

    assert acquired is True
