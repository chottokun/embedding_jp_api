import asyncio
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app, active_requests
import app.main as main_module


@pytest.mark.anyio
async def test_graceful_shutdown_tracking_and_503():
    # Verify in-flight requests are tracked during normal operation
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/health")
        assert res.status_code == 200
        assert res.json() == {"status": "ok"}
        # active_requests should be empty after response finishes
        assert len(active_requests) == 0

    # Simulate server in shutdown phase
    orig_shutting_down = main_module.is_shutting_down
    try:
        main_module.is_shutting_down = True
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.get("/health")
            assert res.status_code == 503
            assert "shutting down" in res.json()["detail"]
    finally:
        main_module.is_shutting_down = orig_shutting_down


@pytest.mark.anyio
async def test_lifespan_drains_active_requests():
    # Simulate a slow request task in active_requests
    drain_completed = False

    async def slow_work():
        nonlocal drain_completed
        await asyncio.sleep(0.05)
        drain_completed = True

    task = asyncio.create_task(slow_work())
    active_requests.add(task)

    try:
        # Run lifespan context
        async with main_module.lifespan(app):
            pass
        # After lifespan exits (shutdown), task should have drained
        assert drain_completed is True
    finally:
        active_requests.discard(task)
        main_module.is_shutting_down = False
        if hasattr(app.state, "tei_client"):
            delattr(app.state, "tei_client")
