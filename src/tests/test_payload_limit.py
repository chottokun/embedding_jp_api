import pytest
import httpx
from unittest.mock import patch
from app.main import app, MAX_PAYLOAD_SIZE


@pytest.mark.anyio
async def test_payload_limit_content_length():
    with patch("app.main.verify_api_key", return_value=None):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/embeddings",
                content=b"a" * (MAX_PAYLOAD_SIZE + 10),
                headers={"Authorization": "Bearer test"},
            )
            assert response.status_code == 413
            assert response.json() == {"detail": "Payload Too Large"}


@pytest.mark.anyio
async def test_payload_limit_within_limit():
    with patch("app.main.verify_api_key", return_value=None):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/embeddings",
                content=b"{}",
                headers={
                    "Authorization": "Bearer test",
                    "Content-Type": "application/json",
                },
            )
            assert response.status_code != 413
