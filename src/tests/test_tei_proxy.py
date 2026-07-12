from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

# Create client with server exception raising disabled to inspect error handlers
client = TestClient(app, raise_server_exceptions=False)


def test_tei_embeddings_proxy_success():
    """Verify that when EMBEDDING_TEI_URL is set, embeddings request is proxied to TEI."""
    # Mock EMBEDDING_TEI_URL and API_KEY configurations
    with (
        patch("app.main.EMBEDDING_TEI_URL", "http://tei-embedding"),
        patch("app.main.API_KEY", None),
        patch("app.main._proxy_to_tei") as mock_proxy,
    ):
        # Configure mock TEI response json
        mock_proxy.return_value = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "cl-nagoya/ruri-v3-30m",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }

        payload = {
            "input": "テスト",
            "model": "cl-nagoya/ruri-v3-30m",
            "input_type": "query",
        }

        response = client.post("/v1/embeddings", json=payload)

        assert response.status_code == 200
        res_json = response.json()
        assert res_json["data"][0]["embedding"] == [0.1, 0.2, 0.3]

        # Verify that _proxy_to_tei was called with correct arguments
        mock_proxy.assert_called_once_with(
            "http://tei-embedding",
            "/v1/embeddings",
            {"input": ["検索クエリ: テスト"], "model": "cl-nagoya/ruri-v3-30m"},
        )


def test_tei_embeddings_proxy_failure():
    """Verify that HTTP errors from TEI are propagated correctly as 500 error."""
    from fastapi import HTTPException

    with (
        patch("app.main.EMBEDDING_TEI_URL", "http://tei-embedding"),
        patch("app.main.API_KEY", None),
        patch("app.main._proxy_to_tei") as mock_proxy,
    ):
        # Simulate proxy function throwing HTTPException (e.g. from 500 remote error)
        mock_proxy.side_effect = HTTPException(
            status_code=500, detail="TEI Internal Crash"
        )

        payload = {"input": "テスト", "model": "cl-nagoya/ruri-v3-30m"}

        response = client.post("/v1/embeddings", json=payload)
        assert response.status_code == 500
        assert "TEI" in response.json()["detail"]


def test_tei_rerank_proxy_success():
    """Verify that when RERANK_TEI_URL is set, rerank request is proxied to TEI."""
    with (
        patch("app.main.RERANK_TEI_URL", "http://tei-rerank"),
        patch("app.main.API_KEY", None),
        patch("app.main._proxy_to_tei") as mock_proxy,
    ):
        # TEI /rerank response format: list of objects with index and score
        mock_proxy.return_value = [
            {"index": 0, "score": 0.95},
            {"index": 1, "score": 0.05},
        ]

        payload = {
            "query": "日本の首都",
            "documents": ["東京", "大阪"],
            "model": "cl-nagoya/ruri-v3-reranker-310m",
            "return_documents": True,
            "top_n": 1,
        }

        response = client.post("/v1/rerank", json=payload)
        assert response.status_code == 200
        res_json = response.json()

        assert len(res_json["data"]) == 1
        assert res_json["data"][0]["document"] == 0
        assert res_json["data"][0]["score"] == 0.95
        assert res_json["data"][0]["text"] == "東京"

        mock_proxy.assert_called_once_with(
            "http://tei-rerank",
            "/rerank",
            {"query": "日本の首都", "texts": ["東京", "大阪"]},
        )


def test_tei_rerank_proxy_failure():
    """Verify that Rerank proxy failure is handled."""
    from fastapi import HTTPException

    with (
        patch("app.main.RERANK_TEI_URL", "http://tei-rerank"),
        patch("app.main.API_KEY", None),
        patch("app.main._proxy_to_tei") as mock_proxy,
    ):
        mock_proxy.side_effect = HTTPException(
            status_code=500, detail="Failed to proxy rerank"
        )

        payload = {
            "query": "query",
            "documents": ["doc1"],
            "model": "cl-nagoya/ruri-v3-reranker-310m",
        }

        response = client.post("/v1/rerank", json=payload)
        assert response.status_code == 500
        assert "proxy" in response.json()["detail"]


def test_proxy_to_tei_error_truncation():
    """Verify that _proxy_to_tei limits the reflected length of response.text on failure."""
    from app.main import _proxy_to_tei
    from fastapi import HTTPException
    import pytest

    # We will mock httpx.Client in _proxy_to_tei to return a non-200 response
    class MockResponse:
        def __init__(self, status_code, text):
            self.status_code = status_code
            self.text = text

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def post(self, url, json):
            # Return response with long text
            return MockResponse(500, "A" * 500)

    with patch("httpx.Client", MockClient):
        with pytest.raises(HTTPException) as exc_info:
            _proxy_to_tei("http://tei-url", "/path", {"data": "test"})

        assert exc_info.value.status_code == 500
        # The detail string should contain truncated response of exactly 200 'A's + "..."
        expected_truncated_text = "A" * 200 + "..."
        assert expected_truncated_text in exc_info.value.detail
        assert len(exc_info.value.detail) < 300

    # Test short error is not truncated and doesn't get "..."
    class MockClientShort:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def post(self, url, json):
            return MockResponse(500, "Short error")

    with patch("httpx.Client", MockClientShort):
        with pytest.raises(HTTPException) as exc_info:
            _proxy_to_tei("http://tei-url", "/path", {"data": "test"})

        assert exc_info.value.status_code == 500
        assert "Short error" in exc_info.value.detail
        assert "..." not in exc_info.value.detail
