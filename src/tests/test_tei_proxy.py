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


def test_tei_proxy_uses_pooled_client():
    """Verify that _proxy_to_tei correctly uses the pooled client from app.state when initialized,

    and gracefully falls back to a local client when not initialized.
    """
    from unittest.mock import MagicMock
    from app.main import _proxy_to_tei, app
    import httpx

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"success": True}

    # 1. Test fallback when app.state does not have tei_client initialized
    if hasattr(app.state, "tei_client"):
        delattr(app.state, "tei_client")

    with patch("httpx.Client") as mock_client_class:
        mock_client_instance = mock_client_class.return_value.__enter__.return_value
        mock_client_instance.post.return_value = mock_response

        res = _proxy_to_tei("http://tei-url", "/path", {"test": "data"})
        assert res == {"success": True}
        mock_client_instance.post.assert_called_once_with(
            "http://tei-url/path", json={"test": "data"}
        )

    # 2. Test when app.state.tei_client IS initialized
    mock_pooled_client = MagicMock(spec=httpx.Client)
    mock_pooled_client.post.return_value = mock_response
    app.state.tei_client = mock_pooled_client

    with patch("httpx.Client") as mock_client_class:
        res = _proxy_to_tei("http://tei-url", "/path", {"test": "data"})
        assert res == {"success": True}
        # Verify that httpx.Client constructor was NOT called (i.e. local client not instantiated)
        mock_client_class.assert_not_called()
        # Verify pooled client was called
        mock_pooled_client.post.assert_called_once_with(
            "http://tei-url/path", json={"test": "data"}
        )

    # Clean up
    if hasattr(app.state, "tei_client"):
        delattr(app.state, "tei_client")
