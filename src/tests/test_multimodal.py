import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.image_utils import is_safe_url_async

client = TestClient(app)

TINY_PNG_B64 = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@pytest.mark.anyio
async def test_is_safe_url_async():
    # Private IP addresses should be blocked
    assert await is_safe_url_async("http://127.0.0.1/test.png") is False
    assert await is_safe_url_async("http://192.168.1.1/test.png") is False
    assert await is_safe_url_async("http://10.0.0.1/test.png") is False


def test_text_only_backward_compatibility():
    with patch("app.main.get_model") as mock_get_model:
        mock_model = MagicMock()
        mock_model.supports_multimodal = False
        mock_model.lock = MagicMock()
        mock_model.tokenizer_lock = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [[0.1, 0.2, 0.3]]
        mock_tokenizer = MagicMock()
        mock_tokenizer.num_special_tokens_to_add.return_value = 2
        mock_tokenizer.return_value = {"input_ids": [[101, 102]]}
        mock_model.tokenizer = mock_tokenizer
        mock_get_model.return_value = mock_model

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "cl-nagoya/ruri-v3-30m",
                "input": "東京の観光地",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]


def test_multimodal_flat_input_success():
    with patch("app.main.get_model") as mock_get_model:
        mock_model = MagicMock()
        mock_model.supports_multimodal = True
        mock_model.encode_multimodal.return_value = [[0.5, 0.6, 0.7]]
        mock_get_model.return_value = mock_model

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "bge-visualized-m3",
                "input": {
                    "text": "青い服",
                    "image_url": TINY_PNG_B64,
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["embedding"] == [0.5, 0.6, 0.7]


def test_multimodal_part_array_input_success():
    with patch("app.main.get_model") as mock_get_model:
        mock_model = MagicMock()
        mock_model.supports_multimodal = True
        mock_model.encode_multimodal.return_value = [[0.8, 0.9, 1.0]]
        mock_get_model.return_value = mock_model

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "bge-visualized-m3",
                "input": [
                    {"type": "text", "text": "赤い車"},
                    {"type": "image_url", "image_url": TINY_PNG_B64},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["embedding"] == [0.8, 0.9, 1.0]


def test_text_only_model_rejects_image_input():
    with patch("app.main.get_model") as mock_get_model:
        mock_model = MagicMock()
        mock_model.supports_multimodal = False
        mock_get_model.return_value = mock_model

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "cl-nagoya/ruri-v3-30m",
                "input": {
                    "image_url": TINY_PNG_B64,
                },
            },
        )
        assert response.status_code == 400
        assert "画像入力をサポートしていません" in response.json()["detail"]


def test_ssrf_url_rejected():
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "bge-visualized-m3",
            "input": {
                "image_url": "http://127.0.0.1/forbidden.jpg",
            },
        },
    )
    assert response.status_code == 400
    assert "拒否されたURL" in response.json()["detail"]


def test_tei_proxy_bypassed_for_multimodal_request():
    with (
        patch("app.main.EMBEDDING_TEI_URL", "http://tei-server:8080"),
        patch("app.main.get_model") as mock_get_model,
        patch("app.main._proxy_to_tei") as mock_tei_proxy,
    ):
        mock_model = MagicMock()
        mock_model.supports_multimodal = True
        mock_model.encode_multimodal.return_value = [[0.1, 0.1, 0.1]]
        mock_get_model.return_value = mock_model

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "bge-visualized-m3",
                "input": {
                    "image_url": TINY_PNG_B64,
                },
            },
        )
        assert response.status_code == 200
        # TEI proxy should NOT be called for image requests
        mock_tei_proxy.assert_not_called()


def test_health_endpoints():
    res1 = client.get("/health")
    assert res1.status_code == 200
    assert res1.json() == {"status": "ok"}

    res2 = client.get("/healthz")
    assert res2.status_code == 200
    assert res2.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_ssrf_redirect_to_private_ip_rejected():
    from app.image_utils import load_image_from_source
    import httpx

    # Mock an HTTP client where safe URL redirects to 127.0.0.1
    redirect_resp = httpx.Response(
        status_code=302,
        headers={"Location": "http://127.0.0.1/private.png"},
        request=httpx.Request("GET", "http://example.com/image.png"),
    )

    class MockAsyncClient:
        def stream(self, method, url, **kwargs):
            class StreamCtx:
                async def __aenter__(self_inner):
                    return redirect_resp

                async def __aexit__(self_inner, *args):
                    pass

            return StreamCtx()

    with patch("app.image_utils.is_safe_url_async") as mock_safe:
        # First request to example.com is safe, but second to 127.0.0.1 is not safe
        mock_safe.side_effect = [True, False]
        with pytest.raises(ValueError, match="拒否されたURL"):
            await load_image_from_source(
                "http://example.com/image.png", MockAsyncClient()
            )


@pytest.mark.anyio
async def test_load_image_from_source_base64_invalid_format():
    from app.image_utils import load_image_from_source
    import httpx

    # Invalid base64 characters
    invalid_b64 = "data:image/png;base64,!!!invalid_base64!!!"
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="Base64画像のデコードに失敗しました"):
            await load_image_from_source(invalid_b64, client)


@pytest.mark.anyio
async def test_load_image_from_source_base64_non_image_data():
    from app.image_utils import load_image_from_source
    import httpx
    import base64

    # Valid base64 encoding of non-image text string
    non_image_b64 = (
        "data:image/png;base64," + base64.b64encode(b"not an image file").decode()
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="Base64画像のデコードに失敗しました"):
            await load_image_from_source(non_image_b64, client)


@pytest.mark.anyio
async def test_load_image_from_source_base64_exceeds_max_size():
    from app.image_utils import load_image_from_source
    import httpx

    with patch("app.image_utils.MAX_FILE_SIZE", 10):
        # TINY_PNG_B64 is larger than 10 bytes after base64 decoding
        async with httpx.AsyncClient() as client:
            with pytest.raises(ValueError, match="画像サイズが上限") as exc_info:
                await load_image_from_source(TINY_PNG_B64, client)
            assert "上限" in str(exc_info.value)


def test_embeddings_endpoint_base64_max_file_size_exceeded():
    from app.image_utils import MAX_FILE_SIZE

    large_b64_source = "data:image/png;base64,large_dummy_data"
    with patch("base64.b64decode") as mock_b64decode:
        mock_b64decode.return_value = b"x" * (MAX_FILE_SIZE + 1)
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "bge-visualized-m3",
                "input": {
                    "text": "テスト",
                    "image_url": large_b64_source,
                },
            },
        )
        assert response.status_code == 400
        assert "画像サイズが上限(15MB)を超えています" in response.json()["detail"]
