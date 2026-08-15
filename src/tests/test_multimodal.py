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
