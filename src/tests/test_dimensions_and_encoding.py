import base64
import struct
import math
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app

client = TestClient(app)


def test_embedding_dimensions_truncation():
    # Mock model returning a 4-dimensional vector: [1.0, 2.0, 3.0, 4.0]
    raw_vector = [1.0, 2.0, 3.0, 4.0]
    mock_model = MagicMock()
    mock_model.supports_multimodal = False
    mock_model.lock = MagicMock()
    mock_model.tokenizer_lock = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [raw_vector]
    mock_tokenizer = MagicMock()
    mock_tokenizer.num_special_tokens_to_add.return_value = 2
    mock_tokenizer.return_value = {"input_ids": [[101, 102]]}
    mock_model.tokenizer = mock_tokenizer

    with patch("app.main.get_model", return_value=mock_model):
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "cl-nagoya/ruri-v3-30m",
                "input": "テスト文章",
                "dimensions": 2,
            },
        )
        assert response.status_code == 200
        data = response.json()
        emb = data["data"][0]["embedding"]
        assert len(emb) == 2
        # Verify L2 normalization: [1.0, 2.0] / sqrt(1^2 + 2^2) = [1 / sqrt(5), 2 / sqrt(5)]
        expected_norm = math.sqrt(1.0**2 + 2.0**2)
        assert pytest.approx(emb[0], rel=1e-4) == 1.0 / expected_norm
        assert pytest.approx(emb[1], rel=1e-4) == 2.0 / expected_norm


def test_embedding_base64_encoding_format():
    raw_vector = [0.25, -0.5, 0.75]
    mock_model = MagicMock()
    mock_model.supports_multimodal = False
    mock_model.lock = MagicMock()
    mock_model.tokenizer_lock = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [raw_vector]
    mock_tokenizer = MagicMock()
    mock_tokenizer.num_special_tokens_to_add.return_value = 2
    mock_tokenizer.return_value = {"input_ids": [[101, 102]]}
    mock_model.tokenizer = mock_tokenizer

    with patch("app.main.get_model", return_value=mock_model):
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "cl-nagoya/ruri-v3-30m",
                "input": "テスト文章",
                "encoding_format": "base64",
            },
        )
        assert response.status_code == 200
        data = response.json()
        b64_str = data["data"][0]["embedding"]
        assert isinstance(b64_str, str)

        # Decode and unpack float32 little endian
        decoded_bytes = base64.b64decode(b64_str)
        unpacked = list(struct.unpack(f"<{len(raw_vector)}f", decoded_bytes))
        for original, unpacked_val in zip(raw_vector, unpacked):
            assert pytest.approx(original, rel=1e-4) == unpacked_val
