from fastapi.testclient import TestClient
from unittest.mock import patch
import numpy as np
from app.main import app
from app.config import EMBEDDING_MODELS

client = TestClient(app)
SUPPORTED_EMBED_MODEL = EMBEDDING_MODELS[0]


@patch("app.main.get_model")
def test_create_embeddings_truncation_logic(mock_get_model):
    """
    Specifically tests the truncation logic in create_embeddings:
    When input tokens exceed (max_seq_length - special_tokens),
    the input should be truncated and decoded back to text before being passed to model.encode.
    """
    mock_model = mock_get_model.return_value

    # Setup constraints: limit will be 10 - 2 = 8
    mock_model.max_seq_length = 10
    mock_model.tokenizer.num_special_tokens_to_add.return_value = 2

    # Mock tokenizer to return 12 tokens (longer than limit of 8)
    def mock_tokenizer_call(batch, **kwargs):
        return {"input_ids": [[1] * 12 for _ in batch]}

    mock_model.tokenizer.side_effect = mock_tokenizer_call

    # Mock decode to return a recognizable string
    mock_model.tokenizer.decode.side_effect = lambda ids: f"truncated_{len(ids)}"

    # Mock model.encode
    mock_model.encode.return_value = np.array([[0.1] * 768])

    # We need a Lock for the mock model
    import threading

    mock_model.lock = threading.Lock()

    request_payload = {
        "input": "this text is way too long",
        "model": SUPPORTED_EMBED_MODEL,
    }

    response = client.post("/v1/embeddings", json=request_payload)

    assert response.status_code == 200

    # 1. Verify truncation happened: model.encode should be called with decoded truncated text
    # The truncated_ids should have length 8, so decoded text should be "truncated_8"
    mock_model.encode.assert_called_once_with(["truncated_8"])

    # 2. Verify usage calculation: 8 truncated tokens + 2 special tokens = 10 total tokens
    data = response.json()
    assert data["usage"]["total_tokens"] == 10
    assert data["usage"]["prompt_tokens"] == 10


@patch("app.main.get_model")
def test_create_embeddings_no_truncation_needed(mock_get_model):
    """
    Verifies that no truncation occurs if tokens are within limits.
    """
    mock_model = mock_get_model.return_value
    mock_model.max_seq_length = 10
    mock_model.tokenizer.num_special_tokens_to_add.return_value = 2
    # limit = 8

    # Mock tokenizer to return 5 tokens (within limit of 8)
    def mock_tokenizer_call(batch, **kwargs):
        return {"input_ids": [[1] * 5 for _ in batch]}

    mock_model.tokenizer.side_effect = mock_tokenizer_call
    mock_model.encode.return_value = np.array([[0.1] * 768])

    import threading

    mock_model.lock = threading.Lock()

    request_payload = {"input": "short text", "model": SUPPORTED_EMBED_MODEL}

    response = client.post("/v1/embeddings", json=request_payload)

    assert response.status_code == 200

    # Should NOT call decode, and should pass original processed input
    assert mock_model.tokenizer.decode.call_count == 0
    mock_model.encode.assert_called_once_with(["short text"])

    # 5 tokens + 2 special = 7
    data = response.json()
    assert data["usage"]["total_tokens"] == 7
