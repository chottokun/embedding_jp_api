from fastapi.testclient import TestClient
from unittest.mock import patch
import numpy as np

# Corrected imports for a 'src' layout
from app.main import app
from app.config import EMBEDDING_MODELS

client = TestClient(app)

SUPPORTED_EMBED_MODEL = EMBEDDING_MODELS[0]


def setup_mock_model(mock_get_model, encode_return=None):
    mock_model = mock_get_model.return_value
    if encode_return is None:
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    else:
        mock_model.encode.return_value = np.array(encode_return)

    # Legacy encode mock for backward compatibility (if any)
    mock_model.tokenizer.encode.side_effect = lambda *args, **kwargs: [1, 2, 3]

    # Batch tokenizer mock
    def mock_tokenizer_call(text, **kwargs):
        if isinstance(text, str):
            # If prefix is present (simple check for test strings), return 6 tokens, else 3
            tokens = [1] * 6 if "検索" in text else [1, 2, 3]
            return {"input_ids": tokens}
        elif isinstance(text, list):
            ids = []
            for t in text:
                tokens = [1] * 6 if "検索" in t else [1, 2, 3]
                ids.append(tokens)
            return {"input_ids": ids}
        return {"input_ids": []}

    mock_model.tokenizer.side_effect = mock_tokenizer_call
    mock_model.tokenizer.num_special_tokens_to_add.return_value = 2
    mock_model.max_seq_length = 8192
    return mock_model


@patch("app.main.get_model")
def test_create_embeddings_single_string_no_prefix_default(mock_get_model):
    """
    Tests the default behavior for a single string input: no prefix should be applied.
    """
    mock_model = setup_mock_model(mock_get_model)

    request_payload = {"input": "今日の天気", "model": SUPPORTED_EMBED_MODEL}

    response = client.post("/v1/embeddings", json=request_payload)

    assert response.status_code == 200
    mock_model.encode.assert_called_once_with(["今日の天気"])
    # 3 tokens (mock) + 2 special tokens = 5
    assert response.json()["usage"]["total_tokens"] == 5


@patch("app.main.get_model")
def test_create_embeddings_ruri_v3_with_prefix_enabled(mock_get_model):
    """
    Tests ruri-v3 model with prefix enabled.
    """
    mock_model = setup_mock_model(mock_get_model)

    request_payload = {
        "input": "今日の天気",
        "model": "cl-nagoya/ruri-v3-30m",
        "apply_ruri_prefix": True,
    }

    response = client.post("/v1/embeddings", json=request_payload)

    assert response.status_code == 200
    mock_model.encode.assert_called_once_with(["検索クエリ: 今日の天気"])
    # 3 tokens (prefix) + 3 tokens (text) + 2 special tokens = 8
    assert response.json()["usage"]["total_tokens"] == 8


@patch("app.main.get_model")
def test_create_embeddings_ruri_v3_with_prefix_enabled_list_input(mock_get_model):
    """
    Tests ruri-v3 model with prefix enabled for a list of strings.
    """
    mock_model = setup_mock_model(
        mock_get_model, encode_return=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )

    request_payload = {
        "input": ["文書A", "文書B"],
        "model": "cl-nagoya/ruri-v3-30m",
        "apply_ruri_prefix": True,
    }

    response = client.post("/v1/embeddings", json=request_payload)

    assert response.status_code == 200
    mock_model.encode.assert_called_once_with(["検索文書: 文書A", "検索文書: 文書B"])
    # 2 inputs * (3 prefix + 3 text + 2 special) = 16
    assert response.json()["usage"]["total_tokens"] == 16


@patch("app.main.get_model")
def test_create_embeddings_non_ruri_v3_with_prefix_flag_no_prefix(mock_get_model):
    """
    Tests that a non-ruri-v3 model does not get prefixes even if the flag is true.
    """
    mock_model = setup_mock_model(mock_get_model)

    # A dummy model that is not ruri-v3
    non_ruri_model = "some-other-model"
    # We need to add this model to the list of supported models for the test
    EMBEDDING_MODELS.append(non_ruri_model)

    request_payload = {
        "input": "今日の天気",
        "model": non_ruri_model,
        "apply_ruri_prefix": True,
    }

    response = client.post("/v1/embeddings", json=request_payload)

    assert response.status_code == 200
    mock_model.encode.assert_called_once_with(["今日の天気"])

    # Clean up the list of supported models
    EMBEDDING_MODELS.pop()
