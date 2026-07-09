from fastapi.testclient import TestClient
from unittest.mock import patch
import numpy as np

from app.main import app
from app.config import EMBEDDING_MODELS, RERANK_MODELS

client = TestClient(app)

SUPPORTED_EMBED_MODEL = EMBEDDING_MODELS[0]
SUPPORTED_RERANK_MODEL = RERANK_MODELS[0]


def setup_mock_model(mock_get_model, model_type="embedding"):
    mock_model = mock_get_model.return_value

    if model_type == "embedding":
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        def mock_tokenizer_call(text, **kwargs):
            if isinstance(text, str):
                tokens = [1] * 10 if "検索" in text else [1, 2, 3]
                return {"input_ids": tokens}
            elif isinstance(text, list):
                ids = [[1] * 10 if "検索" in t else [1, 2, 3] for t in text]
                return {"input_ids": ids}
            return {"input_ids": []}

        mock_model.tokenizer.side_effect = mock_tokenizer_call
        mock_model.tokenizer.num_special_tokens_to_add.return_value = 2
        mock_model.max_seq_length = 8

        # For truncation test
        mock_model.tokenizer.decode.return_value = "truncated text"

    else:  # rerank
        mock_model.predict.return_value = np.array([0.9, 0.8])
        mock_model.tokenizer.encode.side_effect = lambda *args, **kwargs: [1, 2, 3]
        mock_model.tokenizer.num_special_tokens_to_add.return_value = 1

    mock_model.lock = patch("threading.Lock").start()
    mock_model.tokenizer_lock = patch("threading.Lock").start()

    return mock_model


@patch("app.main.get_model")
def test_embeddings_baseline(mock_get_model):
    mock_model = setup_mock_model(mock_get_model, "embedding")

    # Test ruri-v3 prefix handling and truncation
    payload = {
        "input": "test input",
        "model": "cl-nagoya/ruri-v3-30m",
        "apply_ruri_prefix": True,
    }
    response = client.post("/v1/embeddings", json=payload)
    assert response.status_code == 200

    # max_seq_length = 8, special_tokens = 2, limit = 6
    # "検索クエリ: test input" has 10 tokens > 6 tokens
    # It should be truncated
    mock_model.encode.assert_called_with(["truncated text"])
    assert response.json()["usage"]["total_tokens"] == 8  # 6 + 2
    mock_model.tokenizer.decode.assert_called()


@patch("app.main.get_model")
def test_rerank_baseline(mock_get_model):
    setup_mock_model(mock_get_model, "rerank")

    payload = {
        "model": SUPPORTED_RERANK_MODEL,
        "query": "What is the capital of France?",
        "documents": [
            "Paris is the capital of France.",
            "Berlin is the capital of Germany.",
        ],
        "top_n": 1,
    }
    response = client.post("/v1/rerank", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 1
    assert data["data"][0]["document"] == 0
    assert data["usage"]["total_tokens"] > 0
