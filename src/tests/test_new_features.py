import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.config import RERANK_MODELS, EMBEDDING_MODELS
import numpy as np

client = TestClient(app)

@patch('app.main.get_model')
def test_rerank_top_k(mock_get_model):
    mock_model = mock_get_model.return_value
    mock_model.predict.return_value = [0.1, 0.9, 0.5]

    request_payload = {
        "query": "test",
        "documents": ["doc0", "doc1", "doc2"],
        "model": RERANK_MODELS[0],
        "top_k": 2
    }
    response = client.post("/v1/rerank", json=request_payload)
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data) == 2
    assert data[0]["document"] == 1
    assert data[1]["document"] == 2

@patch('app.main.get_model')
def test_rerank_return_documents(mock_get_model):
    mock_model = mock_get_model.return_value
    mock_model.predict.return_value = [0.1, 0.9, 0.5]

    request_payload = {
        "query": "test",
        "documents": ["doc0", "doc1", "doc2"],
        "model": RERANK_MODELS[0],
        "return_documents": True
    }
    response = client.post("/v1/rerank", json=request_payload)
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data) == 3
    assert data[0]["text"] == "doc1"
    assert data[1]["text"] == "doc2"
    assert data[2]["text"] == "doc0"

@patch('app.main.get_model')
def test_embedding_usage(mock_get_model):
    mock_model = mock_get_model.return_value
    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda *args, **kwargs: [1, 2, 3] # returns 3 tokens
    mock_tokenizer.num_special_tokens_to_add.return_value = 2
    mock_model.tokenizer = mock_tokenizer
    mock_model.max_seq_length = 8192
    mock_model.encode.return_value = np.array([[0.1] * 384])

    request_payload = {
        "input": "hello world",
        "model": EMBEDDING_MODELS[0]
    }
    response = client.post("/v1/embeddings", json=request_payload)
    assert response.status_code == 200
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == 3
    assert usage["total_tokens"] == 3
