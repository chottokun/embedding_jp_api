import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

from app.main import app
from app.config import EMBEDDING_MODELS, RERANK_MODELS

client = TestClient(app)

@pytest.fixture
def mock_embedding_model():
    with patch('app.main.get_model') as mock:
        model = MagicMock()
        model.encode.side_effect = lambda x: np.array([[0.1]*10] * (len(x) if isinstance(x, list) else 1))
        
        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda *args, **kwargs: [1, 2, 3] # dummy tokens
        tokenizer.decode.side_effect = lambda *args, **kwargs: "truncated text"
        tokenizer.num_special_tokens_to_add.return_value = 2
        
        model.tokenizer = tokenizer
        model.max_seq_length = 8192
        mock.return_value = model
        yield model

def test_input_type_mapping_query(mock_embedding_model):
    response = client.post("/v1/embeddings", json={
        "input": "test",
        "model": "cl-nagoya/ruri-v3-30m",
        "input_type": "query"
    })
    assert response.status_code == 200
    mock_embedding_model.encode.assert_called_once_with(["検索クエリ: test"])

def test_input_type_mapping_document(mock_embedding_model):
    response = client.post("/v1/embeddings", json={
        "input": "test",
        "model": "cl-nagoya/ruri-v3-30m",
        "input_type": "document"
    })
    assert response.status_code == 200
    mock_embedding_model.encode.assert_called_once_with(["検索文書: test"])

def test_prefix_deduplication(mock_embedding_model):
    response = client.post("/v1/embeddings", json={
        "input": "検索クエリ: test",
        "model": "cl-nagoya/ruri-v3-30m",
        "input_type": "query"
    })
    assert response.status_code == 200
    mock_embedding_model.encode.assert_called_once_with(["検索クエリ: test"])

def test_fallback_logic_string(mock_embedding_model):
    response = client.post("/v1/embeddings", json={
        "input": "test",
        "model": "cl-nagoya/ruri-v3-30m",
        "apply_ruri_prefix": True
    })
    assert response.status_code == 200
    mock_embedding_model.encode.assert_called_once_with(["検索クエリ: test"])

def test_fallback_logic_list(mock_embedding_model):
    response = client.post("/v1/embeddings", json={
        "input": ["test1", "test2"],
        "model": "cl-nagoya/ruri-v3-30m",
        "apply_ruri_prefix": True
    })
    assert response.status_code == 200
    mock_embedding_model.encode.assert_called_once_with(["検索文書: test1", "検索文書: test2"])

def test_truncation_logic(mock_embedding_model):
    # Setup tokenizer to pretend input is very long
    mock_embedding_model.max_seq_length = 10
    # prefix tokens = 3, special tokens = 2, available for text = 5
    # total tokens for "long text" > 5
    def mock_encode(text, **kwargs):
        if "long" in text:
            return [1]*20
        return [1]*3
    mock_embedding_model.tokenizer.encode.side_effect = mock_encode
    mock_embedding_model.tokenizer.decode.side_effect = lambda *args, **kwargs: "truncated"

    response = client.post("/v1/embeddings", json={
        "input": "very long text",
        "model": "cl-nagoya/ruri-v3-30m",
        "input_type": "query"
    })
    assert response.status_code == 200
    # It should call encode with prefix + truncated text
    mock_embedding_model.encode.assert_called_once_with(["検索クエリ: truncated"])

@patch('app.main.get_model')
def test_rerank_top_n(mock_get_model):
    model = MagicMock()
    model.predict.return_value = [0.1, 0.9, 0.5]
    mock_get_model.return_value = model

    response = client.post("/v1/rerank", json={
        "query": "test",
        "documents": ["doc1", "doc2", "doc3"],
        "model": RERANK_MODELS[0],
        "top_n": 2
    })
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data) == 2
    assert data[0]["score"] == 0.9

@patch('app.main.get_model')
def test_rerank_top_k_alias(mock_get_model):
    # Testing that 'top_k' is still accepted as an alias for 'top_n'
    model = MagicMock()
    model.predict.return_value = [0.1, 0.9, 0.5]
    mock_get_model.return_value = model

    response = client.post("/v1/rerank", json={
        "query": "test",
        "documents": ["doc1", "doc2", "doc3"],
        "model": RERANK_MODELS[0],
        "top_k": 1
    })
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data) == 1
