from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app
from app.config import RERANK_MODELS

client = TestClient(app)

SUPPORTED_RERANK_MODEL = RERANK_MODELS[0]


@patch("app.main.get_model")
def test_create_rerank_successful(mock_get_model):
    """
    Tests a successful rerank request.
    Verifies request/response format and that results are sorted.
    """
    # Arrange
    mock_model = mock_get_model.return_value
    # Return scores in an unsorted order to test sorting logic
    mock_model.predict.return_value = [0.1, 0.9, 0.5]

    mock_model.tokenizer.side_effect = lambda queries, docs, **kwargs: {
        "input_ids": [[1, 2, 3, 4, 5] for _ in range(len(queries))]
    }

    query = "AIの未来"
    documents = ["猫について", "AIの進化", "日本の首都"]

    request_payload = {
        "query": query,
        "documents": documents,
        "model": SUPPORTED_RERANK_MODEL,
    }

    # Act
    response = client.post("/v1/rerank", json=request_payload)

    # Assert
    assert response.status_code == 200

    expected_pairs = [[query, doc] for doc in documents]
    mock_model.predict.assert_called_once_with(expected_pairs)

    response_json = response.json()
    assert response_json["query"] == query
    assert response_json["model"] == SUPPORTED_RERANK_MODEL
    assert "usage" in response_json
    assert response_json["usage"]["total_tokens"] > 0

    results = response_json["data"]
    assert len(results) == 3
    assert results[0]["score"] == 0.9
    assert results[0]["document"] == 1
    assert results[1]["score"] == 0.5
    assert results[1]["document"] == 2
    assert results[2]["score"] == 0.1
    assert results[2]["document"] == 0


@patch("app.main.get_model")
def test_rerank_top_n(mock_get_model):
    """
    Tests that top_n parameter correctly limits the number of results
    and returns the highest scoring documents.
    """
    mock_model = mock_get_model.return_value
    # 5 documents with distinct scores
    scores = [0.1, 0.9, 0.5, 0.8, 0.2]
    mock_model.predict.return_value = scores

    mock_model.tokenizer.side_effect = lambda queries, docs, **kwargs: {
        "input_ids": [[1] for _ in range(len(queries))]
    }

    query = "Query"
    documents = ["Doc0", "Doc1", "Doc2", "Doc3", "Doc4"]

    request_payload = {
        "query": query,
        "documents": documents,
        "model": SUPPORTED_RERANK_MODEL,
        "top_n": 3,
    }

    response = client.post("/v1/rerank", json=request_payload)
    assert response.status_code == 200

    results = response.json()["data"]
    assert len(results) == 3

    # Expected top 3:
    # Doc1 (0.9), Doc3 (0.8), Doc2 (0.5)
    assert results[0]["document"] == 1
    assert results[0]["score"] == 0.9

    assert results[1]["document"] == 3
    assert results[1]["score"] == 0.8

    assert results[2]["document"] == 2
    assert results[2]["score"] == 0.5


@patch("app.main.get_model")
def test_rerank_top_n_larger_than_docs(mock_get_model):
    """
    Tests that if top_n is larger than the number of documents,
    it returns all documents sorted.
    """
    mock_model = mock_get_model.return_value
    scores = [0.1, 0.2]
    mock_model.predict.return_value = scores

    mock_model.tokenizer.side_effect = lambda queries, docs, **kwargs: {
        "input_ids": [[1] for _ in range(len(queries))]
    }

    request_payload = {
        "query": "Q",
        "documents": ["D0", "D1"],
        "model": SUPPORTED_RERANK_MODEL,
        "top_n": 10,
    }

    response = client.post("/v1/rerank", json=request_payload)
    assert response.status_code == 200
    results = response.json()["data"]
    assert len(results) == 2
    assert results[0]["score"] == 0.2
    assert results[1]["score"] == 0.1


@patch("app.main.get_model")
def test_rerank_stability(mock_get_model):
    """
    Tests that documents with identical scores preserve their original relative order.
    """
    mock_model = mock_get_model.return_value
    # All scores are identical
    scores = [0.5, 0.5, 0.5]
    mock_model.predict.return_value = scores

    mock_model.tokenizer.side_effect = lambda queries, docs, **kwargs: {
        "input_ids": [[1] for _ in range(len(queries))]
    }

    request_payload = {
        "query": "Q",
        "documents": ["D0", "D1", "D2"],
        "model": SUPPORTED_RERANK_MODEL,
        "top_n": 2,
    }

    response = client.post("/v1/rerank", json=request_payload)
    assert response.status_code == 200
    results = response.json()["data"]

    assert len(results) == 2
    # Should preserve order: D0 then D1
    assert results[0]["document"] == 0
    assert results[1]["document"] == 1


def test_rerank_unsupported_model():
    request_payload = {
        "query": "test",
        "documents": ["doc1"],
        "model": "unsupported-rerank-model",
    }
    response = client.post("/v1/rerank", json=request_payload)
    assert response.status_code == 400
    assert "not found" in response.json()["detail"]
