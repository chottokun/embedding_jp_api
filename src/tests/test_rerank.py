import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app
from app.config import RERANK_MODELS

client = TestClient(app)

SUPPORTED_RERANK_MODEL = RERANK_MODELS[0]

@patch('app.main.get_model')
def test_create_rerank_successful(mock_get_model):
    """
    Tests a successful rerank request.
    Verifies request/response format and that results are sorted.
    """
    # Arrange
    mock_model = mock_get_model.return_value
    # Return scores in an unsorted order to test sorting logic
    mock_model.predict.return_value = [0.1, 0.9, 0.5]

    query = "AIの未来"
    documents = ["猫について", "AIの進化", "日本の首都"]

    request_payload = {
        "query": query,
        "documents": documents,
        "model": SUPPORTED_RERANK_MODEL
    }

    # Act
    response = client.post("/v1/rerank", json=request_payload)

    # Assert
    assert response.status_code == 200

    # Verify the correct pairs were sent to the model
    expected_pairs = [[query, doc] for doc in documents]
    mock_model.predict.assert_called_once_with(expected_pairs)

    response_json = response.json()
    assert response_json["query"] == query
    assert response_json["model"] == SUPPORTED_RERANK_MODEL

    # Check that data is sorted by score descending
    results = response_json["data"]
    assert len(results) == 3
    assert results[0]["score"] == 0.9
    assert results[0]["document"] == 1 # 'AIの進化'
    assert results[1]["score"] == 0.5
    assert results[1]["document"] == 2 # '日本の首都'
    assert results[2]["score"] == 0.1
    assert results[2]["document"] == 0 # '猫について'

def test_rerank_unsupported_model():
    """
    Tests that requesting an unsupported model for rerank returns a 400 error.
    """
    request_payload = {
        "query": "test",
        "documents": ["doc1"],
        "model": "unsupported-rerank-model"
    }
    response = client.post("/v1/rerank", json=request_payload)
    assert response.status_code == 400
    assert "not found" in response.json()["detail"]

@patch('app.main.get_model')
def test_create_rerank_with_top_k(mock_get_model):
    """
    Tests that top_k correctly limits the number of results.
    """
    mock_model = mock_get_model.return_value
    mock_model.predict.return_value = [0.1, 0.9, 0.5]

    request_payload = {
        "query": "AI",
        "documents": ["doc0", "doc1", "doc2"],
        "model": SUPPORTED_RERANK_MODEL,
        "top_k": 2
    }

    response = client.post("/v1/rerank", json=request_payload)
    assert response.status_code == 200
    response_json = response.json()
    assert len(response_json["data"]) == 2
    # Should be the top 2: doc1 (0.9) and doc2 (0.5)
    assert response_json["data"][0]["document"] == 1
    assert response_json["data"][1]["document"] == 2

@patch('app.main.get_model')
def test_create_rerank_with_return_documents(mock_get_model):
    """
    Tests that return_documents correctly includes document text.
    """
    mock_model = mock_get_model.return_value
    mock_model.predict.return_value = [0.1, 0.9]

    documents = ["doc0", "doc1"]
    request_payload = {
        "query": "AI",
        "documents": documents,
        "model": SUPPORTED_RERANK_MODEL,
        "return_documents": True
    }

    response = client.post("/v1/rerank", json=request_payload)
    assert response.status_code == 200
    response_json = response.json()
    assert "text" in response_json["data"][0]
    assert response_json["data"][0]["text"] == documents[1] # doc1 (0.9)
    assert response_json["data"][1]["text"] == documents[0] # doc0 (0.1)
