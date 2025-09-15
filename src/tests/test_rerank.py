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
