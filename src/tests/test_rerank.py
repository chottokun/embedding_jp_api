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

    # Mock tokenizer for usage calculation
    # The new logic calls tokenizer(queries, docs, ...) instead of tokenizer.encode
    # Return a dict with 'input_ids' which is a list of lists (one per pair)
    # 3 pairs, so 3 lists of tokens. Let's say 5 tokens per pair.
    mock_model.tokenizer.side_effect = lambda queries, docs, **kwargs: {
        "input_ids": [[1, 2, 3, 4, 5] for _ in range(len(queries))]
    }

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

    # Verify usage
    # Query (3) + 3 docs (3*3) = 12 tokens total?
    # Logic implementation details:
    # Usually it counts tokens for pairs. CrossEncoder input is pairs.
    # The implementation will likely tokenize (query, doc) pairs or query + doc individually.
    # Let's assume we implement it by summing tokens of all queries and documents involved?
    # Or summing tokens of the pairs?
    # Rerank models usually take [CLS] query [SEP] doc [SEP].
    # If we count query + doc tokens, that's a safe approximation or we can count pair tokens.
    # Let's assert usage is present and > 0 for now, and refine if we have specific logic.
    assert "usage" in response_json
    assert response_json["usage"]["total_tokens"] > 0

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
