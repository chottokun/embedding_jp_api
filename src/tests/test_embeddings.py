import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

# Corrected imports for a 'src' layout
from app.main import app
from app.config import EMBEDDING_MODELS

client = TestClient(app)

SUPPORTED_EMBED_MODEL = EMBEDDING_MODELS[0]

@patch('app.main.get_model')
def test_create_embeddings_single_string(mock_get_model):
    """
    Tests the /v1/embeddings endpoint with a single string input.
    Ensures the '検索クエリ: ' prefix is added.
    """
    # Arrange
    mock_model = mock_get_model.return_value
    mock_model.encode.return_value = [[0.1, 0.2, 0.3]]

    request_payload = {
        "input": "今日の天気",
        "model": SUPPORTED_EMBED_MODEL
    }

    # Act
    response = client.post("/v1/embeddings", json=request_payload)

    # Assert
    assert response.status_code == 200
    mock_model.encode.assert_called_once_with(["検索クエリ: 今日の天気"])

    response_json = response.json()
    assert response_json["model"] == SUPPORTED_EMBED_MODEL
    assert len(response_json["data"]) == 1
    assert response_json["data"][0]["embedding"] == [0.1, 0.2, 0.3]

@patch('app.main.get_model')
def test_create_embeddings_string_array(mock_get_model):
    """
    Tests the /v1/embeddings endpoint with a list of strings.
    Ensures the '検索文書: ' prefix is added to each item.
    """
    # Arrange
    mock_model = mock_get_model.return_value
    mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    request_payload = {
        "input": ["文書A", "文書B"],
        "model": SUPPORTED_EMBED_MODEL
    }

    # Act
    response = client.post("/v1/embeddings", json=request_payload)

    # Assert
    assert response.status_code == 200
    mock_model.encode.assert_called_once_with(["検索文書: 文書A", "検索文書: 文書B"])

    response_json = response.json()
    assert len(response_json["data"]) == 2
    assert response_json["data"][0]["index"] == 0
    assert response_json["data"][1]["index"] == 1

def test_embeddings_unsupported_model():
    """
    Tests that requesting an unsupported model returns a 400 error.
    """
    request_payload = {
        "input": "test",
        "model": "unsupported-model-name"
    }
    response = client.post("/v1/embeddings", json=request_payload)
    assert response.status_code == 400
    assert "not found" in response.json()["detail"]

def test_embeddings_openai_compatible_response():
    """
    Tests if the response schema is compatible with OpenAI's API.
    """
    request_payload = {
        "input": "test input",
        "model": SUPPORTED_EMBED_MODEL
    }
    response = client.post("/v1/embeddings", json=request_payload)
    assert response.status_code == 200

    response_json = response.json()
    assert response_json["object"] == "list"
    assert "data" in response_json
    assert "model" in response_json
    assert "usage" in response_json

    data_item = response_json["data"][0]
    assert data_item["object"] == "embedding"
    assert "embedding" in data_item
    assert "index" in data_item

    usage = response_json["usage"]
    assert "prompt_tokens" in usage
    assert "total_tokens" in usage
