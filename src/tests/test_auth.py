from fastapi.testclient import TestClient
from unittest.mock import patch

# Corrected imports for a 'src' layout
from app.main import app
from app.config import EMBEDDING_MODELS

client = TestClient(app)

SUPPORTED_EMBED_MODEL = EMBEDDING_MODELS[0]


@patch("app.main.get_model")
def test_access_without_api_key_when_required(mock_get_model):
    """
    Test that access is denied when API_KEY is set but no key is provided.
    """
    with patch("app.main.API_KEY", "test-api-key"):
        request_payload = {"input": "test", "model": SUPPORTED_EMBED_MODEL}
        response = client.post("/v1/embeddings", json=request_payload)
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or missing API Key."


@patch("app.main.get_model")
def test_access_with_invalid_api_key(mock_get_model):
    """
    Test that access is denied when an invalid API key is provided.
    """
    with patch("app.main.API_KEY", "test-api-key"):
        request_payload = {"input": "test", "model": SUPPORTED_EMBED_MODEL}
        headers = {"Authorization": "Bearer invalid-key"}
        response = client.post("/v1/embeddings", json=request_payload, headers=headers)
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or missing API Key."


@patch("app.main.get_model")
def test_access_with_valid_api_key(mock_get_model):
    """
    Test that access is granted when a valid API key is provided.
    """
    with patch("app.main.API_KEY", "test-api-key"):
        mock_model = mock_get_model.return_value
        mock_model.tokenizer.side_effect = lambda text, **kwargs: {
            "input_ids": [[1, 2, 3]] if isinstance(text, list) else [1, 2, 3]
        }
        mock_model.tokenizer.num_special_tokens_to_add.return_value = 2
        mock_model.max_seq_length = 8192
        import numpy as np

        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        request_payload = {"input": "test", "model": SUPPORTED_EMBED_MODEL}
        headers = {"Authorization": "Bearer test-api-key"}
        response = client.post("/v1/embeddings", json=request_payload, headers=headers)
        assert response.status_code == 200


def test_access_without_api_key_when_not_required():
    """
    Test that access is granted when API_KEY is NOT set and no key is provided.
    """
    with patch("app.main.API_KEY", None):
        # We need to mock get_model and model behavior if we want a 200
        with patch("app.main.get_model") as mock_get_model:
            mock_model = mock_get_model.return_value
            mock_model.tokenizer.side_effect = lambda text, **kwargs: {
                "input_ids": [[1, 2, 3]] if isinstance(text, list) else [1, 2, 3]
            }
            mock_model.tokenizer.num_special_tokens_to_add.return_value = 2
            mock_model.max_seq_length = 8192
            import numpy as np

            mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

            request_payload = {"input": "test", "model": SUPPORTED_EMBED_MODEL}
            response = client.post("/v1/embeddings", json=request_payload)
            assert response.status_code == 200
