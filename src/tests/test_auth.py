from fastapi.testclient import TestClient
from unittest.mock import patch
import pytest
from app.main import app

client = TestClient(app)

def test_auth_no_key_configured():
    """Access should be granted when no API_KEY is set."""
    with patch("app.main.API_KEY", None):
        # We don't need to mock models for 401/403,
        # but for 200 we might need to mock get_model if it gets that far.
        with patch("app.main.get_model") as mock_get_model:
            mock_model = mock_get_model.return_value
            mock_model.tokenizer.num_special_tokens_to_add.return_value = 2
            mock_model.max_seq_length = 8192
            # Mocking encode and tokenizer as well to avoid errors during inference
            mock_model.tokenizer.side_effect = lambda text, **kwargs: {"input_ids": [[1]]}
            import numpy as np
            mock_model.encode.return_value = np.array([[0.1]])

            response = client.post(
                "/v1/embeddings",
                json={"input": "test", "model": "cl-nagoya/ruri-v3-30m"}
            )
            assert response.status_code == 200

def test_auth_key_configured_missing_in_request():
    """Access should be denied when API_KEY is set but missing in request."""
    with patch("app.main.API_KEY", "secret-key"):
        response = client.post(
            "/v1/embeddings",
            json={"input": "test", "model": "cl-nagoya/ruri-v3-30m"}
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or missing API Key"

def test_auth_key_configured_wrong_key():
    """Access should be denied when API_KEY is set but incorrect key is provided."""
    with patch("app.main.API_KEY", "secret-key"):
        response = client.post(
            "/v1/embeddings",
            json={"input": "test", "model": "cl-nagoya/ruri-v3-30m"},
            headers={"Authorization": "Bearer wrong-key"}
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or missing API Key"

def test_auth_key_configured_correct_key():
    """Access should be granted when API_KEY is set and correct key is provided."""
    with patch("app.main.API_KEY", "secret-key"):
        with patch("app.main.get_model") as mock_get_model:
            mock_model = mock_get_model.return_value
            mock_model.tokenizer.num_special_tokens_to_add.return_value = 2
            mock_model.max_seq_length = 8192
            mock_model.tokenizer.side_effect = lambda text, **kwargs: {"input_ids": [[1]]}
            import numpy as np
            mock_model.encode.return_value = np.array([[0.1]])

            response = client.post(
                "/v1/embeddings",
                json={"input": "test", "model": "cl-nagoya/ruri-v3-30m"},
                headers={"Authorization": "Bearer secret-key"}
            )
            assert response.status_code == 200
