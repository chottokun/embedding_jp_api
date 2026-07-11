import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app, verify_api_key

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
            mock_model.tokenizer.side_effect = lambda text, **kwargs: {
                "input_ids": [[1]]
            }
            import numpy as np

            mock_model.encode.return_value = np.array([[0.1]])

            response = client.post(
                "/v1/embeddings",
                json={"input": "test", "model": "cl-nagoya/ruri-v3-30m"},
            )
            assert response.status_code == 200


def test_auth_key_configured_missing_in_request():
    """Access should be denied when API_KEY is set but missing in request."""
    with patch("app.main.API_KEY", "secret-key"):
        response = client.post(
            "/v1/embeddings", json={"input": "test", "model": "cl-nagoya/ruri-v3-30m"}
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or missing API Key"


def test_auth_key_configured_wrong_key():
    """Access should be denied when API_KEY is set but incorrect key is provided."""
    with patch("app.main.API_KEY", "secret-key"):
        response = client.post(
            "/v1/embeddings",
            json={"input": "test", "model": "cl-nagoya/ruri-v3-30m"},
            headers={"Authorization": "Bearer wrong-key"},
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
            mock_model.tokenizer.side_effect = lambda text, **kwargs: {
                "input_ids": [[1]]
            }
            import numpy as np

            mock_model.encode.return_value = np.array([[0.1]])

            response = client.post(
                "/v1/embeddings",
                json={"input": "test", "model": "cl-nagoya/ruri-v3-30m"},
                headers={"Authorization": "Bearer secret-key"},
            )
            assert response.status_code == 200


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_verify_api_key_no_key_configured_none():
    """verify_api_key should return None when no API_KEY is set and auth is None."""
    with patch("app.main.API_KEY", None):
        result = await verify_api_key(None)
        assert result is None


@pytest.mark.anyio
async def test_verify_api_key_no_key_configured_with_credentials():
    """verify_api_key should return the credentials when no API_KEY is set."""
    auth = HTTPAuthorizationCredentials(scheme="Bearer", credentials="any-key")
    with patch("app.main.API_KEY", None):
        result = await verify_api_key(auth)
        assert result == auth


@pytest.mark.anyio
async def test_verify_api_key_configured_none():
    """verify_api_key should raise 401 HTTPException when API_KEY is set but auth is None."""
    with patch("app.main.API_KEY", "secret-key"):
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(None)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid or missing API Key"
        assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}


@pytest.mark.anyio
async def test_verify_api_key_configured_wrong_key():
    """verify_api_key should raise 401 HTTPException when API_KEY is set but credentials do not match."""
    auth = HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong-key")
    with patch("app.main.API_KEY", "secret-key"):
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(auth)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid or missing API Key"
        assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}


@pytest.mark.anyio
async def test_verify_api_key_configured_correct_key():
    """verify_api_key should return the correct credentials when they match the configured API_KEY."""
    auth = HTTPAuthorizationCredentials(scheme="Bearer", credentials="secret-key")
    with patch("app.main.API_KEY", "secret-key"):
        result = await verify_api_key(auth)
        assert result == auth
