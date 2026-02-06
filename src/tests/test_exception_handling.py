from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app

# Disable raising server exceptions to allow testing the 500 error handler
client = TestClient(app, raise_server_exceptions=False)

def test_http_exception_passed_through():
    """
    Ensure that standard HTTPExceptions (like 400 Bad Request) are not swallowed
    by the global exception handler and still return specific details.
    """
    # Trigger a known 400 error (Invalid Model)
    response = client.post("/v1/embeddings", json={"input": "test", "model": "invalid-model-name"})
    assert response.status_code == 400
    assert "not found" in response.json()["detail"]

@patch("app.main.get_model")
def test_unhandled_exception_caught(mock_get_model):
    """
    Ensure that unexpected exceptions are caught by the global handler,
    returning a 500 status and a generic error message (Fail Closed).
    """
    # Simulate an unexpected error during model loading
    mock_get_model.side_effect = Exception("Unexpected Database Failure")

    # Use a valid model name so we pass the initial validation check
    response = client.post(
        "/v1/embeddings",
        json={"input": "test", "model": "cl-nagoya/ruri-v3-30m"}
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal Server Error"
    # Security check: Ensure the internal error message is NOT leaked
    assert "Unexpected Database Failure" not in response.text
