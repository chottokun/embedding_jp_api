import logging
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.config import EMBEDDING_MODELS, RERANK_MODELS

# Disable raising server exceptions to allow testing the 500 error handler
client = TestClient(app, raise_server_exceptions=False)


def test_http_exception_passed_through():
    """
    Ensure that standard HTTPExceptions (like 400 Bad Request) are not swallowed
    by the global exception handler and still return specific details.
    """
    # Trigger a known 400 error (Invalid Model)
    response = client.post(
        "/v1/embeddings", json={"input": "test", "model": "invalid-model-name"}
    )
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
        "/v1/embeddings", json={"input": "test", "model": "cl-nagoya/ruri-v3-30m"}
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal Server Error"
    # Security check: Ensure the internal error message is NOT leaked
    assert "Unexpected Database Failure" not in response.text


@patch("app.main.get_model")
def test_get_model_value_error_embeddings(mock_get_model):
    """
    Ensure that ValueError during model loading in embeddings is caught
    and returned as a 400 Bad Request.
    """
    mock_get_model.side_effect = ValueError("Custom Model Load Error")

    # Use a valid model name so we pass the initial validation check
    valid_model = EMBEDDING_MODELS[0]
    response = client.post(
        "/v1/embeddings", json={"input": "test", "model": valid_model}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Custom Model Load Error"


@patch("app.main.get_model")
def test_get_model_value_error_rerank(mock_get_model):
    """
    Ensure that ValueError during model loading in reranking is caught
    and returned as a 400 Bad Request.
    """
    mock_get_model.side_effect = ValueError("Custom Rerank Load Error")

    # Use a valid model name so we pass the initial validation check
    valid_model = RERANK_MODELS[0]
    response = client.post(
        "/v1/rerank",
        json={"query": "test", "documents": ["doc"], "model": valid_model},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Custom Rerank Load Error"


@patch("app.main.get_model")
def test_unhandled_exception_logging(mock_get_model, caplog):
    """
    Verify that unhandled exceptions are correctly logged with a stack trace.
    """
    mock_get_model.side_effect = Exception("Logged Exception")

    with caplog.at_level(logging.ERROR):
        client.post(
            "/v1/embeddings", json={"input": "test", "model": "cl-nagoya/ruri-v3-30m"}
        )

    assert "Unhandled exception: Logged Exception" in caplog.text
    # Check that it's logged as an ERROR
    assert any(record.levelname == "ERROR" for record in caplog.records)


@patch("app.main.get_model")
def test_rerank_unhandled_exception_caught(mock_get_model):
    """
    Ensure that unhandled exceptions in the rerank endpoint are also caught.
    """
    mock_get_model.side_effect = Exception("Rerank Failure")

    response = client.post(
        "/v1/rerank",
        json={
            "query": "test",
            "documents": ["doc"],
            "model": "cl-nagoya/ruri-v3-reranker-310m",
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal Server Error"


@patch("app.main.get_model")
def test_security_headers_in_error_response(mock_get_model):
    """
    Ensure security headers are present even in error responses.
    """
    mock_get_model.side_effect = Exception("Header Test Failure")

    response = client.post(
        "/v1/embeddings", json={"input": "test", "model": "cl-nagoya/ruri-v3-30m"}
    )

    assert response.status_code == 500
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-XSS-Protection"] == "1; mode=block"
    assert "Strict-Transport-Security" in response.headers
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "Content-Security-Policy" in response.headers


@patch("app.main.get_model")
def test_inference_exception_caught(mock_get_model):
    """
    Verify that exceptions occurring during model inference (inside model.lock) are caught.
    """
    mock_model = MagicMock()
    # Mock context manager for model.lock
    mock_model.lock.__enter__.return_value = None
    mock_model.lock.__exit__.return_value = None

    mock_tokenizer = MagicMock()
    mock_tokenizer.num_special_tokens_to_add.return_value = 0
    # tokenizer(batch)
    mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}
    mock_model.tokenizer = mock_tokenizer

    # Simulate failure during encode
    mock_model.encode.side_effect = Exception("Inference Failure")

    mock_get_model.return_value = mock_model

    response = client.post(
        "/v1/embeddings", json={"input": "test", "model": "cl-nagoya/ruri-v3-30m"}
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal Server Error"
    assert "Inference Failure" not in response.text
