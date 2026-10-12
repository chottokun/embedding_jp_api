from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_security_headers():
    # Let's try a simple GET on a non-existent route to check headers
    # The middleware should apply to all responses, including 404s.
    response = client.get("/non-existent")

    headers = response.headers

    # Check for common security headers
    assert "X-Content-Type-Options" in headers
    assert headers["X-Content-Type-Options"] == "nosniff"

    assert "X-Frame-Options" in headers
    assert headers["X-Frame-Options"] == "DENY"

    assert "X-XSS-Protection" in headers
    assert headers["X-XSS-Protection"] == "1; mode=block"

    assert "Strict-Transport-Security" in headers
    assert headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"

    assert "Referrer-Policy" in headers
    assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

    assert "Content-Security-Policy" in headers
    assert (
        headers["Content-Security-Policy"]
        == "default-src 'self'; frame-ancestors 'none';"
    )


def test_security_headers_on_500_exception():
    # Test that security headers are present on 500 Internal Server Error responses
    err_client = TestClient(app, raise_server_exceptions=False)
    with patch("app.main.get_model", side_effect=RuntimeError("Unexpected error")):
        response = err_client.post(
            "/v1/embeddings", json={"input": "test", "model": "cl-nagoya/ruri-v3-30m"}
        )

    assert response.status_code == 500
    headers = response.headers

    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert headers.get("X-Frame-Options") == "DENY"
    assert headers.get("X-XSS-Protection") == "1; mode=block"
    assert headers.get("Strict-Transport-Security") == "max-age=31536000; includeSubDomains"
    assert headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert headers.get("Content-Security-Policy") == "default-src 'self'; frame-ancestors 'none';"
