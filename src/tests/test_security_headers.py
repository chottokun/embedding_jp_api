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
    assert headers["Content-Security-Policy"] == "default-src 'self'; frame-ancestors 'none';"
