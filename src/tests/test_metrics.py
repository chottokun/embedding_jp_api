from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_metrics_endpoint_unauthenticated():
    """Verify that /metrics endpoint is accessible without API key and returns prometheus formatted text."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text
    assert "http_request_duration_seconds" in response.text


def test_metrics_middleware_increments_counter():
    """Verify that calling endpoints increments Prometheus metrics."""
    # Trigger /health
    res_health = client.get("/health")
    assert res_health.status_code == 200

    # Trigger /ready
    res_ready = client.get("/ready")
    assert res_ready.status_code == 200

    # Fetch /metrics
    res_metrics = client.get("/metrics")
    assert res_metrics.status_code == 200
    metrics_text = res_metrics.text

    assert 'endpoint="/health"' in metrics_text
    assert 'endpoint="/ready"' in metrics_text
    assert 'http_status="200"' in metrics_text
