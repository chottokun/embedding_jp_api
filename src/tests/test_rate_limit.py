from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app, rate_limiter

client = TestClient(app)


def test_rate_limit_health_endpoints_exempt():
    rate_limiter.requests.clear()
    for _ in range(5):
        res = client.get("/healthz")
        assert res.status_code == 200


def test_rate_limit_enforced_and_retry_after():
    rate_limiter.requests.clear()
    with patch("app.main.RATE_LIMIT_PER_MINUTE", 2):
        rate_limiter.limit = 2

        # 1st request - ok
        res1 = client.get("/v1/models")
        assert res1.status_code == 200

        # 2nd request - ok
        res2 = client.get("/v1/models")
        assert res2.status_code == 200

        # 3rd request - 429 Too Many Requests
        res3 = client.get("/v1/models")
        assert res3.status_code == 429
        assert res3.json() == {"detail": "Too Many Requests"}
        assert "Retry-After" in res3.headers
        assert int(res3.headers["Retry-After"]) >= 1

    # Cleanup
    rate_limiter.requests.clear()
    rate_limiter.limit = 120
