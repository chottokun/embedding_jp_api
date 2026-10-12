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
    orig_limit = rate_limiter.default_limit
    try:
        with patch("app.main.RATE_LIMIT_PER_MINUTE", 2):
            rate_limiter.default_limit = 2

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
    finally:
        rate_limiter.requests.clear()
        rate_limiter.default_limit = orig_limit


def test_get_retry_after_unknown_and_empty_client():
    """Verify get_retry_after returns 0 for unknown client or empty request history."""
    rate_limiter.requests.clear()

    # Unknown client
    assert rate_limiter.get_retry_after("nonexistent_client") == 0

    # Client present but with empty list
    rate_limiter.requests["empty_client"] = []
    assert rate_limiter.get_retry_after("empty_client") == 0


def test_get_retry_after_window_calculation_and_boundaries():
    """Verify retry_after time calculations across window boundaries."""
    rate_limiter.requests.clear()
    client_id = "test_calc_client"
    window = rate_limiter.window  # 60

    # 1. Immediate request (elapsed = 0) -> retry_after should be full window (60)
    with patch("time.time", return_value=1000.0):
        rate_limiter.requests[client_id] = [1000.0]
        assert rate_limiter.get_retry_after(client_id) == window

    # 2. Mid-window request (elapsed = 25s) -> retry_after should be 35
    with patch("time.time", return_value=1025.0):
        rate_limiter.requests[client_id] = [1000.0]
        assert rate_limiter.get_retry_after(client_id) == window - 25

    # 3. Near window boundary (elapsed = 59.9s) -> retry_after should be 1
    with patch("time.time", return_value=1059.9):
        rate_limiter.requests[client_id] = [1000.0]
        assert rate_limiter.get_retry_after(client_id) == 1

    # 4. Exactly at window limit (elapsed = 60s) -> retry_after should be 1 (due to max(1, retry_after))
    with patch("time.time", return_value=1060.0):
        rate_limiter.requests[client_id] = [1000.0]
        assert rate_limiter.get_retry_after(client_id) == 1

    # 5. Past window limit (elapsed = 75s) -> retry_after should be 1 (due to max(1, retry_after))
    with patch("time.time", return_value=1075.0):
        rate_limiter.requests[client_id] = [1000.0]
        assert rate_limiter.get_retry_after(client_id) == 1


def test_get_retry_after_clock_skew_future_timestamps():
    """Verify get_retry_after when request timestamp is in the future (clock skew)."""
    rate_limiter.requests.clear()
    client_id = "future_timestamp_client"
    window = rate_limiter.window  # 60

    # Oldest timestamp is 10s in the future (e.g. 1010.0 vs now=1000.0)
    # retry_after = 60 - int(1000 - 1010) = 60 - (-10) = 70
    with patch("time.time", return_value=1000.0):
        rate_limiter.requests[client_id] = [1010.0]
        assert rate_limiter.get_retry_after(client_id) == window + 10


def test_rate_limit_per_client_api_key():
    """Verify custom per-key rate limits from API_KEYS_MAP."""
    rate_limiter.requests.clear()
    orig_limit = rate_limiter.default_limit
    try:
        # custom-vip allows 3 requests, default allows 1
        with patch("app.main.API_KEYS_MAP", {"custom-vip": 3}):
            rate_limiter.default_limit = 1
            headers = {"Authorization": "Bearer custom-vip"}

            res1 = client.get("/v1/models", headers=headers)
            assert res1.status_code == 200

            res2 = client.get("/v1/models", headers=headers)
            assert res2.status_code == 200

            res3 = client.get("/v1/models", headers=headers)
            assert res3.status_code == 200

            # 4th request breaches the limit of 3
            res4 = client.get("/v1/models", headers=headers)
            assert res4.status_code == 429
    finally:
        rate_limiter.requests.clear()
        rate_limiter.default_limit = orig_limit
