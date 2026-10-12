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


def test_rate_limiter_is_allowed_new_client():
    limiter = rate_limiter.__class__(default_limit=5, window=60)
    client_id = "new_client"
    assert client_id not in limiter.requests
    allowed = limiter.is_allowed(client_id)
    assert allowed is True
    assert client_id in limiter.requests
    assert len(limiter.requests[client_id]) == 1


def test_rate_limiter_eviction_of_old_timestamps():
    limiter = rate_limiter.__class__(default_limit=2, window=60)
    client_id = "evict_client"

    with patch("time.time", return_value=1000.0):
        assert limiter.is_allowed(client_id) is True  # req 1 at t=1000
    with patch("time.time", return_value=1010.0):
        assert limiter.is_allowed(client_id) is True  # req 2 at t=1010
    with patch("time.time", return_value=1020.0):
        assert limiter.is_allowed(client_id) is False  # limit 2 reached

    # Advance time beyond window for 1st request (1000 + 60 = 1060)
    with patch("time.time", return_value=1060.1):
        # 1000.0 is evicted (1060.1 - 1000.0 = 60.1 >= window)
        # 1010.0 is kept (1060.1 - 1010.0 = 50.1 < window)
        assert limiter.is_allowed(client_id) is True
        assert len(limiter.requests[client_id]) == 2
        assert limiter.requests[client_id] == [1010.0, 1060.1]


def test_rate_limiter_custom_limit_override():
    limiter = rate_limiter.__class__(default_limit=10, window=60)
    client_id = "custom_limit_client"

    # Override limit with lower limit (limit=2)
    with patch("time.time", return_value=1000.0):
        assert limiter.is_allowed(client_id, limit=2) is True
        assert limiter.is_allowed(client_id, limit=2) is True
        assert limiter.is_allowed(client_id, limit=2) is False

    # Override limit with higher limit (limit=5)
    with patch("time.time", return_value=1005.0):
        assert limiter.is_allowed(client_id, limit=5) is True
        assert limiter.is_allowed(client_id, limit=5) is True
        assert limiter.is_allowed(client_id, limit=5) is True
        assert limiter.is_allowed(client_id, limit=5) is False

    # Limit = 0 should block immediately
    zero_limit_client = "zero_limit_client"
    with patch("time.time", return_value=1000.0):
        assert limiter.is_allowed(zero_limit_client, limit=0) is False


def test_rate_limiter_max_allowed_boundary():
    limiter = rate_limiter.__class__(default_limit=3, window=60)
    client_id = "boundary_client"

    with patch("time.time", return_value=1000.0):
        assert limiter.is_allowed(client_id) is True  # len 1
        assert limiter.is_allowed(client_id) is True  # len 2
        assert limiter.is_allowed(client_id) is True  # len 3
        assert limiter.is_allowed(client_id) is False  # len 3 >= 3 max_allowed


def test_rate_limiter_get_retry_after():
    limiter = rate_limiter.__class__(default_limit=2, window=60)

    # Unseen client
    with patch("time.time", return_value=1000.0):
        assert limiter.get_retry_after("unknown_client") == 0

    # Client with empty request list
    limiter.requests["empty_client"] = []
    with patch("time.time", return_value=1000.0):
        assert limiter.get_retry_after("empty_client") == 0

    # Client with requests
    client_id = "retry_client"
    with patch("time.time", return_value=1000.0):
        limiter.is_allowed(client_id)  # oldest at 1000.0

    with patch("time.time", return_value=1020.0):
        # window (60) - (1020 - 1000) = 40
        assert limiter.get_retry_after(client_id) == 40

    with patch("time.time", return_value=1059.5):
        # 60 - (1059.5 - 1000) = 0.5 -> int conversion or max(1, ...)
        assert limiter.get_retry_after(client_id) == 1


def test_rate_limiter_client_isolation():
    limiter = rate_limiter.__class__(default_limit=1, window=60)
    with patch("time.time", return_value=1000.0):
        assert limiter.is_allowed("client_a") is True
        assert limiter.is_allowed("client_a") is False
        # client_b should still be allowed
        assert limiter.is_allowed("client_b") is True


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
