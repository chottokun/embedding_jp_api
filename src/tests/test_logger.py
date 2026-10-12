from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_logger_middleware():
    with patch("app.main.logger.info") as mock_logger:
        response = client.get("/health")

        assert response.status_code == 200
        assert "X-Request-ID" in response.headers

        mock_logger.assert_called_once()
        args, kwargs = mock_logger.call_args

        assert "GET /health - 200" in args[0]
        assert kwargs["extra"]["path"] == "/health"
        assert kwargs["extra"]["method"] == "GET"
        assert kwargs["extra"]["status_code"] == 200
        assert "latency" in kwargs["extra"]


def test_x_request_id_passed():
    test_id = "test-request-id-123"
    with patch("app.main.logger.info") as mock_logger:
        response = client.get("/health", headers={"X-Request-ID": test_id})

        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == test_id
        mock_logger.assert_called_once()


def test_json_formatter_redacts_exception_pii():
    import json
    import logging
    import sys
    from app.middleware.logging import JsonFormatter

    formatter = JsonFormatter()

    try:
        raise ValueError("Error with sensitive email user@example.com")
    except ValueError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test_logger",
        level=logging.ERROR,
        pathname="test.py",
        lineno=10,
        msg="An error occurred",
        args=(),
        exc_info=exc_info,
    )

    formatted_json = formatter.format(record)
    log_data = json.loads(formatted_json)

    assert "exception" in log_data
    assert "user@example.com" not in log_data["exception"]
    assert "[REDACTED]" in log_data["exception"]
