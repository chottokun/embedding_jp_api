from unittest.mock import patch
import pytest
from fastapi import Request, Response
from fastapi.testclient import TestClient
from app.main import app
from app.middleware.logging import RequestLoggingMiddleware, request_id_var

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


@pytest.mark.anyio
async def test_logger_middleware_unit_exception():
    # Unit test RequestLoggingMiddleware directly when call_next raises an exception.
    middleware = RequestLoggingMiddleware(app)
    mock_request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/error-test",
            "headers": [(b"x-request-id", b"unit-test-id")],
        }
    )

    async def mock_call_next(req):
        raise RuntimeError("Downstream failure")

    with patch("app.middleware.logging.logger.info") as mock_logger:
        with pytest.raises(RuntimeError, match="Downstream failure"):
            await middleware.dispatch(mock_request, mock_call_next)

        mock_logger.assert_called_once()
        args, kwargs = mock_logger.call_args

        assert "GET /error-test - 500" in args[0]
        assert kwargs["extra"]["path"] == "/error-test"
        assert kwargs["extra"]["method"] == "GET"
        assert kwargs["extra"]["status_code"] == 500
        assert "latency" in kwargs["extra"]

    # Verify contextvar request_id_var was properly reset
    assert request_id_var.get() == ""
