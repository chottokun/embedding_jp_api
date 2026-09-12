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
