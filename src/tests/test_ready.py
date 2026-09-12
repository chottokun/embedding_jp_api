from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Test the /health and /healthz liveness endpoints."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    response_z = client.get("/healthz")
    assert response_z.status_code == 200
    assert response_z.json() == {"status": "ok"}


@patch("torch.cuda.is_available")
def test_ready_endpoint_no_gpu_no_models(mock_cuda):
    """Test the /ready endpoint when no GPU is available and no models are loaded."""
    mock_cuda.return_value = False

    with patch("app.models._model_cache", new={}):
        response = client.get("/ready")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ready"
        assert data["gpu_available"] is False
        assert data["models_loaded"] == []


@patch("torch.cuda.is_available")
def test_ready_endpoint_gpu_and_models(mock_cuda):
    """Test the /ready endpoint when GPU is available and models are loaded."""
    mock_cuda.return_value = True

    mock_cache = {"model_a": MagicMock(), "model_b": MagicMock()}
    with patch("app.models._model_cache", new=mock_cache):
        response = client.get("/ready")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ready"
        assert data["gpu_available"] is True
        assert sorted(data["models_loaded"]) == ["model_a", "model_b"]
