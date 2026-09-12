from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from app.main import app
import app.models as app_models

client = TestClient(app)


def test_model_unload_all():
    app_models._model_cache.clear()
    mock_m1 = MagicMock()
    mock_m2 = MagicMock()
    app_models._model_cache["test-model-1"] = mock_m1
    app_models._model_cache["test-model-2"] = mock_m2

    res = client.post("/v1/models/unload", json={})
    assert res.status_code == 200
    data = res.json()
    assert set(data["unloaded_models"]) == {"test-model-1", "test-model-2"}
    assert isinstance(data["remaining_memory"], int)
    assert len(app_models._model_cache) == 0


def test_model_unload_specific():
    mock_m1 = MagicMock()
    mock_m2 = MagicMock()
    app_models._model_cache["test-model-1"] = mock_m1
    app_models._model_cache["test-model-2"] = mock_m2

    res = client.post("/v1/models/unload", json={"model": "test-model-1"})
    assert res.status_code == 200
    data = res.json()
    assert data["unloaded_models"] == ["test-model-1"]
    assert "test-model-1" not in app_models._model_cache
    assert "test-model-2" in app_models._model_cache

    # Cleanup
    app_models._model_cache.clear()
