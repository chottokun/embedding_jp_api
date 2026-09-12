from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app

client = TestClient(app)


def test_models_endpoint():
    with (
        patch("app.main.EMBEDDING_MODELS", ["model-a", "model-b"]),
        patch("app.main.RERANK_MODELS", ["model-b", "model-c"]),
    ):
        response = client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "list"
        assert "data" in data

        models = data["data"]
        assert len(models) == 3

        model_ids = [m["id"] for m in models]
        assert model_ids == ["model-a", "model-b", "model-c"]

        for m in models:
            assert m["object"] == "model"
            assert "created" in m
            assert isinstance(m["created"], int)
            assert m["owned_by"] == "custom"
            assert m["permission"] == []


@patch("app.main.API_KEY", "test-key")
def test_models_endpoint_auth_missing():
    response = client.get("/v1/models")
    assert response.status_code in (401, 403)


@patch("app.main.API_KEY", "test-key")
def test_models_endpoint_auth_success():
    with (
        patch("app.main.EMBEDDING_MODELS", ["model-a"]),
        patch("app.main.RERANK_MODELS", []),
    ):
        response = client.get(
            "/v1/models", headers={"Authorization": "Bearer test-key"}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "model-a"
