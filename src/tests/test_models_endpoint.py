import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers.models import list_models
from app.schemas import ModelCard, ModelList

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


@patch("app.main.API_KEY", "test-key")
def test_models_endpoint_invalid_auth():
    """Verify HTTP 401 status when an invalid Bearer token is provided."""
    response = client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
    assert response.status_code == 401
    assert "detail" in response.json()


@pytest.mark.anyio
async def test_list_models_direct():
    """Directly test the list_models async function return value and fields."""
    with (
        patch("app.main.EMBEDDING_MODELS", ["model-z", "model-a"]),
        patch("app.main.RERANK_MODELS", ["model-m"]),
    ):
        before = int(time.time())
        result = await list_models()
        after = int(time.time())

        assert isinstance(result, ModelList)
        assert result.object == "list"
        assert len(result.data) == 3
        assert [m.id for m in result.data] == ["model-a", "model-m", "model-z"]
        for card in result.data:
            assert isinstance(card, ModelCard)
            assert card.object == "model"
            assert card.owned_by == "custom"
            assert card.permission == []
            assert before <= card.created <= after


def test_models_endpoint_sorting():
    """Verify that models are sorted alphabetically in the API response."""
    with (
        patch("app.main.EMBEDDING_MODELS", ["zeta-model", "alpha-model"]),
        patch("app.main.RERANK_MODELS", ["beta-model", "gamma-model"]),
    ):
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        model_ids = [m["id"] for m in data["data"]]
        assert model_ids == ["alpha-model", "beta-model", "gamma-model", "zeta-model"]


def test_models_endpoint_empty_list():
    """Verify response when both embedding and rerank model lists are empty."""
    with (
        patch("app.main.EMBEDDING_MODELS", []),
        patch("app.main.RERANK_MODELS", []),
    ):
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert data["data"] == []


@pytest.mark.anyio
async def test_list_models_fallback():
    """Verify fallback mechanism when app.main lacks model attributes."""
    import app.main as main_mod

    orig_emb = getattr(main_mod, "EMBEDDING_MODELS", None)
    orig_rerank = getattr(main_mod, "RERANK_MODELS", None)

    try:
        if hasattr(main_mod, "EMBEDDING_MODELS"):
            delattr(main_mod, "EMBEDDING_MODELS")
        if hasattr(main_mod, "RERANK_MODELS"):
            delattr(main_mod, "RERANK_MODELS")

        with (
            patch("app.routers.models.EMBEDDING_MODELS", ["fallback-emb"]),
            patch("app.routers.models.RERANK_MODELS", ["fallback-rerank"]),
        ):
            result = await list_models()
            model_ids = [m.id for m in result.data]
            assert model_ids == ["fallback-emb", "fallback-rerank"]
    finally:
        if orig_emb is not None:
            main_mod.EMBEDDING_MODELS = orig_emb
        if orig_rerank is not None:
            main_mod.RERANK_MODELS = orig_rerank


@pytest.mark.anyio
async def test_list_models_import_error():
    """Verify fallback mechanism when importing app.main raises ImportError."""
    real_import = (
        __builtins__["__import__"]
        if isinstance(__builtins__, dict)
        else getattr(__builtins__, "__import__")
    )

    def mock_import(name, *args, **kwargs):
        if name == "app.main":
            raise ImportError("Simulated import failure")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        with (
            patch("app.routers.models.EMBEDDING_MODELS", ["fallback-mod1"]),
            patch("app.routers.models.RERANK_MODELS", ["fallback-mod2"]),
        ):
            result = await list_models()
            model_ids = [m.id for m in result.data]
            assert model_ids == ["fallback-mod1", "fallback-mod2"]
