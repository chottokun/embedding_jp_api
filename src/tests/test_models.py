import pytest
from unittest.mock import patch, MagicMock
from app.models import get_model, _model_cache
from app.config import EMBEDDING_MODELS, RERANK_MODELS

@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the model cache before and after each test."""
    # We access the private _model_cache for testing purposes
    _model_cache.clear()
    yield
    _model_cache.clear()

def test_get_model_unsupported_error():
    """Test that get_model raises ValueError for unsupported models."""
    invalid_model = "unsupported-model-name"
    # Ensure it's not in supported models
    assert invalid_model not in EMBEDDING_MODELS
    assert invalid_model not in RERANK_MODELS

    with pytest.raises(ValueError) as excinfo:
        get_model(invalid_model)

    assert f"Model '{invalid_model}' is not supported." in str(excinfo.value)

@patch("torch.cuda.is_available", return_value=False)
@patch("app.models.SentenceTransformer")
def test_get_model_embedding_success(mock_st, mock_cuda):
    """Test that get_model correctly loads an embedding model."""
    if not EMBEDDING_MODELS:
        pytest.skip("No embedding models configured")

    model_name = EMBEDDING_MODELS[0]
    mock_instance = MagicMock()
    mock_st.return_value = mock_instance

    model = get_model(model_name)

    assert model == mock_instance
    mock_st.assert_called_once_with(model_name, device="cpu")
    assert model_name in _model_cache
    assert _model_cache[model_name] == model
    # Check that it has a lock for thread-safety
    assert hasattr(model, "lock")

@patch("torch.cuda.is_available", return_value=True)
@patch("app.models.CrossEncoder")
def test_get_model_rerank_success(mock_ce, mock_cuda):
    """Test that get_model correctly loads a rerank model."""
    if not RERANK_MODELS:
        pytest.skip("No rerank models configured")

    model_name = RERANK_MODELS[0]
    mock_instance = MagicMock()
    mock_ce.return_value = mock_instance

    model = get_model(model_name)

    assert model == mock_instance
    mock_ce.assert_called_once_with(model_name, device="cuda")
    assert model_name in _model_cache
    assert _model_cache[model_name] == model
    # Check that it has a lock for thread-safety
    assert hasattr(model, "lock")

@patch("torch.cuda.is_available", return_value=False)
@patch("app.models.SentenceTransformer")
def test_get_model_caching(mock_st, mock_cuda):
    """Test that get_model caches model instances."""
    if not EMBEDDING_MODELS:
        pytest.skip("No embedding models configured")

    model_name = EMBEDDING_MODELS[0]
    mock_instance = MagicMock()
    mock_st.return_value = mock_instance

    # First call - should instantiate the model
    model1 = get_model(model_name)
    # Second call - should return the cached instance
    model2 = get_model(model_name)

    assert model1 == model2
    assert model1 == mock_instance
    # SentenceTransformer should only be called once due to caching
    mock_st.assert_called_once()
