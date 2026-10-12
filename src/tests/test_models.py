import pytest
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
def test_get_model_embedding_success(mock_st, mock_cuda, caplog):
    """Test that get_model correctly loads an embedding model and logs the output."""
    if not EMBEDDING_MODELS:
        pytest.skip("No embedding models configured")

    model_name = [m for m in EMBEDDING_MODELS if "embeddinggemma" not in m][0]
    mock_instance = MagicMock()
    mock_st.return_value = mock_instance

    with caplog.at_level(logging.INFO):
        model = get_model(model_name)

    assert model == mock_instance
    mock_st.assert_called_once_with(model_name, device="cpu")
    assert model_name in _model_cache
    assert _model_cache[model_name] == model

    # Check that it has locks for thread-safety and they are not mocks
    assert hasattr(model, "lock")
    assert not isinstance(model.lock, MagicMock)
    assert hasattr(model, "tokenizer_lock")
    assert not isinstance(model.tokenizer_lock, MagicMock)

    # Verify that the print replaced logging occurs
    assert f"Loading model '{model_name}' on device 'cpu'..." in caplog.text
    assert f"Model '{model_name}' loaded successfully." in caplog.text


@patch("torch.cuda.is_available", return_value=True)
@patch("app.models.CrossEncoder")
def test_get_model_rerank_success(mock_ce, mock_cuda, caplog):
    """Test that get_model correctly loads a rerank model and logs the output."""
    if not RERANK_MODELS:
        pytest.skip("No rerank models configured")

    model_name = RERANK_MODELS[0]
    mock_instance = MagicMock()
    mock_ce.return_value = mock_instance

    with caplog.at_level(logging.INFO):
        model = get_model(model_name)

    assert model == mock_instance
    mock_ce.assert_called_once_with(model_name, device="cuda")
    assert model_name in _model_cache
    assert _model_cache[model_name] == model

    # Check that it has locks for thread-safety and they are not mocks
    assert hasattr(model, "lock")
    assert not isinstance(model.lock, MagicMock)
    assert hasattr(model, "tokenizer_lock")
    assert not isinstance(model.tokenizer_lock, MagicMock)

    # Verify that the print replaced logging occurs
    assert f"Loading model '{model_name}' on device 'cuda'..." in caplog.text
    assert f"Model '{model_name}' loaded successfully." in caplog.text


@patch("torch.cuda.is_available", return_value=False)
@patch("app.models.SentenceTransformer")
def test_get_model_caching(mock_st, mock_cuda):
    """Test that get_model caches model instances."""
    if not EMBEDDING_MODELS:
        pytest.skip("No embedding models configured")

    model_name = [m for m in EMBEDDING_MODELS if "embeddinggemma" not in m][0]
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


def test_get_model_thread_safety():
    """Test that get_model is thread-safe when loading a new model."""
    if not EMBEDDING_MODELS:
        pytest.skip("No embedding models configured")

    model_name = [m for m in EMBEDDING_MODELS if "embeddinggemma" not in m][0]

    call_count = 0
    call_count_lock = threading.Lock()

    def mocked_st(*args, **kwargs):
        nonlocal call_count
        with call_count_lock:
            call_count += 1
        time.sleep(0.1)  # Simulate slow loading
        return MagicMock()

    with patch("app.models.SentenceTransformer", side_effect=mocked_st):
        with ThreadPoolExecutor(max_workers=5) as executor:
            # All threads try to load the same model
            results = list(executor.map(get_model, [model_name] * 5))

    # All results should be the same instance
    for res in results:
        assert res == results[0]

    # SentenceTransformer should have been called only once despite concurrent calls
    assert call_count == 1


@patch("app.models.SentenceTransformer", side_effect=Exception("Failed to load"))
def test_get_model_load_failure(mock_st):
    """Test that failed loads don't pollute the cache."""
    if not EMBEDDING_MODELS:
        pytest.skip("No embedding models configured")

    model_name = [m for m in EMBEDDING_MODELS if "embeddinggemma" not in m][0]

    with pytest.raises(Exception) as excinfo:
        get_model(model_name)

    assert "Failed to load" in str(excinfo.value)
    # Ensure it's not in cache if it failed to load
    assert model_name not in _model_cache


@patch("app.models.VisualizedBGEEmbeddingModel.__init__", return_value=None)
def test_visualized_bge_encode_multimodal(mock_init):
    """Test VisualizedBGEEmbeddingModel.encode_multimodal helper methods and partition logic."""
    from app.models import VisualizedBGEEmbeddingModel
    from PIL import Image
    import torch

    model_obj = VisualizedBGEEmbeddingModel.__new__(VisualizedBGEEmbeddingModel)
    model_obj.device = "cpu"
    model_obj.lock = threading.RLock()
    model_obj.tokenizer_lock = threading.RLock()

    mock_model = MagicMock()
    model_obj.model = mock_model

    # Mock preprocess_val
    mock_model.preprocess_val.return_value = torch.zeros((3, 224, 224))

    # Mock tokenizer
    def fake_tokenizer(texts, return_tensors=None, padding=None):
        tok = MagicMock()
        tok.to.return_value = tok
        return tok

    mock_model.tokenizer = fake_tokenizer

    # Mock encode outputs
    mock_model.encode_text.return_value = torch.tensor([[0.1, 0.2]])
    mock_model.encode_image.return_value = torch.tensor([[0.3, 0.4]])
    mock_model.encode_mm.return_value = torch.tensor([[0.5, 0.6]])

    img = Image.new("RGB", (10, 10))
    items = [
        ("Hello text", None),          # idx 0: text only
        (None, img),                   # idx 1: image only
        ("Multimodal text", img),      # idx 2: multimodal
        (None, None),                  # idx 3: empty item
    ]

    results = model_obj.encode_multimodal(items)

    assert len(results) == 4
    assert pytest.approx(results[0], rel=1e-5) == [0.1, 0.2]
    assert pytest.approx(results[1], rel=1e-5) == [0.3, 0.4]
    assert pytest.approx(results[2], rel=1e-5) == [0.5, 0.6]
    assert results[3] == []
