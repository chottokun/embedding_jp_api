from unittest.mock import patch, MagicMock
import pytest
from fastapi import HTTPException
from app.main import _get_model_or_400
from app.config import EMBEDDING_MODELS, RERANK_MODELS


@patch("app.main.get_model")
def test_get_model_or_400_embedding_success(mock_get_model):
    """
    Test that calling _get_model_or_400 with a valid embedding model
    correctly calls get_model and returns the model instance.
    """
    mock_model = MagicMock()
    mock_get_model.return_value = mock_model
    valid_model = EMBEDDING_MODELS[0]

    result = _get_model_or_400(valid_model, "embedding")

    mock_get_model.assert_called_once_with(valid_model)
    assert result == mock_model


def test_get_model_or_400_embedding_not_found():
    """
    Test that calling _get_model_or_400 with an unsupported embedding model
    raises an HTTPException with status code 400.
    """
    with pytest.raises(HTTPException) as exc_info:
        _get_model_or_400("unsupported-embedding-model", "embedding")

    assert exc_info.value.status_code == 400
    assert (
        exc_info.value.detail
        == "Model 'unsupported-embedding-model' not found for embeddings."
    )


@patch("app.main.get_model")
def test_get_model_or_400_rerank_success(mock_get_model):
    """
    Test that calling _get_model_or_400 with a valid rerank model
    correctly calls get_model and returns the model instance.
    """
    mock_model = MagicMock()
    mock_get_model.return_value = mock_model
    valid_model = RERANK_MODELS[0]

    result = _get_model_or_400(valid_model, "rerank")

    mock_get_model.assert_called_once_with(valid_model)
    assert result == mock_model


def test_get_model_or_400_rerank_not_found():
    """
    Test that calling _get_model_or_400 with an unsupported rerank model
    raises an HTTPException with status code 400.
    """
    with pytest.raises(HTTPException) as exc_info:
        _get_model_or_400("unsupported-rerank-model", "rerank")

    assert exc_info.value.status_code == 400
    assert (
        exc_info.value.detail
        == "Model 'unsupported-rerank-model' not found for reranks."
    )


def test_get_model_or_400_invalid_model_type():
    """
    Test that calling _get_model_or_400 with an invalid model_type
    uses RERANK_MODELS as a fallback and reports the invalid type in the error.
    """
    invalid_type = "invalid_type"
    with pytest.raises(HTTPException) as exc_info:
        # Since it's not "embedding", it looks up in RERANK_MODELS.
        # "unsupported-model" will not be in RERANK_MODELS, so it raises the error
        # mentioning the model name and the invalid type.
        _get_model_or_400("unsupported-model", invalid_type)

    assert exc_info.value.status_code == 400
    assert (
        exc_info.value.detail
        == f"Model 'unsupported-model' not found for {invalid_type}s."
    )


@patch("app.main.get_model")
def test_get_model_or_400_value_error(mock_get_model):
    """
    Test that if get_model raises a ValueError, _get_model_or_400 catches it
    and raises an HTTPException with status code 400.
    """
    mock_get_model.side_effect = ValueError("Some model load failure message")
    valid_model = EMBEDDING_MODELS[0]

    with pytest.raises(HTTPException) as exc_info:
        _get_model_or_400(valid_model, "embedding")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Some model load failure message"
