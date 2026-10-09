import pytest
from unittest.mock import patch
import numpy as np


@pytest.fixture(autouse=True)
def reset_api_key():
    """
    Ensure API_KEY is None by default for all tests to maintain backward compatibility
    and prevent failures if an API_KEY is set in the environment.
    """
    with patch("app.main.API_KEY", None):
        yield


@pytest.fixture
def valid_auth_headers():
    return {"Authorization": "Bearer test_api_key_secret"}


@pytest.fixture
def dummy_embedding_model():
    with patch("app.main.get_model") as mock_get_model:
        mock_model = mock_get_model.return_value
        mock_model.tokenizer.num_special_tokens_to_add.return_value = 2
        mock_model.max_seq_length = 8192
        mock_model.tokenizer.side_effect = lambda text, **kwargs: {"input_ids": [[1]]}
        mock_model.encode.return_value = np.array([[0.1]])
        yield mock_model


@pytest.fixture
def dummy_rerank_model():
    with patch("app.main.get_model") as mock_get_model:
        mock_model = mock_get_model.return_value
        mock_model.tokenizer.num_special_tokens_to_add.return_value = 2
        mock_model.max_seq_length = 8192
        mock_model.tokenizer.side_effect = lambda text, **kwargs: {"input_ids": [[1]]}
        mock_model.predict.return_value = [0.9, 0.1]
        yield mock_model
