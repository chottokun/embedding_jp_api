import os
import importlib
from unittest.mock import patch, mock_open
import app.config


def test_config_defaults():
    """Test default configuration values when no environment variables are set."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("pathlib.Path.exists", return_value=False):
            importlib.reload(app.config)
            assert app.config.APP_PORT == 8000
            assert app.config.MAX_INPUT_LENGTH == 65536
            assert app.config.MAX_INPUT_ITEMS == 256
            assert app.config.OFFLINE_MODE is False
            assert app.config.SUPPORTED_MODELS == {}
            assert app.config.EMBEDDING_MODELS == []
            assert app.config.RERANK_MODELS == []


def test_config_custom_env():
    """Test configuration with custom environment variables."""
    custom_env = {
        "APP_PORT": "9000",
        "MAX_INPUT_LENGTH": "1000",
        "MAX_INPUT_ITEMS": "50",
        "OFFLINE_MODE": "true",
    }
    with patch.dict(os.environ, custom_env):
        with patch("pathlib.Path.exists", return_value=False):
            importlib.reload(app.config)
            assert app.config.APP_PORT == 9000
            assert app.config.MAX_INPUT_LENGTH == 1000
            assert app.config.MAX_INPUT_ITEMS == 50
            assert app.config.OFFLINE_MODE is True
            # Check if offline mode environment variables were set
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
            assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def test_config_offline_mode_false():
    """Test OFFLINE_MODE=false explicitly."""
    custom_env = {"OFFLINE_MODE": "false"}
    # Clear them first to ensure they aren't there from previous tests if reload failed to isolate
    with patch.dict(os.environ, custom_env):
        if "HF_HUB_OFFLINE" in os.environ:
            del os.environ["HF_HUB_OFFLINE"]
        if "TRANSFORMERS_OFFLINE" in os.environ:
            del os.environ["TRANSFORMERS_OFFLINE"]

        with patch("pathlib.Path.exists", return_value=False):
            importlib.reload(app.config)
            assert app.config.OFFLINE_MODE is False
            assert "HF_HUB_OFFLINE" not in os.environ
            assert "TRANSFORMERS_OFFLINE" not in os.environ


def test_config_models_file_loading():
    """Test loading models from models.yml."""
    yaml_content = """
embedding_models:
  - "model-e-1"
  - "model-e-2"
rerank_models:
  - "model-r-1"
"""
    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            importlib.reload(app.config)
            assert app.config.SUPPORTED_MODELS == {
                "embedding_models": ["model-e-1", "model-e-2"],
                "rerank_models": ["model-r-1"],
            }
            assert app.config.EMBEDDING_MODELS == ["model-e-1", "model-e-2"]
            assert app.config.RERANK_MODELS == ["model-r-1"]


def test_config_models_file_not_found():
    """Test behavior when models.yml does not exist."""
    with patch("pathlib.Path.exists", return_value=False):
        importlib.reload(app.config)
        assert app.config.SUPPORTED_MODELS == {}
        assert app.config.EMBEDDING_MODELS == []
        assert app.config.RERANK_MODELS == []


def test_config_models_file_empty():
    """Test behavior when models.yml is empty."""
    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data="")):
            importlib.reload(app.config)
            assert app.config.SUPPORTED_MODELS == {}
            assert app.config.EMBEDDING_MODELS == []
            assert app.config.RERANK_MODELS == []


def teardown_module(module):
    """Restore config to original state to avoid affecting other tests."""
    importlib.reload(app.config)
