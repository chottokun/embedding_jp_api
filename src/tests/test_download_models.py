import pytest
from unittest.mock import patch, mock_open
from app.download_models import download_models

@patch("app.download_models.MODELS_FILE")
def test_download_models_file_not_found(mock_models_file):
    """Test when the models.yml file does not exist."""
    mock_models_file.exists.return_value = False
    # Use __str__ to avoid issues when the code prints the mock object
    mock_models_file.__str__.return_value = "config/models.yml"

    with pytest.raises(SystemExit) as excinfo:
        download_models()
    assert excinfo.value.code == 1

@patch("app.download_models.MODELS_FILE")
@patch("builtins.open", new_callable=mock_open, read_data="")
@patch("yaml.safe_load")
def test_download_models_empty_file(mock_yaml_load, mock_file, mock_models_file):
    """Test when the models.yml file is empty."""
    mock_models_file.exists.return_value = True
    mock_yaml_load.return_value = None

    with pytest.raises(SystemExit) as excinfo:
        download_models()
    assert excinfo.value.code == 1

@patch("app.download_models.MODELS_FILE")
@patch("builtins.open", new_callable=mock_open, read_data='{"embedding_models": []}')
@patch("yaml.safe_load")
@patch("app.download_models.snapshot_download")
def test_download_models_no_models(mock_snapshot, mock_yaml_load, mock_file, mock_models_file):
    """Test when no models are defined in the config."""
    mock_models_file.exists.return_value = True
    # If safe_load returns an empty dict, download_models should NOT exit if there's at least {}
    # Wait, the code says:
    # if not data:
    #     print(f"Error: {MODELS_FILE} が空です。")
    #     sys.exit(1)
    # {} is falsy in Python! So it will exit.
    mock_yaml_load.return_value = {"some_key": "some_value"}

    # Should return normally without calling snapshot_download
    download_models()
    mock_snapshot.assert_not_called()

@patch("app.download_models.MODELS_FILE")
@patch("builtins.open", new_callable=mock_open, read_data="embedding_models:\n  - model1")
@patch("yaml.safe_load")
@patch("app.download_models.snapshot_download")
def test_download_models_success(mock_snapshot, mock_yaml_load, mock_file, mock_models_file):
    """Test successful download of multiple models."""
    mock_models_file.exists.return_value = True
    mock_yaml_load.return_value = {
        "embedding_models": ["model1", "model2"],
        "rerank_models": ["model3"]
    }

    download_models()

    assert mock_snapshot.call_count == 3
    mock_snapshot.assert_any_call(repo_id="model1")
    mock_snapshot.assert_any_call(repo_id="model2")
    mock_snapshot.assert_any_call(repo_id="model3")

@patch("app.download_models.MODELS_FILE")
@patch("builtins.open", new_callable=mock_open, read_data="embedding_models:\n  - model1")
@patch("yaml.safe_load")
@patch("app.download_models.snapshot_download")
def test_download_models_partial_failure(mock_snapshot, mock_yaml_load, mock_file, mock_models_file):
    """Test that it continues to the next model if one download fails."""
    mock_models_file.exists.return_value = True
    mock_yaml_load.return_value = {
        "embedding_models": ["fail_model", "success_model"]
    }

    def side_effect(repo_id):
        if repo_id == "fail_model":
            raise Exception("Download failed")
        return None

    mock_snapshot.side_effect = side_effect

    # Should not raise exception or exit
    download_models()

    assert mock_snapshot.call_count == 2
    mock_snapshot.assert_any_call(repo_id="fail_model")
    mock_snapshot.assert_any_call(repo_id="success_model")
