import pytest
from unittest.mock import patch, mock_open
from app.download_models import download_models


@patch("builtins.print")
@patch("sys.exit")
@patch("app.download_models.MODELS_FILE")
def test_download_models_missing_config(mock_models_file, mock_exit, mock_print):
    """Test that download_models exits if the config file is missing."""
    mock_models_file.exists.return_value = False
    mock_exit.side_effect = SystemExit(1)

    with pytest.raises(SystemExit) as excinfo:
        download_models()

    assert excinfo.value.code == 1
    mock_exit.assert_called_once_with(1)
    # Check if any call contains the expected error message substring
    printed_messages = [call.args[0] for call in mock_print.call_args_list]
    assert any("が見つかりません。" in str(msg) for msg in printed_messages)


@patch("builtins.print")
@patch("sys.exit")
@patch("app.download_models.yaml.safe_load")
@patch("builtins.open", new_callable=mock_open)
@patch("app.download_models.MODELS_FILE")
def test_download_models_empty_config(
    mock_models_file, mock_file, mock_yaml, mock_exit, mock_print
):
    """Test that download_models exits if the config file is empty."""
    mock_models_file.exists.return_value = True
    mock_yaml.return_value = None
    mock_exit.side_effect = SystemExit(1)

    with pytest.raises(SystemExit) as excinfo:
        download_models()

    assert excinfo.value.code == 1
    mock_exit.assert_called_once_with(1)
    # Check if any call contains the expected error message substring
    printed_messages = [call.args[0] for call in mock_print.call_args_list]
    assert any("が空です。" in str(msg) for msg in printed_messages)


@patch("builtins.print")
@patch("app.download_models.snapshot_download")
@patch("app.download_models.yaml.safe_load")
@patch("builtins.open", new_callable=mock_open)
@patch("app.download_models.MODELS_FILE")
def test_download_models_no_models_defined(
    mock_models_file, mock_file, mock_yaml, mock_snapshot, mock_print
):
    """Test that download_models does nothing if no models are defined in config."""
    mock_models_file.exists.return_value = True
    mock_yaml.return_value = {"other": []}

    download_models()

    mock_snapshot.assert_not_called()
    mock_print.assert_called_with("ダウンロードするモデルが設定されていません。")


@patch("builtins.print")
@patch("app.download_models.snapshot_download")
@patch("app.download_models.yaml.safe_load")
@patch("builtins.open", new_callable=mock_open)
@patch("app.download_models.MODELS_FILE")
def test_download_models_success(
    mock_models_file, mock_file, mock_yaml, mock_snapshot, mock_print
):
    """Test successful download of all models."""
    mock_models_file.exists.return_value = True
    mock_yaml.return_value = {
        "embedding_models": ["model1"],
        "rerank_models": ["model2"],
    }

    download_models()

    assert mock_snapshot.call_count == 2
    mock_snapshot.assert_any_call(repo_id="model1")
    mock_snapshot.assert_any_call(repo_id="model2")
    printed_messages = [
        str(call.args[0]) for call in mock_print.call_args_list if call.args
    ]
    assert any("完了: model1" in msg for msg in printed_messages)
    assert any("完了: model2" in msg for msg in printed_messages)


@patch("builtins.print")
@patch("sys.exit")
@patch("app.download_models.snapshot_download")
@patch("app.download_models.yaml.safe_load")
@patch("builtins.open", new_callable=mock_open)
@patch("app.download_models.MODELS_FILE")
def test_download_models_failure_exits(
    mock_models_file, mock_file, mock_yaml, mock_snapshot, mock_exit, mock_print
):
    """Test that download_models exits with code 1 if a download fails."""
    mock_models_file.exists.return_value = True
    mock_yaml.return_value = {"embedding_models": ["fail_model"]}
    mock_snapshot.side_effect = Exception("Download error")
    mock_exit.side_effect = SystemExit(1)

    with pytest.raises(SystemExit) as excinfo:
        download_models()

    assert excinfo.value.code == 1
    mock_exit.assert_called_once_with(1)

    # Check that error message was printed for fail_model
    printed_messages = [
        str(call.args[0]) for call in mock_print.call_args_list if call.args
    ]
    assert any(
        "fail_model のダウンロードに失敗しました" in msg for msg in printed_messages
    )
