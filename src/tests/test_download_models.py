import pytest
from unittest.mock import patch, mock_open
import os
from unittest.mock import MagicMock
from app.download_models import download_models, verify_offline_loading


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


@patch("builtins.print")
@patch("app.download_models.verify_offline_loading")
@patch("app.download_models.snapshot_download")
@patch("app.download_models.yaml.safe_load")
@patch("builtins.open", new_callable=mock_open)
@patch("app.download_models.MODELS_FILE")
def test_download_models_with_verify_offline(
    mock_models_file, mock_file, mock_yaml, mock_snapshot, mock_verify, mock_print
):
    """Test download_models calls verify_offline_loading when verify_offline=True."""
    mock_models_file.exists.return_value = True
    mock_yaml.return_value = {"embedding_models": ["model1"]}

    download_models(verify_offline=True)

    mock_verify.assert_called_once_with(["model1"])


@patch("builtins.print")
@patch("app.models.get_model")
def test_verify_offline_loading_success(mock_get_model, mock_print):
    """Test verify_offline_loading success path when all models load offline."""
    mock_get_model.return_value = MagicMock()

    test_models = ["model_a", "model_b"]
    verify_offline_loading(test_models)

    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert os.environ.get("HF_DATASETS_OFFLINE") == "1"

    assert mock_get_model.call_count == 2
    mock_get_model.assert_any_call("model_a", device="cpu")
    mock_get_model.assert_any_call("model_b", device="cpu")

    printed_messages = [
        str(call.args[0]) for call in mock_print.call_args_list if call.args
    ]
    assert any(
        "全モデルのオフライン完全ロード検証に成功しました！" in msg for msg in printed_messages
    )


@patch("builtins.print")
@patch("sys.exit")
@patch("app.models.get_model")
def test_verify_offline_loading_exception_failure(
    mock_get_model, mock_exit, mock_print
):
    """Test verify_offline_loading failure when get_model raises an exception."""
    mock_get_model.side_effect = RuntimeError("Model loading error")
    mock_exit.side_effect = SystemExit(1)

    with pytest.raises(SystemExit) as excinfo:
        verify_offline_loading(["fail_model"])

    assert excinfo.value.code == 1
    mock_exit.assert_called_once_with(1)

    printed_messages = [
        str(call.args[0]) for call in mock_print.call_args_list if call.args
    ]
    assert any("FAILED ❌" in msg for msg in printed_messages)


@patch("builtins.print")
@patch("sys.exit")
@patch("app.models.get_model")
def test_verify_offline_loading_none_model_failure(
    mock_get_model, mock_exit, mock_print
):
    """Test verify_offline_loading failure when get_model returns None (AssertionError)."""
    mock_get_model.return_value = None
    mock_exit.side_effect = SystemExit(1)

    with pytest.raises(SystemExit) as excinfo:
        verify_offline_loading(["none_model"])

    assert excinfo.value.code == 1
    mock_exit.assert_called_once_with(1)

    printed_messages = [
        str(call.args[0]) for call in mock_print.call_args_list if call.args
    ]
    assert any("FAILED ❌" in msg for msg in printed_messages)
