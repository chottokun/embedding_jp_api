import pytest
from unittest.mock import patch, mock_open
from app.download_models import download_models

def test_download_models_missing_config(capsys):
    with patch("app.download_models.MODELS_FILE") as mock_file:
        mock_file.exists.return_value = False
        mock_file.__str__.return_value = "config/models.yml"
        with pytest.raises(SystemExit) as e:
            download_models()
        assert e.value.code == 1
        captured = capsys.readouterr()
        assert "config/models.yml が見つかりません。" in captured.out

def test_download_models_empty_config(capsys):
    with patch("app.download_models.MODELS_FILE") as mock_file:
        mock_file.exists.return_value = True
        mock_file.__str__.return_value = "config/models.yml"
        with patch("builtins.open", mock_open(read_data="")):
            with patch("yaml.safe_load", return_value=None):
                with pytest.raises(SystemExit) as e:
                    download_models()
                assert e.value.code == 1
                captured = capsys.readouterr()
                assert "config/models.yml が空です。" in captured.out

def test_download_models_no_models(capsys):
    with patch("app.download_models.MODELS_FILE") as mock_file:
        mock_file.exists.return_value = True
        with patch("builtins.open", mock_open(read_data="{}")):
            with patch("yaml.safe_load", return_value={"embedding_models": [], "rerank_models": []}):
                download_models()
                captured = capsys.readouterr()
                assert "ダウンロードするモデルが設定されていません。" in captured.out

def test_download_models_exception_handling(capsys):
    config_data = {
        "embedding_models": ["model1"],
        "rerank_models": ["model2"]
    }
    with patch("app.download_models.MODELS_FILE") as mock_file:
        mock_file.exists.return_value = True
        with patch("builtins.open", mock_open(read_data="dummy")):
            with patch("yaml.safe_load", return_value=config_data):
                with patch("app.download_models.snapshot_download") as mock_download:
                    # model1 fails, model2 succeeds
                    mock_download.side_effect = [Exception("Download error"), None]

                    download_models()

                    captured = capsys.readouterr()
                    assert "エラー: model1 のダウンロードに失敗しました: Download error" in captured.out
                    assert "完了: model2" in captured.out

def test_download_models_happy_path(capsys):
    config_data = {
        "embedding_models": ["model1"],
        "rerank_models": ["model2"]
    }
    with patch("app.download_models.MODELS_FILE") as mock_file:
        mock_file.exists.return_value = True
        with patch("builtins.open", mock_open(read_data="dummy")):
            with patch("yaml.safe_load", return_value=config_data):
                with patch("app.download_models.snapshot_download") as mock_download:
                    download_models()

                    captured = capsys.readouterr()
                    assert "完了: model1" in captured.out
                    assert "完了: model2" in captured.out
                    assert mock_download.call_count == 2
