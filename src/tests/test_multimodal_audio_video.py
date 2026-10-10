from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

DUMMY_AUDIO_B64 = (
    "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
)
DUMMY_VIDEO_URL = "https://example.com/video.mp4"


def test_unsupported_model_guard():
    """
    Test 1: cl-nagoya/ruri-v3-30m や cl-nagoya/ruri-v3-310m に input_audio や video_url を含むリクエストを送信した場合、
    推論を実行せずに即座に HTTP 400 Bad Request（"Model does not support audio" 等）が返ることを検証。
    """
    with patch("app.main.get_model") as mock_get_model:
        mock_model = MagicMock()
        mock_model.supports_multimodal = False
        mock_get_model.return_value = mock_model

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "cl-nagoya/ruri-v3-30m",
                "input": {"text": "こんにちは", "input_audio": DUMMY_AUDIO_B64},
            },
        )
        assert response.status_code == 400
        assert (
            "audio" in response.json().get("detail", "").lower()
            or "support" in response.json().get("detail", "").lower()
        )


def test_disabled_flag_audio_embedding():
    """
    Test 2: ENABLE_AUDIO_EMBEDDING=false の環境で google/embeddinggemma-2 に対し input_audio を送った場合、
    HTTP 400 Bad Request（"Audio embedding is disabled on this server" 等）が返ることを検証。
    """
    with (
        patch("app.main.get_model") as mock_get_model,
        patch("app.config.ENABLE_AUDIO_EMBEDDING", False, create=True),
    ):
        mock_model = MagicMock()
        mock_model.supports_multimodal = True
        mock_get_model.return_value = mock_model

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "google/embeddinggemma-2",
                "input": {"text": "テスト", "input_audio": DUMMY_AUDIO_B64},
            },
        )
        assert response.status_code == 400
        detail = response.json().get("detail", "").lower()
        assert "disable" in detail or "audio" in detail


def test_schema_validation_audio_video():
    """
    Test 3: EmbeddingRequest で音声・動画を含む正しい JSON がパースできること。
    input_audio のフォーマット不正や過大な入力時にバリデーションエラーが発生すること。
    """
    with patch("app.main.get_model") as mock_get_model:
        mock_model = MagicMock()
        mock_model.supports_multimodal = True
        mock_get_model.return_value = mock_model

        # Test format error
        invalid_audio_b64 = "data:audio/wav;base64,!!!invalid!!!"
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "google/embeddinggemma-2",
                "input": {"text": "テスト", "input_audio": invalid_audio_b64},
            },
        )
        # Should be 422 or 400
        assert response.status_code in (400, 422)

        # Test large input
        large_b64 = "data:audio/wav;base64," + ("A" * 300_000)
        response_large = client.post(
            "/v1/embeddings",
            json={
                "model": "google/embeddinggemma-2",
                "input": {"text": "テスト", "input_audio": large_b64},
            },
        )
        assert response_large.status_code in (400, 413, 422)


def test_happy_path_audio_video():
    """
    Test 4: ENABLE_AUDIO_EMBEDDING=true and ENABLE_VIDEO_EMBEDDING=true の環境で、
    モックされた EmbeddingGemma2Model に対し音声/動画を含むリクエストが正常に 200 OK となり、
    埋め込みベクトルが返却されること。
    """
    with (
        patch("app.main.get_model") as mock_get_model,
        patch("app.config.ENABLE_AUDIO_EMBEDDING", True, create=True),
        patch("app.config.ENABLE_VIDEO_EMBEDDING", True, create=True),
        patch(
            "app.services.embedding.load_audio_from_source",
            return_value=(MagicMock(), 16000),
        ),
        patch(
            "app.services.embedding.load_video_frames_from_source",
            return_value=[MagicMock()],
        ),
    ):
        mock_model = MagicMock()
        mock_model.supports_multimodal = True
        mock_model.supports_audio = True
        mock_model.supports_video = True
        mock_model.encode_multimodal.return_value = [[0.1, 0.2, 0.3]]
        mock_get_model.return_value = mock_model

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "google/embeddinggemma-2",
                "input": {
                    "text": "テスト",
                    "input_audio": DUMMY_AUDIO_B64,
                    "video_url": DUMMY_VIDEO_URL,
                },
            },
        )

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]
