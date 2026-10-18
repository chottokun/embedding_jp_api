import pytest
import httpx
import base64
from fastapi import HTTPException
from PIL import Image
from unittest.mock import patch, MagicMock

import torch
from app.media_utils import (
    _download_media,
    _process_audio,
    _process_video,
)
from app.config import MAX_AUDIO_DURATION_SEC


@pytest.mark.anyio
async def test_download_media_base64():
    client = httpx.AsyncClient()
    test_data = b"fake audio data"
    b64_data = base64.b64encode(test_data).decode("utf-8")
    source = f"data:audio/wav;base64,{b64_data}"

    result = await _download_media(source, client)
    assert result == test_data


@pytest.mark.anyio
async def test_download_media_base64_too_large():
    client = httpx.AsyncClient()
    with patch("app.media_utils.MAX_FILE_SIZE", 10):
        test_data = b"this data is larger than 10 bytes"
        b64_data = base64.b64encode(test_data).decode("utf-8")
        source = f"data:audio/wav;base64,{b64_data}"

        with pytest.raises(HTTPException) as excinfo:
            await _download_media(source, client)
        assert excinfo.value.status_code == 400
        assert "exceeds limit" in excinfo.value.detail


@patch("torchaudio.load")
def test_process_audio(mock_load):
    sample_rate = 48000
    duration = 2
    t = torch.linspace(0, duration, sample_rate * duration)
    waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0).repeat(2, 1)  # stereo

    mock_load.return_value = (waveform, sample_rate)

    processed_waveform, processed_sr = _process_audio(b"fake audio data")

    assert processed_sr == 16000
    assert processed_waveform.shape[0] == 1  # mono
    assert processed_waveform.shape[1] == 16000 * duration


@patch("torchaudio.load")
def test_process_audio_truncate(mock_load):
    sample_rate = 16000
    duration = MAX_AUDIO_DURATION_SEC + 5
    t = torch.linspace(0, duration, sample_rate * duration)
    waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)

    mock_load.return_value = (waveform, sample_rate)

    processed_waveform, processed_sr = _process_audio(b"fake audio data")

    assert processed_sr == 16000
    assert processed_waveform.shape[1] == 16000 * MAX_AUDIO_DURATION_SEC


@patch("av.open")
def test_process_video(mock_open, tmp_path):
    mock_container = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_container

    mock_stream = MagicMock()
    mock_stream.frames = 0  # Forces to read to get length
    mock_container.streams.video = [mock_stream]

    fake_frames = []
    for i in range(20):
        fake_frame = MagicMock()
        fake_image = Image.new("RGB", (10, 10))
        fake_frame.to_image.return_value = fake_image
        fake_frames.append(fake_frame)

    # Return fake_frames for both loop iterations (finding length, decoding)
    mock_container.decode.side_effect = [fake_frames, fake_frames]

    result = _process_video(b"fake video data", max_frames=16)

    assert len(result) == 16
    assert isinstance(result[0], Image.Image)


@pytest.mark.anyio
async def test_download_media_invalid_base64():
    client = httpx.AsyncClient()
    source = "data:audio/wav;base64,invalid!!!base64"

    with pytest.raises(HTTPException) as excinfo:
        await _download_media(source, client)
    assert excinfo.value.status_code == 400
    assert "Failed to decode base64 media" in excinfo.value.detail


@pytest.mark.anyio
async def test_download_media_redirect_missing_location():
    client = httpx.AsyncClient()

    with patch("app.media_utils.resolve_safe_url_async", return_value=(True, "1.2.3.4")):
        mock_resp = MagicMock()
        mock_resp.is_redirect = True
        mock_resp.headers = {}  # missing Location

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__.return_value = mock_resp
        mock_stream_ctx.__aexit__.return_value = None

        with patch("httpx.AsyncClient.stream", return_value=mock_stream_ctx):
            with pytest.raises(HTTPException) as excinfo:
                await _download_media("http://example.com/audio.wav", client)
            assert excinfo.value.status_code == 400
            assert "Redirect Location header missing" in excinfo.value.detail


@pytest.mark.anyio
async def test_download_media_max_redirects_exceeded():
    client = httpx.AsyncClient()

    with patch("app.media_utils.resolve_safe_url_async", return_value=(True, "1.2.3.4")):
        mock_resp = MagicMock()
        mock_resp.is_redirect = True
        mock_resp.headers = {"Location": "http://example.com/redirect"}
        mock_resp.url = httpx.URL("http://example.com/start")

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__.return_value = mock_resp
        mock_stream_ctx.__aexit__.return_value = None

        with patch("httpx.AsyncClient.stream", return_value=mock_stream_ctx):
            with pytest.raises(HTTPException) as excinfo:
                await _download_media("http://example.com/start", client)
            assert excinfo.value.status_code == 400
            assert "Maximum redirects exceeded" in excinfo.value.detail


def test_process_audio_decode_failure():
    with patch("torchaudio.load", side_effect=RuntimeError("Corrupt audio file")):
        with patch("soundfile.read", side_effect=RuntimeError("Cannot read soundfile")):
            with patch("av.open", side_effect=RuntimeError("PyAV failed")):
                with pytest.raises(HTTPException) as excinfo:
                    _process_audio(b"corrupt audio data")
                assert excinfo.value.status_code == 400
                assert "Failed to process audio" in excinfo.value.detail


def test_process_video_no_frames_extracted():
    with patch("av.open") as mock_open:
        mock_container = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_container

        mock_stream = MagicMock()
        mock_stream.frames = 0
        mock_container.streams.video = [mock_stream]
        # Return empty frame list when container.decode is called
        mock_container.decode.return_value = []

        with pytest.raises(HTTPException) as excinfo:
            _process_video(b"empty video data")
        assert excinfo.value.status_code == 400
        assert "Failed to process video" in excinfo.value.detail
