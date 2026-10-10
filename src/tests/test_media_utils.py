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
