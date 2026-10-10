import base64
import io
import logging
from typing import TYPE_CHECKING, List, Tuple

import anyio
import httpx
from fastapi import HTTPException
from PIL import Image

from .config import MAX_AUDIO_DURATION_SEC
from .image_utils import (
    DOWNLOAD_TIMEOUT,
    MAX_FILE_SIZE,
    MAX_REDIRECTS,
    SafeNetworkBackend,
    resolve_safe_url_async,
)

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


async def _download_media(source: str, client: httpx.AsyncClient) -> bytes:
    """
    Downloads media bytes from a base64 string or an HTTP(S) URL.
    Applies SSRF protection and limits the downloaded file size.

    Args:
        source (str): The media source (base64 string or URL).
        client (httpx.AsyncClient): The HTTP client to use for downloading.

    Returns:
        bytes: The downloaded media bytes.

    Raises:
        HTTPException: If the URL is unsafe, download fails, or file size exceeds limits.
    """
    if source.startswith("data:audio") or source.startswith("data:video"):
        try:
            _, b64_data = source.split(",", 1)
            decoded = base64.b64decode(b64_data)
            if len(decoded) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400, detail="Media size exceeds limit (15MB)."
                )
            return decoded
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to decode base64 media: {str(e)}"
            )

    current_url = source
    for _ in range(MAX_REDIRECTS + 1):
        is_safe, safe_ip = await resolve_safe_url_async(current_url)
        if not is_safe or not safe_ip:
            raise HTTPException(
                status_code=400,
                detail=f"URL rejected for security reasons: {current_url}",
            )

        if isinstance(client, httpx.AsyncClient):
            safe_transport = httpx.AsyncHTTPTransport(retries=0)
            original_backend = safe_transport._pool._network_backend
            safe_transport._pool._network_backend = SafeNetworkBackend(
                original_backend, safe_ip
            )
            temp_kwargs = {}
            if hasattr(client, "auth"):
                temp_kwargs["auth"] = client.auth
            if hasattr(client, "headers"):
                temp_kwargs["headers"] = client.headers
            if hasattr(client, "cookies"):
                temp_kwargs["cookies"] = client.cookies
            if hasattr(client, "timeout"):
                temp_kwargs["timeout"] = client.timeout
            if hasattr(client, "max_redirects"):
                temp_kwargs["max_redirects"] = client.max_redirects
            if hasattr(client, "trust_env"):
                temp_kwargs["trust_env"] = client.trust_env
            if hasattr(client, "default_encoding"):
                temp_kwargs["default_encoding"] = client.default_encoding

            client_ctx = httpx.AsyncClient(transport=safe_transport, **temp_kwargs)
        else:
            from contextlib import nullcontext

            client_ctx = nullcontext(client)

        async with client_ctx as safe_client:
            try:
                async with safe_client.stream(
                    "GET", current_url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=False
                ) as resp:
                    if resp.is_redirect:
                        location = resp.headers.get("Location")
                        if not location:
                            raise HTTPException(
                                status_code=400,
                                detail="Redirect Location header missing.",
                            )
                        current_url = str(resp.url.join(location))
                        continue

                    resp.raise_for_status()
                    buffer = bytearray()
                    async for chunk in resp.aiter_bytes():
                        buffer.extend(chunk)
                        if len(buffer) > MAX_FILE_SIZE:
                            raise HTTPException(
                                status_code=400,
                                detail="Media size exceeds limit (15MB).",
                            )

                    return bytes(buffer)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Failed to download media: {str(e)}"
                )

    raise HTTPException(status_code=400, detail="Maximum redirects exceeded.")


def _process_audio(data: bytes) -> Tuple["torch.Tensor", int]:
    """
    Decodes audio bytes, resamples to 16kHz mono, and truncates if it exceeds max duration.

    Args:
        data (bytes): The raw audio bytes.

    Returns:
        tuple[torch.Tensor, int]: The processed audio waveform tensor (1, T) and the sample rate (16000).

    Raises:
        HTTPException: If audio decoding fails or the format is invalid.
    """
    try:
        import torch
        import torchaudio
    except ImportError:
        raise HTTPException(
            status_code=500, detail="Audio processing libraries are not installed."
        )

    try:
        # Try torchaudio.load first (respecting mocks in unit tests)
        try:
            waveform, sample_rate = torchaudio.load(io.BytesIO(data))
        except (ImportError, RuntimeError, Exception):
            # Fallback 1: soundfile
            try:
                import soundfile as sf

                audio_arr, sample_rate = sf.read(io.BytesIO(data), dtype="float32")
                tensor = torch.from_numpy(audio_arr)
                if tensor.ndim == 1:
                    waveform = tensor.unsqueeze(0)
                else:
                    waveform = tensor.transpose(0, 1)
            except Exception:
                # Fallback 2: PyAV
                import av

                with av.open(io.BytesIO(data)) as container:
                    if not container.streams.audio:
                        raise ValueError("No audio streams found in container.")
                    stream = container.streams.audio[0]
                    sample_rate = stream.rate or 16000
                    chunks = []
                    for frame in container.decode(stream):
                        arr = frame.to_ndarray()
                        chunks.append(torch.from_numpy(arr))
                    if not chunks:
                        raise ValueError("No audio frames decoded.")
                    raw_tensor = torch.cat(chunks, dim=-1).to(torch.float32)
                    if raw_tensor.ndim == 1:
                        waveform = raw_tensor.unsqueeze(0)
                    else:
                        waveform = raw_tensor
                    # Normalize float if integer PCM
                    if waveform.abs().max() > 1.0:
                        waveform = waveform / 32768.0

        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to 16kHz
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_sample_rate
            )
            waveform = resampler(waveform)
            sample_rate = target_sample_rate

        # Truncate to MAX_AUDIO_DURATION_SEC
        max_samples = sample_rate * MAX_AUDIO_DURATION_SEC
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

        return waveform, sample_rate
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to process audio: {str(e)}"
        )


def _process_video(data: bytes, max_frames: int = 16) -> List[Image.Image]:
    """
    Decodes video bytes and extracts a representative number of frames (up to max_frames).

    Args:
        data (bytes): The raw video bytes.
        max_frames (int): The maximum number of frames to extract.

    Returns:
        list[PIL.Image.Image]: A list of extracted frames as PIL Images.

    Raises:
        HTTPException: If video decoding fails or the format is invalid.
    """
    try:
        import av
    except ImportError:
        raise HTTPException(
            status_code=500, detail="Video processing libraries are not installed."
        )

    try:
        with av.open(io.BytesIO(data)) as container:
            stream = container.streams.video[0]

            total_frames = stream.frames
            if not total_frames or total_frames <= 0:
                # If stream doesn't report frames accurately, count them
                total_frames = 0
                for _ in container.decode(stream):
                    total_frames += 1
                container.seek(0)

            if total_frames == 0:
                raise ValueError("No video frames could be extracted.")

            # Compute target indices to decode and save
            if total_frames > max_frames:
                target_indices = {
                    int(i * total_frames / max_frames) for i in range(max_frames)
                }
            else:
                target_indices = set(range(total_frames))

            frames = []
            for i, frame in enumerate(container.decode(stream)):
                if i in target_indices:
                    frames.append(frame.to_image())
                if len(frames) >= len(target_indices):
                    break

            return frames
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to process video: {str(e)}"
        )


async def load_audio_from_source(
    source: str, client: httpx.AsyncClient
) -> Tuple["torch.Tensor", int]:
    """
    Loads audio from a Base64 string or an HTTP(S) URL.
    Safely downloads the media, decodes it, converts to 16kHz mono, and truncates if necessary.

    Args:
        source (str): The audio source (Base64 string or URL).
        client (httpx.AsyncClient): The HTTP client for downloading.

    Returns:
        tuple[torch.Tensor, int]: The processed audio waveform tensor and the sample rate (16000).

    Raises:
        HTTPException: For invalid URLs, download failures, or decoding errors.
    """
    try:
        data = await _download_media(source, client)
        return await anyio.to_thread.run_sync(_process_audio, data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio loading failed: {str(e)}")


async def load_video_frames_from_source(
    source: str, client: httpx.AsyncClient, max_frames: int = 16
) -> List[Image.Image]:
    """
    Loads a video from a Base64 string or an HTTP(S) URL and extracts representative frames.
    Safely downloads the media and uniformly samples up to max_frames frames.

    Args:
        source (str): The video source (Base64 string or URL).
        client (httpx.AsyncClient): The HTTP client for downloading.
        max_frames (int): The maximum number of frames to extract (default: 16).

    Returns:
        list[PIL.Image.Image]: A list of extracted frames as PIL Images.

    Raises:
        HTTPException: For invalid URLs, download failures, or decoding errors.
    """
    try:
        data = await _download_media(source, client)

        def process_wrapper():
            return _process_video(data, max_frames)

        return await anyio.to_thread.run_sync(process_wrapper)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Video loading failed: {str(e)}")
