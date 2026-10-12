import base64
import io
import socket
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from PIL import Image, UnidentifiedImageError
import pytest

from app.image_utils import (
    MAX_FILE_SIZE,
    MAX_IMAGE_PIXELS,
    MAX_REDIRECTS,
    SafeNetworkBackend,
    _decode_and_convert_image,
    is_safe_url_async,
    load_image_from_source,
    resolve_safe_url_async,
)


def generate_sample_image_bytes(
    format="PNG", size=(50, 50), color=(255, 0, 0)
) -> bytes:
    """Helper to generate valid image bytes."""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


async def _async_iter(items):
    """Async generator helper for mocking aiter_bytes."""
    for item in items:
        yield item


# ============================================================================
# 1. Tests for resolve_safe_url_async and is_safe_url_async
# ============================================================================


@pytest.mark.anyio
async def test_resolve_safe_url_async_valid_public_ip():
    """Test public URL resolution returns safe=True and resolved IP."""
    addr_info = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]
    with patch("socket.getaddrinfo", return_value=addr_info):
        is_safe, ip = await resolve_safe_url_async("https://example.com/image.png")
        assert is_safe is True
        assert ip == "93.184.216.34"

        safe_bool = await is_safe_url_async("https://example.com/image.png")
        assert safe_bool is True


@pytest.mark.anyio
@pytest.mark.parametrize(
    "private_ip",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "::1",
    ],
)
async def test_resolve_safe_url_async_private_ips(private_ip):
    """Test private, loopback, and link-local IP addresses are blocked."""
    family = socket.AF_INET6 if ":" in private_ip else socket.AF_INET
    addr_info = [(family, socket.SOCK_STREAM, 6, "", (private_ip, 0))]
    with patch("socket.getaddrinfo", return_value=addr_info):
        is_safe, ip = await resolve_safe_url_async("http://internal-host/image.png")
        assert is_safe is False
        assert ip is None

        safe_bool = await is_safe_url_async("http://internal-host/image.png")
        assert safe_bool is False


@pytest.mark.anyio
@pytest.mark.parametrize(
    "invalid_url",
    [
        "ftp://example.com/image.png",
        "file:///etc/passwd",
        "http://",
        "not_a_url",
        "",
    ],
)
async def test_resolve_safe_url_async_invalid_urls(invalid_url):
    """Test invalid schemes, empty hosts, and malformed URLs are blocked."""
    is_safe, ip = await resolve_safe_url_async(invalid_url)
    assert is_safe is False
    assert ip is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "exception_cls",
    [
        socket.gaierror("DNS resolution failed"),
        ValueError("Invalid address format"),
        TypeError("Invalid type"),
    ],
)
async def test_resolve_safe_url_async_exceptions(exception_cls):
    """Test socket / resolution exceptions are safely caught and return (False, None)."""
    with patch("socket.getaddrinfo", side_effect=exception_cls):
        is_safe, ip = await resolve_safe_url_async("https://invalid-domain.example")
        assert is_safe is False
        assert ip is None


# ============================================================================
# 2. Tests for SafeNetworkBackend and _decode_and_convert_image
# ============================================================================


@pytest.mark.anyio
async def test_safe_network_backend():
    """Test SafeNetworkBackend redirects connect_tcp calls to the designated safe IP."""
    mock_backend = AsyncMock()
    safe_backend = SafeNetworkBackend(mock_backend, safe_ip="93.184.216.34")

    # connect_tcp
    await safe_backend.connect_tcp("example.com", 443, timeout=5.0)
    mock_backend.connect_tcp.assert_called_once_with(
        "93.184.216.34",
        443,
        timeout=5.0,
        local_address=None,
        socket_options=None,
    )

    # connect_unix_socket
    await safe_backend.connect_unix_socket("/tmp/socket.sock", timeout=2.0)
    mock_backend.connect_unix_socket.assert_called_once_with(
        "/tmp/socket.sock", timeout=2.0, socket_options=None
    )

    # sleep
    await safe_backend.sleep(0.5)
    mock_backend.sleep.assert_called_once_with(0.5)


def test_decode_and_convert_image_valid():
    """Test _decode_and_convert_image decodes valid image bytes to RGB PIL Image."""
    png_bytes = generate_sample_image_bytes(
        format="PNG", size=(20, 20), color=(0, 255, 0)
    )
    image = _decode_and_convert_image(png_bytes)
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == (20, 20)


def test_decode_and_convert_image_corrupted():
    """Test _decode_and_convert_image raises exception on corrupted bytes."""
    corrupted_bytes = b"not an image"
    with pytest.raises((UnidentifiedImageError, OSError)):
        _decode_and_convert_image(corrupted_bytes)


# ============================================================================
# 3. Tests for load_image_from_source (Base64)
# ============================================================================


@pytest.mark.anyio
async def test_load_image_from_source_base64_success():
    """Test loading valid Base64 image data URL."""
    png_bytes = generate_sample_image_bytes(format="PNG", size=(30, 30))
    b64_str = base64.b64encode(png_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{b64_str}"

    async with httpx.AsyncClient() as client:
        img = await load_image_from_source(data_url, client)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (30, 30)


@pytest.mark.anyio
async def test_load_image_from_source_base64_exceeds_size():
    """Test Base64 image payload exceeding MAX_FILE_SIZE raises ValueError."""
    large_data = b"X" * (MAX_FILE_SIZE + 100)
    b64_str = base64.b64encode(large_data).decode("utf-8")
    data_url = f"data:image/png;base64,{b64_str}"

    async with httpx.AsyncClient() as client:
        with pytest.raises(
            ValueError, match="画像サイズが上限\\(15MB\\)を超えています"
        ):
            await load_image_from_source(data_url, client)


@pytest.mark.anyio
async def test_load_image_from_source_base64_invalid():
    """Test invalid Base64 string raises ValueError."""
    data_url = "data:image/png;base64,invalid_base64_content!!!"

    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="Base64画像のデコードに失敗しました"):
            await load_image_from_source(data_url, client)


# ============================================================================
# 4. Tests for load_image_from_source (URL Download & Redirects)
# ============================================================================


@pytest.mark.anyio
async def test_load_image_from_source_url_ssrf_blocked():
    """Test URL resolving to private IP is rejected by SSRF check."""
    with patch("app.image_utils.resolve_safe_url_async", return_value=(False, None)):
        async with httpx.AsyncClient() as client:
            with pytest.raises(
                ValueError, match="セキュリティ上の理由で拒否されたURLです"
            ):
                await load_image_from_source("http://127.0.0.1/image.png", client)


@pytest.mark.anyio
async def test_load_image_from_source_url_success_mock_client():
    """Test URL stream download happy path with custom/mocked client."""
    img_bytes = generate_sample_image_bytes(format="JPEG", size=(40, 40))

    mock_resp = MagicMock()
    mock_resp.is_redirect = False
    mock_resp.raise_for_status = MagicMock()
    mock_resp.aiter_bytes = MagicMock(return_value=_async_iter([img_bytes]))

    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__.return_value = mock_resp

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_stream_ctx

    with patch(
        "app.image_utils.resolve_safe_url_async",
        return_value=(True, "93.184.216.34"),
    ):
        img = await load_image_from_source("https://example.com/image.jpg", mock_client)
        assert isinstance(img, Image.Image)
        assert img.size == (40, 40)


@pytest.mark.anyio
async def test_load_image_from_source_url_success_real_async_client():
    """Test URL stream download happy path when passing standard httpx.AsyncClient."""
    img_bytes = generate_sample_image_bytes(format="PNG", size=(25, 25))

    mock_resp = MagicMock()
    mock_resp.is_redirect = False
    mock_resp.raise_for_status = MagicMock()
    mock_resp.aiter_bytes = MagicMock(
        return_value=_async_iter([img_bytes[:100], img_bytes[100:]])
    )

    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__.return_value = mock_resp

    with patch(
        "app.image_utils.resolve_safe_url_async",
        return_value=(True, "93.184.216.34"),
    ):
        with patch("httpx.AsyncClient.stream", return_value=mock_stream_ctx):
            real_client = httpx.AsyncClient()
            img = await load_image_from_source(
                "https://example.com/test.png", real_client
            )
            assert isinstance(img, Image.Image)
            assert img.size == (25, 25)


@pytest.mark.anyio
async def test_load_image_from_source_url_exceeds_max_file_size():
    """Test streamed response exceeding MAX_FILE_SIZE raises ValueError."""
    chunk1 = b"A" * 10
    chunk2 = b"B" * 10

    mock_resp = MagicMock()
    mock_resp.is_redirect = False
    mock_resp.raise_for_status = MagicMock()
    mock_resp.aiter_bytes = MagicMock(return_value=_async_iter([chunk1, chunk2]))

    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__.return_value = mock_resp

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_stream_ctx

    with (
        patch(
            "app.image_utils.resolve_safe_url_async",
            return_value=(True, "93.184.216.34"),
        ),
        patch("app.image_utils.MAX_FILE_SIZE", 15),
    ):
        with pytest.raises(
            ValueError, match="画像サイズが上限\\(15MB\\)を超えています"
        ):
            await load_image_from_source("https://example.com/large.png", mock_client)


@pytest.mark.anyio
async def test_load_image_from_source_url_redirect_success():
    """Test following a valid safe redirect location."""
    img_bytes = generate_sample_image_bytes(format="PNG", size=(15, 15))

    # First response: 302 redirect
    mock_redirect_resp = MagicMock()
    mock_redirect_resp.is_redirect = True
    mock_redirect_resp.headers = {"Location": "https://example.com/redirected.png"}
    mock_redirect_resp.url = httpx.URL("https://example.com/initial.png")

    mock_redirect_ctx = AsyncMock()
    mock_redirect_ctx.__aenter__.return_value = mock_redirect_resp

    # Second response: 200 OK
    mock_ok_resp = MagicMock()
    mock_ok_resp.is_redirect = False
    mock_ok_resp.raise_for_status = MagicMock()
    mock_ok_resp.aiter_bytes = MagicMock(return_value=_async_iter([img_bytes]))

    mock_ok_ctx = AsyncMock()
    mock_ok_ctx.__aenter__.return_value = mock_ok_resp

    mock_client = MagicMock()
    mock_client.stream.side_effect = [mock_redirect_ctx, mock_ok_ctx]

    with patch(
        "app.image_utils.resolve_safe_url_async",
        return_value=(True, "93.184.216.34"),
    ):
        img = await load_image_from_source(
            "https://example.com/initial.png", mock_client
        )
        assert isinstance(img, Image.Image)
        assert img.size == (15, 15)
        assert mock_client.stream.call_count == 2


@pytest.mark.anyio
async def test_load_image_from_source_url_redirect_missing_location():
    """Test redirect response without Location header raises ValueError."""
    mock_redirect_resp = MagicMock()
    mock_redirect_resp.is_redirect = True
    mock_redirect_resp.headers = {}

    mock_redirect_ctx = AsyncMock()
    mock_redirect_ctx.__aenter__.return_value = mock_redirect_resp

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_redirect_ctx

    with patch(
        "app.image_utils.resolve_safe_url_async",
        return_value=(True, "93.184.216.34"),
    ):
        with pytest.raises(
            ValueError, match="リダイレクト先Locationヘッダーが指定されていません"
        ):
            await load_image_from_source(
                "https://example.com/bad-redirect.png", mock_client
            )


@pytest.mark.anyio
async def test_load_image_from_source_url_max_redirects_exceeded():
    """Test exceeding MAX_REDIRECTS raises ValueError."""
    mock_redirect_resp = MagicMock()
    mock_redirect_resp.is_redirect = True
    mock_redirect_resp.headers = {"Location": "https://example.com/loop.png"}
    mock_redirect_resp.url = httpx.URL("https://example.com/loop.png")

    mock_redirect_ctx = AsyncMock()
    mock_redirect_ctx.__aenter__.return_value = mock_redirect_resp

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_redirect_ctx

    with patch(
        "app.image_utils.resolve_safe_url_async",
        return_value=(True, "93.184.216.34"),
    ):
        with pytest.raises(ValueError, match="リダイレクト回数が上限を超えました"):
            await load_image_from_source("https://example.com/loop.png", mock_client)

    # MAX_REDIRECTS + 1 = 4 calls total
    assert mock_client.stream.call_count == MAX_REDIRECTS + 1


def test_max_image_pixels_config():
    """Verify decompression bomb guard setting is set to 20 megapixels."""
    assert MAX_IMAGE_PIXELS == 20_000_000
    assert Image.MAX_IMAGE_PIXELS == 20_000_000
