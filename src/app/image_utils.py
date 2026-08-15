import base64
import io
import socket
import ipaddress
from urllib.parse import urlparse
import anyio
import httpx
from PIL import Image

Image.MAX_IMAGE_PIXELS = 20_000_000  # Decompression bomb guard (20 megapixels)
MAX_FILE_SIZE = 15 * 1024 * 1024  # Max 15MB


async def is_safe_url_async(url: str) -> bool:
    """
    SSRF protection: Blocks access to private IP, loopback, and link-local addresses
    using non-blocking async DNS resolution to avoid blocking the event loop.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False

        # Perform DNS resolution in a worker thread to prevent blocking the asyncio loop
        addr_info = await anyio.to_thread.run_sync(
            socket.getaddrinfo, parsed.hostname, None
        )
        for family, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            ip_obj = ipaddress.ip_address(ip_str)
            if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
                return False
        return True
    except Exception:
        return False


async def load_image_from_source(source: str, client: httpx.AsyncClient) -> Image.Image:
    """
    Loads and converts an image from Base64 or HTTP(S) URL into PIL Image (RGB format).
    Enforces stream chunk byte size checks to prevent OOM / DoS.
    """
    if source.startswith("data:image"):
        try:
            _, b64_data = source.split(",", 1)
            decoded = base64.b64decode(b64_data)
            if len(decoded) > MAX_FILE_SIZE:
                raise ValueError("画像サイズが上限(15MB)を超えています。")
            image = Image.open(io.BytesIO(decoded))
            image.load()
            return image.convert("RGB")
        except Exception as e:
            raise ValueError(f"Base64画像のデコードに失敗しました: {str(e)}")

    # Stream download with safe redirect validation to prevent SSRF redirect bypass
    current_url = source
    max_redirects = 3
    for _ in range(max_redirects + 1):
        if not await is_safe_url_async(current_url):
            raise ValueError(f"セキュリティ上の理由で拒否されたURLです: {current_url}")

        async with client.stream(
            "GET", current_url, timeout=10.0, follow_redirects=False
        ) as resp:
            if resp.is_redirect:
                location = resp.headers.get("Location")
                if not location:
                    raise ValueError(
                        "リダイレクト先Locationヘッダーが指定されていません。"
                    )
                current_url = str(resp.url.join(location))
                continue

            resp.raise_for_status()
            buffer = bytearray()
            async for chunk in resp.aiter_bytes():
                buffer.extend(chunk)
                if len(buffer) > MAX_FILE_SIZE:
                    raise ValueError("画像サイズが上限(15MB)を超えています。")

            image = Image.open(io.BytesIO(buffer))
            image.load()
            return image.convert("RGB")

    raise ValueError("リダイレクト回数が上限を超えました。")
