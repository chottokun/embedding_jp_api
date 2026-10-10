"""Real binary multimodal E2E verification test.

Generates real WAV audio and real MP4 video data and sends them to the API server
with ENABLE_AUDIO_EMBEDDING=true and ENABLE_VIDEO_EMBEDDING=true enabled.
Verifies actual end-to-end vector generation, dimensions, and L2 normalization.
"""

import base64
import io
import math
import os
import struct
import subprocess
import sys
import time
import wave
import av
import httpx
from PIL import Image

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8009
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


def create_real_wav_base64(duration_sec: float = 1.0, sample_rate: int = 16000) -> str:
    """Generate a real 16kHz mono PCM WAV file in Base64 format."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        num_samples = int(duration_sec * sample_rate)
        # Generate 440Hz sine wave samples
        raw_samples = bytearray()
        for i in range(num_samples):
            val = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
            raw_samples.extend(struct.pack("<h", val))
        wav_file.writeframes(raw_samples)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:audio/wav;base64,{b64}"


def create_real_mp4_base64(
    num_frames: int = 4, width: int = 64, height: int = 64
) -> str:
    """Generate a real MP4 video file in Base64 format using PyAV."""
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    stream = container.add_stream("h264", rate=2)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    for i in range(num_frames):
        color = ((i * 60) % 255, (100 + i * 40) % 255, 200)
        img = Image.new("RGB", (width, height), color=color)
        frame = av.VideoFrame.from_image(img)
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:video/mp4;base64,{b64}"


def create_real_png_base64() -> str:
    """Generate a real PNG image in Base64 format."""
    img = Image.new("RGB", (64, 64), color=(255, 128, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def wait_for_server(timeout_sec: float = 30.0) -> bool:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout_sec:
        try:
            r = httpx.get(f"{BASE_URL}/readyz", timeout=1.0)
            if r.status_code == 200 and r.json().get("status") == "ready":
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def run_real_multimodal_tests():
    print("=" * 70)
    print("🚀 Starting Real Data Multimodal E2E Verification")
    print("=" * 70)

    audio_b64 = create_real_wav_base64(duration_sec=0.5)
    video_b64 = create_real_mp4_base64(num_frames=4)
    image_b64 = create_real_png_base64()

    print("Generated Real Media Samples:")
    print(f"  • Real Audio (WAV 16kHz PCM): {len(audio_b64)} chars base64")
    print(f"  • Real Video (MP4 H264):     {len(video_b64)} chars base64")
    print(f"  • Real Image (PNG RGB):      {len(image_b64)} chars base64")

    # 1. Real Audio + Text Embedding
    print("\n[1/4] Testing Real Audio + Text Embedding (google/embeddinggemma-2)...")
    payload = {
        "model": "google/embeddinggemma-2",
        "input": {
            "text": "音声を聴いて内容を分類してください。",
            "input_audio": audio_b64,
        },
    }
    t0 = time.perf_counter()
    r = httpx.post(f"{BASE_URL}/v1/embeddings", json=payload, timeout=45.0)
    lat_audio = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200, f"Audio embedding failed ({r.status_code}): {r.text}"
    emb_data = r.json()["data"][0]["embedding"]
    dim = len(emb_data)
    norm = math.sqrt(sum(x * x for x in emb_data))
    assert dim == 768, f"Expected 768d, got {dim}"
    assert math.isclose(norm, 1.0, abs_tol=1e-2), (
        f"Expected normalized L2 norm ~1.0, got {norm}"
    )
    print(
        f"  ✓ Audio + Text -> 200 OK (Latency: {lat_audio:.1f}ms, Dimension: {dim}, L2 Norm: {norm:.4f})"
    )

    # 2. Real Video + Text Embedding
    print("\n[2/4] Testing Real Video + Text Embedding (google/embeddinggemma-2)...")
    payload = {
        "model": "google/embeddinggemma-2",
        "input": {
            "text": "動画の内容を検索します。",
            "video_url": video_b64,
        },
    }
    t0 = time.perf_counter()
    r = httpx.post(f"{BASE_URL}/v1/embeddings", json=payload, timeout=45.0)
    lat_video = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200, f"Video embedding failed ({r.status_code}): {r.text}"
    emb_data = r.json()["data"][0]["embedding"]
    dim = len(emb_data)
    norm = math.sqrt(sum(x * x for x in emb_data))
    assert dim == 768, f"Expected 768d, got {dim}"
    assert math.isclose(norm, 1.0, abs_tol=1e-2), (
        f"Expected normalized L2 norm ~1.0, got {norm}"
    )
    print(
        f"  ✓ Video + Text -> 200 OK (Latency: {lat_video:.1f}ms, Dimension: {dim}, L2 Norm: {norm:.4f})"
    )

    # 3. Full Multimodal (Text + Image + Audio + Video)
    print("\n[3/4] Testing Full Multimodal (Text + Image + Audio + Video)...")
    payload = {
        "model": "google/embeddinggemma-2",
        "input": [
            {"type": "text", "text": "マルチメディア総合検索"},
            {"type": "image_url", "image_url": image_b64},
            {"type": "input_audio", "input_audio": audio_b64},
            {"type": "video_url", "video_url": video_b64},
        ],
    }
    t0 = time.perf_counter()
    r = httpx.post(f"{BASE_URL}/v1/embeddings", json=payload, timeout=45.0)
    lat_full = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200, f"Full multimodal failed ({r.status_code}): {r.text}"
    emb_data = r.json()["data"][0]["embedding"]
    dim = len(emb_data)
    norm = math.sqrt(sum(x * x for x in emb_data))
    assert dim == 768, f"Expected 768d, got {dim}"
    print(
        f"  ✓ Full Multimodal -> 200 OK (Latency: {lat_full:.1f}ms, Dimension: {dim}, L2 Norm: {norm:.4f})"
    )

    # 4. Matryoshka Representation Learning (MRL: 256d) with Real Media
    print("\n[4/4] Testing MRL Dimensionality Reduction (256d) on Real Audio...")
    payload = {
        "model": "google/embeddinggemma-2",
        "input": {"text": "MRL圧縮テスト", "input_audio": audio_b64},
        "dimensions": 256,
        "encoding_format": "float",
    }
    t0 = time.perf_counter()
    r = httpx.post(f"{BASE_URL}/v1/embeddings", json=payload, timeout=45.0)
    lat_mrl = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200, f"MRL failed ({r.status_code}): {r.text}"
    emb_data = r.json()["data"][0]["embedding"]
    dim = len(emb_data)
    norm = math.sqrt(sum(x * x for x in emb_data))
    assert dim == 256, f"Expected 256d, got {dim}"
    assert math.isclose(norm, 1.0, abs_tol=1e-2), (
        f"Expected normalized L2 norm ~1.0, got {norm}"
    )
    print(
        f"  ✓ MRL 256d Audio -> 200 OK (Latency: {lat_mrl:.1f}ms, Dimension: {dim}, L2 Norm: {norm:.4f})"
    )

    print("\n" + "=" * 70)
    print("🎉 All Real Data Multimodal E2E Tests PASSED Perfectly!")
    print("=" * 70)


def main():
    env = os.environ.copy()
    env["APP_PORT"] = str(SERVER_PORT)
    env["PYTHONPATH"] = "src"
    env["ENABLE_AUDIO_EMBEDDING"] = "true"
    env["ENABLE_VIDEO_EMBEDDING"] = "true"

    cmd = [
        "uv",
        "run",
        "uvicorn",
        "app.main:app",
        "--host",
        SERVER_HOST,
        "--port",
        str(SERVER_PORT),
        "--log-level",
        "warning",
    ]
    print(f"[INFO] Launching server with Audio & Video ENABLED: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env)

    try:
        if not wait_for_server(timeout_sec=30):
            print("[ERROR] Server failed to start.")
            sys.exit(1)
        run_real_multimodal_tests()
    finally:
        print("[INFO] Shutting down server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("[INFO] Server stopped gracefully.")


if __name__ == "__main__":
    main()
