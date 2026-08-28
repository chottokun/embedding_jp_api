"""
Integration tests for Real Multimodal (Diagram + Text) Embeddings with bge-visualized-m3.
Tests various diagram types, formats, edge cases, and schema structures.
"""

import base64
import io
import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from app.main import app

pytestmark = pytest.mark.integration

client = TestClient(app)
AUTH_HEADERS = {"Authorization": "Bearer test_api_key_secret"}


def _create_sample_diagram(title: str = "Sample Architecture") -> str:
    """Helper to generate a small sample PNG diagram Base64 Data URL."""
    img = Image.new("RGB", (200, 120), color=(240, 245, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [20, 20, 180, 100], fill=(33, 150, 243), outline=(25, 118, 210), width=2
    )
    draw.text((35, 50), title, fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def test_multimodal_real_flat_format():
    """Test Flat schema with real diagram + Japanese description."""
    img_b64 = _create_sample_diagram("API Gateway")
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "bge-visualized-m3",
            "input": {
                "text": "API Gatewayとマイクロサービスのシステム構成図",
                "image_url": img_b64,
            },
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "bge-visualized-m3"
    assert len(data["data"]) == 1
    assert len(data["data"][0]["embedding"]) == 1024


def test_multimodal_real_content_part_format():
    """Test OpenAI ContentPart array schema with real diagram + text."""
    img_b64 = _create_sample_diagram("Vector Store")
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "bge-visualized-m3",
            "input": [
                {"type": "text", "text": "ベクトルデータベースのクラスタ構成図"},
                {"type": "image_url", "image_url": {"url": img_b64}},
            ],
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 1
    assert len(data["data"][0]["embedding"]) == 1024


def test_multimodal_real_image_only_empty_text():
    """Test image-only input with empty or None text."""
    img_b64 = _create_sample_diagram("Diagram Only")
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "bge-visualized-m3",
            "input": {
                "text": "",
                "image_url": img_b64,
            },
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"][0]["embedding"]) == 1024


def test_multimodal_real_batch_input():
    """Test batch multimodal input (multiple diagram+text pairs)."""
    img1 = _create_sample_diagram("Diagram A")
    img2 = _create_sample_diagram("Diagram B")
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "bge-visualized-m3",
            "input": [
                {"text": "構成図A", "image_url": img1},
                {"text": "構成図B", "image_url": img2},
            ],
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 2
    assert len(data["data"][0]["embedding"]) == 1024
    assert len(data["data"][1]["embedding"]) == 1024


def test_multimodal_real_rgba_transparency():
    """Test RGBA transparent PNG diagram handling."""
    img = Image.new("RGBA", (150, 100), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 130, 80], fill=(255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    img_b64 = f"data:image/png;base64,{b64}"

    response = client.post(
        "/v1/embeddings",
        json={
            "model": "bge-visualized-m3",
            "input": {"text": "透過PNG図面", "image_url": img_b64},
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert len(response.json()["data"][0]["embedding"]) == 1024


def test_multimodal_real_invalid_base64_returns_400():
    """Test that corrupted/invalid Base64 returns 400 Bad Request."""
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "bge-visualized-m3",
            "input": {
                "text": "破損画像",
                "image_url": "data:image/png;base64,not_valid_b64!!",
            },
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 400
    assert (
        "Base64" in response.json()["detail"] or "デコード" in response.json()["detail"]
    )


def test_multimodal_real_text_only_model_rejection():
    """Test that sending an image to a text-only model returns 400 Bad Request."""
    img_b64 = _create_sample_diagram("Diagram")
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "cl-nagoya/ruri-v3-30m",
            "input": {"text": "テスト", "image_url": img_b64},
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 400
    assert "画像入力をサポートしていません" in response.json()["detail"]
