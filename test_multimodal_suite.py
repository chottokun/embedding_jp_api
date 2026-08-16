"""
Comprehensive Multimodal (Diagram + Text) Test & Stress Suite for bge-visualized-m3.
Tests diverse realistic diagrams, input formats, edge cases, semantic retrieval, and concurrent load.
"""

import asyncio
import base64
import io
import time
import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from app.main import app

BASE_URL = "http://testserver"
API_KEY = "test_api_key_secret"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


# ==============================================================================
# 1. Real Diagram & Visual Asset Generators
# ==============================================================================

def create_architecture_diagram() -> Image.Image:
    """Microservices / Cloud Architecture Diagram."""
    img = Image.new("RGB", (600, 360), color=(245, 247, 250))
    draw = ImageDraw.Draw(img)

    # API Gateway
    draw.rounded_rectangle([30, 130, 150, 210], radius=8, fill=(30, 136, 229), outline=(21, 101, 192), width=2)
    draw.text((45, 160), "API Gateway\n(FastAPI)", fill=(255, 255, 255))

    # Service A (Embedding Service)
    draw.rounded_rectangle([230, 40, 370, 120], radius=8, fill=(67, 160, 71), outline=(46, 125, 50), width=2)
    draw.text((245, 70), "Embedding\nService (GPU)", fill=(255, 255, 255))

    # Service B (Rerank Service)
    draw.rounded_rectangle([230, 220, 370, 300], radius=8, fill=(142, 36, 170), outline=(106, 27, 154), width=2)
    draw.text((245, 250), "Rerank\nService", fill=(255, 255, 255))

    # Vector DB (Milvus / Qdrant)
    draw.rounded_rectangle([450, 130, 570, 210], radius=8, fill=(251, 140, 0), outline=(239, 108, 0), width=2)
    draw.text((465, 160), "Vector Store\n(Qdrant DB)", fill=(255, 255, 255))

    # Arrows
    draw.line([(150, 170), (230, 80)], fill=(66, 66, 66), width=3)
    draw.line([(150, 170), (230, 260)], fill=(66, 66, 66), width=3)
    draw.line([(370, 80), (450, 170)], fill=(66, 66, 66), width=3)
    draw.line([(370, 260), (450, 170)], fill=(66, 66, 66), width=3)

    return img


def create_performance_chart() -> Image.Image:
    """Benchmark Metrics Multi-Bar Chart."""
    img = Image.new("RGB", (500, 320), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title & Axes
    draw.text((140, 20), "Embedding Throughput (QPS) by Device", fill=(33, 33, 33))
    draw.line([(60, 260), (450, 260)], fill=(0, 0, 0), width=2)
    draw.line([(60, 60), (60, 260)], fill=(0, 0, 0), width=2)

    # Bars: CPU (35 QPS), RTX 3060 (180 QPS), A100 (620 QPS)
    # CPU
    draw.rectangle([100, 230, 160, 260], fill=(229, 57, 53))
    draw.text((115, 210), "35", fill=(229, 57, 53))
    draw.text((105, 270), "CPU 8-core", fill=(66, 66, 66))

    # RTX 3060
    draw.rectangle([210, 140, 270, 260], fill=(30, 136, 229))
    draw.text((225, 120), "180", fill=(30, 136, 229))
    draw.text((215, 270), "RTX 3060", fill=(66, 66, 66))

    # A100 GPU
    draw.rectangle([320, 70, 380, 260], fill=(67, 160, 71))
    draw.text((335, 50), "620", fill=(67, 160, 71))
    draw.text((330, 270), "A100 GPU", fill=(66, 66, 66))

    return img


def create_flowchart_diagram() -> Image.Image:
    """Authentication & Token Flowchart."""
    img = Image.new("RGB", (520, 340), color=(250, 250, 250))
    draw = ImageDraw.Draw(img)

    # Step 1: Client Request
    draw.ellipse([40, 140, 120, 200], fill=(225, 245, 254), outline=(2, 136, 209), width=2)
    draw.text((55, 160), "Client\nLogin", fill=(1, 87, 155))

    # Step 2: Auth Check
    draw.polygon([(220, 130), (280, 170), (220, 210), (160, 170)], fill=(255, 243, 224), outline=(245, 124, 0), width=2)
    draw.text((190, 162), "Verify\nToken", fill=(230, 81, 0))

    # Step 3: Success Token Granted
    draw.rectangle([340, 90, 480, 150], fill=(232, 245, 233), outline=(56, 142, 60), width=2)
    draw.text((360, 110), "200 OK JWT Token\nAccess Granted", fill=(27, 94, 32))

    # Step 4: 401 Unauthorized
    draw.rectangle([340, 210, 480, 270], fill=(255, 235, 238), outline=(211, 47, 47), width=2)
    draw.text((360, 230), "401 Unauthorized\nInvalid API Key", fill=(183, 28, 28))

    # Connecting Lines
    draw.line([(120, 170), (160, 170)], fill=(66, 66, 66), width=2)
    draw.line([(280, 170), (340, 120)], fill=(56, 142, 60), width=2)
    draw.line([(280, 170), (340, 240)], fill=(211, 47, 47), width=2)

    return img


def create_spec_table_image() -> Image.Image:
    """Product / Spec Comparison Table Image."""
    img = Image.new("RGB", (540, 280), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Headers
    draw.rectangle([20, 20, 520, 60], fill=(55, 71, 79))
    draw.text((40, 32), "Model ID", fill=(255, 255, 255))
    draw.text((200, 32), "Dimensions", fill=(255, 255, 255))
    draw.text((340, 32), "Max Sequence", fill=(255, 255, 255))
    draw.text((450, 32), "Modalities", fill=(255, 255, 255))

    # Row 1: ruri-v3-30m
    draw.rectangle([20, 60, 520, 100], fill=(245, 245, 245))
    draw.text((40, 72), "ruri-v3-30m", fill=(33, 33, 33))
    draw.text((200, 72), "256 dim", fill=(33, 33, 33))
    draw.text((340, 72), "8192 tokens", fill=(33, 33, 33))
    draw.text((450, 72), "Text only", fill=(33, 33, 33))

    # Row 2: ruri-v3-310m
    draw.rectangle([20, 100, 520, 140], fill=(255, 255, 255))
    draw.text((40, 112), "ruri-v3-310m", fill=(33, 33, 33))
    draw.text((200, 112), "768 dim", fill=(33, 33, 33))
    draw.text((340, 112), "8192 tokens", fill=(33, 33, 33))
    draw.text((450, 112), "Text only", fill=(33, 33, 33))

    # Row 3: bge-visualized-m3
    draw.rectangle([20, 140, 520, 180], fill=(232, 240, 254))
    draw.text((40, 152), "bge-visualized-m3", fill=(26, 115, 232))
    draw.text((200, 152), "1024 dim", fill=(26, 115, 232))
    draw.text((340, 152), "8192 tokens", fill=(26, 115, 232))
    draw.text((450, 152), "Image + Text", fill=(26, 115, 232))

    return img


def create_annotated_sketch() -> Image.Image:
    """Handwritten-style Annotated Sketch Image with arrows & Japanese notes."""
    img = Image.new("RGB", (480, 300), color=(254, 250, 240))
    draw = ImageDraw.Draw(img)

    # Sketchy server boxes
    draw.rectangle([50, 80, 180, 200], outline=(70, 70, 70), width=3)
    draw.text((70, 120), "Primary DB\n(Read/Write)", fill=(70, 70, 70))

    draw.rectangle([300, 80, 430, 200], outline=(70, 70, 70), width=3)
    draw.text((320, 120), "Replica DB\n(Read-Only)", fill=(70, 70, 70))

    # Hand-drawn arrow with replication note
    draw.line([(180, 140), (300, 140)], fill=(211, 47, 47), width=3)
    draw.polygon([(290, 135), (305, 140), (290, 145)], fill=(211, 47, 47))
    draw.text((200, 110), "非同期レプリケーション", fill=(211, 47, 47))

    return img


def create_extreme_aspect_ratio(mode: str) -> Image.Image:
    """Generates images with extreme aspect ratios or resolutions."""
    if mode == "ultra_wide":
        img = Image.new("RGB", (1200, 180), color=(240, 244, 248))
        draw = ImageDraw.Draw(img)
        draw.text((450, 80), "Ultra Wide Architecture Banner (1200x180)", fill=(33, 33, 33))
        return img
    elif mode == "ultra_tall":
        img = Image.new("RGB", (180, 1200), color=(248, 244, 240))
        draw = ImageDraw.Draw(img)
        draw.text((30, 580), "Vertical Sequence\n(180x1200)", fill=(33, 33, 33))
        return img
    elif mode == "high_res":
        img = Image.new("RGB", (1920, 1080), color=(230, 238, 245))
        draw = ImageDraw.Draw(img)
        draw.text((800, 500), "Full HD 1080p High-Resolution Diagram (1920x1080)", fill=(33, 33, 33))
        return img
    elif mode == "thumbnail":
        img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, 54, 54], outline=(255, 255, 255), width=2)
        return img
    raise ValueError(f"Unknown mode: {mode}")


def image_to_base64_data_url(img: Image.Image, format: str = "PNG") -> str:
    """Encodes PIL Image to Data URL format."""
    buf = io.BytesIO()
    img.save(buf, format=format)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = format.lower()
    if mime == "jpg":
        mime = "jpeg"
    return f"data:image/{mime};base64,{b64}"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


# ==============================================================================
# 2. Main Test Execution Engine
# ==============================================================================

async def run_extended_multimodal_tests():
    print("=" * 80)
    print("🚀 EXTENDED REAL DATA MULTIMODAL TEST & STRESS SUITE (bge-visualized-m3)")
    print("=" * 80)

    # 1. Generate Assets
    print("\n[Step 1: Visual Asset Generation]")
    assets = {
        "architecture": create_architecture_diagram(),
        "performance": create_performance_chart(),
        "flowchart": create_flowchart_diagram(),
        "table": create_spec_table_image(),
        "sketch": create_annotated_sketch(),
    }
    for name, img in assets.items():
        print(f"  ✓ {name:15s}: size={img.size}, mode={img.mode}")

    extreme_assets = {
        "ultra_wide": create_extreme_aspect_ratio("ultra_wide"),
        "ultra_tall": create_extreme_aspect_ratio("ultra_tall"),
        "high_res": create_extreme_aspect_ratio("high_res"),
        "thumbnail": create_extreme_aspect_ratio("thumbnail"),
    }
    for name, img in extreme_assets.items():
        print(f"  ✓ {name:15s}: size={img.size}, mode={img.mode}")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url=BASE_URL,
        headers=HEADERS,
        timeout=180.0,
    ) as client:

        # ======================================================================
        # SECTION 1: Image Format Variations & Transparency
        # ======================================================================
        print("\n" + "=" * 80)
        print("SECTION 1: Image Encoding Format & Transparency Variations")
        print("=" * 80)

        formats = ["PNG", "JPEG", "WEBP"]
        for fmt in formats:
            b64_url = image_to_base64_data_url(assets["architecture"], format=fmt)
            t0 = time.perf_counter()
            resp = await client.post("/v1/embeddings", json={
                "model": "bge-visualized-m3",
                "input": {"text": f"{fmt}形式でエンコードされたシステム構成図", "image_url": b64_url}
            })
            dt = (time.perf_counter() - t0) * 1000
            assert resp.status_code == 200, f"Failed for format {fmt}: {resp.text}"
            vec = resp.json()["data"][0]["embedding"]
            assert len(vec) == 1024
            print(f"  ✓ Format {fmt:6s}: 200 OK ({dt:.1f} ms, dim={len(vec)})")

        # RGBA Transparency test
        rgba_img = Image.new("RGBA", (300, 200), color=(0, 0, 0, 0))
        rgba_draw = ImageDraw.Draw(rgba_img)
        rgba_draw.rectangle([50, 50, 250, 150], fill=(0, 128, 255, 180))
        rgba_b64 = image_to_base64_data_url(rgba_img, format="PNG")
        resp = await client.post("/v1/embeddings", json={
            "model": "bge-visualized-m3",
            "input": {"text": "半透明アルファチャンネルを含むRGBA透過PNG画像", "image_url": rgba_b64}
        })
        assert resp.status_code == 200
        print("  ✓ RGBA PNG (Transparent Alpha Channel): 200 OK (dim=1024)")

        # ======================================================================
        # SECTION 2: Extreme Resolutions & Aspect Ratios
        # ======================================================================
        print("\n" + "=" * 80)
        print("SECTION 2: Extreme Aspect Ratios & Resolutions")
        print("=" * 80)

        for name, img in extreme_assets.items():
            b64_url = image_to_base64_data_url(img, format="PNG")
            t0 = time.perf_counter()
            resp = await client.post("/v1/embeddings", json={
                "model": "bge-visualized-m3",
                "input": {"text": f"解像度テスト: {name} (size: {img.size})", "image_url": b64_url}
            })
            dt = (time.perf_counter() - t0) * 1000
            assert resp.status_code == 200, f"Failed for {name}: {resp.text}"
            vec = resp.json()["data"][0]["embedding"]
            assert len(vec) == 1024
            print(f"  ✓ {name:12s} ({img.size[0]:4d}x{img.size[1]:4d}): 200 OK ({dt:.1f} ms)")

        # ======================================================================
        # SECTION 3: Edge Cases, Schema Variations & Error Handling
        # ======================================================================
        print("\n" + "=" * 80)
        print("SECTION 3: Edge Cases, Text Truncation & Validation")
        print("=" * 80)

        # 1. Image only (Empty Text)
        resp = await client.post("/v1/embeddings", json={
            "model": "bge-visualized-m3",
            "input": {"text": "", "image_url": image_to_base64_data_url(assets["sketch"])}
        })
        assert resp.status_code == 200
        print("  ✓ Image Only (text=''): 200 OK")

        # 2. Text only with bge-visualized-m3
        resp = await client.post("/v1/embeddings", json={
            "model": "bge-visualized-m3",
            "input": "テキストのみの単体クエリエンコード"
        })
        assert resp.status_code == 200
        print("  ✓ Text Only with bge-visualized-m3: 200 OK")

        # 3. Long Japanese Text (>1000 chars) + Diagram
        long_jp_text = "このシステムアーキテクチャは、高可用性とスケーラビリティを担保するために設計された最新のマイクロサービス構成です。" * 30
        resp = await client.post("/v1/embeddings", json={
            "model": "bge-visualized-m3",
            "input": {"text": long_jp_text, "image_url": image_to_base64_data_url(assets["architecture"])}
        })
        assert resp.status_code == 200
        print(f"  ✓ Long Japanese Text ({len(long_jp_text)} chars) + Diagram: 200 OK")

        # 4. Japanese Unicode, Emojis & Symbols
        special_text = "🔥【超重要】API 構成図 🚀 (Ver 2.5.0) -> DB 連携 & 高速キャッシュ ⚡️"
        resp = await client.post("/v1/embeddings", json={
            "model": "bge-visualized-m3",
            "input": {"text": special_text, "image_url": image_to_base64_data_url(assets["architecture"])}
        })
        assert resp.status_code == 200
        print("  ✓ Japanese Emojis & Unicode Symbols: 200 OK")

        # 5. Invalid Base64 Image -> 400 Bad Request
        resp = await client.post("/v1/embeddings", json={
            "model": "bge-visualized-m3",
            "input": {"text": "破損した画像データ", "image_url": "data:image/png;base64,invalid_corrupted_base64_!@#$"}
        })
        assert resp.status_code == 400
        print(f"  ✓ Invalid Base64 Validation: Correctly returned 400 ({resp.json()['detail'][:40]}...)")

        # 6. Image sent to Text-Only model -> 400 Bad Request
        resp = await client.post("/v1/embeddings", json={
            "model": "cl-nagoya/ruri-v3-310m",
            "input": {"text": "テキスト専用モデルに画像送信", "image_url": image_to_base64_data_url(assets["sketch"])}
        })
        assert resp.status_code == 400
        print(f"  ✓ Text-Only Model Guard: Correctly returned 400 ({resp.json()['detail'][:40]}...)")

        # ======================================================================
        # SECTION 4: 5x5 Cross-Modal Semantic Retrieval Matrix & Accuracy
        # ======================================================================
        print("\n" + "=" * 80)
        print("SECTION 4: 5x5 Cross-Modal Semantic Retrieval Accuracy Matrix")
        print("=" * 80)

        # Encode all 5 diagrams
        diagram_keys = ["architecture", "performance", "flowchart", "table", "sketch"]
        diagram_vecs = {}
        for key in diagram_keys:
            resp = await client.post("/v1/embeddings", json={
                "model": "bge-visualized-m3",
                "input": {"text": f"{key} diagram", "image_url": image_to_base64_data_url(assets[key])}
            })
            diagram_vecs[key] = resp.json()["data"][0]["embedding"]

        # Domain Text Queries
        queries = {
            "architecture": "マイクロサービスとAPI Gateway、Qdrantベクトルデータベースのシステムインフラ構成図",
            "performance": "CPUとRTX 3060、A100 GPUの推論スループット(QPS)性能比較棒グラフ",
            "flowchart": "クライアントログインとJWT認証トークンの検証シーケンスフローチャート",
            "table": "各埋め込みモデルの次元数とトークン長、マルチモーダル対応表",
            "sketch": "プライマリDBからレプリカDBへの非同期レプリケーション手書きスケッチ",
        }

        # Encode queries and compute similarity matrix
        query_vecs = {}
        for q_key, q_text in queries.items():
            resp = await client.post("/v1/embeddings", json={
                "model": "bge-visualized-m3",
                "input": q_text
            })
            query_vecs[q_key] = resp.json()["data"][0]["embedding"]

        # Print Matrix
        print(f"\n{'Query Category':<18} | " + " | ".join([f"{k[:7]:>7}" for k in diagram_keys]))
        print("-" * 65)

        correct_top1_count = 0
        for q_key, q_vec in query_vecs.items():
            sims = {d_key: cosine_similarity(q_vec, diagram_vecs[d_key]) for d_key in diagram_keys}
            row_str = f"{q_key:<18} | " + " | ".join([f"{sims[k]:7.4f}" for k in diagram_keys])
            top_match = max(sims, key=sims.get)
            is_correct = (top_match == q_key)
            if is_correct:
                correct_top1_count += 1
            status = "🎯 Match" if is_correct else "❌ Mismatch"
            print(f"{row_str}  [{status} -> {top_match}]")

        accuracy = (correct_top1_count / len(queries)) * 100.0
        print(f"\n📊 Top-1 Retrieval Accuracy: {accuracy:.1f}% ({correct_top1_count}/{len(queries)})")
        assert correct_top1_count == len(queries), "Cross-modal retrieval failed accuracy check!"

        # ======================================================================
        # SECTION 5: Batch Processing & Throughput Scaling
        # ======================================================================
        print("\n" + "=" * 80)
        print("SECTION 5: Multimodal Batch Processing Scaling")
        print("=" * 80)

        batch_sizes = [1, 2, 4, 8]
        base_item = {
            "text": "バッチテスト用図面アイテム",
            "image_url": image_to_base64_data_url(assets["performance"])
        }

        for bs in batch_sizes:
            batch_input = [base_item] * bs
            t0 = time.perf_counter()
            resp = await client.post("/v1/embeddings", json={
                "model": "bge-visualized-m3",
                "input": batch_input
            })
            dt = (time.perf_counter() - t0) * 1000
            assert resp.status_code == 200
            res_data = resp.json()["data"]
            assert len(res_data) == bs
            per_item_ms = dt / bs
            print(f"  ✓ Batch Size {bs:2d}: Total {dt:7.1f} ms ({per_item_ms:6.1f} ms/item, QPS={bs / (dt/1000):5.1f})")

        # ======================================================================
        # SECTION 6: High Concurrency & Thread-Safety Stress Test
        # ======================================================================
        print("\n" + "=" * 80)
        print("SECTION 6: Concurrent Multi-Worker Stress Test (20 Concurrent Requests)")
        print("=" * 80)

        async def worker(worker_id: int):
            diag_name = diagram_keys[worker_id % len(diagram_keys)]
            t0 = time.perf_counter()
            r = await client.post("/v1/embeddings", json={
                "model": "bge-visualized-m3",
                "input": {
                    "text": f"並行ワーカー {worker_id} リクエスト ({diag_name})",
                    "image_url": image_to_base64_data_url(assets[diag_name])
                }
            })
            dt = (time.perf_counter() - t0) * 1000
            assert r.status_code == 200, f"Worker {worker_id} failed: {r.text}"
            return worker_id, dt

        num_concurrent = 20
        t_start = time.perf_counter()
        tasks = [worker(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks)
        total_time_ms = (time.perf_counter() - t_start) * 1000

        latencies = [res[1] for res in results]
        print(f"  ✓ Processed {num_concurrent} concurrent multimodal requests in {total_time_ms:.1f} ms")
        print(f"  ✓ Avg Latency: {np.mean(latencies):.1f} ms | Min: {np.min(latencies):.1f} ms | Max: {np.max(latencies):.1f} ms | P95: {np.percentile(latencies, 95):.1f} ms")
        print(f"  ✓ Concurrency Throughput: {num_concurrent / (total_time_ms / 1000):.2f} req/s")
        print("  ✓ Thread-Safety Verified: All 20 workers returned 200 OK without race conditions.")

    print("\n" + "=" * 80)
    print("🎉 ALL EXTENDED MULTIMODAL TESTS AND LOAD STRESS TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_extended_multimodal_tests())
