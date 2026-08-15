import asyncio
import base64
import io
import time
import httpx
import numpy as np
from PIL import Image, ImageDraw

BASE_URL = "http://127.0.0.1:8000"
API_KEY = "test_api_key_secret"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

def create_diagram_image(diag_type: str) -> str:
    """Create a real PIL diagram image and return as Base64 Data URL."""
    img = Image.new("RGB", (400, 260), color=(248, 249, 250))
    draw = ImageDraw.Draw(img)

    if diag_type == "system_architecture":
        # Draw System Architecture Diagram (API Gateway -> Load Balancer -> DB)
        draw.rectangle([30, 40, 150, 100], fill=(66, 133, 244), outline=(26, 115, 232))
        draw.text((45, 65), "API Gateway", fill=(255, 255, 255))

        draw.rectangle([230, 40, 360, 100], fill=(52, 168, 83), outline=(30, 142, 62))
        draw.text((245, 65), "Vector Store", fill=(255, 255, 255))

        draw.rectangle([130, 160, 270, 220], fill=(251, 188, 4), outline=(242, 153, 0))
        draw.text((150, 185), "Worker Nodes", fill=(32, 33, 36))

        draw.line([(150, 70), (230, 70)], fill=(32, 33, 36), width=3)
        draw.line([(90, 100), (160, 160)], fill=(32, 33, 36), width=3)
        draw.line([(295, 100), (240, 160)], fill=(32, 33, 36), width=3)

    elif diag_type == "bar_chart":
        # Draw Performance Bar Chart
        draw.rectangle([50, 180, 100, 220], fill=(234, 67, 53))  # CPU
        draw.text((55, 225), "CPU", fill=(0, 0, 0))

        draw.rectangle([150, 80, 200, 220], fill=(66, 133, 244))  # GPU
        draw.text((155, 225), "GPU", fill=(0, 0, 0))

        draw.rectangle([250, 40, 300, 220], fill=(52, 168, 83))  # TPU
        draw.text((255, 225), "TPU", fill=(0, 0, 0))

        draw.line([(30, 220), (350, 220)], fill=(0, 0, 0), width=2)
        draw.line([(30, 20), (30, 220)], fill=(0, 0, 0), width=2)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    a = np.array(v1)
    b = np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

async def run_multimodal_real_data_tests():
    print("=" * 80)
    print("🖼️ REAL DATA MULTIMODAL (DIAGRAM + TEXT) EMBEDDING VERIFICATION")
    print("=" * 80)

    arch_diagram_b64 = create_diagram_image("system_architecture")
    chart_diagram_b64 = create_diagram_image("bar_chart")

    print("\n[Data Preparation]")
    print(f"  ✓ Created System Architecture Diagram (Base64 length: {len(arch_diagram_b64)})")
    print(f"  ✓ Created Performance Bar Chart (Base64 length: {len(chart_diagram_b64)})")

    async with httpx.AsyncClient(base_url=BASE_URL, headers=HEADERS, timeout=60.0) as client:
        # Test 1: Flat Format (Architecture Diagram + Japanese Description)
        print("\n[Test 1] Flat Schema: System Architecture Diagram + Japanese Description")
        t0 = time.perf_counter()
        resp1 = await client.post("/v1/embeddings", json={
            "model": "bge-visualized-m3",
            "input": {
                "text": "APIゲートウェイとベクトルDBを連携させたマイクロサービスシステム構成図",
                "image_url": arch_diagram_b64
            }
        })
        assert resp1.status_code == 200, f"Failed: {resp1.status_code} - {resp1.text}"
        data1 = resp1.json()
        vec1 = data1["data"][0]["embedding"]
        lat1 = (time.perf_counter() - t0) * 1000
        print(f"  ✓ Status 200 OK ({lat1:.1f} ms)")
        print(f"  ✓ Output Dimension: {len(vec1)} (Expected: 1024)")
        print(f"  ✓ First 5 elements: {[round(x, 4) for x in vec1[:5]]}")

        # Test 2: OpenAI ContentPart Array Format (Bar Chart + Japanese Description)
        print("\n[Test 2] OpenAI ContentPart Format: Performance Bar Chart + Japanese Description")
        t0 = time.perf_counter()
        resp2 = await client.post("/v1/embeddings", json={
            "model": "bge-visualized-m3",
            "input": [
                {"type": "text", "text": "CPU、GPU、TPUの処理スループットを比較した棒グラフ"},
                {"type": "image_url", "image_url": chart_diagram_b64}
            ]
        })
        assert resp2.status_code == 200, f"Failed: {resp2.status_code} - {resp2.text}"
        data2 = resp2.json()
        vec2 = data2["data"][0]["embedding"]
        lat2 = (time.perf_counter() - t0) * 1000
        print(f"  ✓ Status 200 OK ({lat2:.1f} ms)")
        print(f"  ✓ Output Dimension: {len(vec2)} (Expected: 1024)")
        print(f"  ✓ First 5 elements: {[round(x, 4) for x in vec2[:5]]}")

        # Test 3: Text-Only Query for Matching
        print("\n[Test 3] Semantic Cross-Modal Retrieval Relevance Test")
        # Query A: "システムアーキテクチャ・設計図" (Should match Architecture Diagram)
        resp_qA = await client.post("/v1/embeddings", json={
            "model": "bge-visualized-m3",
            "input": "クラウドインフラのシステム構成とマイクロサービス設計図"
        })
        vec_qA = resp_qA.json()["data"][0]["embedding"]

        # Query B: "ベンチマーク性能比較グラフ" (Should match Performance Bar Chart)
        resp_qB = await client.post("/v1/embeddings", json={
            "model": "bge-visualized-m3",
            "input": "ハードウェア別スループット性能評価の棒グラフ"
        })
        vec_qB = resp_qB.json()["data"][0]["embedding"]

        sim_A_to_arch = cosine_similarity(vec_qA, vec1)
        sim_A_to_chart = cosine_similarity(vec_qA, vec2)
        sim_B_to_arch = cosine_similarity(vec_qB, vec1)
        sim_B_to_chart = cosine_similarity(vec_qB, vec2)

        print("\n[Retrieval Similarity Matrix]")
        print(f"  • Query A (システム構成)  -> Architecture Diagram: {sim_A_to_arch:.4f}  | Bar Chart: {sim_A_to_chart:.4f}")
        print(f"  • Query B (性能比較グラフ) -> Architecture Diagram: {sim_B_to_arch:.4f}  | Bar Chart: {sim_B_to_chart:.4f}")

        # Assert correct matching
        assert sim_A_to_arch > sim_A_to_chart, "Query A should match Architecture diagram higher than Chart!"
        assert sim_B_to_chart > sim_B_to_arch, "Query B should match Bar Chart higher than Architecture diagram!"
        print("  ✓ Correct semantic separation and cross-modal retrieval confirmed! 🎯")

        # Test 4: Batch Multimodal Encoding (Both Diagrams simultaneously in one request)
        print("\n[Test 4] Batch Multimodal Request (Multiple Diagram+Text items in one batch)")
        t0 = time.perf_counter()
        resp_batch = await client.post("/v1/embeddings", json={
            "model": "bge-visualized-m3",
            "input": [
                {
                    "text": "システム構成図",
                    "image_url": arch_diagram_b64
                },
                {
                    "text": "性能比較グラフ",
                    "image_url": chart_diagram_b64
                }
            ]
        })
        assert resp_batch.status_code == 200
        batch_data = resp_batch.json()
        assert len(batch_data["data"]) == 2
        lat_batch = (time.perf_counter() - t0) * 1000
        print(f"  ✓ Batch returned {len(batch_data['data'])} vectors in {lat_batch:.1f} ms")

    print("\n" + "=" * 80)
    print("🎉 REAL DATA MULTIMODAL VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(run_multimodal_real_data_tests())
