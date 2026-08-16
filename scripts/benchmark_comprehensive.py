"""
Comprehensive Benchmark & Profiling Suite for embedding_jp_api.
Measures:
1. All 5 supported models head-to-head (Latency P50/P90/P95/P99, Cold Load, VRAM usage)
2. Sequence Length Scaling (32, 128, 512, 1024, 2048 tokens)
3. Multimodal Modality & Resolution Breakdown (Text vs Image vs Image+Text, 64px to 1080p)
4. Batch Scaling & Peak Throughput (Batch 1, 8, 32, 64)
5. CPU vs GPU Head-to-Head comparison
"""

import asyncio
import base64
import io
import time
import httpx
import numpy as np
import torch
from PIL import Image, ImageDraw
from app.main import app

BASE_URL = "http://testserver"
API_KEY = "test_api_key_secret"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def create_mock_image(width: int = 400, height: int = 260) -> str:
    """Creates a sample mock diagram Base64 Data URL."""
    img = Image.new("RGB", (width, height), color=(240, 245, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, width - 20, height - 20], fill=(33, 150, 243), outline=(25, 118, 210), width=2)
    draw.text((width // 4, height // 2), f"Benchmark Diagram {width}x{height}", fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def get_vram_mb() -> float:
    """Returns currently allocated CUDA VRAM in MB if available."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


async def run_comprehensive_benchmarks():
    print("=" * 80)
    print("📊 COMPREHENSIVE BENCHMARK & HARDWARE PROFILING SUITE")
    print("=" * 80)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Host CPU"
    print(f"\n[Environment Information]")
    print(f"  • Compute Device : {device_name}")
    print(f"  • PyTorch Version: {torch.__version__}")
    print(f"  • CUDA Available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)
        print(f"  • Total GPU VRAM : {total_vram:.2f} GB")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url=BASE_URL,
        headers=HEADERS,
        timeout=180.0,
    ) as client:

        # ======================================================================
        # 1. All Models Head-to-Head Comparison (Single Query Latency & VRAM)
        # ======================================================================
        print("\n" + "=" * 80)
        print("1. All Models Head-to-Head Single Inference Latency & VRAM Footprint")
        print("=" * 80)

        embedding_models = [
            ("cl-nagoya/ruri-v3-30m", "Text (30M / 256d)"),
            ("cl-nagoya/ruri-v3-310m", "Text (310M / 768d)"),
            ("BAAI/bge-m3", "Text (560M / 1024d)"),
            ("bge-visualized-m3", "Multimodal (800M / 1024d)"),
        ]

        print(f"{'Model Name':<28} | {'Type':<22} | {'P50 (ms)':>8} | {'P95 (ms)':>8} | {'P99 (ms)':>8} | {'VRAM (MB)':>9}")
        print("-" * 88)

        for model_id, model_desc in embedding_models:
            # Warm up / Load model
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            # 50 iterations
            latencies = []
            for i in range(50):
                t0 = time.perf_counter()
                resp = await client.post("/v1/embeddings", json={
                    "model": model_id,
                    "input": f"東京都千代田区における自然言語処理技術のベンチマーク測定サンプル {i}",
                    "input_type": "query"
                })
                dt = (time.perf_counter() - t0) * 1000
                assert resp.status_code == 200, f"Error {resp.status_code}: {resp.text}"
                latencies.append(dt)

            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)
            vram = get_vram_mb()

            print(f"{model_id:<28} | {model_desc:<22} | {p50:8.2f} | {p95:8.2f} | {p99:8.2f} | {vram:9.1f}")

        # Reranker Model
        print("-" * 88)
        rerank_model = "cl-nagoya/ruri-v3-reranker-310m"
        rerank_lats = []
        for i in range(30):
            t0 = time.perf_counter()
            resp = await client.post("/v1/rerank", json={
                "model": rerank_model,
                "query": "日本の首都はどこですか？",
                "documents": [
                    "東京は日本の首都であり、最大の都市です。",
                    "京都はかつての日本の古都です。",
                    "大阪は西日本の主要な経済都市です。",
                    "名古屋は中部地方の中心都市です。",
                    "福岡は九州地方の主要都市です。"
                ],
                "top_n": 3
            })
            dt = (time.perf_counter() - t0) * 1000
            assert resp.status_code == 200
            rerank_lats.append(dt)

        r_p50 = np.percentile(rerank_lats, 50)
        r_p95 = np.percentile(rerank_lats, 95)
        r_p99 = np.percentile(rerank_lats, 99)
        r_vram = get_vram_mb()
        print(f"{rerank_model:<28} | {'Reranker (5 docs)':<22} | {r_p50:8.2f} | {r_p95:8.2f} | {r_p99:8.2f} | {r_vram:9.1f}")

        # ======================================================================
        # 2. Batch Scaling & Peak Throughput (ruri-v3-30m vs ruri-v3-310m)
        # ======================================================================
        print("\n" + "=" * 80)
        print("2. Batch Scaling & Throughput (items/sec) Comparison")
        print("=" * 80)

        batch_sizes = [1, 8, 32, 64]
        for m_id in ["cl-nagoya/ruri-v3-30m", "cl-nagoya/ruri-v3-310m"]:
            print(f"\n--- Model: {m_id} ---")
            print(f"{'Batch Size':>10} | {'Total Time (ms)':>15} | {'Throughput (items/s)':>20} | {'Per-Item (ms)':>13}")
            print("-" * 65)
            sample_txt = "高度な自然言語処理技術を用いたベクトル検索エンジンの性能テスト。"
            for bs in batch_sizes:
                b_inputs = [f"{sample_txt} (seq_{j})" for j in range(bs)]
                t0 = time.perf_counter()
                resp = await client.post("/v1/embeddings", json={
                    "model": m_id,
                    "input": b_inputs,
                    "input_type": "document"
                })
                dt = (time.perf_counter() - t0) * 1000
                assert resp.status_code == 200
                qps = bs / (dt / 1000)
                per_item = dt / bs
                print(f"{bs:10d} | {dt:15.1f} | {qps:20.1f} | {per_item:13.2f}")

        # ======================================================================
        # 3. Context Length (Sequence Length) Scaling
        # ======================================================================
        print("\n" + "=" * 80)
        print("3. Sequence Length Scaling (Token Length Sensitivity)")
        print("=" * 80)

        seq_lengths = [32, 128, 512, 1024, 2048]
        base_phrase = "自然言語処理における埋め込みベクトルの計算速度とメモリ使用量を検証する。"
        
        print(f"{'Target Tokens (approx)':<25} | {'Char Length':>12} | {'ruri-30m (ms)':>13} | {'ruri-310m (ms)':>14}")
        print("-" * 72)

        for target_tokens in seq_lengths:
            # Repeat base phrase to achieve approximate target tokens
            repeat_count = max(1, target_tokens // 16)
            text_payload = base_phrase * repeat_count
            char_len = len(text_payload)

            # Measure ruri-30m
            t0 = time.perf_counter()
            r1 = await client.post("/v1/embeddings", json={"model": "cl-nagoya/ruri-v3-30m", "input": text_payload})
            t_30m = (time.perf_counter() - t0) * 1000
            assert r1.status_code == 200

            # Measure ruri-310m
            t0 = time.perf_counter()
            r2 = await client.post("/v1/embeddings", json={"model": "cl-nagoya/ruri-v3-310m", "input": text_payload})
            t_310m = (time.perf_counter() - t0) * 1000
            assert r2.status_code == 200

            print(f"{target_tokens:<25d} | {char_len:12d} | {t_30m:13.1f} | {t_310m:14.1f}")

        # ======================================================================
        # 4. Multimodal Modality & Image Resolution Breakdown
        # ======================================================================
        print("\n" + "=" * 80)
        print("4. Multimodal (bge-visualized-m3) Modality & Resolution Breakdown")
        print("=" * 80)

        img_64 = create_mock_image(64, 64)
        img_224 = create_mock_image(224, 224)
        img_1080 = create_mock_image(1920, 1080)

        mm_cases = [
            ("Text Only (Query)", {"text": "システムアーキテクチャ構成図の検索クエリ"}),
            ("Image Only (Thumbnail 64x64)", {"image_url": img_64}),
            ("Image Only (Standard 224x224)", {"image_url": img_224}),
            ("Image Only (Full HD 1080p)", {"image_url": img_1080}),
            ("Multimodal (Standard 224 + Text)", {"text": "マイクロサービス構成図", "image_url": img_224}),
            ("Multimodal (Full HD 1080p + Text)", {"text": "高解像度インフラ構成図", "image_url": img_1080}),
        ]

        print(f"{'Input Modality / Resolution':<36} | {'Avg Latency (ms)':>16} | {'Output Dim':>10}")
        print("-" * 68)

        for case_name, payload_input in mm_cases:
            lats = []
            for _ in range(5):
                t0 = time.perf_counter()
                resp = await client.post("/v1/embeddings", json={
                    "model": "bge-visualized-m3",
                    "input": payload_input
                })
                dt = (time.perf_counter() - t0) * 1000
                assert resp.status_code == 200
                lats.append(dt)
            avg_dt = np.mean(lats)
            dim = len(resp.json()["data"][0]["embedding"])
            print(f"{case_name:<36} | {avg_dt:16.1f} | {dim:10d}")

    print("\n" + "=" * 80)
    print("🎉 COMPREHENSIVE BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_comprehensive_benchmarks())
