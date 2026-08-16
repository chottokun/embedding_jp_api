import asyncio
import time
import httpx
import statistics

BASE_URL = "http://127.0.0.1:8000"
API_KEY = "test_api_key_secret"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

async def benchmark_endpoint():
    print("=" * 80)
    print("📊 API LATENCY & THROUGHPUT BENCHMARK SUITE")
    print("=" * 80)

    async with httpx.AsyncClient(base_url=BASE_URL, headers=HEADERS, timeout=60.0) as client:
        # 1. Warm-up
        print("\n[Step 1] Warming up model endpoints...")
        await client.post("/v1/embeddings", json={"model": "cl-nagoya/ruri-v3-30m", "input": "ウォームアップ"})
        print("  ✓ Warm-up complete")

        # 2. Single Query Latency Benchmark (100 sequential requests)
        print("\n[Step 2] Measuring Single Text Embedding Latency (100 sequential requests)...")
        latencies = []
        for i in range(100):
            t0 = time.perf_counter()
            resp = await client.post("/v1/embeddings", json={
                "model": "cl-nagoya/ruri-v3-30m",
                "input": f"日本語クエリのベンチマーク測定テスト {i}",
                "input_type": "query"
            })
            assert resp.status_code == 200
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)

        p50 = statistics.median(latencies)
        p90 = sorted(latencies)[int(len(latencies) * 0.90)]
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        avg = statistics.mean(latencies)
        stdev = statistics.stdev(latencies)

        print(f"  • Sample Count : {len(latencies)}")
        print(f"  • Avg Latency  : {avg:.2f} ms (±{stdev:.2f} ms)")
        print(f"  • P50 (Median) : {p50:.2f} ms")
        print(f"  • P90 Latency  : {p90:.2f} ms")
        print(f"  • P95 Latency  : {p95:.2f} ms")
        print(f"  • P99 Latency  : {p99:.2f} ms")
        print(f"  • Min / Max    : {min(latencies):.2f} ms / {max(latencies):.2f} ms")

        # 3. Batch Scaling Benchmark (Batch size = 1, 8, 32, 64)
        print("\n[Step 3] Measuring Batch Size Scaling Performance...")
        sample_sentence = "東京都千代田区千代田1番1号に位置する日本の名所。"
        for batch_size in [1, 8, 32, 64]:
            batch_input = [f"{sample_sentence} (idx={j})" for j in range(batch_size)]
            t0 = time.perf_counter()
            resp = await client.post("/v1/embeddings", json={
                "model": "cl-nagoya/ruri-v3-30m",
                "input": batch_input
            })
            elapsed = time.perf_counter() - t0
            assert resp.status_code == 200
            data = resp.json()
            total_tokens = data["usage"]["total_tokens"]
            print(f"  • Batch Size {batch_size:2d}: {elapsed*1000:6.1f} ms | Throughput: {batch_size/elapsed:6.1f} items/sec | Token Rate: {total_tokens/elapsed:7.1f} tok/sec")

        # 4. Reranking Benchmark
        print("\n[Step 4] Measuring Reranker Latency (cl-nagoya/ruri-v3-reranker-310m)...")
        docs = [
            "東京は日本の首都であり、最大の都市です。",
            "京都は日本の古都として親しまれています。",
            "富士山は日本で最も高い山です。",
            "北海道は日本最北端の島で、広大な自然を有します。",
            "沖縄は日本南西部の島々で、美しいサンゴ礁が広がります。"
        ]
        rerank_lats = []
        for i in range(20):
            t0 = time.perf_counter()
            resp = await client.post("/v1/rerank", json={
                "model": "cl-nagoya/ruri-v3-reranker-310m",
                "query": "日本の首都はどこですか？",
                "documents": docs
            })
            assert resp.status_code == 200
            rerank_lats.append((time.perf_counter() - t0) * 1000)

        print(f"  • Rerank (5 docs) Avg Latency: {statistics.mean(rerank_lats):.2f} ms (P50={statistics.median(rerank_lats):.2f} ms, P95={sorted(rerank_lats)[int(len(rerank_lats)*0.95)]:.2f} ms)")

    print("\n" + "=" * 80)
    print("✨ BENCHMARK SUITE COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(benchmark_endpoint())
