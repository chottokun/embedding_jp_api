import asyncio
import time
import random
import statistics
import httpx

BASE_URL = "http://127.0.0.1:8009"
API_KEY = "loadtest_key"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

TINY_PNG_B64 = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

TEXT_SAMPLES = [
    "自然言語処理モデルの並行推論ベンチマーク",
    "マイクロサービス間の通信レイテンシ最適化",
    "Dockerコンテナ環境での高負荷ストレステスト",
    "高並行リクエストにおけるデッドロック検出とスレッドセーフティ",
    "検索拡張生成（RAG）のための日本語リランカー検証",
]


async def send_request(client: httpx.AsyncClient, req_id: int):
    # Randomly pick endpoint type
    req_type = random.choice(
        ["emb_single", "emb_batch", "emb_multimodal", "rerank", "health"]
    )
    t0 = time.perf_counter()
    try:
        if req_type == "emb_single":
            resp = await client.post(
                "/v1/embeddings",
                json={
                    "model": "cl-nagoya/ruri-v3-30m",
                    "input": random.choice(TEXT_SAMPLES),
                    "input_type": "query",
                },
                headers=HEADERS,
            )
        elif req_type == "emb_batch":
            resp = await client.post(
                "/v1/embeddings",
                json={
                    "model": "cl-nagoya/ruri-v3-30m",
                    "input": [random.choice(TEXT_SAMPLES) for _ in range(3)],
                    "input_type": "document",
                },
                headers=HEADERS,
            )
        elif req_type == "emb_multimodal":
            resp = await client.post(
                "/v1/embeddings",
                json={
                    "model": "cl-nagoya/ruri-v3-30m",
                    "input": {
                        "image_url": TINY_PNG_B64
                    },  # Expected to return 400 safely without crashing server
                },
                headers=HEADERS,
            )
            # 400 is expected for text-only model receiving image
            if resp.status_code == 400:
                elapsed = time.perf_counter() - t0
                return True, elapsed, req_type, resp.status_code
        elif req_type == "rerank":
            resp = await client.post(
                "/v1/rerank",
                json={
                    "model": "cl-nagoya/ruri-v3-reranker-310m",
                    "query": "高負荷テストの検証",
                    "documents": TEXT_SAMPLES[:4],
                    "top_n": 2,
                },
                headers=HEADERS,
            )
        else:  # health
            resp = await client.get("/health")

        elapsed = time.perf_counter() - t0
        is_success = resp.status_code == 200
        return is_success, elapsed, req_type, resp.status_code
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return False, elapsed, req_type, str(e)


async def main():
    print("=" * 65)
    print("🚀 Running Intensive High-Concurrency Load & Stress Test (100 Concurrent)")
    print("=" * 65)

    # Warm-up call
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120.0) as client:
        print("Warming up models in container...")
        await client.get("/health")
        await client.post(
            "/v1/embeddings",
            json={"model": "cl-nagoya/ruri-v3-30m", "input": "ウォームアップ"},
            headers=HEADERS,
        )
        await client.post(
            "/v1/rerank",
            json={
                "model": "cl-nagoya/ruri-v3-reranker-310m",
                "query": "ウォームアップ",
                "documents": ["テスト1", "テスト2"],
            },
            headers=HEADERS,
        )
        print("Warmup complete.\n")

        # Stress Test: 100 simultaneous concurrent requests
        TOTAL_REQUESTS = 100
        print(f"Launching {TOTAL_REQUESTS} simultaneous concurrent requests...")
        t_start = time.perf_counter()
        tasks = [send_request(client, i) for i in range(TOTAL_REQUESTS)]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - t_start

    successes = [r for r in results if r[0]]
    failures = [r for r in results if not r[0]]
    latencies = [r[1] for r in results]

    avg_lat = statistics.mean(latencies)
    median_lat = statistics.median(latencies)
    p95_lat = sorted(latencies)[int(len(latencies) * 0.95)]
    p99_lat = sorted(latencies)[int(len(latencies) * 0.99)]
    rps = TOTAL_REQUESTS / total_time

    print("\n" + "=" * 65)
    print("📊 STRESS TEST RESULTS SUMMARY")
    print("=" * 65)
    print(f"• Total Requests Sent : {TOTAL_REQUESTS}")
    print(
        f"• Successful Requests : {len(successes)} ({len(successes) / TOTAL_REQUESTS * 100:.1f}%)"
    )
    print(
        f"• Failed / Crashed    : {len(failures)} ({len(failures) / TOTAL_REQUESTS * 100:.1f}%)"
    )
    print(f"• Total Test Duration : {total_time:.2f} seconds")
    print(f"• Throughput (RPS)    : {rps:.2f} req/sec")
    print(f"• Latency (Average)   : {avg_lat * 1000:.1f} ms")
    print(f"• Latency (Median P50): {median_lat * 1000:.1f} ms")
    print(f"• Latency (P95)       : {p95_lat * 1000:.1f} ms")
    print(f"• Latency (P99)       : {p99_lat * 1000:.1f} ms")
    print("=" * 65)

    if failures:
        print("❌ Failure Details:")
        for f in failures[:5]:
            print(f"  - Type: {f[2]}, Status/Error: {f[3]}")
    else:
        print("✅ 100% SUCCESS: Zero crashes, zero deadlocks, and zero 500 errors!")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
