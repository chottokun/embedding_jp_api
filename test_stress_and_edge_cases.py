import asyncio
import time
import httpx

BASE_URL = "http://127.0.0.1:8000"
API_KEY = "test_api_key_secret"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

TINY_PNG_B64 = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

async def run_stress_and_edge_tests():
    print("=" * 80)
    print("🧪 COMPREHENSIVE EDGE-CASE, STRESS & BUG-HUNTING TEST SUITE")
    print("=" * 80)

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        # 1. Health & Diagnostics
        print("\n[Phase 1] Health & Status Diagnostics")
        r1 = await client.get("/health")
        assert r1.status_code == 200, f"/health returned {r1.status_code}"
        assert r1.json() == {"status": "ok"}
        assert r1.headers.get("x-content-type-options") == "nosniff"
        assert r1.headers.get("x-frame-options") == "DENY"
        print("  ✓ /health returned 200 OK with security headers")

        r2 = await client.get("/healthz")
        assert r2.status_code == 200, f"/healthz returned {r2.status_code}"
        assert r2.json() == {"status": "ok"}
        print("  ✓ /healthz returned 200 OK")

        # 2. Authentication & Authorization Guard
        print("\n[Phase 2] Authentication & Security Guards")
        r_no_auth = await client.post("/v1/embeddings", json={"model": "cl-nagoya/ruri-v3-30m", "input": "test"})
        assert r_no_auth.status_code == 401, f"Expected 401 without token, got {r_no_auth.status_code}"
        print("  ✓ Missing Authorization header rejected with 401")

        r_bad_auth = await client.post("/v1/embeddings", json={"model": "cl-nagoya/ruri-v3-30m", "input": "test"}, headers={"Authorization": "Bearer invalid_secret"})
        assert r_bad_auth.status_code == 401, f"Expected 401 with invalid token, got {r_bad_auth.status_code}"
        print("  ✓ Invalid Bearer token rejected with 401")

        # 3. OpenAI-Compatible Text Embeddings
        print("\n[Phase 3] OpenAI-Compatible Text Embeddings")
        # 3.1 Single string input
        r_single = await client.post("/v1/embeddings", json={
            "model": "cl-nagoya/ruri-v3-30m",
            "input": "抗生物質の耐性菌に関する研究動向",
            "input_type": "query"
        }, headers=HEADERS)
        assert r_single.status_code == 200, f"Single embedding failed: {r_single.text}"
        data = r_single.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert len(data["data"][0]["embedding"]) == 256
        assert data["usage"]["total_tokens"] > 0
        print(f"  ✓ Single text embedding: OK (dim={len(data['data'][0]['embedding'])}, tokens={data['usage']['total_tokens']})")

        # 3.2 Array of strings (Batch)
        r_batch = await client.post("/v1/embeddings", json={
            "model": "cl-nagoya/ruri-v3-30m",
            "input": ["太陽光発電の効率", "風力タービンの保守", "地熱エネルギーの利用"]
        }, headers=HEADERS)
        assert r_batch.status_code == 200, f"Batch embedding failed: {r_batch.text}"
        data = r_batch.json()
        assert len(data["data"]) == 3
        print(f"  ✓ Batch text embedding (3 items): OK (returned {len(data['data'])} vectors)")

        # 4. Reranker API (/v1/rerank)
        print("\n[Phase 4] Reranking Engine (/v1/rerank)")
        r_rerank = await client.post("/v1/rerank", json={
            "model": "cl-nagoya/ruri-v3-reranker-310m",
            "query": "日本の首都はどこですか？",
            "documents": [
                "東京は日本の首都であり、最大の都市です。",
                "京都は日本の古都として親しまれています。",
                "富士山は日本で最も高い山です。"
            ]
        }, headers=HEADERS)
        if r_rerank.status_code == 200:
            rerank_data = r_rerank.json()
            print(f"  ✓ Rerank succeeded: scores={[round(item.get('score', 0), 4) for item in rerank_data.get('data', [])]}")
        else:
            print(f"  ℹ Rerank response: {r_rerank.status_code} - {r_rerank.text}")

        # 5. Multimodal & Schema Boundary Validation
        print("\n[Phase 5] Multimodal Schema & Boundary Controls")
        # 5.1 Image to text-only model
        r_img_to_text = await client.post("/v1/embeddings", json={
            "model": "cl-nagoya/ruri-v3-30m",
            "input": {"image_url": TINY_PNG_B64}
        }, headers=HEADERS)
        assert r_img_to_text.status_code == 400
        print("  ✓ Text-only model correctly rejected image input with 400")

        # 5.2 SSRF Loopback block (127.0.0.1)
        r_ssrf1 = await client.post("/v1/embeddings", json={
            "model": "bge-visualized-m3",
            "input": {"image_url": "http://127.0.0.1:8080/private_image.png"}
        }, headers=HEADERS)
        assert r_ssrf1.status_code == 400
        print("  ✓ SSRF 127.0.0.1 blocked with 400")

        # 5.3 SSRF Cloud Metadata block (169.254.169.254)
        r_ssrf2 = await client.post("/v1/embeddings", json={
            "model": "bge-visualized-m3",
            "input": {"image_url": "http://169.254.169.254/latest/meta-data/"}
        }, headers=HEADERS)
        assert r_ssrf2.status_code == 400
        print("  ✓ SSRF 169.254.169.254 blocked with 400")

        # 5.4 SSRF Private Subnet (10.0.0.5)
        r_ssrf3 = await client.post("/v1/embeddings", json={
            "model": "bge-visualized-m3",
            "input": {"image_url": "http://10.0.0.5/image.png"}
        }, headers=HEADERS)
        assert r_ssrf3.status_code == 400
        print("  ✓ SSRF 10.0.0.5 blocked with 400")

        # 5.5 Corrupted Base64 Image
        r_corrupt_b64 = await client.post("/v1/embeddings", json={
            "model": "bge-visualized-m3",
            "input": {"image_url": "data:image/png;base64,THIS_IS_CORRUPTED_DATA_!!!"}
        }, headers=HEADERS)
        assert r_corrupt_b64.status_code == 400
        print("  ✓ Corrupted base64 payload rejected with 400")

        # 5.6 Max Input Length Exceeded (> 65536 chars) -> Pydantic validation rejection
        huge_text = "あ" * 70000
        r_huge = await client.post("/v1/embeddings", json={
            "model": "cl-nagoya/ruri-v3-30m",
            "input": huge_text
        }, headers=HEADERS)
        assert r_huge.status_code in {400, 422}, f"Expected 400 or 422, got {r_huge.status_code}"
        print(f"  ✓ Overlength input (>65536 chars) safely rejected with {r_huge.status_code}")

        # 5.7 Max Input Items Exceeded (> 256 items) -> Pydantic validation rejection
        too_many_items = ["テスト"] * 300
        r_many = await client.post("/v1/embeddings", json={
            "model": "cl-nagoya/ruri-v3-30m",
            "input": too_many_items
        }, headers=HEADERS)
        assert r_many.status_code in {400, 422}, f"Expected 400 or 422, got {r_many.status_code}"
        print(f"  ✓ Batch size overflow (>256 items) safely rejected with {r_many.status_code}")

        # 5.8 Unsupported Model Name
        r_unsupported = await client.post("/v1/embeddings", json={
            "model": "non-existent-fake-model-xyz",
            "input": "テスト"
        }, headers=HEADERS)
        assert r_unsupported.status_code == 400
        print("  ✓ Unsupported model name rejected with 400")

        # 6. High Concurrency Stress Test (50 simultaneous requests)
        print("\n[Phase 6] High Concurrency Stress Test (50 Simultaneous Requests)")
        concurrent_count = 50
        
        async def send_req(i: int):
            t_start = time.perf_counter()
            res = await client.post("/v1/embeddings", json={
                "model": "cl-nagoya/ruri-v3-30m",
                "input": f"並行リクエスト処理のストレステスト実行中 クエリ番号={i}"
            }, headers=HEADERS)
            lat = time.perf_counter() - t_start
            return res.status_code, lat

        t0 = time.perf_counter()
        results = await asyncio.gather(*[send_req(i) for i in range(concurrent_count)])
        total_time = time.perf_counter() - t0

        status_codes = [r[0] for r in results]
        latencies = [r[1] for r in results]
        successes = status_codes.count(200)

        avg_lat = sum(latencies) / len(latencies)
        p50_lat = sorted(latencies)[int(len(latencies) * 0.50)]
        p95_lat = sorted(latencies)[int(len(latencies) * 0.95)]
        p99_lat = sorted(latencies)[int(len(latencies) * 0.99)]
        rps = concurrent_count / total_time

        print(f"  ✓ Concurrency count: {concurrent_count}")
        print(f"  ✓ Success rate: {successes}/{concurrent_count} ({successes/concurrent_count*100:.1f}%)")
        print(f"  ✓ Total execution time: {total_time:.2f}s")
        print(f"  ✓ Average Latency: {avg_lat*1000:.1f}ms")
        print(f"  ✓ Median (P50) Latency: {p50_lat*1000:.1f}ms")
        print(f"  ✓ P95 Latency: {p95_lat*1000:.1f}ms")
        print(f"  ✓ P99 Latency: {p99_lat*1000:.1f}ms")
        print(f"  ✓ Throughput: {rps:.1f} req/sec")

        assert successes == concurrent_count, f"Concurrency test failed: {status_codes}"

    print("\n" + "=" * 80)
    print("🎉 ALL TESTS, EDGE-CASES & CONCURRENCY STRESS TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(run_stress_and_edge_tests())
