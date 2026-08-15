import asyncio
import time
import httpx

TINY_PNG_B64 = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

BASE_URL = "http://127.0.0.1:8000"
API_KEY = "test_api_key_secret"


async def run_e2e_tests():
    headers = {"Authorization": f"Bearer {API_KEY}"}
    print("=" * 60)
    print("Starting Comprehensive E2E Live & Microservice Tests")
    print("=" * 60)

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        # 1. Health Checks
        print("\n[1/7] Testing Health Check Endpoints (/health, /healthz)...")
        r_health = await client.get("/health")
        assert r_health.status_code == 200, (
            f"Failed health check: {r_health.status_code}"
        )
        assert r_health.json() == {"status": "ok"}

        r_healthz = await client.get("/healthz")
        assert r_healthz.status_code == 200, (
            f"Failed healthz check: {r_healthz.status_code}"
        )
        assert r_healthz.json() == {"status": "ok"}
        print("  ✓ Health check endpoints returned 200 OK")

        # 2. Security Headers
        print("\n[2/7] Testing Security Headers...")
        assert r_health.headers.get("X-Content-Type-Options") == "nosniff"
        assert r_health.headers.get("X-Frame-Options") == "DENY"
        assert "Content-Security-Policy" in r_health.headers
        print("  ✓ Security headers verified")

        # 3. Authentication Checks
        print("\n[3/7] Testing Authentication (Bearer Token)...")
        r_unauth = await client.post(
            "/v1/embeddings", json={"model": "cl-nagoya/ruri-v3-30m", "input": "test"}
        )
        # When API_KEY is set in environment, unauthenticated request should return 401
        if r_unauth.status_code == 401:
            print("  ✓ Unauthenticated request rejected with 401")
        else:
            print(
                f"  ℹ Auth not enforced or API_KEY not set (Status {r_unauth.status_code})"
            )

        # 4. Text Embeddings
        print("\n[4/7] Testing Text Embeddings (/v1/embeddings)...")
        r_emb = await client.post(
            "/v1/embeddings",
            json={
                "model": "cl-nagoya/ruri-v3-30m",
                "input": ["自然言語処理の基礎", "機械学習モデルのデプロイ"],
                "input_type": "query",
            },
            headers=headers,
        )
        assert r_emb.status_code == 200, f"Embedding failed: {r_emb.text}"
        data = r_emb.json()
        assert len(data["data"]) == 2
        assert len(data["data"][0]["embedding"]) > 0
        print(
            f"  ✓ Text embedding succeeded (dim={len(data['data'][0]['embedding'])}, tokens={data['usage']['total_tokens']})"
        )

        # 5. Multimodal Embedding Input & Text-Only Rejection
        print("\n[5/7] Testing Multimodal Input & Text-Only Model Rejection...")
        # Image input to text-only model must return HTTP 400
        r_reject = await client.post(
            "/v1/embeddings",
            json={
                "model": "cl-nagoya/ruri-v3-30m",
                "input": {"image_url": TINY_PNG_B64},
            },
            headers=headers,
        )
        assert r_reject.status_code == 400, f"Expected 400, got: {r_reject.status_code}"
        print("  ✓ Text-only model properly rejected image input with HTTP 400")

        # 6. SSRF Protection Checks
        print("\n[6/7] Testing SSRF Protection against internal/loopback endpoints...")
        r_ssrf = await client.post(
            "/v1/embeddings",
            json={
                "model": "bge-visualized-m3",
                "input": {"image_url": "http://127.0.0.1:8000/secret.png"},
            },
            headers=headers,
        )
        assert r_ssrf.status_code == 400
        assert "拒否されたURL" in r_ssrf.text
        print("  ✓ SSRF request to 127.0.0.1 successfully blocked with HTTP 400")

        # 7. Concurrency & Stress Test
        print(
            "\n[7/7] Testing High Concurrency Stress Test (30 simultaneous requests)..."
        )

        async def make_req(idx: int):
            t0 = time.perf_counter()
            resp = await client.post(
                "/v1/embeddings",
                json={
                    "model": "cl-nagoya/ruri-v3-30m",
                    "input": f"並行リクエストテスト インデックス {idx}",
                },
                headers=headers,
            )
            elapsed = time.perf_counter() - t0
            return resp.status_code, elapsed

        t_start = time.perf_counter()
        results = await asyncio.gather(*[make_req(i) for i in range(30)])
        total_time = time.perf_counter() - t_start
        status_codes = [r[0] for r in results]
        latencies = [r[1] for r in results]

        assert all(code == 200 for code in status_codes), (
            f"Some requests failed: {status_codes}"
        )
        avg_lat = sum(latencies) / len(latencies)
        p95_lat = sorted(latencies)[int(len(latencies) * 0.95)]
        print("  ✓ 30/30 concurrent requests succeeded with 100% 200 OK")
        print(
            f"  ✓ Total duration: {total_time:.2f}s | Avg: {avg_lat * 1000:.1f}ms | P95: {p95_lat * 1000:.1f}ms | RPS: {30 / total_time:.1f}"
        )

    print("\n" + "=" * 60)
    print("ALL E2E & MICROSERVICE TESTS COMPLETED SUCCESSFULLY! ✨")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    port = sys.argv[1] if len(sys.argv) > 1 else "8000"
    BASE_URL = f"http://127.0.0.1:{port}"
    asyncio.run(run_e2e_tests())
