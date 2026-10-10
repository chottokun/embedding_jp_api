#!/usr/bin/env python3
"""
Live E2E Verification Script for Japanese Embedding & Reranking API.
Validates all endpoints, real model inferences, security headers, metrics, and error handling.
"""

import os
import sys
import time
import subprocess
import httpx


SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8008
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


def wait_for_server(timeout_sec: int = 30) -> bool:
    """Waits until the server is responsive on /healthz."""
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            r = httpx.get(f"{BASE_URL}/healthz", timeout=1.0)
            if r.status_code == 200:
                return True
        except Exception:
            time.sleep(0.5)
    return False


def run_live_tests():
    print("=" * 60)
    print("🚀 Starting Live E2E Service Verification")
    print("=" * 60)

    # 1. Health & Readiness Probes
    print("\n[1/7] Testing Health & Readiness Probes...")
    r = httpx.get(f"{BASE_URL}/healthz")
    assert r.status_code == 200, f"Healthz failed: {r.status_code}"
    print("  ✓ GET /healthz -> 200 OK")

    r = httpx.get(f"{BASE_URL}/readyz")
    assert r.status_code == 200, f"Readyz failed: {r.status_code}"
    print(f"  ✓ GET /readyz -> 200 OK (Data: {r.json()})")

    # 2. Security Headers
    print("\n[2/7] Testing Security Headers Enforcement...")
    r = httpx.get(f"{BASE_URL}/healthz")
    headers = r.headers
    assert headers.get("x-content-type-options") == "nosniff", (
        "Missing X-Content-Type-Options"
    )
    assert headers.get("x-frame-options") == "DENY", "Missing X-Frame-Options"
    assert (
        "Strict-Transport-Security" in headers or "strict-transport-security" in headers
    ), "Missing HSTS"
    print("  ✓ Security Headers Verified (nosniff, DENY, HSTS present)")

    # 3. Models Endpoint
    print("\n[3/7] Testing Models Listing Endpoint...")
    r = httpx.get(f"{BASE_URL}/v1/models")
    assert r.status_code == 200, f"Models failed: {r.status_code}"
    models_data = r.json()
    model_ids = [m["id"] for m in models_data.get("data", [])]
    assert len(model_ids) > 0, "No models returned"
    assert "cl-nagoya/ruri-v3-30m" in model_ids, "ruri-v3-30m missing"
    print(f"  ✓ GET /v1/models -> 200 OK (Found {len(model_ids)} models)")

    # 4. Text Embedding Inference (Real Model: ruri-v3-30m)
    print("\n[4/7] Testing Real Text Embedding Inference...")
    payload = {
        "model": "cl-nagoya/ruri-v3-30m",
        "input": ["自然言語処理の進化", "機械学習とディープラーニング"],
        "encoding_format": "float",
    }
    t0 = time.perf_counter()
    r = httpx.post(f"{BASE_URL}/v1/embeddings", json=payload, timeout=30.0)
    latency_emb = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200, f"Embedding failed ({r.status_code}): {r.text}"
    emb_data = r.json()
    assert len(emb_data["data"]) == 2, "Expected 2 embeddings"
    dim = len(emb_data["data"][0]["embedding"])
    print(
        f"  ✓ POST /v1/embeddings -> 200 OK (Latency: {latency_emb:.1f}ms, Dimension: {dim})"
    )

    # 5. Reranking Inference (Real Model: ruri-v3-reranker-310m)
    print("\n[5/7] Testing Real Cross-Encoder Reranking Inference...")
    rerank_payload = {
        "model": "cl-nagoya/ruri-v3-reranker-310m",
        "query": "日本の首都はどこですか？",
        "documents": [
            "日本の首都は東京であり、政治と経済の中心地です。",
            "富士山は日本で最も高い山です。",
            "京都はかつて日本の都として栄えました。",
        ],
        "top_n": 3,
        "return_documents": True,
    }
    t0 = time.perf_counter()
    r = httpx.post(f"{BASE_URL}/v1/rerank", json=rerank_payload, timeout=30.0)
    latency_rerank = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200, f"Rerank failed ({r.status_code}): {r.text}"
    rerank_data = r.json()
    assert len(rerank_data["data"]) == 3, "Expected 3 reranked results"
    top_doc = rerank_data["data"][0]["text"]
    assert "東京" in top_doc, f"Top ranked document unexpected: {top_doc}"
    print(
        f"  ✓ POST /v1/rerank -> 200 OK (Latency: {latency_rerank:.1f}ms, Top result: {top_doc[:20]}...)"
    )

    # 6. Prometheus Metrics Validation
    print("\n[6/7] Testing Prometheus Metrics...")
    r = httpx.get(f"{BASE_URL}/metrics")
    assert r.status_code == 200, f"Metrics failed: {r.status_code}"
    metrics_text = r.text
    assert "http_requests_total" in metrics_text, "Missing http_requests_total"
    assert "http_request_duration_seconds" in metrics_text, (
        "Missing http_request_duration_seconds"
    )
    assert "http_request_batch_size" in metrics_text, "Missing http_request_batch_size"
    print("  ✓ GET /metrics -> 200 OK (Prometheus metrics verified)")

    # 7. Payload Limit Validation
    print("\n[7/8] Testing Payload Size Limit (413 Payload Too Large)...")
    large_payload = b"x" * (11 * 1024 * 1024)  # 11MB (limit is 10MB)
    r = httpx.post(f"{BASE_URL}/v1/embeddings", content=large_payload, timeout=10.0)
    assert r.status_code == 413, f"Expected 413, got {r.status_code}"
    print("  ✓ Large payload rejected with 413 Payload Too Large")

    # 8. Multimodal Capability Guard Verification
    print("\n[8/8] Testing Multimodal Audio/Video Capability Guards...")
    dummy_audio = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
    # 8a: Unsupported model guard (Ruri cannot accept audio)
    r = httpx.post(
        f"{BASE_URL}/v1/embeddings",
        json={"model": "cl-nagoya/ruri-v3-30m", "input": {"text": "hello", "input_audio": dummy_audio}},
    )
    assert r.status_code == 400, f"Expected 400 for audio with Ruri, got {r.status_code}"
    print("  ✓ Unsupported model audio rejected with 400 Bad Request")

    # 8b: Server flag disabled guard (EmbeddingGemma audio rejected when ENABLE_AUDIO_EMBEDDING=false)
    r = httpx.post(
        f"{BASE_URL}/v1/embeddings",
        json={"model": "google/embeddinggemma-2", "input": {"text": "hello", "input_audio": dummy_audio}},
    )
    assert r.status_code == 400, f"Expected 400 for disabled audio, got {r.status_code}"
    print("  ✓ Disabled audio feature rejected with 400 Bad Request")

    print("\n" + "=" * 60)
    print("🎉 All 8 Live Service Verification Tests PASSED Perfectly!")
    print("=" * 60)


def main():
    env = os.environ.copy()
    env["APP_PORT"] = str(SERVER_PORT)
    env["PYTHONPATH"] = "src"

    # Launch server process
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
        "info",
    ]
    print(f"[INFO] Starting API server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env)

    try:
        if not wait_for_server(timeout_sec=30):
            print("[ERROR] Server failed to start within timeout.")
            sys.exit(1)
        run_live_tests()
    finally:
        print("[INFO] Terminating server process...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("[INFO] Server stopped gracefully.")


if __name__ == "__main__":
    main()
