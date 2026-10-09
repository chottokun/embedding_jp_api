"""Comprehensive load test runner for embedding_jp_api.

Measures concurrency handling, throughput (RPS), latencies (p50, p95, p99),
and zero-error reliability under sustained load using asyncio and httpx.
"""

import asyncio
import os
import subprocess
import sys
import time
from dataclasses import dataclass
import httpx
import numpy as np

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8008
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


@dataclass
class TestResult:
    endpoint: str
    status_code: int
    duration_ms: float
    error: str | None = None


@dataclass
class BenchmarkSummary:
    endpoint: str
    total_requests: int
    success_count: int
    failure_count: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    rps: float


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


async def worker_task(
    client: httpx.AsyncClient,
    queue: asyncio.Queue,
    results: list[TestResult],
) -> None:
    while not queue.empty():
        req_type, payload = await queue.get()
        t0 = time.perf_counter()
        endpoint = (
            f"/v1/{req_type}"
            if req_type in ["embeddings", "rerank"]
            else f"/{req_type}"
        )
        url = f"{BASE_URL}{endpoint}"
        try:
            if req_type in ["embeddings", "rerank"]:
                resp = await client.post(url, json=payload, timeout=30.0)
            else:
                resp = await client.get(url, timeout=10.0)
            duration_ms = (time.perf_counter() - t0) * 1000
            results.append(
                TestResult(
                    endpoint=endpoint,
                    status_code=resp.status_code,
                    duration_ms=duration_ms,
                    error=None
                    if resp.status_code == 200
                    else f"HTTP {resp.status_code}: {resp.text[:100]}",
                )
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - t0) * 1000
            results.append(
                TestResult(
                    endpoint=endpoint,
                    status_code=0,
                    duration_ms=duration_ms,
                    error=str(e),
                )
            )
        finally:
            queue.task_done()


async def run_benchmark(
    concurrency: int = 10, total_requests: int = 100
) -> list[TestResult]:
    queue: asyncio.Queue = asyncio.Queue()

    # Prepare diverse realistic request workloads
    embedding_payloads = [
        {
            "model": "cl-nagoya/ruri-v3-30m",
            "input": "人工知能の急速な進歩により社会が変革しています。",
        },
        {
            "model": "cl-nagoya/ruri-v3-30m",
            "input": ["機械学習の基礎", "深層学習と自然言語処理の応用"],
        },
        {
            "model": "cl-nagoya/ruri-v3-30m",
            "input": "FastAPIとPydanticによるハイパフォーマンスAPI設計。",
        },
    ]
    rerank_payloads = [
        {
            "model": "cl-nagoya/ruri-v3-reranker-310m",
            "query": "日本の首都",
            "documents": [
                "東京は日本の首都です。",
                "富士山は日本一の山です。",
                "京都は歴史的な古都です。",
            ],
        },
        {
            "model": "cl-nagoya/ruri-v3-reranker-310m",
            "query": "Python パフォーマンス改善",
            "documents": [
                "CythonやPyPyの活用",
                "非同期処理の導入",
                "アルゴリズムの見直し",
            ],
        },
    ]

    for i in range(total_requests):
        choice = i % 4
        if choice in (0, 1):
            queue.put_nowait(
                ("embeddings", embedding_payloads[i % len(embedding_payloads)])
            )
        elif choice == 2:
            queue.put_nowait(("rerank", rerank_payloads[i % len(rerank_payloads)]))
        else:
            queue.put_nowait(("healthz", None))

    results: list[TestResult] = []
    limits = httpx.Limits(
        max_connections=concurrency * 2, max_keepalive_connections=concurrency
    )
    async with httpx.AsyncClient(limits=limits) as client:
        workers = [
            asyncio.create_task(worker_task(client, queue, results))
            for _ in range(concurrency)
        ]
        await queue.join()
        for w in workers:
            w.cancel()

    return results


def summarize_results(
    results: list[TestResult], elapsed_total: float
) -> list[BenchmarkSummary]:
    summaries: list[BenchmarkSummary] = []
    by_endpoint: dict[str, list[TestResult]] = {}
    for r in results:
        by_endpoint.setdefault(r.endpoint, []).append(r)

    # Also compute aggregate
    by_endpoint["[ALL COMBINED]"] = results

    for endpoint, res_list in by_endpoint.items():
        durations = [r.duration_ms for r in res_list]
        successes = [r for r in res_list if r.status_code == 200]
        failures = [r for r in res_list if r.status_code != 200]

        summaries.append(
            BenchmarkSummary(
                endpoint=endpoint,
                total_requests=len(res_list),
                success_count=len(successes),
                failure_count=len(failures),
                mean_ms=float(np.mean(durations)),
                p50_ms=float(np.percentile(durations, 50)),
                p90_ms=float(np.percentile(durations, 90)),
                p95_ms=float(np.percentile(durations, 95)),
                p99_ms=float(np.percentile(durations, 99)),
                max_ms=float(np.max(durations)),
                rps=len(res_list) / elapsed_total,
            )
        )
    return summaries


def main():
    concurrency = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    total_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    print("=" * 70)
    print("🔥 Starting Sustained High-Concurrency Load Test")
    print(f"   Concurrency: {concurrency} workers")
    print(f"   Total Requests: {total_requests}")
    print("=" * 70)

    rate_limit = os.environ.get("RATE_LIMIT_PER_MINUTE", "10000")
    env = os.environ.copy()
    env["APP_PORT"] = str(SERVER_PORT)
    env["PYTHONPATH"] = "src"
    env["RATE_LIMIT_PER_MINUTE"] = rate_limit

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
    proc = subprocess.Popen(cmd, env=env)

    try:
        if not wait_for_server(timeout_sec=30):
            print("[ERROR] Server failed to start.")
            sys.exit(1)

        # Warmup
        print("[INFO] Warmup single request...")
        httpx.post(
            f"{BASE_URL}/v1/embeddings",
            json={"model": "cl-nagoya/ruri-v3-30m", "input": "Warmup test"},
            timeout=30.0,
        )

        print("[INFO] Running concurrent load testing...")
        t_start = time.perf_counter()
        results = asyncio.run(
            run_benchmark(concurrency=concurrency, total_requests=total_requests)
        )
        t_elapsed = time.perf_counter() - t_start

        summaries = summarize_results(results, t_elapsed)

        print("\n" + "=" * 70)
        print("📊 LOAD TEST RESULTS SUMMARY")
        print("=" * 70)
        print(
            f"Total Time: {t_elapsed:.2f}s | Overall RPS: {len(results) / t_elapsed:.2f} req/s\n"
        )

        print(
            f"{'Endpoint':<18} | {'Total':>5} | {'Pass':>5} | {'Fail':>4} | {'Mean(ms)':>8} | {'P50(ms)':>8} | {'P95(ms)':>8} | {'P99(ms)':>8} | {'RPS':>6}"
        )
        print("-" * 88)
        for s in summaries:
            print(
                f"{s.endpoint:<18} | {s.total_requests:>5} | {s.success_count:>5} | {s.failure_count:>4} | {s.mean_ms:>8.1f} | {s.p50_ms:>8.1f} | {s.p95_ms:>8.1f} | {s.p99_ms:>8.1f} | {s.rps:>6.1f}"
            )

        # Print failures if any
        failures = [r for r in results if r.status_code != 200]
        if failures:
            print("\n❌ Failures detected:")
            for f in failures[:5]:
                print(f"  {f.endpoint} -> Status {f.status_code}: {f.error}")
            sys.exit(1)
        else:
            print("\n✅ Zero Errors: 100% Success Rate under high concurrent load!")
            print("=" * 70)

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
