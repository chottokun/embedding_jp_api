"""Orchestrated GPU Locust Load Testing Suite for embedding_jp_api.

Runs comprehensive load testing on NVIDIA GPU (RTX 3060) across multiple
concurrency tiers (e.g. 10, 30, 60 users) covering:
- Single text embedding (Ruri-310m)
- Batch text embedding (Ruri-310m)
- Ruri task prefix embedding
- Gemma-2 text embedding
- Matryoshka 256d & base64 format
- Multimodal image embedding (BGE-M3 Visualized & Gemma-2)
- Multimodal audio & video embedding (Gemma-2)
- Cross-encoder rerank (Ruri-310m-reranker)
- Health / Readiness probes
"""

import os
import subprocess
import sys
import time
from pathlib import Path
import httpx
import pandas as pd

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8008
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


def wait_for_server(timeout_sec: float = 60.0) -> bool:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout_sec:
        try:
            r = httpx.get(f"{BASE_URL}/ready", timeout=2.0)
            if r.status_code == 200 and r.json().get("status") == "ready":
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def get_gpu_status() -> str:
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        parts = res.stdout.strip().split(",")
        if len(parts) == 3:
            used, total, util = parts[0].strip(), parts[1].strip(), parts[2].strip()
            return f"VRAM: {used}MB / {total}MB ({float(used) / float(total) * 100:.1f}%), GPU Util: {util}%"
    except Exception as e:
        return f"GPU query error: {e}"
    return "Unknown"


def warmup():
    print("[INFO] Pre-warming models on GPU...")
    # 1. Warmup ruri
    httpx.post(
        f"{BASE_URL}/v1/embeddings",
        json={"model": "cl-nagoya/ruri-v3-310m", "input": "ウォームアップクエリ"},
        timeout=30.0,
    )
    # 2. Warmup gemma-2
    httpx.post(
        f"{BASE_URL}/v1/embeddings",
        json={"model": "google/embeddinggemma-2", "input": "ウォームアップテキスト"},
        timeout=30.0,
    )
    # 3. Warmup rerank
    httpx.post(
        f"{BASE_URL}/v1/rerank",
        json={
            "model": "cl-nagoya/ruri-v3-reranker-310m",
            "query": "ウォームアップ",
            "documents": ["ドキュメント1", "ドキュメント2"],
        },
        timeout=30.0,
    )
    # 4. Warmup multimodal bge-visualized-m3
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from scripts.locustfile import DUMMY_IMAGE

        httpx.post(
            f"{BASE_URL}/v1/embeddings",
            json={
                "model": "bge-visualized-m3",
                "input": {"text": "ウォームアップ", "image_url": DUMMY_IMAGE},
            },
            timeout=30.0,
        )
    except Exception as e:
        print(f"[WARN] BGE multimodal warmup skipped/failed: {e}")
    print("[INFO] Warmup complete. Current GPU status:", get_gpu_status())


def run_tier(users: int, spawn_rate: int, run_time: str, prefix: str):
    print("\n" + "=" * 70)
    print(
        f"🚀 Running Locust Tier: Users={users}, SpawnRate={spawn_rate}, Duration={run_time}"
    )
    print(f"   Pre-test GPU Status: {get_gpu_status()}")
    print("=" * 70)

    csv_prefix = f"locust_results_{prefix}"
    tier_env = os.environ.copy()
    tier_env["ENABLE_AUDIO_EMBEDDING"] = "true"
    tier_env["ENABLE_VIDEO_EMBEDDING"] = "true"

    cmd = [
        "uv",
        "run",
        "locust",
        "-f",
        "scripts/locustfile.py",
        "--headless",
        "--users",
        str(users),
        "--spawn-rate",
        str(spawn_rate),
        "--run-time",
        run_time,
        "--host",
        BASE_URL,
        "--csv",
        csv_prefix,
    ]

    res = subprocess.run(cmd, env=tier_env)
    if res.returncode != 0:
        print(f"[WARN] Locust exited with code {res.returncode}")
    print(f"   Post-test GPU Status: {get_gpu_status()}")

    stats_file = f"{csv_prefix}_stats.csv"
    if Path(stats_file).exists():
        df = pd.read_csv(stats_file)
        print("\n" + "-" * 70)
        print(f"📊 Tier Summary (Users={users}):")
        print("-" * 70)
        for _, row in df.iterrows():
            name = row["Name"]
            count = row["Request Count"]
            fails = row["Failure Count"]
            rps = row.get("Requests/s", 0)
            p50 = row.get("50%", 0)
            p95 = row.get("95%", 0)
            p99 = row.get("99%", 0)
            print(
                f"  {name:<32} | Req: {count:>5} | Fail: {fails:>3} | RPS: {rps:>5.1f} | P50: {p50:>6.1f}ms | P95: {p95:>7.1f}ms | P99: {p99:>7.1f}ms"
            )
    return stats_file


def main():
    print("=" * 70)
    print("🔥 Starting Comprehensive GPU Locust Load Testing Suite")
    print("=" * 70)

    env = os.environ.copy()
    env["APP_PORT"] = str(SERVER_PORT)
    env["PYTHONPATH"] = "src"
    env["RATE_LIMIT_PER_MINUTE"] = "100000"
    env["MAX_CONCURRENT_INFERENCES"] = "8"
    env["INFERENCE_SEMAPHORE_TIMEOUT_SECONDS"] = "60.0"
    env["ENABLE_AUDIO_EMBEDDING"] = "true"
    env["ENABLE_VIDEO_EMBEDDING"] = "true"

    server_cmd = [
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

    print(f"[INFO] Spawning API Server on GPU (port {SERVER_PORT})...")
    proc = subprocess.Popen(server_cmd, env=env)

    try:
        if not wait_for_server(timeout_sec=60):
            print("[ERROR] Server failed to start within timeout.")
            sys.exit(1)

        warmup()

        tiers = [
            {"users": 10, "spawn_rate": 5, "run_time": "35s"},
            {"users": 30, "spawn_rate": 10, "run_time": "40s"},
            {"users": 60, "spawn_rate": 15, "run_time": "45s"},
        ]

        for t in tiers:
            run_tier(
                users=t["users"],
                spawn_rate=t["spawn_rate"],
                run_time=t["run_time"],
                prefix=f"u{t['users']}",
            )
            time.sleep(3)

        print("\n" + "=" * 70)
        print("✅ All Locust Load Test Tiers Completed Successfully!")
        print("=" * 70)

    finally:
        print("[INFO] Terminating API Server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
