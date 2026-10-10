import argparse
import subprocess
import pandas as pd
from pathlib import Path


def print_summary_metrics(stats_file: str):
    try:
        df = pd.read_csv(stats_file)
        # Filter for the aggregated row, or drop it
        agg_row = df[df["Name"] == "Aggregated"]
        if agg_row.empty:
            print("No aggregated metrics found.")
            return

        print("\n" + "=" * 50)
        print(" LOCUST BENCHMARK SUMMARY")
        print("=" * 50)

        req_count = agg_row["Request Count"].values[0]
        fails = agg_row["Failure Count"].values[0]
        rps = agg_row["Requests/s"].values[0]
        p50 = agg_row["50%"].values[0]
        p95 = agg_row["95%"].values[0]
        p99 = agg_row["99%"].values[0]

        err_rate = (fails / req_count * 100) if req_count > 0 else 0

        print(f"Total Requests: {req_count}")
        print(f"Total Failures: {fails}")
        print(f"Error Rate:     {err_rate:.2f}%")
        print(f"RPS:            {rps:.2f}")
        print(f"P50 Latency:    {p50} ms")
        print(f"P95 Latency:    {p95} ms")
        print(f"P99 Latency:    {p99} ms")
        print("=" * 50 + "\n")
    except Exception as e:
        print(f"Failed to parse or print metrics: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run Locust benchmark suite.")
    parser.add_argument(
        "--host", default="http://localhost:8000", help="Host URL to test"
    )
    parser.add_argument(
        "--users", type=int, default=10, help="Number of concurrent users"
    )
    parser.add_argument("--spawn-rate", type=int, default=2, help="Spawn rate")
    parser.add_argument("--run-time", default="30s", help="Run time (e.g., 30s, 1m)")
    args = parser.parse_args()

    cmd = [
        "uv",
        "run",
        "locust",
        "-f",
        "scripts/locustfile.py",
        "--headless",
        "--users",
        str(args.users),
        "--spawn-rate",
        str(args.spawn_rate),
        "--run-time",
        args.run_time,
        "--host",
        args.host,
        "--csv",
        "locust_results",
    ]

    print(f"Running command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Locust run failed: {e}")
    finally:
        stats_file = "locust_results_stats.csv"
        if Path(stats_file).exists():
            print_summary_metrics(stats_file)
        else:
            print("Could not find Locust results CSV file.")


if __name__ == "__main__":
    main()
