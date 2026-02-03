import time
import random
import heapq


def benchmark_topk():
    print("Running Top-K Benchmark...")
    print(
        f"{'Num Docs':<10} | {'Top N':<5} | {'Method':<10} | {'Time (ms)':<10} | {'Speedup':<10}"
    )
    print("-" * 60)

    scenarios = [
        (100, 10),
        (1000, 10),
        (10000, 10),
        (1000, 100),
        (10000, 100),
        (100000, 10),  # Larger scale to see asymptotic difference clearly
    ]

    for num_docs, top_n in scenarios:
        # Create dummy data
        results = [{"document": i, "score": random.random()} for i in range(num_docs)]

        # Method 1: Sorted + Slice (Current)
        start_time = time.perf_counter()
        for _ in range(100):  # Repeat for stability
            sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
            _ = sorted_results[:top_n]
        end_time = time.perf_counter()
        avg_time_sorted = (end_time - start_time) / 100 * 1000  # ms

        # Method 2: Heapq (Proposed)
        start_time = time.perf_counter()
        for _ in range(100):
            _ = heapq.nlargest(top_n, results, key=lambda x: x["score"])
        end_time = time.perf_counter()
        avg_time_heapq = (end_time - start_time) / 100 * 1000  # ms

        speedup = avg_time_sorted / avg_time_heapq if avg_time_heapq > 0 else 0.0

        print(f"{num_docs:<10} | {top_n:<5} | {'Sorted':<10} | {avg_time_sorted:.4f}")
        print(
            f"{'':<10} | {'':<5} | {'Heapq':<10} | {avg_time_heapq:.4f}     | {speedup:.2f}x"
        )
        print("-" * 60)


if __name__ == "__main__":
    benchmark_topk()
