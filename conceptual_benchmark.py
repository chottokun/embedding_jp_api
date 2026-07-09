import time
import threading
import concurrent.futures


class MockModel:
    def __init__(self):
        self.lock = threading.Lock()

    def tokenize(self, data):
        # Simulated CPU-heavy task (e.g. tokenization)
        time.sleep(0.05)
        return len(data) * 10

    def predict(self, data):
        # Simulated Model Inference task (needs lock)
        time.sleep(0.1)
        return [0.5] * len(data)


def baseline_call(model, data):
    with model.lock:
        tokens = model.tokenize(data)
        result = model.predict(data)
    return tokens, result


def optimized_call(model, data):
    tokens = model.tokenize(data)  # Outside lock
    with model.lock:
        result = model.predict(data)
    return tokens, result


def run_test(name, func, model, num_requests):
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [
            executor.submit(func, model, ["doc"] * 10) for _ in range(num_requests)
        ]
        concurrent.futures.wait(futures)
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.4f}s")
    return elapsed


if __name__ == "__main__":
    model = MockModel()
    num_req = 10
    print(f"Running benchmark with {num_req} concurrent requests...")
    t1 = run_test("Baseline (Inside Lock)", baseline_call, model, num_req)
    t2 = run_test("Optimized (Outside Lock)", optimized_call, model, num_req)
    improvement = (t1 - t2) / t1 * 100
    print(f"Improvement: {improvement:.2f}%")
