import time
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import os
import sys

# Set environment variable before importing app
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src to sys.path to ensure imports work
sys.path.append(os.path.join(os.getcwd(), "src"))

from app.main import app


def benchmark_rerank_logic(num_docs=200, use_batch=True):
    client = TestClient(app)

    # Mock model and tokenizer
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Simulate tokenization time
    def slow_encode(text, **kwargs):
        # Small delay to simulate individual work
        time.sleep(0.001)
        return [1, 2, 3]

    mock_tokenizer.encode.side_effect = slow_encode
    mock_tokenizer.num_special_tokens_to_add.return_value = 2

    if use_batch:
        # Simulate batch tokenization where we do it in a single fast call
        # e.g., 0.01s sleep instead of 0.2s (200 * 0.001s)
        def batch_tokenizer(text, **kwargs):
            time.sleep(0.01)
            return {"input_ids": [[1, 2, 3] for _ in text]}

        mock_tokenizer.side_effect = batch_tokenizer
    else:
        # Disable batch tokenizer to force the fallback loop path
        mock_tokenizer.side_effect = None

    mock_model.tokenizer = mock_tokenizer

    # Simulate model prediction time
    def fast_predict(pairs, **kwargs):
        return [0.5 for _ in range(len(pairs))]

    mock_model.predict.side_effect = fast_predict
    mock_model.lock = MagicMock()
    mock_model.lock.__enter__.return_value = None
    mock_model.lock.__exit__.return_value = None

    with patch("app.main.get_model", return_value=mock_model):
        payload = {
            "query": "test query",
            "documents": [f"doc{i}" for i in range(num_docs)],
            "model": "cl-nagoya/ruri-v3-reranker-310m",
        }

        start_time = time.perf_counter()
        response = client.post("/v1/rerank", json=payload)
        end_time = time.perf_counter()

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")

    total_time = end_time - start_time
    return total_time


if __name__ == "__main__":
    # Warmup
    benchmark_rerank_logic(num_docs=10, use_batch=True)
    benchmark_rerank_logic(num_docs=10, use_batch=False)

    print("Benchmarking with 200 documents...")
    iterations = 5

    # Measure batch (optimized)
    total_batch = 0
    for i in range(iterations):
        total_batch += benchmark_rerank_logic(num_docs=200, use_batch=True)
    avg_batch = total_batch / iterations
    print(f"Average batch (optimized) time: {avg_batch:.4f} seconds")

    # Measure loop (fallback/baseline)
    total_loop = 0
    for i in range(iterations):
        total_loop += benchmark_rerank_logic(num_docs=200, use_batch=False)
    avg_loop = total_loop / iterations
    print(f"Average loop (fallback/baseline) time: {avg_loop:.4f} seconds")

    improvement = ((avg_loop - avg_batch) / avg_loop) * 100
    print(f"Performance improvement: {improvement:.2f}%")
