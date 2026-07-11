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


def benchmark_rerank_logic(num_docs=100):
    client = TestClient(app)

    # Mock model and tokenizer
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Simulate tokenization time
    def slow_encode(text, **kwargs):
        # Small delay to simulate work
        time.sleep(0.001)
        return [1, 2, 3]

    mock_tokenizer.encode.side_effect = slow_encode
    mock_tokenizer.num_special_tokens_to_add.return_value = 2
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
    print(
        f"Total time for 1 request with {num_docs} documents: {total_time:.4f} seconds"
    )
    return total_time


if __name__ == "__main__":
    # Warmup
    benchmark_rerank_logic(num_docs=10)

    # Measure
    print("Starting baseline benchmark...")
    total_baseline = 0
    iterations = 5
    for i in range(iterations):
        total_baseline += benchmark_rerank_logic(num_docs=200)

    print(f"Average baseline time: {total_baseline / iterations:.4f} seconds")
