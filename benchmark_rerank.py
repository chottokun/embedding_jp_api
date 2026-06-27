
import time
import threading
import concurrent.futures
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import os

# Set environment variable before importing app
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.app.main import app

def benchmark_rerank(num_requests=10, concurrency=5):
    client = TestClient(app)

    # Mock model and tokenizer
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Simulate tokenization time
    def slow_tokenizer(*args, **kwargs):
        time.sleep(0.1)
        return {"input_ids": [[1, 2, 3] for _ in range(len(args[0]))]}

    mock_tokenizer.side_effect = slow_tokenizer
    mock_model.tokenizer = mock_tokenizer

    # Simulate model prediction time
    def slow_predict(*args, **kwargs):
        time.sleep(0.2)
        return [0.5 for _ in range(len(args[0]))]

    mock_model.predict.side_effect = slow_predict
    mock_model.lock = threading.Lock()

    with patch("src.app.main.get_model", return_value=mock_model):
        payload = {
            "query": "test query",
            "documents": ["doc1", "doc2", "doc3"],
            "model": "cl-nagoya/ruri-v3-reranker-310m"
        }

        start_time = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(client.post, "/v1/rerank", json=payload) for _ in range(num_requests)]
            _ = [f.result() for f in futures]

        end_time = time.perf_counter()

    total_time = end_time - start_time
    print(f"Total time for {num_requests} requests with concurrency {concurrency}: {total_time:.4f} seconds")
    return total_time

if __name__ == "__main__":
    print("Starting baseline benchmark...")
    benchmark_rerank(num_requests=10, concurrency=5)
