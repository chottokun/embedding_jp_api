
import time
import threading
import concurrent.futures
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import os
import torch

# Set environment variable before importing app
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.app.main import app

def benchmark_rerank(num_requests=10, concurrency=5):
    client = TestClient(app)

    # Mock model and tokenizer
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Simulate tokenization time for __call__ (batch tokenization)
    def slow_call(batch_queries=None, batch_docs=None, *args, **kwargs):
        time.sleep(0.05) # 50ms per batch call
        count = len(batch_queries)
        # Return tensors as the real tokenizer would with return_tensors="pt"
        return {
            "input_ids": torch.zeros((count, 10), dtype=torch.long),
            "attention_mask": torch.ones((count, 10), dtype=torch.long)
        }

    mock_tokenizer.side_effect = slow_call
    mock_model.tokenizer = mock_tokenizer

    # Mock inner model
    inner_model = MagicMock()
    def slow_forward(**kwargs):
        time.sleep(0.1) # 100ms for inference
        count = kwargs["input_ids"].shape[0]
        output = MagicMock()
        output.logits = torch.zeros((count, 1))
        return output

    inner_model.side_effect = slow_forward
    mock_model.model = inner_model
    mock_model._target_device = "cpu"
    mock_model.default_activation_function = None

    mock_model.lock = threading.Lock()
    mock_model.tokenizer_lock = threading.Lock()

    with patch("src.app.main.get_model", return_value=mock_model):
        payload = {
            "query": "test query",
            "documents": ["doc1", "doc2", "doc3"],
            "model": "cl-nagoya/ruri-v3-reranker-310m"
        }

        # Optimized logic:
        # Outside lock: 0.05s (tokenize)
        # Inside lock: 0.1s (inference)
        # Total time per request: 0.15s
        # Sequentialized bottleneck (lock): 10 * 0.1s = 1.0s
        # Total expected time: ~1.0s + some overhead and the initial 0.05s of the first few requests.

        start_time = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(client.post, "/v1/rerank", json=payload) for _ in range(num_requests)]
            results = [f.result() for f in futures]

        end_time = time.perf_counter()

    total_time = end_time - start_time
    print(f"Total time for {num_requests} requests with concurrency {concurrency}: {total_time:.4f} seconds")
    return total_time

if __name__ == "__main__":
    print("Starting optimized benchmark...")
    benchmark_rerank(num_requests=10, concurrency=5)
