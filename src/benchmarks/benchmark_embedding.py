import time
import sys
import os
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from app.main import app
from app.config import EMBEDDING_MODELS

# --- Mocking ---

class MockTokenizer:
    def __init__(self):
        pass

    def encode(self, text, add_special_tokens=False):
        # Simulate O(N) work
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join([chr(t) for t in tokens])

    def num_special_tokens_to_add(self, pair):
        return 0

    def __call__(self, text, padding=False, truncation=False, max_length=None, add_special_tokens=False, return_tensors=None):
        # Supports both single string and list of strings (batch)
        if isinstance(text, str):
            input_ids = self.encode(text)
        elif isinstance(text, list):
            input_ids = [self.encode(t) for t in text]
        else:
            input_ids = []

        # Simple dict-like object
        return {"input_ids": input_ids}

class MockModel:
    def __init__(self):
        self.tokenizer = MockTokenizer()
        self.max_seq_length = 8192

    def encode(self, inputs):
        # Simulate encoding time (lightweight)
        # inputs is a list of strings
        # Return fake embeddings
        return np.random.rand(len(inputs), 10)

# --- Benchmark ---

def run_benchmark():
    # Setup
    model_name = "cl-nagoya/ruri-v3-30m"
    if model_name not in EMBEDDING_MODELS:
        EMBEDDING_MODELS.append(model_name)

    client = TestClient(app)

    # Payload
    # 100 inputs, each ~1000 chars
    input_texts = ["A" * 1000 for _ in range(100)]
    payload = {
        "input": input_texts,
        "model": model_name
    }

    # Patch get_model to return our mock
    with patch("app.main.get_model", return_value=MockModel()):
        # Warmup
        client.post("/v1/embeddings", json={"input": "test", "model": model_name})

        # Measure
        start_time = time.time()
        for _ in range(50): # 50 iterations
            response = client.post("/v1/embeddings", json=payload)
            assert response.status_code == 200
        end_time = time.time()

    total_time = end_time - start_time
    print(f"Benchmark Result: {total_time:.4f} seconds for 50 batches of 100 items.")

if __name__ == "__main__":
    run_benchmark()
