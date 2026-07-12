import threading
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pytest
import numpy as np

from app.main import app

client = TestClient(app)

@pytest.mark.anyio
def test_high_concurrency_thread_safety_mock():
    """
    Verify that concurrent threads calling embeddings and rerank do not cause
    tokenizer race conditions ('Already borrowed') or deadlocks.
    """
    # 1. Mock get_model to return a mock model that simulates tokenizer/inference calls
    with patch("app.main.get_model") as mock_get_model:
        mock_model = mock_get_model.return_value
        
        # Setup threading locks on the mock
        mock_model.lock = threading.Lock()
        mock_model.tokenizer_lock = threading.Lock()
        mock_model.max_seq_length = 8192
        
        # Mock encoding response
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        # Mock predict response for rerank
        mock_model.predict.return_value = [0.9, 0.1]
        
        # Configure mock tokenizer to mimic slow Rust tokenization
        def mock_tokenize_call(texts, *args, **kwargs):
            # Tokenizer fast mapping output format
            if isinstance(texts, list):
                return {"input_ids": [[1, 2, 3] for _ in range(len(texts))]}
            return [1, 2, 3]
            
        mock_model.tokenizer.side_effect = mock_tokenize_call
        mock_model.tokenizer.num_special_tokens_to_add.return_value = 2
        mock_model.tokenizer.encode.side_effect = lambda text, *args, **kwargs: [1, 2, 3]

        # Ensure RURI_PREFIX_MAP check passes
        with patch("app.main.EMBEDDING_MODELS", ["cl-nagoya/ruri-v3-30m"]):
            with patch("app.main.RERANK_MODELS", ["cl-nagoya/ruri-v3-reranker-310m"]):
                errors = []
                
                # Worker task simulating FastAPI endpoint processing via TestClient
                def worker(thread_idx):
                    try:
                        # Authenticate bypass or headers (verify_api_key bypassed if API_KEY not set)
                        if thread_idx % 2 == 0:
                            response = client.post(
                                "/v1/embeddings",
                                json={"input": ["テスト入力文"] * 10, "model": "cl-nagoya/ruri-v3-30m"},
                            )
                        else:
                            response = client.post(
                                "/v1/rerank",
                                json={
                                    "query": "テストクエリ",
                                    "documents": ["ドキュメントA", "ドキュメントB"],
                                    "model": "cl-nagoya/ruri-v3-reranker-310m",
                                },
                            )
                        if response.status_code != 200:
                            errors.append(f"HTTP {response.status_code}: {response.text}")
                    except Exception as e:
                        errors.append(str(e))

                # Spawn multiple parallel threads
                threads = []
                for i in range(16):
                    t = threading.Thread(target=worker, args=(i,))
                    threads.append(t)
                    t.start()
                    
                for t in threads:
                    t.join()
                
                # Assert no errors occurred during concurrent executions
                assert len(errors) == 0, f"Concurrency test failed with errors: {errors}"
