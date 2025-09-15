import numpy as np
from .config import EMBEDDING_MODELS, RERANK_MODELS

# --- Mock Model Implementations ---

class MockSentenceTransformer:
    """A mock class for sentence_transformers.SentenceTransformer."""
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        print(f"MockEmbeddingModel '{model_name}' loaded on '{device}'.")

    def encode(self, sentences: list[str], convert_to_tensor: bool = False):
        """Returns dummy embeddings."""
        # Return a fixed-size vector for each sentence.
        embedding_dim = 256  # As per ruri-v3-30m
        embeddings = [np.random.rand(embedding_dim).tolist() for _ in sentences]
        print(f"Encoded {len(sentences)} sentences with mock model '{self.model_name}'.")
        return embeddings

class MockCrossEncoder:
    """A mock class for sentence_transformers.CrossEncoder."""
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        print(f"MockRerankModel '{model_name}' loaded on '{device}'.")

    def predict(self, pairs: list[list[str]]):
        """Returns dummy scores."""
        # Generate scores based on a simple logic for predictability in tests.
        # For example, score is higher if the query words appear in the document.
        scores = []
        for query, doc in pairs:
            score = 0.1
            if query.lower() in doc.lower():
                score = 0.9
            scores.append(score)
        print(f"Predicted scores for {len(pairs)} pairs with mock model '{self.model_name}'.")
        return scores

# --- Model Loader (Factory) ---

_model_cache = {}

def get_model(model_name: str):
    """
    Factory function to get a model instance.
    For TDD, this returns mock models.
    It caches model instances to avoid reloading.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    if model_name in EMBEDDING_MODELS:
        # In a real scenario, we would add `device=...`
        model = MockSentenceTransformer(model_name=model_name)
        _model_cache[model_name] = model
        return model

    if model_name in RERANK_MODELS:
        model = MockCrossEncoder(model_name=model_name)
        _model_cache[model_name] = model
        return model

    raise ValueError(f"Model '{model_name}' is not supported.")
