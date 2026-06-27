import time
import torch
from sentence_transformers import CrossEncoder
from app.schemas import RerankRequest

def benchmark():
    model_name = "BAAI/bge-reranker-base"
    # Mocking what get_model does
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(model_name, device=device)

    query = "What is the capital of France?"
    documents = ["Paris is the capital of France."] * 100

    # Warmup
    model.predict([[query, doc] for doc in documents])

    start_time = time.time()
    for _ in range(10):
        # Simulation of the current logic in main.py
        pairs = [[query, doc] for doc in documents]

        # Token count logic
        tokenizer = model.tokenizer
        total_tokens = 0
        for pair in pairs:
            q_tokens = len(tokenizer.encode(pair[0], add_special_tokens=False))
            d_tokens = len(tokenizer.encode(pair[1], add_special_tokens=False))
            total_tokens += (
                q_tokens + d_tokens + tokenizer.num_special_tokens_to_add(True)
            )

        scores = model.predict(pairs)
    end_time = time.time()

    print(f"Time taken for 10 iterations (current): {end_time - start_time:.4f}s")

if __name__ == "__main__":
    benchmark()
