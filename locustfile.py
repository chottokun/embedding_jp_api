import random
from locust import HttpUser, task, between

# --- Sample Data Pool ---
# A small pool of data to be used in the load tests.
EMBEDDING_INPUTS = [
    "今日の天気は晴れです。",
    "最新のAI技術について教えてください。",
    "このドキュメントを要約して。",
    "自然言語処理とは何ですか？",
    ["これは最初の文書です。", "これは2番目の文書で、少し長いです。", "そして3番目。"]
]

RERANK_QUERIES = [
    "AIの未来について",
    "日本の首都",
    "猫の生態"
]
RERANK_DOCS = [
    "これは猫についての文章です。",
    "人工知能は今後の社会を大きく変えるでしょう。",
    "日本の首都は東京です。",
    "犬は人間の最良の友です。",
    "機械学習はAIのサブセットです。"
]

# Supported models from our config. We will randomly pick one.
# In a real test, you might want to target specific models.
EMBEDDING_MODELS = ["cl-nagoya/ruri-v3-30m", "cl-nagoya/ruri-v3-310m"]
RERANK_MODELS = ["cl-nagoya/ruri-v3-reranker-310m"]


class ApiUser(HttpUser):
    """
    A user that simulates requests to the embedding and rerank APIs.

    How to run this test:
    1. Make sure the FastAPI server is running.
       (e.g., poetry run uvicorn src.app.main:app --port 8000)
    2. Run Locust from the command line:
       poetry run locust -f locustfile.py --host http://localhost:8000
    3. Open your web browser to http://localhost:8089 and start the test.
    """
    wait_time = between(1, 5)  # Users wait 1-5 seconds between tasks

    @task(3)
    def get_embeddings(self):
        """Task to call the /v1/embeddings endpoint."""
        input_data = random.choice(EMBEDDING_INPUTS)
        input_type = random.choice(["query", "document", "classification", "clustering", "sts", None])
        
        payload = {
            "input": input_data,
            "model": random.choice(EMBEDDING_MODELS),
            "input_type": input_type,
            "apply_ruri_prefix": random.choice([True, False])
        }
        self.client.post("/v1/embeddings", json=payload, name="/v1/embeddings")

    @task(1)
    def get_rerank(self):
        """Task to call the /v1/rerank endpoint."""
        # Select 3 random documents for reranking
        documents = random.sample(RERANK_DOCS, 3)

        payload = {
            "query": random.choice(RERANK_QUERIES),
            "documents": documents,
            "model": RERANK_MODELS[0],
            "top_n": random.choice([None, 1, 2]),
            "return_documents": random.choice([True, False])
        }
        self.client.post("/v1/rerank", json=payload, name="/v1/rerank")
