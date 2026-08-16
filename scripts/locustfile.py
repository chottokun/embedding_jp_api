import os
import random
from locust import HttpUser, task, between

# --- Sample Data Pool ---
# A small pool of data to be used in the load tests.
EMBEDDING_INPUTS = [
    "今日の天気は晴れです。",
    "最新のAI技術について教えてください。",
    "このドキュメントを要約して。",
    "自然言語処理とは何ですか？",
    ["これは最初の文書です。", "これは2番目の文書で、少し長いです。", "そして3番目。"],
]

RERANK_QUERIES = ["AIの未来について", "日本の首都", "猫の生態"]
RERANK_DOCS = [
    "これは猫についての文章です。",
    "人工知能は今後の社会を大きく変えるでしょう。",
    "日本の首都は東京です。",
    "犬は人間の最良の友です。",
    "機械学習はAIのサブセットです。",
]

# Supported models from our config. We will randomly pick one.
EMBEDDING_MODELS = ["cl-nagoya/ruri-v3-30m", "cl-nagoya/ruri-v3-310m"]
MULTIMODAL_MODELS = ["bge-visualized-m3"]
RERANK_MODELS = ["cl-nagoya/ruri-v3-reranker-310m"]

TINY_PNG_B64 = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class ApiUser(HttpUser):
    """
    A user that simulates requests to the embedding, rerank, and multimodal APIs.
    """

    wait_time = between(0.1, 1.0)

    @task(3)
    def get_embeddings(self):
        """Task to call the /v1/embeddings endpoint with text inputs."""
        input_data = random.choice(EMBEDDING_INPUTS)
        input_type = random.choice(
            ["query", "document", "classification", "clustering", "sts", None]
        )

        payload = {
            "input": input_data,
            "model": random.choice(EMBEDDING_MODELS),
            "input_type": input_type,
            "apply_ruri_prefix": random.choice([True, False]),
        }
        headers = {}
        api_key = os.getenv("API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client.post(
            "/v1/embeddings", json=payload, name="/v1/embeddings-text", headers=headers
        )

    @task(2)
    def get_multimodal_embeddings(self):
        """Task to call /v1/embeddings with multimodal inputs."""
        payload = {
            "model": random.choice(MULTIMODAL_MODELS),
            "input": {
                "text": random.choice(EMBEDDING_INPUTS[0:4]),
                "image_url": TINY_PNG_B64,
            },
        }
        headers = {}
        api_key = os.getenv("API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client.post(
            "/v1/embeddings",
            json=payload,
            name="/v1/embeddings-multimodal",
            headers=headers,
        )

    @task(2)
    def get_rerank(self):
        """Task to call the /v1/rerank endpoint."""
        documents = random.sample(RERANK_DOCS, 3)

        payload = {
            "query": random.choice(RERANK_QUERIES),
            "documents": documents,
            "model": RERANK_MODELS[0],
            "top_n": random.choice([None, 1, 2]),
            "return_documents": random.choice([True, False]),
        }
        headers = {}
        api_key = os.getenv("API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client.post("/v1/rerank", json=payload, name="/v1/rerank", headers=headers)

    @task(1)
    def check_health(self):
        """Task to check health endpoint."""
        self.client.get("/health", name="/health")
