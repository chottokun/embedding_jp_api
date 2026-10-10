import os
import random
import io
import base64
import wave
import numpy as np
from PIL import Image

try:
    import av
except ImportError:
    av = None

from locust import HttpUser, task, between


# ---------------------------------------------------------------------------
# Dummy Data Generation
# ---------------------------------------------------------------------------
def generate_dummy_image_b64() -> str:
    img = Image.new(
        "RGB",
        (32, 32),
        color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def generate_dummy_audio_b64() -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        # 0.1 seconds of audio
        w.writeframes(b"\x00" * 3200)
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def generate_dummy_video_b64() -> str:
    if not av:
        return ""
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    stream = container.add_stream("h264", rate=10)
    stream.width = 64
    stream.height = 64
    stream.pix_fmt = "yuv420p"

    for i in range(5):
        img_data = np.zeros((64, 64, 3), dtype=np.uint8)
        img_data[:, :] = [random.randint(0, 255), 100, 200]
        frame = av.VideoFrame.from_ndarray(img_data, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    return "data:video/mp4;base64," + base64.b64encode(buf.getvalue()).decode()


# Pre-generate some dummy media to save CPU during load test
DUMMY_IMAGE = generate_dummy_image_b64()
DUMMY_AUDIO = generate_dummy_audio_b64()
DUMMY_VIDEO = generate_dummy_video_b64()

# ---------------------------------------------------------------------------
# Sample Data Pool
# ---------------------------------------------------------------------------
TEXT_INPUTS = [
    "今日の天気は晴れです。",
    "最新のAI技術について教えてください。",
    "このドキュメントを要約して。",
    "自然言語処理とは何ですか？",
    "Locustを用いた負荷テストの手法とベストプラクティス",
    "The quick brown fox jumps over the lazy dog.",
]

BATCH_TEXT_INPUTS = [
    ["これは最初の文書です。", "これは2番目の文書で、少し長いです。", "そして3番目。"],
    ["日本の首都はどこですか？", "東京です。", "富士山は日本で一番高い山です。"],
]

RERANK_QUERIES = ["AIの未来について", "日本の首都", "猫の生態"]
RERANK_DOCS = [
    "これは猫についての文章です。",
    "人工知能は今後の社会を大きく変えるでしょう。",
    "日本の首都は東京です。",
    "犬は人間の最良の友です。",
    "機械学習はAIのサブセットです。",
]

# Models (could be parameterized but these are standard supported)
TEXT_EMBEDDING_MODELS = ["cl-nagoya/ruri-v3-310m"]
GEMMA2_MODELS = ["google/embeddinggemma-2"]
MULTIMODAL_MODELS = ["bge-visualized-m3"]
RERANK_MODELS = ["cl-nagoya/ruri-v3-reranker-310m"]


class ApiLoadTestUser(HttpUser):
    """
    Comprehensive load testing user.
    """

    # Think time between requests (adjust based on load test goals)
    wait_time = between(0.05, 0.5)

    def on_start(self):
        self.headers = {}
        api_key = os.getenv("API_KEY")
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    @task(5)
    def test_text_embedding_single(self):
        """Standard single text embedding."""
        payload = {
            "input": random.choice(TEXT_INPUTS),
            "model": random.choice(TEXT_EMBEDDING_MODELS),
        }
        self.client.post(
            "/v1/embeddings",
            json=payload,
            headers=self.headers,
            name="/v1/embeddings (single)",
        )

    @task(3)
    def test_text_embedding_batch(self):
        """Batch text embedding."""
        payload = {
            "input": random.choice(BATCH_TEXT_INPUTS),
            "model": random.choice(TEXT_EMBEDDING_MODELS),
        }
        self.client.post(
            "/v1/embeddings",
            json=payload,
            headers=self.headers,
            name="/v1/embeddings (batch)",
        )

    @task(2)
    def test_text_embedding_ruri_prefix(self):
        """Ruri text embedding with task prefix."""
        payload = {
            "input": random.choice(TEXT_INPUTS),
            "model": random.choice(TEXT_EMBEDDING_MODELS),
            "input_type": random.choice(["query", "document"]),
            "apply_ruri_prefix": True,
        }
        self.client.post(
            "/v1/embeddings",
            json=payload,
            headers=self.headers,
            name="/v1/embeddings (ruri prefix)",
        )

    @task(2)
    def test_text_embedding_gemma2(self):
        """Gemma-2 text embedding with instructions."""
        payload = {
            "input": random.choice(TEXT_INPUTS),
            "model": GEMMA2_MODELS[0],
            "input_type": random.choice(["query", "document"]),
        }
        self.client.post(
            "/v1/embeddings",
            json=payload,
            headers=self.headers,
            name="/v1/embeddings (gemma-2)",
        )

    @task(2)
    def test_text_embedding_matryoshka_base64(self):
        """Text embedding with dimensionality reduction (256d) and base64 encoding format."""
        payload = {
            "input": random.choice(TEXT_INPUTS),
            "model": random.choice(TEXT_EMBEDDING_MODELS),
            "dimensions": 256,
            "encoding_format": "base64",
        }
        self.client.post(
            "/v1/embeddings",
            json=payload,
            headers=self.headers,
            name="/v1/embeddings (matryoshka/b64)",
        )

    @task(3)
    def test_multimodal_image_bge(self):
        """Multimodal image + text embedding (BGE-M3 Visualized format)."""
        payload = {
            "model": MULTIMODAL_MODELS[0],
            "input": {
                "text": random.choice(TEXT_INPUTS),
                "image_url": DUMMY_IMAGE,
            },
        }
        self.client.post(
            "/v1/embeddings",
            json=payload,
            headers=self.headers,
            name="/v1/embeddings (image bge-m3)",
        )

    @task(2)
    def test_multimodal_gemma2_openai_format(self):
        """Multimodal text+image using OpenAI content part array (Gemma-2)."""
        payload = {
            "model": GEMMA2_MODELS[0],
            "input": [
                {"type": "text", "text": random.choice(TEXT_INPUTS)},
                {"type": "image_url", "image_url": {"url": DUMMY_IMAGE}},
            ],
        }
        self.client.post(
            "/v1/embeddings",
            json=payload,
            headers=self.headers,
            name="/v1/embeddings (image gemma2)",
        )

    @task(1)
    def test_multimodal_audio_gemma2(self):
        """Audio multimodal embedding (Gemma-2)."""
        payload = {
            "model": GEMMA2_MODELS[0],
            "input": [
                {"type": "text", "text": "Describe this audio"},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": DUMMY_AUDIO.split(",")[-1],
                        "format": "wav",
                    },
                },
            ],
        }
        self.client.post(
            "/v1/embeddings",
            json=payload,
            headers=self.headers,
            name="/v1/embeddings (audio gemma2)",
        )

    @task(1)
    def test_multimodal_video_gemma2(self):
        """Video multimodal embedding (Gemma-2)."""
        if not DUMMY_VIDEO:
            return
        payload = {
            "model": GEMMA2_MODELS[0],
            "input": [
                {"type": "text", "text": "What is in this video?"},
                {"type": "video_url", "video_url": {"url": DUMMY_VIDEO}},
            ],
        }
        self.client.post(
            "/v1/embeddings",
            json=payload,
            headers=self.headers,
            name="/v1/embeddings (video gemma2)",
        )

    @task(4)
    def test_rerank(self):
        """Reranking documents against a query."""
        docs = random.sample(RERANK_DOCS, k=random.randint(2, 4))
        payload = {
            "model": random.choice(RERANK_MODELS),
            "query": random.choice(RERANK_QUERIES),
            "documents": docs,
            "top_n": random.choice([None, 1, 2]),
            "return_documents": random.choice([True, False]),
        }
        self.client.post(
            "/v1/rerank", json=payload, headers=self.headers, name="/v1/rerank"
        )

    @task(1)
    def check_healthz(self):
        """Liveness probe."""
        self.client.get("/healthz", name="/healthz")

    @task(1)
    def check_ready(self):
        """Readiness probe."""
        self.client.get("/ready", name="/ready")
