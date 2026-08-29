from fastapi.testclient import TestClient
from app.main import app, get_embedding_service, get_rerank_service
from app.services import MockEmbeddingService, MockRerankService

client = TestClient(app)
AUTH_HEADERS = {"Authorization": "Bearer test_api_key_secret"}


def test_mock_embedding_service_fast():
    # Override service dependencies with mock implementations
    app.dependency_overrides[get_embedding_service] = lambda: MockEmbeddingService(
        vector_dim=1024
    )
    try:
        response = client.post(
            "/v1/embeddings",
            json={"model": "bge-visualized-m3", "input": "Fast Mock Unit Test"},
            headers=AUTH_HEADERS,
        )
        assert response.status_code == 200
        res = response.json()
        assert res["model"] == "bge-visualized-m3"
        assert len(res["data"]) == 1
        assert len(res["data"][0]["embedding"]) == 1024
        assert res["data"][0]["embedding"][0] == 0.1
    finally:
        app.dependency_overrides.clear()


def test_mock_rerank_service_fast():
    app.dependency_overrides[get_rerank_service] = lambda: MockRerankService()
    try:
        response = client.post(
            "/v1/rerank",
            json={
                "model": "cl-nagoya/ruri-v3-reranker-310m",
                "query": "Fast Query",
                "documents": ["Doc A", "Doc B"],
                "return_documents": True,
            },
            headers=AUTH_HEADERS,
        )
        assert response.status_code == 200
        res = response.json()
        assert res["model"] == "cl-nagoya/ruri-v3-reranker-310m"
        assert len(res["data"]) == 2
        assert res["data"][0]["score"] == 1.0
        assert res["data"][0]["text"] == "Doc A"
    finally:
        app.dependency_overrides.clear()
