import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.config import LOGIT_GATE_MODELS

client = TestClient(app)

SUPPORTED_LOGIT_GATE_MODEL = (
    LOGIT_GATE_MODELS[0] if LOGIT_GATE_MODELS else "Qwen/Qwen2.5-1.5B-Instruct"
)


@pytest.fixture
def mock_logit_gate_model():
    model = MagicMock()
    model.tokenizer_lock = MagicMock()
    model.tokenizer_lock.__enter__ = MagicMock(return_value=None)
    model.tokenizer_lock.__exit__ = MagicMock(return_value=None)
    model.tokenizer = MagicMock()
    model.tokenizer.encode.return_value = [1, 2, 3]
    model.tokenizer.num_special_tokens_to_add.return_value = 0
    return model


@pytest.fixture
def mock_gate_results():
    return [
        {"logit_margin": -1.0, "passed": False},
        {"logit_margin": 2.0, "passed": True},
        {"logit_margin": 0.5, "passed": True},
    ]


@pytest.fixture
def mock_containment_scores():
    return [0.0, 0.8, 1.0]


@patch("app.services.rerank.get_validated_model")
@patch("app.services.rerank.LogitGateService")
@patch("app.services.rerank.AsciiMatcher")
def test_create_rerank_logit_gate(
    mock_matcher_cls,
    mock_gate_cls,
    mock_get_model,
    mock_logit_gate_model,
    mock_gate_results,
    mock_containment_scores,
):
    mock_get_model.return_value = mock_logit_gate_model

    mock_gate_instance = MagicMock()
    mock_gate_instance.predict_margins.return_value = mock_gate_results
    mock_gate_cls.return_value = mock_gate_instance

    mock_matcher_instance = MagicMock()
    mock_matcher_instance.score_documents.return_value = mock_containment_scores
    mock_matcher_cls.return_value = mock_matcher_instance

    query = "test error 404"
    documents = ["no match", "error 404 found", "404 completely"]

    request_payload = {
        "query": query,
        "documents": documents,
        "model": SUPPORTED_LOGIT_GATE_MODEL,
    }

    response = client.post("/v1/rerank", json=request_payload)

    assert response.status_code == 200
    response_json = response.json()
    assert response_json["model"] == SUPPORTED_LOGIT_GATE_MODEL

    results = response_json["data"]
    assert len(results) == 3

    # Doc 1: margin 2.0 + (1.2 * 0.8) = 2.96
    # Doc 2: margin 0.5 + (1.2 * 1.0) = 1.7
    # Doc 0: margin -1.0 + (1.2 * 0.0) = -1.0
    assert results[0]["document"] == 1
    assert results[1]["document"] == 2
    assert results[2]["document"] == 0

    for r in results:
        assert "passed" in r
        assert "logit_margin" in r
        assert "containment_score" in r
        assert "entropy" in r


@patch("app.services.rerank.get_validated_model")
@patch("app.services.rerank.LogitGateService")
@patch("app.services.rerank.AsciiMatcher")
def test_create_rerank_logit_gate_drop_failed(
    mock_matcher_cls,
    mock_gate_cls,
    mock_get_model,
    mock_logit_gate_model,
    mock_gate_results,
    mock_containment_scores,
):
    mock_get_model.return_value = mock_logit_gate_model

    mock_gate_instance = MagicMock()
    mock_gate_instance.predict_margins.return_value = mock_gate_results
    mock_gate_cls.return_value = mock_gate_instance

    mock_matcher_instance = MagicMock()
    mock_matcher_instance.score_documents.return_value = mock_containment_scores
    mock_matcher_cls.return_value = mock_matcher_instance

    request_payload = {
        "query": "query",
        "documents": ["d0", "d1", "d2"],
        "model": SUPPORTED_LOGIT_GATE_MODEL,
        "drop_failed": True,
        "threshold": 0.5,
    }

    response = client.post("/v1/rerank", json=request_payload)

    assert response.status_code == 200
    results = response.json()["data"]

    # only two items should pass based on margins
    # Doc 1 score > 0.5
    # Doc 2 score > 0.5
    # Doc 0 score < 0.5
    assert len(results) == 2
    assert results[0]["document"] == 1
    assert results[1]["document"] == 2
