from fastapi.testclient import TestClient

from app.config import LOGIT_GATE_MODELS
from app.dependencies.services import get_decisions_service
from app.main import app
from app.services.mock import MockDecisionsService

client = TestClient(app)

SUPPORTED_MODEL = LOGIT_GATE_MODELS[0]


def test_decisions_success_all_question_types():
    """
    Tests POST /v1/decisions with valid model and predicate, choice, score, refusal question types.
    """
    app.dependency_overrides[get_decisions_service] = MockDecisionsService

    try:
        request_payload = {
            "model": SUPPORTED_MODEL,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "テストテキスト"}],
                }
            ],
            "questions": [
                {
                    "type": "predicate",
                    "name": "test_bool",
                    "instructions": "条件に合うか？",
                },
                {
                    "type": "choice",
                    "name": "test_class",
                    "instructions": "分類",
                    "options": ["Option A", "Option B"],
                },
                {
                    "type": "score",
                    "name": "test_score",
                    "instructions": "評価スコア",
                    "levels": [
                        {"index": 0, "description": "Low"},
                        {"index": 1, "description": "Medium"},
                        {"index": 2, "description": "High"},
                    ],
                },
                {
                    "type": "unknown_type",
                    "name": "test_unknown",
                    "instructions": "未知のタイプ",
                },
            ],
        }

        response = client.post("/v1/decisions", json=request_payload)

        assert response.status_code == 200
        res_data = response.json()
        assert "answers" in res_data
        answers = res_data["answers"]
        assert len(answers) == 4

        # 1. Predicate
        assert answers[0]["name"] == "test_bool"
        assert answers[0]["type"] == "predicate"
        assert answers[0]["probability"] == 0.85

        # 2. Choice
        assert answers[1]["name"] == "test_class"
        assert answers[1]["type"] == "choice"
        assert answers[1]["choice"] == "Option A"
        assert answers[1]["probabilities"] == {"Option A": 1.0, "Option B": 0.0}
        assert answers[1]["confidence"] == 0.95

        # 3. Score
        assert answers[2]["name"] == "test_score"
        assert answers[2]["type"] == "score"
        assert answers[2]["score"] == 1.5
        assert len(answers[2]["probabilities"]) == 3
        assert abs(answers[2]["probabilities"][0] - 1 / 3) < 1e-5

        # 4. Refusal for unsupported question type
        assert answers[3]["name"] == "test_unknown"
        assert answers[3]["type"] == "refusal"
        assert "Unsupported question type: unknown_type" in answers[3]["refusal"]
    finally:
        app.dependency_overrides.clear()


def test_decisions_empty_options_and_levels_refusal():
    """
    Tests that choice with empty options and score with empty levels return RefusalAnswer.
    """
    app.dependency_overrides[get_decisions_service] = MockDecisionsService

    try:
        request_payload = {
            "model": SUPPORTED_MODEL,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "テストテキスト"}],
                }
            ],
            "questions": [
                {
                    "type": "choice",
                    "name": "test_empty_choice",
                    "instructions": "分類",
                    # options omitted / None
                },
                {
                    "type": "score",
                    "name": "test_empty_score",
                    "instructions": "評価スコア",
                    # levels omitted / None
                },
            ],
        }

        response = client.post("/v1/decisions", json=request_payload)

        assert response.status_code == 200
        res_data = response.json()
        answers = res_data["answers"]
        assert len(answers) == 2

        assert answers[0]["type"] == "refusal"
        assert "Choice question requires non-empty options" in answers[0]["refusal"]

        assert answers[1]["type"] == "refusal"
        assert "Score question requires non-empty levels" in answers[1]["refusal"]
    finally:
        app.dependency_overrides.clear()


def test_decisions_unsupported_model():
    """
    Tests POST /v1/decisions with unsupported model name (must raise 400 Bad Request).
    """
    request_payload = {
        "model": "unsupported-model-xyz",
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "test"}],
            }
        ],
        "questions": [
            {"type": "predicate", "name": "test_bool", "instructions": "test"}
        ],
    }

    response = client.post("/v1/decisions", json=request_payload)

    assert response.status_code == 400
    assert "not found" in response.json()["detail"].lower()


def test_decisions_invalid_request_schema():
    """
    Tests POST /v1/decisions with missing required fields (returns 422 Unprocessable Entity).
    """
    request_payload = {
        "model": SUPPORTED_MODEL,
        # Missing 'input' and 'questions'
    }

    response = client.post("/v1/decisions", json=request_payload)
    assert response.status_code == 422
