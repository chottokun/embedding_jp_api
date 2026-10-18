from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app

client = TestClient(app)


def test_decisions_success_all_question_types():
    """
    Tests POST /v1/decisions with valid gpt-6-luna model and predicate, choice, score, refusal question types.
    """
    request_payload = {
        "model": "gpt-6-luna",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "テストテキスト"}
                ]
            }
        ],
        "questions": [
            {
                "type": "predicate",
                "name": "test_bool",
                "instructions": "条件に合うか？"
            },
            {
                "type": "choice",
                "name": "test_class",
                "instructions": "分類",
                "options": ["Option A", "Option B"]
            },
            {
                "type": "score",
                "name": "test_score",
                "instructions": "評価スコア",
                "levels": [
                    {"index": 0, "description": "Low"},
                    {"index": 1, "description": "Medium"},
                    {"index": 2, "description": "High"}
                ]
            },
            {
                "type": "unknown_type",
                "name": "test_unknown",
                "instructions": "未知のタイプ"
            }
        ]
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
    assert abs(answers[2]["probabilities"][0] - 1/3) < 1e-5

    # 4. Refusal for unsupported question type
    assert answers[3]["name"] == "test_unknown"
    assert answers[3]["type"] == "refusal"
    assert "Unsupported question type: unknown_type" in answers[3]["refusal"]


def test_decisions_unsupported_model():
    """
    Tests POST /v1/decisions with unsupported model name (must raise 400 Bad Request).
    """
    request_payload = {
        "model": "gpt-4o",
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "test"}]
            }
        ],
        "questions": [
            {"type": "predicate", "name": "test_bool", "instructions": "test"}
        ]
    }

    response = client.post("/v1/decisions", json=request_payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "Only 'gpt-6-luna' model is supported in Decisions API."


def test_decisions_invalid_request_schema():
    """
    Tests POST /v1/decisions with missing required fields (returns 422 Unprocessable Entity).
    """
    request_payload = {
        "model": "gpt-6-luna",
        # Missing 'input' and 'questions'
    }

    response = client.post("/v1/decisions", json=request_payload)
    assert response.status_code == 422
