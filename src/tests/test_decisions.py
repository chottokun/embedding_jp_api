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


def test_decisions_explicit_empty_options_and_levels_array_refusal():
    """
    Adversarial test: Passing options=[] or levels=[] explicitly in JSON should return RefusalAnswer
    rather than crashing or failing schema validation unexpectedly.
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
                    "name": "empty_list_choice",
                    "instructions": "分類",
                    "options": [],
                },
                {
                    "type": "score",
                    "name": "empty_list_score",
                    "instructions": "評価スコア",
                    "levels": [],
                },
            ],
        }

        response = client.post("/v1/decisions", json=request_payload)
        assert response.status_code == 200
        answers = response.json()["answers"]
        assert answers[0]["type"] == "refusal"
        assert answers[1]["type"] == "refusal"
    finally:
        app.dependency_overrides.clear()


def test_decisions_service_real_logic_with_mock_model():
    """
    Tests the concrete DecisionsService logic (predicate, choice with ASCII boost, score, refusal)
    without calling calibrate() unnecessarily on every request.
    """
    from unittest.mock import MagicMock, patch
    from app.services.decisions import DecisionsService

    mock_model = MagicMock()
    mock_model.tokenizer_lock = MagicMock()
    mock_model.lock = MagicMock()

    service = DecisionsService(model_loader=lambda name: mock_model)

    app.dependency_overrides[get_decisions_service] = lambda: service

    try:
        with patch("app.services.decisions.LogitGateService") as mock_gate_cls:
            mock_gate = MagicMock()

            def fake_predict_margins(query: str, documents: list[str]):
                if "Option B" in query:
                    return [{"logit_margin": 2.0, "sufficiency_prob": 0.88}]
                if "Option A" in query:
                    return [{"logit_margin": -1.0, "sufficiency_prob": 0.27}]
                if "Score level 2" in query:
                    return [{"logit_margin": 3.0, "sufficiency_prob": 0.95}]
                if "Score level 0" in query:
                    return [{"logit_margin": -2.0, "sufficiency_prob": 0.12}]
                return [{"logit_margin": 1.5, "sufficiency_prob": 0.81}]

            mock_gate.predict_margins.side_effect = fake_predict_margins
            mock_gate_cls.return_value = mock_gate

            request_payload = {
                "model": SUPPORTED_MODEL,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "エラーコード ABC-999 が発生しました。",
                            }
                        ],
                    }
                ],
                "questions": [
                    {
                        "type": "predicate",
                        "name": "has_error_code",
                        "instructions": "エラーコード ABC-999 に関する記述があるか？",
                    },
                    {
                        "type": "choice",
                        "name": "best_category",
                        "instructions": "カテゴリを選択",
                        "options": ["Option A", "Option B"],
                    },
                    {
                        "type": "score",
                        "name": "severity",
                        "instructions": "重要度",
                        "levels": [
                            {"index": 0, "description": "Low"},
                            {"index": 2, "description": "High"},
                        ],
                    },
                ],
            }

            response = client.post("/v1/decisions", json=request_payload)
            assert response.status_code == 200
            answers = response.json()["answers"]

            # calibrate() must NOT be called per request
            mock_gate.calibrate.assert_not_called()

            # 1. Predicate (with ASCII boost on ABC-999, probability should be > sigmoid(1.5) = 0.8175)
            assert answers[0]["type"] == "predicate"
            assert answers[0]["probability"] > 0.85

            # 2. Choice (Option B has higher margin than Option A)
            assert answers[1]["type"] == "choice"
            assert answers[1]["choice"] == "Option B"
            assert (
                answers[1]["probabilities"]["Option B"]
                > answers[1]["probabilities"]["Option A"]
            )
            assert abs(sum(answers[1]["probabilities"].values()) - 1.0) < 1e-5

            # 3. Score (Level 2 has higher margin than Level 0, expected score closer to 2.0)
            assert answers[2]["type"] == "score"
            assert 1.0 < answers[2]["score"] <= 2.0
            assert len(answers[2]["probabilities"]) == 2
            assert abs(sum(answers[2]["probabilities"]) - 1.0) < 1e-5
    finally:
        app.dependency_overrides.clear()


def test_decisions_dos_validation_limits():
    """
    Adversarial test: Empty input list, empty questions list, or overly long strings must return 422.
    """
    from app.config import MAX_INPUT_LENGTH

    # Empty questions list
    res_empty_q = client.post(
        "/v1/decisions",
        json={
            "model": SUPPORTED_MODEL,
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "ok"}]}
            ],
            "questions": [],
        },
    )
    assert res_empty_q.status_code == 422

    # Oversized text exceeding MAX_INPUT_LENGTH
    res_oversized = client.post(
        "/v1/decisions",
        json={
            "model": SUPPORTED_MODEL,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "a" * (MAX_INPUT_LENGTH + 1)}
                    ],
                }
            ],
            "questions": [{"type": "predicate", "name": "q1", "instructions": "check"}],
        },
    )
    assert res_oversized.status_code == 422


def test_decisions_choice_duplicate_options_normalized_probabilities():
    """
    Adversarial test: When duplicate options are passed in a choice question,
    probabilities dictionary values must still sum to 1.0 (deduplicating options in order).
    """
    from unittest.mock import MagicMock, patch
    from app.services.decisions import DecisionsService

    mock_model = MagicMock()
    mock_model.tokenizer_lock = MagicMock()
    mock_model.lock = MagicMock()

    service = DecisionsService(model_loader=lambda name: mock_model)
    app.dependency_overrides[get_decisions_service] = lambda: service

    try:
        with patch("app.services.decisions.LogitGateService") as mock_gate_cls:
            mock_gate = MagicMock()
            mock_gate.predict_margins.return_value = [
                {"logit_margin": 1.0, "sufficiency_prob": 0.73}
            ]
            mock_gate_cls.return_value = mock_gate

            res = client.post(
                "/v1/decisions",
                json={
                    "model": SUPPORTED_MODEL,
                    "input": [
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": "テスト"}],
                        }
                    ],
                    "questions": [
                        {
                            "type": "choice",
                            "name": "dup_choice",
                            "instructions": "選択",
                            "options": ["Option A", "Option A", "Option B"],
                        }
                    ],
                },
            )
            assert res.status_code == 200
            ans = res.json()["answers"][0]
            assert ans["type"] == "choice"
            assert abs(sum(ans["probabilities"].values()) - 1.0) < 1e-5
    finally:
        app.dependency_overrides.clear()


def test_decisions_semaphore_timeout_returns_503():
    """
    Adversarial test: When inference_semaphore times out, /v1/decisions must return 503 Service Unavailable.
    """
    from unittest.mock import patch

    app.dependency_overrides[get_decisions_service] = MockDecisionsService
    try:
        with patch("app.routers.decisions.asyncio.timeout", side_effect=TimeoutError):
            res = client.post(
                "/v1/decisions",
                json={
                    "model": SUPPORTED_MODEL,
                    "input": [
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": "テスト"}],
                        }
                    ],
                    "questions": [
                        {
                            "type": "predicate",
                            "name": "q1",
                            "instructions": "テスト",
                        }
                    ],
                },
            )
            assert res.status_code == 503
            assert "timeout" in res.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()
