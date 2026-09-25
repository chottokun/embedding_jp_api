import pytest
from unittest.mock import MagicMock, patch
import torch
import math

from app.services.logit_gate import LogitGateService
from app.config import (
    LOGIT_GATE_POS_TOKENS,
    LOGIT_GATE_NEG_TOKENS,
    LOGIT_GATE_MAX_DOC_CHARS,
    LOGIT_GATE_BATCH_SIZE,
)


@pytest.fixture
def mock_model_wrapper():
    wrapper = MagicMock()
    wrapper.device = "cpu"
    wrapper.tokenizer = MagicMock()
    # Mock tokenizer encode to return some dummy ids
    wrapper.tokenizer.encode.side_effect = lambda text, **kwargs: [
        ord(c) for c in text.strip()
    ]
    wrapper.tokenizer.apply_chat_template.side_effect = lambda msgs, **kwargs: (
        "\n".join([m["content"] for m in msgs])
    )

    # Mock the __call__ of tokenizer for batch processing
    def mock_tokenizer_call(texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        batch_size = len(texts)

        class BatchEncoding(dict):
            def to(self, device):
                return self

        return BatchEncoding({"input_ids": torch.zeros(batch_size, 10)})

    wrapper.tokenizer.side_effect = mock_tokenizer_call

    wrapper.model = MagicMock()
    # Mock model output: batch_size, seq_len, vocab_size
    mock_output = MagicMock()

    # Pre-fill logits such that POS tokens have higher values than NEG tokens
    # Assume pos_tokens are 'Yes', 'yes', 'はい'.
    # Because of 'はい', ord('は') can be up to ~12356, so we need a larger vocab size.
    VOCAB_SIZE = 20000
    torch.randn(2, 10, VOCAB_SIZE)  # batch of 2, seq_len of 10

    # We will overwrite the model to return this on call
    def mock_forward(**kwargs):
        input_ids = kwargs.get("input_ids", torch.zeros(1, 10))
        batch_size = (
            input_ids.shape[0]
            if isinstance(input_ids, torch.Tensor)
            else len(input_ids)
        )
        if hasattr(input_ids, "shape") and len(input_ids.shape) == 1:
            batch_size = 1  # edge case handling if list of size 1

        # Return logits where POS tokens are highly probable
        logits = torch.zeros(batch_size, 10, VOCAB_SIZE)

        # Determine pos/neg token ids based on our dummy tokenizer
        pos_ids = [ord(c) for c in "Yes".strip()] + [ord(c) for c in "yes".strip()]
        neg_ids = [ord(c) for c in "No".strip()] + [ord(c) for c in "no".strip()]

        for b in range(batch_size):
            for pid in pos_ids:
                if pid < VOCAB_SIZE:
                    logits[b, -1, pid] = 5.0  # high probability
            for nid in neg_ids:
                if nid < VOCAB_SIZE:
                    logits[b, -1, nid] = -5.0  # low probability

        mock_output.logits = logits
        return mock_output

    wrapper.model.side_effect = mock_forward
    return wrapper


def test_logit_gate_init(mock_model_wrapper):
    service = LogitGateService(mock_model_wrapper)
    assert len(service.pos_token_ids) > 0
    assert len(service.neg_token_ids) > 0
    assert service.baseline_margin == 0.0


def test_build_prompt(mock_model_wrapper):
    service = LogitGateService(mock_model_wrapper)
    query = "What is the capital of France?"
    doc = "Paris is the capital."

    # Since we mocked apply_chat_template to just join content:
    prompt = service._build_prompt(query, doc)
    assert "What is the capital of France?" in prompt
    assert "Paris is the capital." in prompt


def test_build_prompt_truncation(mock_model_wrapper):
    service = LogitGateService(mock_model_wrapper)
    query = "Q"
    doc = "A" * (LOGIT_GATE_MAX_DOC_CHARS + 100)
    prompt = service._build_prompt(query, doc)

    # It should truncate doc to LOGIT_GATE_MAX_DOC_CHARS
    assert len(doc[:LOGIT_GATE_MAX_DOC_CHARS]) == LOGIT_GATE_MAX_DOC_CHARS
    assert "A" * LOGIT_GATE_MAX_DOC_CHARS in prompt
    assert "A" * (LOGIT_GATE_MAX_DOC_CHARS + 1) not in prompt


def test_calibrate(mock_model_wrapper):
    service = LogitGateService(mock_model_wrapper)
    baseline = service.calibrate()

    # Since we mocked the logits, z_pos should be around log(len(pos_ids) * e^5) = 5 + log(len)
    # and z_neg should be around log(len(neg_ids) * e^-5) = -5 + log(len)
    # baseline = z_pos - z_neg ~= 10
    assert baseline > 0
    assert service.baseline_margin == baseline


def test_predict_margins(mock_model_wrapper):
    service = LogitGateService(mock_model_wrapper)
    # Calibrate first to set baseline
    service.calibrate()

    docs = ["Doc 1", "Doc 2", "Doc 3"]

    with patch.object(
        mock_model_wrapper, "model", side_effect=mock_model_wrapper.model
    ):
        results = service.predict_margins("Test query", docs)

        assert len(results) == 3
        for i, res in enumerate(results):
            assert res["document_index"] == i
            assert "logit_margin" in res
            assert "sufficiency_prob" in res
            assert res["text"] == docs[i]

            # Since we mock the same logits for calibrate and predict, delta_z should be 0
            # which means p_sufficient = sigmoid(0) = 0.5, and binary entropy should be 1.0
            assert math.isclose(res["logit_margin"], 0.0, abs_tol=1e-5)
            assert math.isclose(res["sufficiency_prob"], 0.5, abs_tol=1e-5)
            assert "entropy" in res
            assert math.isclose(res["entropy"], 1.0, abs_tol=1e-5)


def test_binary_entropy():
    from app.services.logit_gate import _binary_entropy

    # Maximum uncertainty at p=0.5
    assert math.isclose(_binary_entropy(0.5), 1.0, abs_tol=1e-5)
    # Zero uncertainty at boundaries
    assert math.isclose(_binary_entropy(0.0), 0.0, abs_tol=1e-4)
    assert math.isclose(_binary_entropy(1.0), 0.0, abs_tol=1e-4)
    # Intermediate values
    assert 0.0 < _binary_entropy(0.8) < 1.0


def test_mini_batching(mock_model_wrapper):
    service = LogitGateService(mock_model_wrapper)

    # Generate 20 documents, batch size is default LOGIT_GATE_BATCH_SIZE (e.g., 8)
    docs = [f"Doc {i}" for i in range(20)]

    with patch.object(
        service, "model", side_effect=mock_model_wrapper.model
    ) as mock_model_call:
        results = service.predict_margins("Query", docs)

        assert len(results) == 20
        # Expected calls = ceil(20 / 8) = 3
        expected_calls = math.ceil(20 / LOGIT_GATE_BATCH_SIZE)
        assert mock_model_call.call_count == expected_calls


def test_config_loading():
    # Test that config variables are loaded (using defaults or env vars)
    assert isinstance(LOGIT_GATE_POS_TOKENS, list)
    assert isinstance(LOGIT_GATE_NEG_TOKENS, list)
    assert isinstance(LOGIT_GATE_MAX_DOC_CHARS, int)
    assert isinstance(LOGIT_GATE_BATCH_SIZE, int)
    assert len(LOGIT_GATE_POS_TOKENS) > 0
    assert len(LOGIT_GATE_NEG_TOKENS) > 0


@patch("app.models.unload_model")
def test_model_unloading(mock_unload):
    mock_unload.return_value = (["Qwen/Qwen2.5-1.5B-Instruct"], 1024)
    from app.models import unload_model

    unloaded, memory = unload_model("Qwen/Qwen2.5-1.5B-Instruct")
    assert "Qwen/Qwen2.5-1.5B-Instruct" in unloaded
    assert memory == 1024
