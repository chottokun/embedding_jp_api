from fastapi.testclient import TestClient
from unittest.mock import patch
import numpy as np
from app.main import app
from app.config import EMBEDDING_MODELS, RURI_PREFIX_MAP

client = TestClient(app)


# Helper to mock model
def setup_mock_model(mock_get_model):
    mock_model = mock_get_model.return_value
    mock_model.encode.return_value = np.array([[0.1, 0.2]])
    # Mock tokenizer
    # Assume 1 char = 1 token for simplicity in testing logic
    mock_model.tokenizer.encode.side_effect = lambda text, add_special_tokens=True: [
        ord(c) for c in text
    ]
    mock_model.tokenizer.decode.side_effect = lambda tokens: "".join(
        [chr(t) for t in tokens]
    )
    mock_model.tokenizer.num_special_tokens_to_add.return_value = 0  # Simplify
    mock_model.max_seq_length = 100  # Default to large enough for basic tests

    def mock_tokenizer_call(text, **kwargs):
        if isinstance(text, str):
            ids = [ord(c) for c in text]
            return {"input_ids": ids}
        elif isinstance(text, list):
            ids = [[ord(c) for c in t] for t in text]
            return {"input_ids": ids}
        return {"input_ids": []}

    mock_model.tokenizer.side_effect = mock_tokenizer_call
    return mock_model


@patch("app.main.get_model")
def test_ruri_prefix_fallback_logic(mock_get_model):
    """
    Verifies the controversial fallback logic:
    String input -> Query prefix
    List input -> Document prefix
    """
    mock_model = setup_mock_model(mock_get_model)
    ruri_model = "cl-nagoya/ruri-v3-30m"
    if ruri_model not in EMBEDDING_MODELS:
        EMBEDDING_MODELS.append(ruri_model)

    # Case 1: String input (should be Query)
    client.post(
        "/v1/embeddings",
        json={"input": "hello", "model": ruri_model, "apply_ruri_prefix": True},
    )

    # Check what was passed to encode
    # "検索クエリ: " is the prefix for query
    query_prefix = RURI_PREFIX_MAP["query"]
    expected_text = f"{query_prefix}hello"
    mock_model.encode.assert_called_with([expected_text])

    # Case 2: List input (should be Document)
    client.post(
        "/v1/embeddings",
        json={"input": ["doc1"], "model": ruri_model, "apply_ruri_prefix": True},
    )

    doc_prefix = RURI_PREFIX_MAP["document"]
    expected_text_doc = f"{doc_prefix}doc1"
    mock_model.encode.assert_called_with([expected_text_doc])


@patch("app.main.get_model")
def test_truncation_logic_preserves_prefix(mock_get_model):
    """
    Verifies that when text is too long, it is truncated,
    BUT the prefix is preserved at the beginning.
    """
    mock_model = setup_mock_model(mock_get_model)
    # limit is 10 tokens (chars)
    # prefix is "P:" (length 2)
    # text is "ABCDEFGHIJK" (length 11)
    # Total 13 > 10.
    # Should truncate text to fit: 10 - 2 = 8 chars.
    # Result should be "P:ABCDEFGH"

    ruri_model = "cl-nagoya/ruri-v3-30m"

    # Override tokenizer behavior for this test to be predictable
    # We use a simple length based mock

    # We cheat a bit by setting RURI_PREFIX_MAP temporarily or just relying on known one
    # query prefix is "検索クエリ: " which is 6 chars.
    # Let's use input_type to force a specific prefix if we want, or just use query

    long_text = "A" * 20

    # We need to make sure the prefix + text > max_seq_length
    # Mock max_seq_length = 10
    mock_model.max_seq_length = 10

    # The prefix "検索クエリ: " is 7 chars long.
    # So available for text = 10 - 7 = 3 chars.
    # So expected is "検索クエリ: AAA"

    client.post(
        "/v1/embeddings",
        json={
            "input": long_text,
            "model": ruri_model,
            "apply_ruri_prefix": True,
            "input_type": "query",
        },
    )

    expected_prefix = RURI_PREFIX_MAP["query"]
    expected_text = f"{expected_prefix}AAA"

    # We need to verify what was passed to encode.
    # Note: encode is called with a list of strings.
    args, _ = mock_model.encode.call_args
    passed_text = args[0][0]

    assert passed_text == expected_text
    assert len(passed_text) == 10  # Should meet the limit
