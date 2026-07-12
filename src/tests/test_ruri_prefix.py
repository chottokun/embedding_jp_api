from app.main import _determine_ruri_prefix
from app.schemas import EmbeddingRequest
from app.config import RURI_PREFIX_MAP


def test_determine_ruri_prefix_non_ruri_model():
    # Model without "ruri-v3" in its name should return an empty prefix
    request = EmbeddingRequest(
        input="test",
        model="some-other-model",
        input_type="query",
        apply_ruri_prefix=True,
    )
    assert _determine_ruri_prefix(request) == ""


def test_determine_ruri_prefix_ruri_with_mapped_input_types():
    # Valid input_type mapped in RURI_PREFIX_MAP
    for input_type, expected_prefix in RURI_PREFIX_MAP.items():
        request = EmbeddingRequest(
            input="test",
            model="cl-nagoya/ruri-v3-30m",
            input_type=input_type,
            apply_ruri_prefix=False,  # Even if False, mapped input_type should take precedence
        )
        assert _determine_ruri_prefix(request) == expected_prefix


def test_determine_ruri_prefix_ruri_unmapped_input_type_apply_false():
    # Unmapped input_type with apply_ruri_prefix=False should return ""
    request = EmbeddingRequest(
        input="test",
        model="cl-nagoya/ruri-v3-30m",
        input_type="unmapped_type",
        apply_ruri_prefix=False,
    )
    assert _determine_ruri_prefix(request) == ""


def test_determine_ruri_prefix_ruri_unmapped_input_type_apply_true_str():
    # Unmapped input_type with apply_ruri_prefix=True and string input should return query prefix
    request = EmbeddingRequest(
        input="test",
        model="cl-nagoya/ruri-v3-30m",
        input_type="unmapped_type",
        apply_ruri_prefix=True,
    )
    assert _determine_ruri_prefix(request) == RURI_PREFIX_MAP["query"]


def test_determine_ruri_prefix_ruri_unmapped_input_type_apply_true_list():
    # Unmapped input_type with apply_ruri_prefix=True and list input should return document prefix
    request = EmbeddingRequest(
        input=["test"],
        model="cl-nagoya/ruri-v3-30m",
        input_type="unmapped_type",
        apply_ruri_prefix=True,
    )
    assert _determine_ruri_prefix(request) == RURI_PREFIX_MAP["document"]


def test_determine_ruri_prefix_ruri_no_input_type_apply_false():
    # No input_type and apply_ruri_prefix=False should return ""
    request = EmbeddingRequest(
        input="test",
        model="cl-nagoya/ruri-v3-30m",
        input_type=None,
        apply_ruri_prefix=False,
    )
    assert _determine_ruri_prefix(request) == ""


def test_determine_ruri_prefix_ruri_no_input_type_apply_true_str():
    # No input_type, apply_ruri_prefix=True, and string input should return query prefix
    request = EmbeddingRequest(
        input="test",
        model="cl-nagoya/ruri-v3-30m",
        input_type=None,
        apply_ruri_prefix=True,
    )
    assert _determine_ruri_prefix(request) == RURI_PREFIX_MAP["query"]


def test_determine_ruri_prefix_ruri_no_input_type_apply_true_list():
    # No input_type, apply_ruri_prefix=True, and list input should return document prefix
    request = EmbeddingRequest(
        input=["test"],
        model="cl-nagoya/ruri-v3-30m",
        input_type=None,
        apply_ruri_prefix=True,
    )
    assert _determine_ruri_prefix(request) == RURI_PREFIX_MAP["document"]
