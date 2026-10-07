import pytest
from unittest.mock import patch, MagicMock
from app.models import EmbeddingGemma2Model
import torch
from app.schemas import EmbeddingRequest
from app.services.embedding import _determine_gemma2_prompt, _format_embedding


@pytest.fixture
def mock_sentence_transformer():
    with patch("app.models.SentenceTransformer") as mock:
        yield mock


def test_embeddinggemma2_init_max_seq_length(mock_sentence_transformer):
    model = EmbeddingGemma2Model(device="cpu", dtype=torch.float32)
    assert model.max_seq_length == 8192
    assert model.model.max_seq_length == 8192


@patch("torch.cuda.is_available", return_value=False)
def test_embeddinggemma2_init_float16_fallback(mock_cuda, mock_sentence_transformer):
    # Simulate CPU fallback
    model_cpu = EmbeddingGemma2Model(device="cpu", dtype=torch.float16)
    assert model_cpu.dtype == torch.float32

    # Simulate GPU fallback
    mock_cuda.return_value = True
    with patch("sentence_transformers.base.model.BaseModel.to") as _:
        model_cuda = EmbeddingGemma2Model(device="cuda", dtype=torch.float16)
        assert model_cuda.dtype == torch.bfloat16


def test_embeddinggemma2_encode_multimodal(mock_sentence_transformer):
    model = EmbeddingGemma2Model(device="cpu", dtype=torch.float32)

    # mock encode
    mock_encode = MagicMock(return_value=torch.tensor([[0.1, 0.2]]))
    model.model.encode = mock_encode

    img = MagicMock()
    items = [("test", None), (None, img), ("test2", img)]

    embeddings = model.encode_multimodal(items, prompt_name="SearchQuery")

    # Check processed items
    expected_items = ["test", {"image": img}, {"text": "test2 <|image|>", "image": img}]

    mock_encode.assert_called_once()
    args, kwargs = mock_encode.call_args
    assert args[0] == expected_items
    assert kwargs["prompt_name"] is None  # Since there are images, prompt is suppressed
    assert kwargs["normalize_embeddings"] is True
    import math

    assert math.isclose(embeddings[0][0], 0.1, abs_tol=1e-5)
    assert math.isclose(embeddings[0][1], 0.2, abs_tol=1e-5)


def test_embeddinggemma2_encode_text_only(mock_sentence_transformer):
    model = EmbeddingGemma2Model(device="cpu", dtype=torch.float32)
    mock_encode = MagicMock(return_value=torch.tensor([[0.1, 0.2]]))
    model.model.encode = mock_encode

    items = [("test", None)]
    model.encode_multimodal(items, prompt_name="SearchQuery")

    args, kwargs = mock_encode.call_args
    assert kwargs["prompt_name"] == "SearchQuery"  # Applied because text-only


def test_determine_gemma2_prompt():
    req = EmbeddingRequest(
        model="google/embeddinggemma-2", input="text", input_type="query"
    )
    assert _determine_gemma2_prompt(req) == "SearchQuery"

    req.input_type = "document"
    assert _determine_gemma2_prompt(req) == "Document"

    req.input_type = "classification"
    assert _determine_gemma2_prompt(req) == "Classification"

    req.input_type = "clustering"
    assert _determine_gemma2_prompt(req) == "Clustering"

    req.input_type = "other"
    assert _determine_gemma2_prompt(req) is None


def test_embeddinggemma2_mrl_and_base64():
    # Test Matryoshka dimensionality reduction and base64 encoding formatting
    # As requested by the user: verify MRL (Matryoshka) and Base64 works.

    # 1. Test dimensionality reduction to 128 and normalisation
    vector = [0.1] * 768
    dim = 128
    formatted_vector = _format_embedding(vector, dim, encoding_format="float")
    assert len(formatted_vector) == dim

    # Ensure L2 normalized
    import math

    norm = math.sqrt(sum(x * x for x in formatted_vector))
    assert math.isclose(norm, 1.0, abs_tol=1e-5)

    # 2. Test base64 encoding format
    import base64
    import struct

    expected_packed = struct.pack(f"<{dim}f", *formatted_vector)
    expected_b64 = base64.b64encode(expected_packed).decode("utf-8")

    b64_vector = _format_embedding(vector, dim, encoding_format="base64")
    assert b64_vector == expected_b64
