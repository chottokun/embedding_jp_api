import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import HTTPException

from app.schemas import EmbeddingRequest
from app.services.embedding import (
    EmbeddingService,
    _format_embeddings_batch,
    _determine_ruri_prefix,
    _apply_prefix,
    _determine_gemma2_prompt,
    _normalize_raw_inputs,
)


def test_format_embeddings_batch_matryoshka_and_base64():
    # 4D vector, reduce to 2D
    vectors = np.array([[3.0, 4.0, 0.0, 0.0]], dtype=np.float32)

    # Matryoshka reduction to 2 dimensions (vector [3, 4] normalized to [0.6, 0.8])
    reduced = _format_embeddings_batch(vectors, dimensions=2, encoding_format="float")
    assert len(reduced[0]) == 2
    np.testing.assert_allclose(reduced[0], [0.6, 0.8], rtol=1e-5)

    # Base64 format
    b64_res = _format_embeddings_batch(vectors, dimensions=2, encoding_format="base64")
    assert isinstance(b64_res[0], str)


def test_determine_ruri_prefix_and_apply_prefix():
    req = EmbeddingRequest(
        input="テスト",
        model="cl-nagoya/ruri-v3-30m",
        input_type="query",
    )
    prefix = _determine_ruri_prefix(req)
    assert prefix == "検索クエリ: "

    res = _apply_prefix(["テスト"], prefix)
    assert res == ["検索クエリ: テスト"]

    # Don't duplicate if prefix already exists
    res_dup = _apply_prefix(["検索クエリ: テスト"], prefix)
    assert res_dup == ["検索クエリ: テスト"]


def test_determine_gemma2_prompt():
    req = EmbeddingRequest(
        input="test",
        model="google/embeddinggemma-2-9b",
        input_type="document",
    )
    prompt = _determine_gemma2_prompt(req)
    assert prompt == "Document"


def test_normalize_raw_inputs():
    assert _normalize_raw_inputs([]) == []
    assert _normalize_raw_inputs("single text") == ["single text"]
    assert _normalize_raw_inputs(["a", "b"]) == ["a", "b"]


@pytest.mark.anyio
async def test_embedding_service_unsupported_model():
    service = EmbeddingService()
    req = EmbeddingRequest(input="test", model="unsupported-model-x")

    with pytest.raises(HTTPException) as excinfo:
        await service.create_embeddings(req)
    assert excinfo.value.status_code == 400
    assert "not found for embeddings" in excinfo.value.detail


@pytest.mark.anyio
async def test_embedding_service_tei_proxy():
    mock_proxy = AsyncMock(return_value={
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
        "model": "cl-nagoya/ruri-v3-30m",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    })
    service = EmbeddingService(proxy_to_tei_func=mock_proxy)

    req = EmbeddingRequest(input="test query", model="cl-nagoya/ruri-v3-30m")

    with patch("app.main.EMBEDDING_TEI_URL", "http://tei-server:8080"):
        with patch("app.main._proxy_to_tei", mock_proxy):
            resp = await service.create_embeddings(req)
            assert resp.model == "cl-nagoya/ruri-v3-30m"
            assert len(resp.data) == 1
            assert resp.data[0].embedding == [0.1, 0.2, 0.3]


@pytest.mark.anyio
async def test_embedding_service_text_embedding_success():
    mock_model = MagicMock()
    mock_model.max_seq_length = 8192
    mock_model.tokenizer.num_special_tokens_to_add.return_value = 2
    mock_model.tokenizer.return_value = {"input_ids": [[1, 2, 3]]}
    mock_model.encode.return_value = np.array([[0.5, 0.6, 0.7]])

    service = EmbeddingService(model_loader=lambda name: mock_model)
    req = EmbeddingRequest(input="hello world", model="cl-nagoya/ruri-v3-30m")

    with patch("app.main.EMBEDDING_TEI_URL", None):
        resp = await service.create_embeddings(req)
        assert resp.model == "cl-nagoya/ruri-v3-30m"
        assert resp.data[0].embedding == [0.5, 0.6, 0.7]
        assert resp.usage.total_tokens == 5
