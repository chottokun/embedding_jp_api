import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from app.schemas import RerankRequest
from app.services.rerank import RerankService, _calculate_rerank_tokens
from app.config import RERANK_MODELS, LOGIT_GATE_MODELS


@pytest.mark.anyio
async def test_rerank_tei_proxy_return_documents_toggle():
    """Verify TEI proxy handling when return_documents is True vs False."""
    mock_proxy = AsyncMock(
        return_value=[
            {"index": 1, "score": 0.85},
            {"index": 0, "score": 0.42},
        ]
    )
    service = RerankService(proxy_to_tei_func=mock_proxy)

    req_with_docs = RerankRequest(
        query="python rerank",
        documents=["doc A", "doc B"],
        model=RERANK_MODELS[0],
        return_documents=True,
    )

    with (
        patch("app.services.rerank.RERANK_TEI_URL", "http://tei-rerank"),
        patch("app.main._proxy_to_tei", mock_proxy),
    ):
        res_with_docs = await service.create_rerank(req_with_docs)

    assert len(res_with_docs.data) == 2
    assert res_with_docs.data[0].document == 1
    assert res_with_docs.data[0].text == "doc B"
    assert res_with_docs.data[1].document == 0
    assert res_with_docs.data[1].text == "doc A"

    req_no_docs = RerankRequest(
        query="python rerank",
        documents=["doc A", "doc B"],
        model=RERANK_MODELS[0],
        return_documents=False,
    )

    with (
        patch("app.services.rerank.RERANK_TEI_URL", "http://tei-rerank"),
        patch("app.main._proxy_to_tei", mock_proxy),
    ):
        res_no_docs = await service.create_rerank(req_no_docs)

    assert len(res_no_docs.data) == 2
    assert res_no_docs.data[0].text is None
    assert res_no_docs.data[1].text is None


@pytest.mark.anyio
async def test_rerank_logit_gate_ascii_boost_disabled():
    """Verify Logit Gate behavior when use_ascii_boost=False."""
    mock_model = MagicMock()
    mock_model.tokenizer_lock = MagicMock()
    mock_model.lock = MagicMock()
    mock_model.tokenizer.encode.side_effect = lambda text, **kw: [1, 2, 3]

    mock_loader = MagicMock(return_value=mock_model)
    service = RerankService(model_loader=mock_loader)

    with (
        patch(
            "app.services.rerank.LogitGateService.predict_margins",
            return_value=[{"logit_margin": 0.0}, {"logit_margin": 1.0}],
        ),
        patch("app.services.rerank.RERANK_TEI_URL", None),
    ):
        req = RerankRequest(
            query="test query with ABC-123",
            documents=["doc with ABC-123", "another doc"],
            model=LOGIT_GATE_MODELS[0],
            use_ascii_boost=False,
        )
        res = await service.create_rerank(req)

    assert len(res.data) == 2
    for d in res.data:
        assert d.logit_margin in (0.0, 1.0)


@pytest.mark.anyio
async def test_rerank_logit_gate_custom_threshold_and_drop_failed():
    """Verify Logit Gate threshold filtering when drop_failed is True."""
    mock_model = MagicMock()
    mock_model.tokenizer_lock = MagicMock()
    mock_model.lock = MagicMock()
    mock_model.tokenizer.encode.side_effect = lambda text, **kw: [1, 2, 3]

    mock_loader = MagicMock(return_value=mock_model)
    service = RerankService(model_loader=mock_loader)

    with (
        patch(
            "app.services.rerank.LogitGateService.predict_margins",
            return_value=[{"logit_margin": 2.0}, {"logit_margin": -2.0}],
        ),
        patch(
            "app.services.rerank.AsciiMatcher.score_documents",
            return_value=[0.0, 0.0],
        ),
        patch("app.services.rerank.RERANK_TEI_URL", None),
    ):
        req = RerankRequest(
            query="query",
            documents=["high score doc", "low score doc"],
            model=LOGIT_GATE_MODELS[0],
            threshold=0.5,
            drop_failed=True,
        )
        res = await service.create_rerank(req)

    assert len(res.data) == 1
    assert res.data[0].document == 0
    assert res.data[0].passed is True


def test_calculate_rerank_tokens_fallback_on_exception():
    """Verify _calculate_rerank_tokens falls back to per-document tokenization when batch encoding raises Exception."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_model.tokenizer = mock_tokenizer
    mock_model.tokenizer_lock = MagicMock()

    mock_tokenizer.encode.side_effect = lambda text, add_special_tokens=False: [1, 2]
    mock_tokenizer.num_special_tokens_to_add.return_value = 2
    mock_tokenizer.side_effect = RuntimeError("Batch encoding failed")

    usage = _calculate_rerank_tokens(mock_model, "query string", ["doc one", "doc two"])
    assert usage.prompt_tokens == 12
    assert usage.total_tokens == 12


@pytest.mark.anyio
async def test_rerank_unsupported_model_exception():
    """Verify HTTPException is raised for invalid model."""
    service = RerankService()
    req = RerankRequest(
        query="query",
        documents=["doc"],
        model="non-existent-model",
    )
    with pytest.raises(HTTPException) as exc_info:
        await service.create_rerank(req)

    assert exc_info.value.status_code == 400
    assert "not found for reranks" in exc_info.value.detail
