import heapq
from typing import Any, List, Optional
from fastapi import HTTPException

from .base import BaseRerankService
from ..schemas import RerankRequest, RerankResponse, RerankData, Usage
from ..config import RERANK_MODELS


def _calculate_rerank_tokens(model: Any, query: str, documents: List[str]) -> Usage:
    with model.tokenizer_lock:
        tokenizer = model.tokenizer
        q_tokens = len(tokenizer.encode(query, add_special_tokens=False))
        special_tokens = tokenizer.num_special_tokens_to_add(True)

        try:
            from collections.abc import Mapping

            encodings = tokenizer(documents, add_special_tokens=False)
            if not isinstance(encodings, Mapping) or "input_ids" not in encodings:
                raise ValueError("Unexpected tokenizer output format")

            total_tokens = sum(
                q_tokens + len(d_ids) + special_tokens
                for d_ids in encodings["input_ids"]
            )
        except Exception:
            total_tokens = 0
            for doc in documents:
                d_tokens = len(tokenizer.encode(doc, add_special_tokens=False))
                total_tokens += q_tokens + d_tokens + special_tokens

        return Usage(prompt_tokens=total_tokens, total_tokens=total_tokens)


def _sort_and_format_rerank_results(
    results: List[dict], top_n: Optional[int]
) -> List[RerankData]:
    if top_n is not None:
        sorted_results = heapq.nlargest(
            top_n, results, key=lambda x: (x["score"], -x["document"])
        )
    else:
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

    return [RerankData(**result) for result in sorted_results]


def _get_model_or_400(model_name: str) -> Any:
    from app.main import get_model

    if model_name not in RERANK_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not found for reranks.",
        )
    try:
        return get_model(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class RerankService(BaseRerankService):
    """
    Default production implementation of BaseRerankService.
    """

    def __init__(self, proxy_to_tei_func: Optional[Any] = None):
        self.proxy_to_tei_func = proxy_to_tei_func

    async def create_rerank(self, request: RerankRequest) -> RerankResponse:
        import app.main as main_mod

        if request.model not in RERANK_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' not found for reranks.",
            )

        tei_url = getattr(main_mod, "RERANK_TEI_URL", None)
        proxy_func = getattr(main_mod, "_proxy_to_tei", self.proxy_to_tei_func)

        if tei_url and proxy_func:
            tei_results = proxy_func(
                tei_url,
                "/rerank",
                {"query": request.query, "texts": request.documents},
            )
            results = []
            for item in tei_results:
                idx = item["index"]
                score = item["score"]
                result_item = {"document": idx, "score": float(score)}
                if request.return_documents:
                    result_item["text"] = request.documents[idx]
                results.append(result_item)

            response_data = _sort_and_format_rerank_results(results, request.top_n)
            usage = Usage(prompt_tokens=0, total_tokens=0)
            return RerankResponse(
                query=request.query,
                data=response_data,
                model=request.model,
                usage=usage,
            )

        model = _get_model_or_400(request.model)

        pairs = [[request.query, doc] for doc in request.documents]
        usage = _calculate_rerank_tokens(model, request.query, request.documents)

        with model.lock, model.tokenizer_lock:
            scores = model.predict(pairs)

        results = []
        for i, score in enumerate(scores):
            result_item = {"document": i, "score": float(score)}
            if request.return_documents:
                result_item["text"] = request.documents[i]
            results.append(result_item)

        response_data = _sort_and_format_rerank_results(results, request.top_n)

        return RerankResponse(
            query=request.query,
            data=response_data,
            model=request.model,
            usage=usage,
        )
