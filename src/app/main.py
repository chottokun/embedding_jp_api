# ruff: noqa: E402
import os
from typing import Any, List, Tuple, Optional

# Disable tokenizer parallelism to prevent "Already Borrowed" errors and deadlocks
# in multi-process/multi-threaded environments.
# Set at the very beginning to ensure libraries read this correctly during import.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import heapq
import logging
import re
import secrets
import traceback
from contextlib import asynccontextmanager

import anyio
import httpx
from PIL import Image
from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .image_utils import load_image_from_source
from .schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
    Usage,
    RerankRequest,
    RerankResponse,
    RerankData,
    FlatMultimodalItem,
    ContentPartText,
    ContentPartImage,
    ImageUrl,
)
from .models import get_model
from .config import (
    EMBEDDING_MODELS,
    RERANK_MODELS,
    RURI_PREFIX_MAP,
    API_KEY,
    EMBEDDING_TEI_URL,
    RERANK_TEI_URL,
)


def redact_pii(text: str) -> str:
    """
    Redacts common PII from a string.
    Currently masks email addresses.
    """
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    return re.sub(email_pattern, "[REDACTED]", text)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    # Initialize global HTTP client with connection pooling for TEI proxy requests
    app_instance.state.tei_client = httpx.Client(timeout=30.0)
    try:
        yield
    finally:
        app_instance.state.tei_client.close()


app = FastAPI(title="OpenAI-Compatible API", lifespan=lifespan)


@app.get("/health", tags=["Health"])
@app.get("/healthz", tags=["Health"])
async def health_check():
    """
    Liveness / readiness probe for microservice orchestrators and Docker health checks.
    """
    return {"status": "ok"}


# Authentication dependency
security = HTTPBearer(auto_error=False)


async def verify_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Security(security),
):
    if API_KEY:
        if auth is None or not secrets.compare_digest(auth.credentials, API_KEY):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API Key",
                headers={"WWW-Authenticate": "Bearer"},
            )
    return auth


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """
    Middleware that adds security headers to every response.
    """
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; frame-ancestors 'none';"
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb_str = traceback.format_exc()
    redacted_exc = redact_pii(str(exc))
    redacted_tb = redact_pii(tb_str)

    await anyio.to_thread.run_sync(
        lambda: logging.error(
            f"Unhandled exception: {redacted_exc}\n{redacted_tb}", exc_info=False
        )
    )
    response = JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; frame-ancestors 'none';"
    )
    return response


def _proxy_to_tei(tei_url: str, path: str, json_data: dict) -> Any:
    """
    Helper to send a POST request to TEI and return the JSON response.
    """
    try:
        shared_client = getattr(app.state, "tei_client", None)
        if shared_client is not None:
            response = shared_client.post(f"{tei_url}{path}", json=json_data)
        else:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(f"{tei_url}{path}", json=json_data)

        if response.status_code != 200:
            error_msg = response.text
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            raise HTTPException(
                status_code=500,
                detail=f"TEI Proxy Error ({response.status_code}): {error_msg}",
            )
        return response.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to proxy request to TEI: {str(e)}"
        )


def _get_model_or_400(model_name: str, model_type: str) -> Any:
    """
    Helper to get a model or raise a 400 HTTPException.
    """
    supported_models = EMBEDDING_MODELS if model_type == "embedding" else RERANK_MODELS
    if model_name not in supported_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not found for {model_type}s.",
        )

    try:
        return get_model(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _determine_ruri_prefix(request: EmbeddingRequest) -> str:
    """
    Determines the prefix for ruri-v3 models based on request parameters.
    """
    prefix = ""
    if "ruri-v3" in request.model:
        if request.input_type in RURI_PREFIX_MAP:
            prefix = RURI_PREFIX_MAP[request.input_type]
        elif request.apply_ruri_prefix:
            if isinstance(request.input, str):
                prefix = RURI_PREFIX_MAP["query"]
            else:
                prefix = RURI_PREFIX_MAP["document"]
    return prefix


def _apply_prefix(inputs: List[str], prefix: str) -> List[str]:
    """
    Applies a prefix to each input string if it doesn't already start with it.
    """
    if not prefix:
        return inputs
    return [text if text.startswith(prefix) else f"{prefix}{text}" for text in inputs]


def _tokenize_and_truncate_embeddings(
    model: Any, inputs: List[str]
) -> Tuple[List[str], Usage]:
    """
    Handles batch tokenization, truncation, and usage calculation for embeddings.
    """
    max_seq_length = getattr(model, "max_seq_length", 8192)
    if not isinstance(max_seq_length, int):
        max_seq_length = 8192
    tokenizer = model.tokenizer
    processed_inputs = list(inputs)

    with model.tokenizer_lock:
        total_tokens = 0
        special_tokens_count = tokenizer.num_special_tokens_to_add(False)
        if not isinstance(special_tokens_count, int):
            special_tokens_count = 2
        limit = max_seq_length - special_tokens_count

        batch_size = 256
        for i in range(0, len(processed_inputs), batch_size):
            batch = processed_inputs[i : i + batch_size]
            encodings = tokenizer(batch, add_special_tokens=False)

            for j, ids in enumerate(encodings["input_ids"]):
                if len(ids) > limit:
                    truncated_ids = ids[:limit]
                    truncated_text = tokenizer.decode(truncated_ids)
                    processed_inputs[i + j] = truncated_text
                    total_tokens += len(truncated_ids) + special_tokens_count
                else:
                    total_tokens += len(ids) + special_tokens_count

        usage = Usage(prompt_tokens=total_tokens, total_tokens=total_tokens)
    return processed_inputs, usage


def _calculate_rerank_tokens(model: Any, query: str, documents: List[str]) -> Usage:
    """
    Calculates token usage for reranking based on query and documents.
    """
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
    """
    Sorts and filters rerank results by score (descending order).
    """
    if top_n is not None:
        sorted_results = heapq.nlargest(
            top_n, results, key=lambda x: (x["score"], -x["document"])
        )
    else:
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

    return [RerankData(**result) for result in sorted_results]


def _proxy_rerank_to_tei(request: RerankRequest) -> RerankResponse:
    """
    Proxies a reranking request to TEI.
    """
    tei_results = _proxy_to_tei(
        RERANK_TEI_URL,
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
        query=request.query, data=response_data, model=request.model, usage=usage
    )


def _normalize_raw_inputs(input_data: Any) -> list:
    """
    Normalizes input into a flat list of input items.
    """
    if isinstance(input_data, list):
        if not input_data:
            return []
        if all(
            isinstance(x, (ContentPartText, ContentPartImage))
            or (isinstance(x, dict) and x.get("type") in {"text", "image_url"})
            for x in input_data
        ):
            return [input_data]
        return input_data
    return [input_data]


async def parse_input_item(
    item: Any, client: httpx.AsyncClient
) -> Tuple[Optional[str], Optional[Image.Image]]:
    """
    Parses a single input item into a (text, PIL.Image) tuple.
    """
    if isinstance(item, str):
        return item, None

    if isinstance(item, FlatMultimodalItem):
        text = item.text
        img = None
        if item.image_url:
            url_val = (
                item.image_url.url
                if isinstance(item.image_url, ImageUrl)
                else item.image_url
            )
            img = await load_image_from_source(url_val, client)
        return text, img

    if isinstance(item, list):
        text_parts = []
        img = None
        for part in item:
            if isinstance(part, ContentPartText) or (
                isinstance(part, dict) and part.get("type") == "text"
            ):
                text_val = (
                    part.text
                    if isinstance(part, ContentPartText)
                    else part.get("text", "")
                )
                text_parts.append(text_val)
            elif isinstance(part, ContentPartImage) or (
                isinstance(part, dict) and part.get("type") == "image_url"
            ):
                img_data = (
                    part.image_url
                    if isinstance(part, ContentPartImage)
                    else part.get("image_url")
                )
                url_val = (
                    img_data.url
                    if isinstance(img_data, ImageUrl)
                    else (
                        img_data.get("url") if isinstance(img_data, dict) else img_data
                    )
                )
                img = await load_image_from_source(url_val, client)
        text = "\n".join(text_parts) if text_parts else None
        return text, img

    raise ValueError("不正な入力形式です。")


@app.post(
    "/v1/embeddings",
    response_model=EmbeddingResponse,
    dependencies=[Depends(verify_api_key)],
)
async def create_embeddings(request: EmbeddingRequest):
    """
    Creates embeddings for the given input, following OpenAI's API format.
    Supports text-only and multimodal (image/composite) inputs.
    """
    if request.model not in EMBEDDING_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not found for embeddings.",
        )

    raw_items = _normalize_raw_inputs(request.input)

    try:
        async with httpx.AsyncClient() as client:
            tasks = [parse_input_item(item, client) for item in raw_items]
            parsed_items: List[
                Tuple[Optional[str], Optional[Image.Image]]
            ] = await asyncio.gather(*tasks)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    has_image = any(img is not None for _, img in parsed_items)

    # TEI Proxy check: if EMBEDDING_TEI_URL is set and request is text-only, proxy to TEI
    if EMBEDDING_TEI_URL and not has_image:
        inputs = [text for text, _ in parsed_items if text is not None]
        prefix = _determine_ruri_prefix(request)
        processed_inputs = _apply_prefix(inputs, prefix)
        data = _proxy_to_tei(
            EMBEDDING_TEI_URL,
            "/v1/embeddings",
            {"input": processed_inputs, "model": request.model},
        )
        return EmbeddingResponse(**data)

    model = _get_model_or_400(request.model, "embedding")
    is_multimodal = getattr(model, "supports_multimodal", False) is True

    # Guard against sending image inputs to text-only models
    if has_image and not is_multimodal:
        raise HTTPException(
            status_code=400,
            detail=f"モデル '{request.model}' は画像入力をサポートしていません。bge-visualized-m3 などのマルチモーダル対応モデルを指定してください。",
        )

    # Multimodal model encoding
    if is_multimodal:
        prefix = _determine_ruri_prefix(request)
        processed_items = []
        for text, img in parsed_items:
            if text:
                text = _apply_prefix([text], prefix)[0]
            processed_items.append((text, img))

        embeddings = await anyio.to_thread.run_sync(
            model.encode_multimodal, processed_items
        )
        response_data = [
            EmbeddingData(embedding=emb, index=i) for i, emb in enumerate(embeddings)
        ]
        usage = Usage(prompt_tokens=0, total_tokens=0)
        return EmbeddingResponse(data=response_data, model=request.model, usage=usage)

    # Standard text-only embedding model pipeline
    inputs = [text if text is not None else "" for text, _ in parsed_items]
    prefix = _determine_ruri_prefix(request)
    processed_inputs = _apply_prefix(inputs, prefix)

    processed_inputs, usage = _tokenize_and_truncate_embeddings(model, processed_inputs)

    def _run_inference():
        with model.lock, model.tokenizer_lock:
            return model.encode(processed_inputs)

    vectors = await anyio.to_thread.run_sync(_run_inference)

    response_data = [
        EmbeddingData(embedding=vector, index=i)
        for i, vector in enumerate(vectors.tolist())
    ]

    return EmbeddingResponse(data=response_data, model=request.model, usage=usage)


@app.post(
    "/v1/rerank",
    response_model=RerankResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(verify_api_key)],
)
def create_rerank(request: RerankRequest):
    """
    Reranks a list of documents for a given query.
    """
    if request.model not in RERANK_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not found for reranks.",
        )

    if RERANK_TEI_URL:
        return _proxy_rerank_to_tei(request)

    model = _get_model_or_400(request.model, "rerank")

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
        query=request.query, data=response_data, model=request.model, usage=usage
    )
