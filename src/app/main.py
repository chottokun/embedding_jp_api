import os
from typing import Any, List, Tuple

# Disable tokenizer parallelism to prevent "Already Borrowed" errors and deadlocks
# in multi-process/multi-threaded environments.
# Set at the very beginning to ensure libraries read this correctly during import.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import heapq
import logging
import anyio
from typing import Optional
import re
import traceback

def redact_pii(text: str) -> str:
    """
    Redacts common PII from a string.
    Currently masks email addresses.
    """
    # Simple email regex
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    return re.sub(email_pattern, "[REDACTED]", text)


from .schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
    Usage,
    RerankRequest,
    RerankResponse,
    RerankData,
)
from .models import get_model
from .config import EMBEDDING_MODELS, RERANK_MODELS, RURI_PREFIX_MAP, API_KEY

app = FastAPI(title="OpenAI-Compatible API")

# Authentication dependency
security = HTTPBearer(auto_error=False)


async def verify_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Security(security),
):
    if API_KEY:
        if auth is None or auth.credentials != API_KEY:
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
    # Obtain the full traceback as a string
    tb_str = traceback.format_exc()

    # Redact PII from the exception message and traceback
    redacted_exc = redact_pii(str(exc))
    redacted_tb = redact_pii(tb_str)

    # Log the redacted error details
    # We use run_sync in a thread pool to avoid blocking the event loop
    # during potentially slow logging operations.
    await anyio.to_thread.run_sync(
        lambda: logging.error(
            f"Unhandled exception: {redacted_exc}\n{redacted_tb}", exc_info=False
        )
    )
    # Return a generic error message to the client to avoid leaking internal details
    response = JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )
    # Security headers for error responses
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    # Note: Use standard header name
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; frame-ancestors 'none';"
    )
    return response


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
            # Fallback logic based on input shape (compatibility mode)
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
    tokenizer = model.tokenizer
    processed_inputs = list(inputs)  # Make a copy to avoid side effects

    with model.tokenizer_lock:
        total_tokens = 0
        special_tokens_count = tokenizer.num_special_tokens_to_add(False)
        limit = max_seq_length - special_tokens_count

        # Process in batches to avoid OOM on huge payloads
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


@app.post(
    "/v1/embeddings",
    response_model=EmbeddingResponse,
    dependencies=[Depends(verify_api_key)],
)
def create_embeddings(request: EmbeddingRequest):
    """
    Creates embeddings for the given input, following OpenAI's API format.
    """
    model = _get_model_or_400(request.model, "embedding")

    inputs = request.input if isinstance(request.input, list) else [request.input]

    prefix = _determine_ruri_prefix(request)
    processed_inputs = _apply_prefix(inputs, prefix)

    # 1. Tokenize, truncate and calculate usage
    # This runs under model.tokenizer_lock but outside model.lock to improve concurrency
    processed_inputs, usage = _tokenize_and_truncate_embeddings(model, processed_inputs)

    # 2. Model inference under a single lock
    with model.lock:
        vectors = model.encode(processed_inputs)

    # Create response data
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
    model = _get_model_or_400(request.model, "rerank")

    # Reranking and token count within the same lock
    with model.lock:
        # Prepare pairs for the cross-encoder
        pairs = [[request.query, doc] for doc in request.documents]

        # Calculate token usage
        tokenizer = model.tokenizer
        total_tokens = 0
        for pair in pairs:
            # For cross-encoders, we usually count both parts
            q_tokens = len(tokenizer.encode(pair[0], add_special_tokens=False))
            d_tokens = len(tokenizer.encode(pair[1], add_special_tokens=False))
            total_tokens += (
                q_tokens + d_tokens + tokenizer.num_special_tokens_to_add(True)
            )

        usage = Usage(prompt_tokens=total_tokens, total_tokens=total_tokens)

        # Get scores
        scores = model.predict(pairs)

    # Combine documents with their scores
    results = []
    for i, score in enumerate(scores):
        result_item = {"document": i, "score": float(score)}
        if request.return_documents:
            result_item["text"] = request.documents[i]
        results.append(result_item)

    # Sort results by score in descending order
    # Optimization: Use heapq.nlargest for top_n which is O(N log k) instead of O(N log N)
    if request.top_n is not None:
        # Use (score, -index) tuple key to ensure stability (deterministic tie-breaking)
        # matching Python's stable sort behavior where earlier indices come first for same score.
        sorted_results = heapq.nlargest(
            request.top_n, results, key=lambda x: (x["score"], -x["document"])
        )
    else:
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

    # Format for response schema
    response_data = [RerankData(**result) for result in sorted_results]

    return RerankResponse(
        query=request.query, data=response_data, model=request.model, usage=usage
    )
