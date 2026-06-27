import os

# Disable tokenizer parallelism to prevent "Already Borrowed" errors and deadlocks
# in multi-process/multi-threaded environments.
# Set at the very beginning to ensure libraries read this correctly during import.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import heapq
import logging
import anyio

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
from .config import EMBEDDING_MODELS, RERANK_MODELS, RURI_PREFIX_MAP

app = FastAPI(title="OpenAI-Compatible API")


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
    # Log the full error with stack trace
    # We use run_sync in a thread pool to avoid blocking the event loop
    # during potentially slow logging operations.
    await anyio.to_thread.run_sync(
        lambda: logging.error(f"Unhandled exception: {exc}", exc_info=True)
    )
    # Return a generic error message to the client to avoid leaking internal details
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )


def _get_model_or_400(model_name: str, allowed_models: list[str]):
    """
    Validates the model name and retrieves the model instance.
    Raises HTTPException if the model is not found or fails to load.
    """
    if model_name not in allowed_models:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found.")

    try:
        return get_model(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _determine_ruri_prefix(
    model_name: str,
    input_data: str | list[str],
    input_type: str | None,
    apply_ruri_prefix: bool,
) -> str:
    """
    Determines the prefix for ruri-v3 models based on request parameters and input shape.
    """
    if "ruri-v3" not in model_name:
        return ""

    if input_type in RURI_PREFIX_MAP:
        return RURI_PREFIX_MAP[input_type]

    if apply_ruri_prefix:
        # Fallback logic based on input shape (compatibility mode)
        if isinstance(input_data, str):
            return RURI_PREFIX_MAP["query"]
        else:
            return RURI_PREFIX_MAP["document"]

    return ""


def _apply_prefix(inputs: list[str], prefix: str) -> list[str]:
    """
    Applies the given prefix to each input string if it doesn't already start with it.
    """
    if not prefix:
        return inputs

    return [text if text.startswith(prefix) else f"{prefix}{text}" for text in inputs]


def _tokenize_and_truncate_embeddings(
    tokenizer, inputs: list[str], max_seq_length: int
) -> tuple[list[str], int]:
    """
    Batch tokenizes inputs, truncates if necessary, and calculates total token count.
    """
    total_tokens = 0
    special_tokens_count = tokenizer.num_special_tokens_to_add(False)
    limit = max_seq_length - special_tokens_count
    processed_inputs = list(inputs)

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

    return processed_inputs, total_tokens


def _calculate_rerank_usage(tokenizer, query: str, documents: list[str]) -> int:
    """
    Calculates total token usage for a rerank request.
    """
    total_tokens = 0
    special_tokens_to_add = tokenizer.num_special_tokens_to_add(True)
    for doc in documents:
        # For cross-encoders, we usually count both parts
        q_tokens = len(tokenizer.encode(query, add_special_tokens=False))
        d_tokens = len(tokenizer.encode(doc, add_special_tokens=False))
        total_tokens += q_tokens + d_tokens + special_tokens_to_add
    return total_tokens


def _sort_rerank_results(results: list[dict], top_n: int | None) -> list[dict]:
    """
    Sorts rerank results by score in descending order.
    """
    if top_n is not None:
        # Use (score, -index) tuple key to ensure stability (deterministic tie-breaking)
        return heapq.nlargest(
            top_n, results, key=lambda x: (x["score"], -x["document"])
        )
    else:
        return sorted(results, key=lambda x: x["score"], reverse=True)


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def create_embeddings(request: EmbeddingRequest):
    """
    Creates embeddings for the given input, following OpenAI's API format.
    """
    model = _get_model_or_400(request.model, EMBEDDING_MODELS)
    inputs = request.input if isinstance(request.input, list) else [request.input]

    prefix = _determine_ruri_prefix(
        request.model, request.input, request.input_type, request.apply_ruri_prefix
    )
    processed_inputs = _apply_prefix(inputs, prefix)

    with model.lock:
        processed_inputs, total_tokens = _tokenize_and_truncate_embeddings(
            model.tokenizer, processed_inputs, getattr(model, "max_seq_length", 8192)
        )
        usage = Usage(prompt_tokens=total_tokens, total_tokens=total_tokens)
        vectors = model.encode(processed_inputs)

    response_data = [
        EmbeddingData(embedding=vector.tolist(), index=i)
        for i, vector in enumerate(vectors)
    ]
    return EmbeddingResponse(data=response_data, model=request.model, usage=usage)


@app.post("/v1/rerank", response_model=RerankResponse)
def create_rerank(request: RerankRequest):
    """
    Reranks a list of documents for a given query.
    """
    model = _get_model_or_400(request.model, RERANK_MODELS)

    with model.lock:
        total_tokens = _calculate_rerank_usage(
            model.tokenizer, request.query, request.documents
        )
        usage = Usage(prompt_tokens=total_tokens, total_tokens=total_tokens)

        pairs = [[request.query, doc] for doc in request.documents]
        scores = model.predict(pairs)

    results = []
    for i, score in enumerate(scores):
        result_item = {"document": i, "score": float(score)}
        if request.return_documents:
            result_item["text"] = request.documents[i]
        results.append(result_item)

    sorted_results = _sort_rerank_results(results, request.top_n)
    response_data = [RerankData(**result) for result in sorted_results]

    return RerankResponse(
        query=request.query, data=response_data, model=request.model, usage=usage
    )
