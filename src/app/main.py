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


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def create_embeddings(request: EmbeddingRequest):
    """
    Creates embeddings for the given input, following OpenAI's API format.
    """
    if request.model not in EMBEDDING_MODELS:
        raise HTTPException(
            status_code=400, detail=f"Model '{request.model}' not found for embeddings."
        )

    try:
        model = get_model(request.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    inputs = request.input if isinstance(request.input, list) else [request.input]

    max_seq_length = getattr(model, "max_seq_length", 8192)
    tokenizer = model.tokenizer

    # Optimization: Determine prefix once per request
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

    # 1. Prepare strings with prefixes
    # If the text already starts with the prefix, we don't add it again.
    if prefix:
        processed_inputs = [
            text if text.startswith(prefix) else f"{prefix}{text}" for text in inputs
        ]
    else:
        processed_inputs = inputs

    # Get embeddings and process tokens under a single lock to avoid "Already borrowed" inside library calls
    with model.lock:
        # 2. Batch tokenize to calculate usage and truncate if necessary
        # Doing this inside model.lock ensures the tokenizer state is safe
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

        # 3. Model inference
        vectors = model.encode(processed_inputs)

    # Create response data
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
    if request.model not in RERANK_MODELS:
        raise HTTPException(
            status_code=400, detail=f"Model '{request.model}' not found for reranking."
        )

    try:
        model = get_model(request.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

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
