from fastapi import FastAPI, HTTPException
from typing import List, Union, Tuple

from .schemas import (
    EmbeddingRequest, EmbeddingResponse, EmbeddingData, Usage,
    RerankRequest, RerankResponse, RerankData
)
from .models import get_model
from .config import EMBEDDING_MODELS, RERANK_MODELS, RURI_PREFIX_MAP

app = FastAPI(title="OpenAI-Compatible API")

def _prepare_embedding_input(text: str, request: EmbeddingRequest, tokenizer, max_seq_length: int) -> Tuple[str, int]:
    """
    Prepares the input text for embedding:
    1. Determines and applies prefix (if applicable).
    2. Truncates the text if it exceeds the model's maximum sequence length, preserving the prefix.
    3. Calculates token usage.

    Returns:
        Tuple[str, int]: The processed text and the total token count.
    """
    prefix = ""

    # 1. Determine prefix
    if "ruri-v3" in request.model:
        if request.input_type in RURI_PREFIX_MAP:
            prefix = RURI_PREFIX_MAP[request.input_type]
        elif request.apply_ruri_prefix:
            # Fallback logic based on input shape (compatibility mode)
            if isinstance(request.input, str):
                prefix = RURI_PREFIX_MAP["query"]
            else:
                prefix = RURI_PREFIX_MAP["document"]

    # 2. Prefix processing
    # If the text already starts with the prefix, we don't add it again.
    effective_prefix = prefix if (prefix and not text.startswith(prefix)) else ""

    prefix_tokens = tokenizer.encode(effective_prefix, add_special_tokens=False) if effective_prefix else []
    text_tokens = tokenizer.encode(text, add_special_tokens=False)

    special_tokens_count = tokenizer.num_special_tokens_to_add(False)

    available_for_text = max_seq_length - len(prefix_tokens) - special_tokens_count

    # Ensure available_for_text is non-negative to avoid negative slicing
    available_for_text = max(0, available_for_text)

    if len(text_tokens) > available_for_text:
        # Truncate text
        text_tokens = text_tokens[:available_for_text]
        truncated_text = tokenizer.decode(text_tokens)
        final_text = f"{effective_prefix}{truncated_text}"
    else:
        final_text = f"{effective_prefix}{text}"

    total_tokens = len(prefix_tokens) + len(text_tokens) + special_tokens_count

    return final_text, total_tokens

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def create_embeddings(request: EmbeddingRequest):
    """
    Creates embeddings for the given input, following OpenAI's API format.
    """
    if request.model not in EMBEDDING_MODELS:
        raise HTTPException(status_code=400, detail=f"Model '{request.model}' not found for embeddings.")

    try:
        model = get_model(request.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    inputs = request.input if isinstance(request.input, list) else [request.input]
    
    processed_inputs = []
    max_seq_length = getattr(model, "max_seq_length", 8192)
    tokenizer = model.tokenizer

    total_tokens = 0

    for text in inputs:
        final_text, tokens = _prepare_embedding_input(text, request, tokenizer, max_seq_length)
        processed_inputs.append(final_text)
        total_tokens += tokens

    usage = Usage(prompt_tokens=total_tokens, total_tokens=total_tokens)

    # Get embeddings
    vectors = model.encode(processed_inputs)

    # Create response data
    response_data = [
        EmbeddingData(embedding=vector.tolist(), index=i) for i, vector in enumerate(vectors)
    ]

    return EmbeddingResponse(
        data=response_data,
        model=request.model,
        usage=usage
    )

@app.post("/v1/rerank", response_model=RerankResponse)
def create_rerank(request: RerankRequest):
    """
    Reranks a list of documents for a given query.
    """
    if request.model not in RERANK_MODELS:
        raise HTTPException(status_code=400, detail=f"Model '{request.model}' not found for reranking.")

    try:
        model = get_model(request.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Prepare pairs for the cross-encoder
    pairs = [[request.query, doc] for doc in request.documents]

    # Calculate token usage
    tokenizer = model.tokenizer
    total_tokens = 0
    for query_text, doc_text in pairs:
        tokens = tokenizer.encode(query_text, doc_text)
        total_tokens += len(tokens)

    usage = Usage(prompt_tokens=total_tokens, total_tokens=total_tokens)

    # Get scores from the model
    scores = model.predict(pairs)

    # Combine documents with their scores
    results = []
    for i, score in enumerate(scores):
        result_item = {"document": i, "score": float(score)}
        if request.return_documents:
            result_item["text"] = request.documents[i]
        results.append(result_item)

    # Sort results by score in descending order
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

    # Apply top_n if specified
    if request.top_n is not None:
        sorted_results = sorted_results[:request.top_n]

    # Format for response schema
    response_data = [RerankData(**result) for result in sorted_results]

    return RerankResponse(
        query=request.query,
        data=response_data,
        model=request.model,
        usage=usage
    )
