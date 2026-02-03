from fastapi import FastAPI, HTTPException

from .schemas import (
    EmbeddingRequest, EmbeddingResponse, EmbeddingData, Usage,
    RerankRequest, RerankResponse, RerankData
)
from .models import get_model
from .config import EMBEDDING_MODELS, RERANK_MODELS, RURI_PREFIX_MAP

app = FastAPI(title="OpenAI-Compatible API")

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
        prefix = ""
        # 1. Determine prefix
        if "ruri-v3" in request.model:
            if request.input_type in RURI_PREFIX_MAP:
                prefix = RURI_PREFIX_MAP[request.input_type]
            elif request.apply_ruri_prefix:
                # Fallback logic
                if isinstance(request.input, str):
                    prefix = RURI_PREFIX_MAP["query"]
                else:
                    prefix = RURI_PREFIX_MAP["document"]
        
        # 2. Prefix deduplication
        if prefix and text.startswith(prefix):
            final_text = text
        else:
            final_text = f"{prefix}{text}"

        # 3. Truncation logic (8192 tokens)
        # Tokenize prefix and text separately to ensure prefix is preserved
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False) if prefix else []
        text_tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # Safety margin for special tokens (e.g. [CLS], [SEP])
        special_tokens_count = tokenizer.num_special_tokens_to_add(False) # Usually 2
        available_for_text = max_seq_length - len(prefix_tokens) - special_tokens_count
        
        if len(text_tokens) > available_for_text:
            text_tokens = text_tokens[:available_for_text]
            truncated_text = tokenizer.decode(text_tokens)
            final_text = f"{prefix}{truncated_text}"
        
        processed_inputs.append(final_text)

        # Calculate usage based on the tokens we already have (approximate but efficient)
        # Note: This ignores potential subword merging at the prefix-text boundary.
        total_tokens += (len(prefix_tokens) + len(text_tokens) + special_tokens_count)

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

    # Batch processing for token counting to improve performance and manage memory
    batch_size = 256

    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i : i + batch_size]
        batch_queries = [p[0] for p in batch_pairs]
        batch_docs = [p[1] for p in batch_pairs]

        encodings = tokenizer(
            batch_queries,
            batch_docs,
            add_special_tokens=True
        )

        for input_ids in encodings['input_ids']:
            total_tokens += len(input_ids)

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
