from fastapi import FastAPI, HTTPException

from .schemas import (
    EmbeddingRequest, EmbeddingResponse, EmbeddingData, Usage,
    RerankRequest, RerankResponse, RerankData
)
from .models import get_model
from .config import EMBEDDING_MODELS, RERANK_MODELS

app = FastAPI(title="OpenAI-Compatible API")

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Creates embeddings for the given input, following OpenAI's API format.
    """
    if request.model not in EMBEDDING_MODELS:
        raise HTTPException(status_code=400, detail=f"Model '{request.model}' not found for embeddings.")

    try:
        model = get_model(request.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Conditionally apply prefixes for ruri-v3 model
    inputs_to_encode = request.input if isinstance(request.input, list) else [request.input]

    if "ruri-v3" in request.model and request.apply_ruri_prefix:
        if isinstance(request.input, str):
            inputs_to_encode = [f"検索クエリ: {request.input}"]
        else:
            inputs_to_encode = [f"検索文書: {text}" for text in request.input]

    # Get embeddings
    vectors = model.encode(inputs_to_encode)

    # Create response data
    response_data = [
        EmbeddingData(embedding=vector, index=i) for i, vector in enumerate(vectors)
    ]

    # Dummy usage data for now. In a real scenario, this would be calculated.
    usage = Usage(prompt_tokens=0, total_tokens=0)

    return EmbeddingResponse(
        data=response_data,
        model=request.model,
        usage=usage
    )

@app.post("/v1/rerank", response_model=RerankResponse)
async def create_rerank(request: RerankRequest):
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

    # Get scores from the mock model
    scores = model.predict(pairs)

    # Combine documents with their scores
    results = []
    for i, score in enumerate(scores):
        results.append({"document": i, "score": score})

    # Sort results by score in descending order
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

    # Format for response schema
    response_data = [RerankData(**result) for result in sorted_results]

    return RerankResponse(
        query=request.query,
        data=response_data,
        model=request.model
    )
