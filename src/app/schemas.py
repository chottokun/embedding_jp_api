from pydantic import BaseModel, Field
from typing import List, Union, Optional

# --- For /v1/embeddings ---

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str
    user: Optional[str] = None
    apply_ruri_prefix: bool = False

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Usage

# --- For /v1/rerank ---
# Schemas for the rerank endpoint will be added in the corresponding step.
class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    model: str
    top_k: Optional[int] = None
    return_documents: Optional[bool] = None

class RerankData(BaseModel):
    document: int # As per the doc, this is the index
    score: float
    text: Optional[str] = None

class RerankResponse(BaseModel):
    query: str
    data: List[RerankData]
    model: str
