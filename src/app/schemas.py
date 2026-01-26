from pydantic import BaseModel, Field, ConfigDict
from typing import List, Union, Optional

# --- For /v1/embeddings ---

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str
    user: Optional[str] = None
    input_type: Optional[str] = Field(
        None, 
        description="Type of the input. Maps to Ruri-v3 prefixes: query, document, classification, clustering, sts."
    )
    instruction: Optional[str] = Field(
        None,
        description="Specific instruction for the model. For future use with instruction-based models."
    )
    apply_ruri_prefix: bool = Field(
        False, 
        description="Automatically apply prefixes based on input shape if true (fallback/compatibility)."
    )

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
    top_n: Optional[int] = Field(None, validation_alias="top_k")
    return_documents: Optional[bool] = None

    model_config = ConfigDict(populate_by_name=True)

class RerankData(BaseModel):
    document: int # As per the doc, this is the index
    score: float
    text: Optional[str] = None

class RerankResponse(BaseModel):
    query: str
    data: List[RerankData]
    model: str
