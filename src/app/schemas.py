from pydantic import BaseModel, Field, ConfigDict, StringConstraints
from typing import List, Union, Optional, Annotated

from .config import MAX_INPUT_LENGTH, MAX_INPUT_ITEMS

# --- Security Types ---
LimitedString = Annotated[str, StringConstraints(max_length=MAX_INPUT_LENGTH)]

# --- For /v1/embeddings ---


class EmbeddingRequest(BaseModel):
    input: Union[
        LimitedString,
        Annotated[List[LimitedString], Field(max_length=MAX_INPUT_ITEMS)]
    ]
    model: str
    user: Optional[str] = None
    input_type: Optional[str] = Field(
        None,
        description="Type of the input. Maps to Ruri-v3 prefixes: query, document, classification, clustering, sts.",
    )
    instruction: Optional[str] = Field(
        None,
        description="Specific instruction for the model. For future use with instruction-based models.",
    )
    apply_ruri_prefix: bool = Field(
        False,
        description="Automatically apply prefixes based on input shape if true (fallback/compatibility).",
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
    query: LimitedString
    documents: Annotated[
        List[LimitedString],
        Field(
            max_length=MAX_INPUT_ITEMS,
            description="List of documents to rerank. Limited to MAX_INPUT_ITEMS to prevent DoS.",
        ),
    ]
    model: str
    top_n: Optional[int] = Field(None, validation_alias="top_k")
    return_documents: Optional[bool] = None

    model_config = ConfigDict(populate_by_name=True)


class RerankData(BaseModel):
    document: int  # As per the doc, this is the index
    score: float
    text: Optional[LimitedString] = None


class RerankResponse(BaseModel):
    query: LimitedString
    data: List[RerankData]
    model: str
    usage: Optional[Usage] = None
