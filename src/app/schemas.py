from pydantic import BaseModel, Field, ConfigDict, StringConstraints
from typing import Union, Optional, Annotated, Literal

from .config import MAX_INPUT_LENGTH, MAX_INPUT_ITEMS

# --- Security Types ---
LimitedString = Annotated[
    str, StringConstraints(min_length=1, max_length=MAX_INPUT_LENGTH)
]

# Base64/URL image source constraint (up to ~18MB Base64 string length)
ImageSourceString = Annotated[
    str, StringConstraints(min_length=1, max_length=25_000_000)
]


# Multimodal text constraint (allows empty string for image-only items)
MultimodalText = Annotated[str, StringConstraints(max_length=MAX_INPUT_LENGTH)]


# --- Multimodal Schemas ---
class ImageUrl(BaseModel):
    url: ImageSourceString
    detail: Optional[str] = "auto"


class FlatMultimodalItem(BaseModel):
    text: Optional[MultimodalText] = None
    image_url: Optional[Union[ImageUrl, ImageSourceString]] = None


class ContentPartText(BaseModel):
    type: Literal["text"]
    text: MultimodalText


class ContentPartImage(BaseModel):
    type: Literal["image_url"]
    image_url: Union[ImageUrl, ImageSourceString]


ContentPart = Union[ContentPartText, ContentPartImage]
SingleInputItem = Union[
    LimitedString,
    FlatMultimodalItem,
    Annotated[list[ContentPart], Field(min_length=1)],
]

# --- For /v1/embeddings ---


class EmbeddingRequest(BaseModel):
    input: Union[
        SingleInputItem,
        # Limit list size to prevent memory exhaustion (DoS)
        Annotated[
            list[SingleInputItem], Field(min_length=1, max_length=MAX_INPUT_ITEMS)
        ],
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
    embedding: list[float]
    index: int


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: Usage


# --- For /v1/rerank ---
class RerankRequest(BaseModel):
    query: LimitedString
    # Limit list size to prevent memory exhaustion (DoS)
    documents: Annotated[
        list[LimitedString],
        Field(
            min_length=1,
            max_length=MAX_INPUT_ITEMS,
            description="List of documents to rerank. Limited to MAX_INPUT_ITEMS to prevent DoS.",
        ),
    ]
    model: str
    top_n: Optional[int] = Field(
        None, validation_alias="top_k", ge=0, le=MAX_INPUT_ITEMS
    )
    return_documents: Optional[bool] = None

    model_config = ConfigDict(populate_by_name=True)


class RerankData(BaseModel):
    document: int  # As per the doc, this is the index
    score: float
    text: Optional[LimitedString] = None


class RerankResponse(BaseModel):
    query: LimitedString
    data: list[RerankData]
    model: str
    usage: Optional[Usage] = None
