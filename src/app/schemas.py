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
    ] = Field(
        description="Input text to embed, encoded as a string or array of strings/multimodal items.",
        examples=[["Hello world", "How are you?"]],
    )
    model: str = Field(description="ID of the model to use.", examples=["BAAI/bge-m3"])
    user: Optional[str] = Field(
        default=None,
        description="A unique identifier representing your end-user.",
        examples=["user-1234"],
    )
    input_type: Optional[str] = Field(
        default=None,
        description="Type of the input. Maps to Ruri-v3 prefixes: query, document, classification, clustering, sts.",
        examples=["document"],
    )
    instruction: Optional[str] = Field(
        default=None,
        description="Specific instruction for the model. For future use with instruction-based models.",
        examples=["Represent the document for retrieval: "],
    )
    apply_ruri_prefix: bool = Field(
        default=False,
        description="Automatically apply prefixes based on input shape if true (fallback/compatibility).",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "input": ["Hello world", "How are you?"],
                    "model": "BAAI/bge-m3",
                    "input_type": "document",
                }
            ]
        }
    )


class EmbeddingData(BaseModel):
    object: str = Field(
        default="embedding", description="The object type, which is always 'embedding'."
    )
    embedding: list[float] = Field(
        description="The embedding vector, which is a list of floats."
    )
    index: int = Field(
        description="The index of the embedding in the list of embeddings."
    )


class Usage(BaseModel):
    prompt_tokens: int = Field(description="Number of tokens in the prompt.")
    total_tokens: int = Field(
        description="Total number of tokens used in the request (prompt + completion)."
    )


class EmbeddingResponse(BaseModel):
    object: str = Field(
        default="list", description="The object type, which is always 'list'."
    )
    data: list[EmbeddingData] = Field(description="A list of embedding objects.")
    model: str = Field(description="The ID of the model used.")
    usage: Usage = Field(description="Usage statistics for the request.")


# --- For /v1/rerank ---
class RerankRequest(BaseModel):
    query: LimitedString = Field(
        description="The search query.", examples=["What is the capital of France?"]
    )
    # Limit list size to prevent memory exhaustion (DoS)
    documents: Annotated[
        list[LimitedString],
        Field(
            min_length=1,
            max_length=MAX_INPUT_ITEMS,
            description="List of documents to rerank. Limited to MAX_INPUT_ITEMS to prevent DoS.",
            examples=[
                ["Paris is the capital of France.", "Berlin is the capital of Germany."]
            ],
        ),
    ]
    model: str = Field(
        description="ID of the model to use.", examples=["BAAI/bge-reranker-v2-m3"]
    )
    top_n: Optional[int] = Field(
        default=None,
        validation_alias="top_k",
        ge=0,
        le=MAX_INPUT_ITEMS,
        description="The number of most relevant documents to return.",
        examples=[1],
    )
    return_documents: Optional[bool] = Field(
        default=None,
        description="If true, returns the document text along with the score.",
        examples=[True],
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "query": "What is the capital of France?",
                    "documents": [
                        "Paris is the capital of France.",
                        "Berlin is the capital of Germany.",
                    ],
                    "model": "BAAI/bge-reranker-v2-m3",
                    "top_n": 1,
                    "return_documents": True,
                }
            ]
        },
    )


class RerankData(BaseModel):
    document: int = Field(description="The index of the document in the original list.")
    score: float = Field(description="The relevance score of the document.")
    text: Optional[LimitedString] = Field(
        default=None,
        description="The text of the document, if `return_documents` is true.",
    )


class RerankResponse(BaseModel):
    query: LimitedString = Field(description="The original search query.")
    data: list[RerankData] = Field(description="A list of ranked documents.")
    model: str = Field(description="The ID of the model used.")
    usage: Optional[Usage] = Field(
        default=None, description="Usage statistics for the request."
    )


# --- Error Responses ---
class ErrorResponse(BaseModel):
    detail: str = Field(
        description="A detailed human-readable error message.",
        examples=["Invalid API Key"],
    )


# --- Models API ---
class ModelCard(BaseModel):
    id: str = Field(description="The model identifier.", examples=["BAAI/bge-m3"])
    object: str = Field(
        default="model", description="The object type, which is always 'model'."
    )
    created: int = Field(
        default=0,
        description="The Unix timestamp (in seconds) when the model was created.",
    )
    owned_by: str = Field(
        default="organization", description="The organization that owns the model."
    )


class ModelList(BaseModel):
    object: str = Field(
        default="list", description="The object type, which is always 'list'."
    )
    data: list[ModelCard] = Field(description="A list of model objects.")
