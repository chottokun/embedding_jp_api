"""Pydantic schemas for API request and response models."""

from pydantic import BaseModel, Field, ConfigDict, StringConstraints
from typing import Union, Optional, Annotated, Literal, Any

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
    """Schema representing an image URL and its detail level.

    Attributes:
        url (ImageSourceString): The URL or Base64 encoded string of the image.
        detail (Optional[str]): Detail level of the image, e.g., 'auto', 'low', 'high'. Defaults to 'auto'.
    """

    url: ImageSourceString = Field(
        description="The URL or Base64 encoded string of the image.",
        examples=["data:image/jpeg;base64,..."],
    )
    detail: Optional[str] = Field(
        "auto", description="Detail level of the image.", examples=["auto"]
    )


class FlatMultimodalItem(BaseModel):
    """Schema for a flat multimodal item containing optional text and image URL.

    Attributes:
        text (Optional[MultimodalText]): The text part of the multimodal item.
        image_url (Optional[Union[ImageUrl, ImageSourceString]]): The image part of the multimodal item.
    """

    text: Optional[MultimodalText] = Field(
        None,
        description="The text part of the multimodal item.",
        examples=["What is in this image?"],
    )
    image_url: Optional[Union[ImageUrl, ImageSourceString]] = Field(
        None,
        description="The image part of the multimodal item.",
        examples=["data:image/jpeg;base64,..."],
    )


class ContentPartText(BaseModel):
    """Schema for a text content part in a multimodal request.

    Attributes:
        type (Literal["text"]): The type of the content part, must be 'text'.
        text (MultimodalText): The text content.
    """

    type: Literal["text"] = Field(
        description="The type of the content part, must be 'text'.", examples=["text"]
    )
    text: MultimodalText = Field(
        description="The text content.", examples=["What is in this image?"]
    )


class ContentPartImage(BaseModel):
    """Schema for an image content part in a multimodal request.

    Attributes:
        type (Literal["image_url"]): The type of the content part, must be 'image_url'.
        image_url (Union[ImageUrl, ImageSourceString]): The image content.
    """

    type: Literal["image_url"] = Field(
        description="The type of the content part, must be 'image_url'.",
        examples=["image_url"],
    )
    image_url: Union[ImageUrl, ImageSourceString] = Field(
        description="The image content.", examples=["data:image/jpeg;base64,..."]
    )


ContentPart = Union[ContentPartText, ContentPartImage]

SingleInputItem = Union[
    LimitedString,
    FlatMultimodalItem,
    Annotated[list[ContentPart], Field(min_length=1)],
]


# --- For /v1/embeddings ---
class EmbeddingRequest(BaseModel):
    """Schema for embedding API requests.

    Attributes:
        input (Union[SingleInputItem, list[SingleInputItem]]): The input text(s) or multimodal item(s) to embed.
        model (str): The name of the model to use.
        user (Optional[str]): The user ID associated with the request.
        input_type (Optional[str]): Type of the input, e.g., 'query', 'document'.
        instruction (Optional[str]): Specific instruction for instruction-based models.
        apply_ruri_prefix (bool): Whether to apply ruri-style prefix automatically.
        dimensions (Optional[int]): Number of dimensions for the output embeddings.
        encoding_format (Literal["float", "base64"]): Format to return embeddings in.
    """

    input: Union[
        SingleInputItem,
        # Limit list size to prevent memory exhaustion (DoS)
        Annotated[
            list[SingleInputItem], Field(min_length=1, max_length=MAX_INPUT_ITEMS)
        ],
    ] = Field(
        description="The input text(s) or multimodal item(s) to embed.",
        examples=["日本語のテキスト埋め込みテスト"],
    )
    model: str = Field(
        description="The name of the model to use.",
        examples=["cl-nagoya/ruri-v3-small"],
    )
    user: Optional[str] = Field(
        None,
        description="The user ID associated with the request.",
        examples=["user-1234"],
    )
    input_type: Optional[str] = Field(
        None,
        description="Type of the input. Maps to Ruri-v3 prefixes: query, document, classification, clustering, sts.",
        examples=["query"],
    )
    instruction: Optional[str] = Field(
        None,
        description="Specific instruction for the model. For future use with instruction-based models.",
        examples=["Represent this sentence for searching relevant passages: "],
    )
    apply_ruri_prefix: bool = Field(
        False,
        description="Automatically apply prefixes based on input shape if true (fallback/compatibility).",
    )
    dimensions: Optional[int] = Field(
        None,
        ge=1,
        description="The number of dimensions the resulting output embeddings should have. Supports Matryoshka models.",
        examples=[512],
    )
    encoding_format: Literal["float", "base64"] = Field(
        "float",
        description="The format to return the embeddings in. Can be either float or base64.",
        examples=["float"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "input": "日本語のテキスト埋め込みテスト",
                "model": "cl-nagoya/ruri-v3-small",
                "input_type": "query",
                "dimensions": 512,
                "encoding_format": "float",
            }
        }
    )


class EmbeddingData(BaseModel):
    """Schema for a single embedding result data.

    Attributes:
        object (str): The object type, always 'embedding'.
        embedding (Union[list[float], str]): The embedding values, either as a list of floats or base64 string.
        index (int): The index of the input this embedding corresponds to.
    """

    object: str = Field(
        "embedding", description="The object type.", examples=["embedding"]
    )
    embedding: Union[list[float], str] = Field(
        description="The embedding values, either as a list of floats or base64 string.",
        examples=[[0.01, 0.02, 0.03]],
    )
    index: int = Field(
        description="The index of the input this embedding corresponds to.",
        examples=[0],
    )


class Usage(BaseModel):
    """Schema for usage statistics.

    Attributes:
        prompt_tokens (int): The number of tokens in the prompt.
        total_tokens (int): The total number of tokens used.
    """

    prompt_tokens: int = Field(
        description="The number of tokens in the prompt.", examples=[10]
    )
    total_tokens: int = Field(
        description="The total number of tokens used.", examples=[10]
    )


class EmbeddingResponse(BaseModel):
    """Schema for the embedding API response.

    Attributes:
        object (str): The object type, always 'list'.
        data (list[EmbeddingData]): The list of embedding results.
        model (str): The model used to generate embeddings.
        usage (Usage): Usage statistics for the request.
    """

    object: str = Field("list", description="The object type.", examples=["list"])
    data: list[EmbeddingData] = Field(description="The list of embedding results.")
    model: str = Field(
        description="The model used to generate embeddings.",
        examples=["cl-nagoya/ruri-v3-small"],
    )
    usage: Usage = Field(description="Usage statistics for the request.")


# --- For /v1/rerank ---
class RerankRequest(BaseModel):
    """Schema for reranking API requests.

    Attributes:
        query (LimitedString): The query to rerank documents against.
        documents (list[LimitedString]): The list of documents to rerank.
        model (str): The name of the reranking model.
        top_n (Optional[int]): Number of top documents to return.
        return_documents (Optional[bool]): Whether to include document text in the response.
        threshold (Optional[float]): Sufficiency probability threshold.
        drop_failed (bool): Whether to drop documents below the threshold.
        use_ascii_boost (Optional[bool]): Whether to use ASCII 3-gram boost.
    """

    query: LimitedString = Field(
        description="The query to rerank documents against.",
        examples=["日本の首都は？"],
    )
    # Limit list size to prevent memory exhaustion (DoS)
    documents: Annotated[
        list[LimitedString],
        Field(
            min_length=1,
            max_length=MAX_INPUT_ITEMS,
            description="List of documents to rerank. Limited to MAX_INPUT_ITEMS to prevent DoS.",
            examples=[
                [
                    "東京都は日本の首都であり、最大の都市です。",
                    "京都府は日本の古都として知られています。",
                ]
            ],
        ),
    ]
    model: str = Field(
        description="The name of the reranking model.",
        examples=["BAAI/bge-reranker-v2-m3"],
    )
    top_n: Optional[int] = Field(
        None,
        validation_alias="top_k",
        ge=0,
        le=MAX_INPUT_ITEMS,
        description="Number of top documents to return.",
        examples=[2],
    )
    return_documents: Optional[bool] = Field(
        None,
        description="Whether to include document text in the response.",
        examples=[True],
    )
    threshold: Optional[float] = Field(
        None,
        description="Sufficiency probability threshold (defaults to LOGIT_GATE_THRESHOLD)",
        examples=[0.5],
    )
    drop_failed: bool = Field(
        False,
        description="If True, documents with score < threshold are removed. If False (default), all documents are kept and sorted.",
    )
    use_ascii_boost: Optional[bool] = Field(
        None,
        description="Enable/disable ASCII 3-gram boost (None uses LOGIT_GATE_ASCII_BOOST_WEIGHT > 0)",
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "query": "日本の首都は？",
                "documents": [
                    "東京都は日本の首都であり、最大の都市です。",
                    "京都府は日本の古都として知られています。",
                    "富士山は日本で最も高い山です。",
                ],
                "model": "BAAI/bge-reranker-v2-m3",
                "top_n": 2,
                "return_documents": True,
            }
        },
    )


class RerankData(BaseModel):
    """Schema for a single reranked document result.

    Attributes:
        document (int): The index of the document in the original request.
        score (float): The reranking score.
        text (Optional[LimitedString]): The text of the document, if requested.
        passed (Optional[bool]): Whether the document passed the threshold.
        logit_margin (Optional[float]): The logit margin for hybrid reranking.
        containment_score (Optional[float]): The containment score for hybrid reranking.
        entropy (Optional[float]): The entropy for hybrid reranking.
    """

    document: int = Field(
        description="The index of the document in the original request.", examples=[0]
    )
    score: float = Field(description="The reranking score.", examples=[0.95])
    text: Optional[LimitedString] = Field(
        None,
        description="The text of the document, if requested.",
        examples=["東京都は日本の首都であり、最大の都市です。"],
    )
    passed: Optional[bool] = Field(
        None, description="Whether the document passed the threshold.", examples=[True]
    )
    logit_margin: Optional[float] = Field(
        None, description="The logit margin for hybrid reranking.", examples=[1.5]
    )
    containment_score: Optional[float] = Field(
        None, description="The containment score for hybrid reranking.", examples=[0.8]
    )
    entropy: Optional[float] = Field(
        None, description="The entropy for hybrid reranking.", examples=[0.2]
    )


class RerankResponse(BaseModel):
    """Schema for the reranking API response.

    Attributes:
        query (LimitedString): The original query.
        data (list[RerankData]): The list of reranked document results.
        model (str): The model used for reranking.
        usage (Optional[Usage]): Usage statistics for the request.
    """

    query: LimitedString = Field(
        description="The original query.", examples=["日本の首都は？"]
    )
    data: list[RerankData] = Field(description="The list of reranked document results.")
    model: str = Field(
        description="The model used for reranking.",
        examples=["BAAI/bge-reranker-v2-m3"],
    )
    usage: Optional[Usage] = Field(
        None, description="Usage statistics for the request."
    )


# --- For /v1/models ---
class ModelCard(BaseModel):
    """Schema for a single model card.

    Attributes:
        id (str): The model identifier.
        object (str): The object type, always 'model'.
        created (int): The Unix timestamp when the model was created.
        owned_by (str): The owner of the model, defaults to 'custom'.
        permission (list[Any]): The permissions for the model.
    """

    id: str = Field(
        description="The model identifier.", examples=["cl-nagoya/ruri-v3-small"]
    )
    object: str = Field("model", description="The object type.", examples=["model"])
    created: int = Field(
        description="The Unix timestamp when the model was created.",
        examples=[1677610602],
    )
    owned_by: str = Field(
        "custom", description="The owner of the model.", examples=["custom"]
    )
    permission: list[Any] = Field(
        default_factory=list, description="The permissions for the model."
    )


class ModelList(BaseModel):
    """Schema for a list of available models.

    Attributes:
        object (str): The object type, always 'list'.
        data (list[ModelCard]): The list of available models.
    """

    object: str = Field("list", description="The object type.", examples=["list"])
    data: list[ModelCard] = Field(description="The list of available models.")


# --- For /v1/models/unload ---
class UnloadRequest(BaseModel):
    """Schema for model unloading API request.

    Attributes:
        model (Optional[str]): The name of the model to unload.
    """

    model: Optional[str] = Field(
        None,
        description="The name of the model to unload. If omitted, all models are unloaded.",
        examples=["cl-nagoya/ruri-v3-small"],
    )


class UnloadResponse(BaseModel):
    """Schema for model unloading API response.

    Attributes:
        unloaded_models (list[str]): The names of the unloaded models.
        remaining_memory (int): The amount of memory remaining after unloading.
    """

    unloaded_models: list[str] = Field(
        description="The names of the unloaded models.",
        examples=[["cl-nagoya/ruri-v3-small"]],
    )
    remaining_memory: int = Field(
        description="The amount of memory remaining after unloading in bytes.",
        examples=[1024000],
    )


# --- OpenAPI Error Schema ---
class ErrorResponse(BaseModel):
    """Schema for standard error responses.

    Attributes:
        detail (str): Detailed human-readable error message.
    """

    detail: str = Field(
        description="Detailed human-readable error message explaining the failure.",
        examples=["Invalid or missing API Key"],
    )
