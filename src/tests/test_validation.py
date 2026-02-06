import pytest
from pydantic import ValidationError
from app.schemas import EmbeddingRequest, RerankRequest
from app.config import MAX_INPUT_LENGTH, MAX_INPUT_ITEMS

def test_embedding_request_valid():
    """Test valid EmbeddingRequest inputs."""
    EmbeddingRequest(input="hello", model="model")
    EmbeddingRequest(input=["hello"], model="model")

    # Boundary check: exact limits should be allowed
    EmbeddingRequest(input="a" * MAX_INPUT_LENGTH, model="model")

def test_embedding_request_string_too_long():
    """Test EmbeddingRequest with input string exceeding max length."""
    # We construct a string slightly longer than the limit
    long_string = "a" * (MAX_INPUT_LENGTH + 1)
    with pytest.raises(ValidationError) as excinfo:
        EmbeddingRequest(input=long_string, model="model")
    assert "String should have at most" in str(excinfo.value)

def test_embedding_request_list_item_too_long():
    """Test EmbeddingRequest with a list item exceeding max length."""
    long_string = "a" * (MAX_INPUT_LENGTH + 1)
    with pytest.raises(ValidationError) as excinfo:
        EmbeddingRequest(input=[long_string], model="model")
    assert "String should have at most" in str(excinfo.value)

def test_embedding_request_list_too_many_items():
    """Test EmbeddingRequest with too many items in input list."""
    items = ["a"] * (MAX_INPUT_ITEMS + 1)
    with pytest.raises(ValidationError) as excinfo:
        EmbeddingRequest(input=items, model="model")
    assert "List should have at most" in str(excinfo.value)

def test_rerank_request_valid():
    """Test valid RerankRequest inputs."""
    RerankRequest(query="hello", documents=["doc"], model="model")
    # Boundary check
    RerankRequest(query="a" * MAX_INPUT_LENGTH, documents=["doc"], model="model")
    RerankRequest(query="hello", documents=["a" * MAX_INPUT_LENGTH], model="model")

def test_rerank_request_query_too_long():
    """Test RerankRequest with query exceeding max length."""
    long_string = "a" * (MAX_INPUT_LENGTH + 1)
    with pytest.raises(ValidationError) as excinfo:
        RerankRequest(query=long_string, documents=["doc"], model="model")
    assert "String should have at most" in str(excinfo.value)

def test_rerank_request_document_too_long():
    """Test RerankRequest with a document exceeding max length."""
    long_string = "a" * (MAX_INPUT_LENGTH + 1)
    with pytest.raises(ValidationError) as excinfo:
        RerankRequest(query="hello", documents=[long_string], model="model")
    assert "String should have at most" in str(excinfo.value)

def test_rerank_request_documents_too_many_items():
    """Test RerankRequest with too many documents."""
    items = ["doc"] * (MAX_INPUT_ITEMS + 1)
    with pytest.raises(ValidationError) as excinfo:
        RerankRequest(query="hello", documents=items, model="model")
    assert "List should have at most" in str(excinfo.value)
