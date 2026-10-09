from typing import Any
import httpx
from fastapi import HTTPException

from app.config import EMBEDDING_MODELS, RERANK_MODELS
from app.models import get_model
from app.services import (
    BaseEmbeddingService,
    BaseRerankService,
    EmbeddingService,
    RerankService,
)
from app.services.base import get_validated_model


async def _proxy_to_tei(tei_url: str, path: str, json_data: dict) -> Any:
    """Helper to send an async POST request to TEI and return the JSON response."""
    try:
        # Avoid importing app globally to prevent circular imports
        from app.main import app

        shared_client = getattr(app.state, "tei_client", None)
        if (
            shared_client is not None
            and getattr(shared_client, "is_closed", False) is not True
        ):
            response = await shared_client.post(f"{tei_url}{path}", json=json_data)
        else:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{tei_url}{path}", json=json_data)

        if response.status_code != 200:
            error_msg = response.text
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            raise HTTPException(
                status_code=500,
                detail=f"TEI Proxy Error ({response.status_code}): {error_msg}",
            )
        return response.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to proxy request to TEI: {str(e)}"
        )


def _get_model_or_400(model_name: str, model_type: str) -> Any:
    """Helper for backwards compatibility with legacy tests calling _get_model_or_400."""
    supported_models = EMBEDDING_MODELS if model_type == "embedding" else RERANK_MODELS
    loader = get_model
    try:
        import app.main as main_mod

        loader = getattr(main_mod, "get_model", loader)
    except ImportError:
        pass
    return get_validated_model(model_name, supported_models, model_type, loader=loader)


def get_embedding_service() -> BaseEmbeddingService:
    loader = get_model
    proxy_func = _proxy_to_tei
    try:
        import app.main as main_mod

        loader = getattr(main_mod, "get_model", loader)
        proxy_func = getattr(main_mod, "_proxy_to_tei", proxy_func)
    except ImportError:
        pass
    return EmbeddingService(proxy_to_tei_func=proxy_func, model_loader=loader)


def get_rerank_service() -> BaseRerankService:
    loader = get_model
    proxy_func = _proxy_to_tei
    try:
        import app.main as main_mod

        loader = getattr(main_mod, "get_model", loader)
        proxy_func = getattr(main_mod, "_proxy_to_tei", proxy_func)
    except ImportError:
        pass
    return RerankService(proxy_to_tei_func=proxy_func, model_loader=loader)
