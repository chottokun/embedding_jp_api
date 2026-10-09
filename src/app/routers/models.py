import time
from fastapi import APIRouter, Depends

from app.schemas import ModelCard, ModelList, UnloadRequest, UnloadResponse
from app.models import unload_model
from app.config import EMBEDDING_MODELS, RERANK_MODELS
from app.dependencies.auth import verify_api_key

router = APIRouter(prefix="/v1", tags=["Models"])


@router.get(
    "/models",
    response_model=ModelList,
    dependencies=[Depends(verify_api_key)],
    summary="List available models",
    description="Lists all currently supported embedding and reranking models in OpenAI format.",
    responses={
        401: {"description": "Invalid or missing Bearer API key"},
        429: {"description": "Rate limit exceeded (Too Many Requests)"},
    },
)
async def list_models():
    """
    Lists all available models in the OpenAI-compatible format.
    """
    embedding_models = EMBEDDING_MODELS
    rerank_models = RERANK_MODELS
    try:
        import app.main as main_mod

        embedding_models = getattr(main_mod, "EMBEDDING_MODELS", embedding_models)
        rerank_models = getattr(main_mod, "RERANK_MODELS", rerank_models)
    except ImportError:
        pass

    all_models = set(embedding_models + rerank_models)
    current_time = int(time.time())

    models = [
        ModelCard(
            id=model_id,
            created=current_time,
        )
        for model_id in sorted(list(all_models))
    ]

    return ModelList(data=models)


@router.post(
    "/models/unload",
    response_model=UnloadResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Unload a specific model from cache",
    description=(
        "Unloads a specific model from memory/VRAM, triggering garbage collection "
        "and CUDA cache clearance."
    ),
    responses={
        401: {"description": "Invalid or missing Bearer API key"},
        429: {"description": "Rate limit exceeded (Too Many Requests)"},
    },
)
async def unload_models(request: UnloadRequest):
    """
    Unloads a specific model from cache, freeing memory / VRAM.
    """
    unloader = unload_model
    try:
        import app.main as main_mod

        unloader = getattr(main_mod, "unload_model", unloader)
    except ImportError:
        pass

    unloaded_models, remaining_memory = unloader(request.model)
    return UnloadResponse(
        unloaded_models=unloaded_models,
        remaining_memory=remaining_memory,
    )
