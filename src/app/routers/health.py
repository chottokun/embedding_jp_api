from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

router = APIRouter(tags=["Health"])


@router.get("/health")
@router.get("/healthz")
async def health_check():
    """
    Liveness probe for microservice orchestrators and Docker health checks.
    """
    return {"status": "ok"}


@router.get("/ready")
@router.get("/readyz")
async def readiness_check():
    """
    Readiness probe verifying model loading status and GPU availability.
    """
    import torch
    from app.models import _model_cache

    return {
        "status": "ready",
        "gpu_available": torch.cuda.is_available(),
        "models_loaded": list(_model_cache.keys()),
    }


@router.get("/metrics", tags=["Metrics"])
async def metrics():
    """
    Exposes Prometheus metrics.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
