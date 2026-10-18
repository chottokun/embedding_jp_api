from fastapi import APIRouter, Depends, HTTPException

from app.schemas import DecisionsRequest, DecisionsResponse, ErrorResponse
from app.dependencies.auth import verify_api_key

router = APIRouter(prefix="/v1", tags=["Decisions"])


@router.post(
    "/decisions",
    response_model=DecisionsResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Create decision answers for given input context and questions",
    description="OpenAI Decisions API compatible endpoint returning decision probabilities and choices.",
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Unsupported model or invalid parameters",
        },
        401: {
            "model": ErrorResponse,
            "description": "Invalid or missing Bearer API key",
        },
    },
)
async def create_decision(request: DecisionsRequest):
    """
    Evaluates questions against input context and produces decision responses.
    """
    if request.model != "gpt-6-luna":
        raise HTTPException(
            status_code=400,
            detail="Only 'gpt-6-luna' model is supported in Decisions API.",
        )

    output_answers = []

    for q in request.questions:
        if q.type == "predicate":
            output_answers.append(
                {
                    "name": q.name,
                    "type": "predicate",
                    "probability": 0.85,
                }
            )
        elif q.type == "choice":
            selected = q.options[0] if q.options else "other"
            options_list = q.options or [selected]
            probs = {
                opt: (1.0 if idx == 0 else 0.0)
                for idx, opt in enumerate(options_list)
            }
            output_answers.append(
                {
                    "name": q.name,
                    "type": "choice",
                    "choice": selected,
                    "probabilities": probs,
                    "confidence": 0.95,
                }
            )
        elif q.type == "score":
            num_levels = len(q.levels) if q.levels else 1
            dummy_probs = [1.0 / num_levels] * num_levels
            output_answers.append(
                {
                    "name": q.name,
                    "type": "score",
                    "score": 1.5,
                    "probabilities": dummy_probs,
                }
            )
        else:
            output_answers.append(
                {
                    "name": q.name,
                    "type": "refusal",
                    "refusal": f"Unsupported question type: {q.type}",
                }
            )

    return {"answers": output_answers}
