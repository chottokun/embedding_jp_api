from typing import List
from .base import BaseEmbeddingService, BaseRerankService, BaseDecisionsService
from ..schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
    Usage,
    RerankRequest,
    RerankResponse,
    RerankData,
    DecisionsRequest,
    DecisionsResponse,
    PredicateAnswer,
    ChoiceAnswer,
    ScoreAnswer,
    RefusalAnswer,
)


class MockEmbeddingService(BaseEmbeddingService):
    """
    Mock Embedding Service for fast CI and unit tests without loading actual models or weights.
    Returns normalized dummy vectors.
    """

    def __init__(self, vector_dim: int = 1024):
        self.vector_dim = vector_dim

    async def create_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        input_data = request.input
        if isinstance(input_data, list):
            # Check if it's a list of ContentParts (single item) vs list of SingleInputItems
            if input_data and (
                isinstance(input_data[0], dict)
                and "type" in input_data[0]
                or hasattr(input_data[0], "type")
            ):
                num_items = 1
            else:
                num_items = len(input_data)
        else:
            num_items = 1

        dummy_vector = [0.1] * self.vector_dim
        response_data = [
            EmbeddingData(embedding=dummy_vector, index=i) for i in range(num_items)
        ]
        usage = Usage(prompt_tokens=10, total_tokens=10)
        return EmbeddingResponse(data=response_data, model=request.model, usage=usage)


class MockRerankService(BaseRerankService):
    """
    Mock Rerank Service for fast CI and unit tests without loading actual models.
    """

    async def create_rerank(self, request: RerankRequest) -> RerankResponse:
        results: List[RerankData] = []
        for i, doc in enumerate(request.documents):
            score = float(len(request.documents) - i) / len(request.documents)
            text_val = doc if request.return_documents else None
            results.append(RerankData(document=i, score=score, text=text_val))

        if request.top_n is not None:
            results = results[: request.top_n]

        usage = Usage(prompt_tokens=10, total_tokens=10)
        return RerankResponse(
            query=request.query, data=results, model=request.model, usage=usage
        )


class MockDecisionsService(BaseDecisionsService):
    """
    Mock Decisions Service for fast CI and unit tests without loading actual models.
    """

    async def create_decisions(self, request: DecisionsRequest) -> DecisionsResponse:
        output_answers = []
        for q in request.questions:
            if q.type == "predicate":
                output_answers.append(PredicateAnswer(name=q.name, probability=0.85))
            elif q.type == "choice":
                if not q.options:
                    output_answers.append(
                        RefusalAnswer(
                            name=q.name,
                            refusal="Choice question requires non-empty options.",
                        )
                    )
                else:
                    selected = q.options[0]
                    probs = {
                        opt: (1.0 if idx == 0 else 0.0)
                        for idx, opt in enumerate(q.options)
                    }
                    output_answers.append(
                        ChoiceAnswer(
                            name=q.name,
                            choice=selected,
                            probabilities=probs,
                            confidence=0.95,
                        )
                    )
            elif q.type == "score":
                if not q.levels:
                    output_answers.append(
                        RefusalAnswer(
                            name=q.name,
                            refusal="Score question requires non-empty levels.",
                        )
                    )
                else:
                    num_levels = len(q.levels)
                    probs = [1.0 / num_levels] * num_levels
                    output_answers.append(
                        ScoreAnswer(
                            name=q.name,
                            score=1.5,
                            probabilities=probs,
                        )
                    )
            else:
                output_answers.append(
                    RefusalAnswer(
                        name=q.name,
                        refusal=f"Unsupported question type: {q.type}",
                    )
                )

        return DecisionsResponse(answers=output_answers)
