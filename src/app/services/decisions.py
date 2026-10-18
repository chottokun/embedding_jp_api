import anyio
import numpy as np
from typing import Any, Callable, Optional

from app.config import (
    LOGIT_GATE_MODELS,
    LOGIT_GATE_ASCII_MIN_TOKEN_LEN,
    LOGIT_GATE_ASCII_BOOST_WEIGHT,
)
from app.schemas import (
    DecisionsRequest,
    DecisionsResponse,
    DecisionInputMessage,
    PredicateAnswer,
    ChoiceAnswer,
    ScoreAnswer,
    RefusalAnswer,
)
from app.services.ascii_matcher import AsciiMatcher
from app.services.base import BaseDecisionsService, get_validated_model
from app.services.logit_gate import LogitGateService, _sigmoid


def _extract_context(input_messages: list[DecisionInputMessage]) -> str:
    text_parts = []
    for msg in input_messages:
        for item in msg.content:
            if item.text:
                text_parts.append(item.text)
    return "\n\n".join(text_parts) if text_parts else ""


def _softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    arr = np.array(logits, dtype=np.float64)
    exp_arr = np.exp(arr - np.max(arr))
    probs = exp_arr / np.sum(exp_arr)
    return probs.tolist()


class DecisionsService(BaseDecisionsService):
    """
    Production Decisions Service utilizing LogitGateService and AsciiMatcher
    for probability and logit margin computations on decision questions.
    """

    def __init__(self, model_loader: Optional[Callable[[str], Any]] = None):
        self.model_loader = model_loader

    async def create_decisions(self, request: DecisionsRequest) -> DecisionsResponse:
        model_wrapper = get_validated_model(
            request.model,
            LOGIT_GATE_MODELS,
            "decisions",
            loader=self.model_loader,
        )

        def _evaluate_sync() -> DecisionsResponse:
            gate_service = LogitGateService(model_wrapper)
            matcher = AsciiMatcher(min_token_len=LOGIT_GATE_ASCII_MIN_TOKEN_LEN)
            beta = LOGIT_GATE_ASCII_BOOST_WEIGHT

            context = _extract_context(request.input)
            output_answers = []

            for q in request.questions:
                if q.type == "predicate":
                    res = gate_service.predict_margins(
                        query=q.instructions, documents=[context]
                    )
                    delta_z = res[0]["logit_margin"] if res else 0.0
                    containment = matcher.score_documents(q.instructions, [context])[0]
                    prob = _sigmoid(delta_z + (beta * containment))
                    output_answers.append(
                        PredicateAnswer(name=q.name, probability=float(prob))
                    )

                elif q.type == "choice":
                    if not q.options:
                        output_answers.append(
                            RefusalAnswer(
                                name=q.name,
                                refusal="Choice question requires non-empty options.",
                            )
                        )
                    else:
                        unique_options = list(dict.fromkeys(q.options))
                        opt_margins = []
                        for opt in unique_options:
                            query_opt = f"{q.instructions}\nChoice option: {opt}"
                            res = gate_service.predict_margins(
                                query=query_opt, documents=[context]
                            )
                            delta_z = res[0]["logit_margin"] if res else 0.0
                            containment = matcher.score_documents(query_opt, [context])[
                                0
                            ]
                            opt_margins.append(delta_z + (beta * containment))

                        probs_list = _softmax(opt_margins)
                        probs_dict = {
                            opt: float(p) for opt, p in zip(unique_options, probs_list)
                        }

                        best_opt = max(probs_dict, key=lambda k: probs_dict[k])
                        confidence = probs_dict[best_opt]

                        output_answers.append(
                            ChoiceAnswer(
                                name=q.name,
                                choice=best_opt,
                                probabilities=probs_dict,
                                confidence=confidence,
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
                        lvl_margins = []
                        for lvl in q.levels:
                            query_lvl = f"{q.instructions}\nScore level {lvl.index}: {lvl.description}"
                            res = gate_service.predict_margins(
                                query=query_lvl, documents=[context]
                            )
                            delta_z = res[0]["logit_margin"] if res else 0.0
                            containment = matcher.score_documents(query_lvl, [context])[
                                0
                            ]
                            lvl_margins.append(delta_z + (beta * containment))

                        probs_list = _softmax(lvl_margins)
                        expected_score = float(
                            sum(lvl.index * p for lvl, p in zip(q.levels, probs_list))
                        )

                        output_answers.append(
                            ScoreAnswer(
                                name=q.name,
                                score=expected_score,
                                probabilities=[float(p) for p in probs_list],
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

        return await anyio.to_thread.run_sync(_evaluate_sync)
