import torch
import math
from typing import Any, List, Dict
from ..config import (
    LOGIT_GATE_POS_TOKENS,
    LOGIT_GATE_NEG_TOKENS,
    LOGIT_GATE_MAX_DOC_CHARS,
    LOGIT_GATE_BATCH_SIZE,
)


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _binary_entropy(p: float, eps: float = 1e-12) -> float:
    """Normalized binary Shannon entropy in range [0.0, 1.0]."""
    p_clamped = max(eps, min(1.0 - eps, p))
    h = -(
        p_clamped * math.log2(p_clamped)
        + (1.0 - p_clamped) * math.log2(1.0 - p_clamped)
    )
    return max(0.0, min(1.0, float(h)))


class LogitGateService:
    def __init__(self, model_wrapper: Any):
        self.model_wrapper = model_wrapper
        self.tokenizer = model_wrapper.tokenizer
        self.model = model_wrapper.model
        self.device = model_wrapper.device

        # Pre-compute token IDs
        self.pos_token_ids = self._get_token_ids(LOGIT_GATE_POS_TOKENS)
        self.neg_token_ids = self._get_token_ids(LOGIT_GATE_NEG_TOKENS)

        self.baseline_margin: float = 0.0

    def _get_token_ids(self, tokens: List[str]) -> List[int]:
        token_ids = set()
        for t in tokens:
            # We add dummy prefix to avoid stripping issues, taking the first generated token
            ids = self.tokenizer.encode(" " + t, add_special_tokens=False)
            if ids:
                token_ids.add(ids[0])
            # Also try without prefix
            ids = self.tokenizer.encode(t, add_special_tokens=False)
            if ids:
                token_ids.add(ids[0])
        return list(token_ids)

    def _build_prompt(self, query: str, document: str) -> str:
        truncated_doc = document[:LOGIT_GATE_MAX_DOC_CHARS]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert retrieval verifier. "
                    "Determine whether the Context contains sufficient information to directly answer the Question. "
                    "Answer only 'Yes' or 'No'."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nContext: {truncated_doc}",
            },
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        # Fallback to standard ChatML
        return (
            "<|im_start|>system\nYou are an expert retrieval verifier. "
            "Determine whether the Context contains sufficient information to directly answer the Question. "
            "Answer only 'Yes' or 'No'.<|im_end|>\n"
            f"<|im_start|>user\nQuestion: {query}\n\nContext: {truncated_doc}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def calibrate(self, empty_query: str = "", empty_document: str = "") -> float:
        prompt = self._build_prompt(empty_query, empty_document)

        with self.model_wrapper.tokenizer_lock:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with self.model_wrapper.lock:
            with torch.no_grad():
                outputs = self.model(**inputs)
                next_token_logits = outputs.logits[0, -1, :]

        pos_logits = next_token_logits[self.pos_token_ids]
        neg_logits = next_token_logits[self.neg_token_ids]

        z_pos = torch.logsumexp(pos_logits, dim=-1).item()
        z_neg = torch.logsumexp(neg_logits, dim=-1).item()

        self.baseline_margin = z_pos - z_neg
        return self.baseline_margin

    def predict_margins(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        results = []

        for i in range(0, len(documents), LOGIT_GATE_BATCH_SIZE):
            batch_docs = documents[i : i + LOGIT_GATE_BATCH_SIZE]
            batch_indices = list(range(i, i + len(batch_docs)))

            prompts = [self._build_prompt(query, doc) for doc in batch_docs]

            with self.model_wrapper.tokenizer_lock:
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
                    self.device
                )

            with self.model_wrapper.lock:
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    next_token_logits = outputs.logits[:, -1, :]

            for j, logits in enumerate(next_token_logits):
                pos_logits = logits[self.pos_token_ids]
                neg_logits = logits[self.neg_token_ids]

                z_pos = torch.logsumexp(pos_logits, dim=-1).item()
                z_neg = torch.logsumexp(neg_logits, dim=-1).item()

                delta_z = z_pos - z_neg - self.baseline_margin
                p_sufficient = _sigmoid(delta_z)

                results.append(
                    {
                        "document_index": batch_indices[j],
                        "logit_margin": delta_z,
                        "sufficiency_prob": p_sufficient,
                        "entropy": _binary_entropy(p_sufficient),
                        "text": batch_docs[j],
                    }
                )

        return results
