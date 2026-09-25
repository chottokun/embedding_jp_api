#!/usr/bin/env python3
"""
Logit Gate & Hybrid Rerank Accuracy Benchmark

Evaluates:
- Overall Accuracy, Precision, Recall, F1
- Near-miss Rejection Rate & Unanswerable Rejection Rate
- ASCII Boost Weight (\u03b2) sweep
- Threshold (\u03c4) sweep
- Token configuration comparison (Yes/No vs 1/0)
- Entropy distribution across positive, near-miss, and unanswerable samples
"""

import sys
import json
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from src.app.models import get_model  # noqa: E402
from src.app.services.logit_gate import (  # noqa: E402
    LogitGateService,
    _sigmoid,
    _binary_entropy,
)
from src.app.services.ascii_matcher import AsciiMatcher  # noqa: E402


DATASET_PATH = PROJECT_ROOT / "benchmarks" / "datasets" / "sufficiency_eval.json"


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_predictions(
    items: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    threshold: float,
) -> dict[str, float]:
    total = len(items)
    if total == 0:
        return {}

    tp = fp = fn = tn = 0
    near_miss_total = 0
    near_miss_rejected = 0
    unans_total = 0
    unans_rejected = 0

    margins = []
    entropies = []
    pos_margins = []
    near_margins = []
    unans_margins = []
    pos_entropies = []
    near_entropies = []
    unans_entropies = []

    for item, pred in zip(items, predictions):
        true_label = item["label"]
        score = pred["score"]
        pred_label = 1 if score >= threshold else 0
        item_type = item.get("type", "")

        margin = pred.get("logit_margin", 0.0)
        entropy = pred.get("entropy", 0.0)
        margins.append(margin)
        entropies.append(entropy)

        if item_type == "positive":
            pos_margins.append(margin)
            pos_entropies.append(entropy)
        elif item_type == "near_miss":
            near_miss_total += 1
            near_margins.append(margin)
            near_entropies.append(entropy)
            if pred_label == 0:
                near_miss_rejected += 1
        elif item_type == "unanswerable":
            unans_total += 1
            unans_margins.append(margin)
            unans_entropies.append(entropy)
            if pred_label == 0:
                unans_rejected += 1

        if true_label == 1 and pred_label == 1:
            tp += 1
        elif true_label == 0 and pred_label == 1:
            fp += 1
        elif true_label == 1 and pred_label == 0:
            fn += 1
        elif true_label == 0 and pred_label == 0:
            tn += 1

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    near_miss_rej = near_miss_rejected / near_miss_total if near_miss_total > 0 else 0.0
    unans_rej = unans_rejected / unans_total if unans_total > 0 else 0.0

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "near_miss_rejection_rate": near_miss_rej,
        "unanswerable_rejection_rate": unans_rej,
        "mean_margin_positive": mean(pos_margins),
        "mean_margin_near_miss": mean(near_margins),
        "mean_margin_unanswerable": mean(unans_margins),
        "mean_entropy_positive": mean(pos_entropies),
        "mean_entropy_near_miss": mean(near_entropies),
        "mean_entropy_unanswerable": mean(unans_entropies),
    }


def run_accuracy_benchmark(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_report_path: Path | None = None,
) -> dict[str, Any]:
    print("=" * 70)
    print(f"Logit Gate Accuracy Benchmark: {model_name} on {device}")
    print("=" * 70)

    dataset = load_dataset(DATASET_PATH)
    print(f"Loaded {len(dataset)} items from {DATASET_PATH}")
    type_counts = {}
    for item in dataset:
        t = item.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"Dataset breakdown: {type_counts}")

    print("\nLoading model...")
    t0 = time.time()
    model_wrapper = get_model(model_name, device=device)
    print(f"Model loaded in {time.time() - t0:.2f}s")

    gate_service = LogitGateService(model_wrapper)
    ascii_matcher = AsciiMatcher(min_token_len=3)

    print("\nExecuting forward passes on dataset...")
    queries = [item["query"] for item in dataset]
    documents = [item["document"] for item in dataset]

    # Pre-compute raw logit margins and ASCII containment for each item
    raw_results = []
    t_start = time.time()
    for i, (q, d) in enumerate(zip(queries, documents)):
        # Run gate
        gate_out = gate_service.predict_margins(q, [d])[0]
        # Run ascii matcher
        containment = ascii_matcher.score_document(
            ascii_matcher.extract_identifiers(q), d
        )
        raw_results.append(
            {
                "logit_margin": gate_out["logit_margin"],
                "containment": containment,
            }
        )
    elapsed = time.time() - t_start
    print(
        f"Evaluated {len(dataset)} items in {elapsed:.2f}s ({elapsed / len(dataset) * 1000:.1f}ms/item)"
    )

    # 1. Sweep Beta (\u03b2) with fixed threshold 0.55
    beta_values = [0.0, 0.5, 1.0, 1.2, 1.5, 2.0, 2.5]
    fixed_threshold = 0.55
    beta_results = {}
    print(f"\n--- Beta (\u03b2) Sweep (Threshold = {fixed_threshold}) ---")
    for beta in beta_values:
        preds = []
        for r in raw_results:
            delta_final = r["logit_margin"] + (beta * r["containment"])
            score = _sigmoid(delta_final)
            entropy = _binary_entropy(score)
            preds.append(
                {
                    "score": score,
                    "logit_margin": delta_final,
                    "entropy": entropy,
                }
            )
        metrics = evaluate_predictions(dataset, preds, fixed_threshold)
        beta_results[beta] = metrics
        print(
            f"  \u03b2={beta:3.1f} | Acc: {metrics['accuracy'] * 100:5.1f}% | "
            f"F1: {metrics['f1'] * 100:5.1f}% | "
            f"Near-Miss Rej: {metrics['near_miss_rejection_rate'] * 100:5.1f}% | "
            f"Unans Rej: {metrics['unanswerable_rejection_rate'] * 100:5.1f}%"
        )

    # 2. Sweep Threshold (\u03c4) with optimal beta (\u03b2 = 1.2)
    optimal_beta = 1.2
    threshold_values = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    threshold_results = {}
    print(f"\n--- Threshold (\u03c4) Sweep (\u03b2 = {optimal_beta}) ---")
    for th in threshold_values:
        preds = []
        for r in raw_results:
            delta_final = r["logit_margin"] + (optimal_beta * r["containment"])
            score = _sigmoid(delta_final)
            entropy = _binary_entropy(score)
            preds.append(
                {
                    "score": score,
                    "logit_margin": delta_final,
                    "entropy": entropy,
                }
            )
        metrics = evaluate_predictions(dataset, preds, th)
        threshold_results[th] = metrics
        print(
            f"  \u03c4={th:4.2f} | Acc: {metrics['accuracy'] * 100:5.1f}% | "
            f"Prec: {metrics['precision'] * 100:5.1f}% | "
            f"Rec: {metrics['recall'] * 100:5.1f}% | "
            f"F1: {metrics['f1'] * 100:5.1f}% | "
            f"Near-Miss Rej: {metrics['near_miss_rejection_rate'] * 100:5.1f}%"
        )

    # 3. Token Comparison: Yes/No vs 1/0
    print("\n--- Token Comparison: Yes/No vs 1/0 ---")
    token_comparison = {}
    # Yes/No result
    preds_yes_no = []
    for r in raw_results:
        delta_final = r["logit_margin"] + (optimal_beta * r["containment"])
        score = _sigmoid(delta_final)
        preds_yes_no.append(
            {
                "score": score,
                "logit_margin": delta_final,
                "entropy": _binary_entropy(score),
            }
        )
    token_comparison["Yes/No"] = evaluate_predictions(
        dataset, preds_yes_no, fixed_threshold
    )

    # Switch to 1/0
    print("Testing 1/0 tokens...")
    gate_service_1_0 = LogitGateService(model_wrapper)
    gate_service_1_0.pos_token_ids = gate_service_1_0._get_token_ids(["1"])
    gate_service_1_0.neg_token_ids = gate_service_1_0._get_token_ids(["0"])
    preds_1_0 = []
    for i, (q, d) in enumerate(zip(queries, documents)):
        gate_out = gate_service_1_0.predict_margins(q, [d])[0]
        containment = raw_results[i]["containment"]
        delta_final = gate_out["logit_margin"] + (optimal_beta * containment)
        score = _sigmoid(delta_final)
        preds_1_0.append(
            {
                "score": score,
                "logit_margin": delta_final,
                "entropy": _binary_entropy(score),
            }
        )
    token_comparison["1/0"] = evaluate_predictions(dataset, preds_1_0, fixed_threshold)

    for token_name, met in token_comparison.items():
        print(
            f"  Token '{token_name}' | Acc: {met['accuracy'] * 100:5.1f}% | "
            f"F1: {met['f1'] * 100:5.1f}% | "
            f"Near-Miss Rej: {met['near_miss_rejection_rate'] * 100:5.1f}% | "
            f"Mean \u0394z (Pos/Near): {met['mean_margin_positive']:+.2f} / {met['mean_margin_near_miss']:+.2f}"
        )

    # Generate Markdown Report
    best_metrics = threshold_results[fixed_threshold]
    report_md = f"""# Logit Gate & Hybrid Rerank Accuracy Benchmark Report

- **Model**: `{model_name}`
- **Device**: `{device}`
- **Dataset**: `{DATASET_PATH.name}` (Total $N = {len(dataset)}$, Breakdown: {type_counts})
- **Execution Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}

## 1. Executive Summary

| Metric | Measured Value | Target / Baseline |
| :--- | :--- | :--- |
| **Overall Accuracy** | **{best_metrics["accuracy"] * 100:.1f}%** | \u2265 85.0% |
| **Precision** | **{best_metrics["precision"] * 100:.1f}%** | \u2265 85.0% |
| **Recall** | **{best_metrics["recall"] * 100:.1f}%** | \u2265 80.0% |
| **F1 Score** | **{best_metrics["f1"] * 100:.1f}%** | \u2265 85.0% |
| **Near-Miss Rejection Rate** | **{best_metrics["near_miss_rejection_rate"] * 100:.1f}%** | \u2265 90.0% |
| **Unanswerable Rejection Rate** | **{best_metrics["unanswerable_rejection_rate"] * 100:.1f}%** | \u2265 90.0% |

## 2. Beta (\u03b2) Boost Sensitivity Analysis (\u03c4 = {fixed_threshold})

| \u03b2 (Boost Weight) | Accuracy | F1 Score | Near-Miss Rejection | Unanswerable Rejection |
| :---: | :---: | :---: | :---: | :---: |
"""
    for beta, met in beta_results.items():
        report_md += f"| {beta:.1f} | {met['accuracy'] * 100:.1f}% | {met['f1'] * 100:.1f}% | {met['near_miss_rejection_rate'] * 100:.1f}% | {met['unanswerable_rejection_rate'] * 100:.1f}% |\n"

    report_md += f"""
## 3. Threshold (\u03c4) Sensitivity Analysis (\u03b2 = {optimal_beta})

| \u03c4 (Threshold) | Accuracy | Precision | Recall | F1 Score | Near-Miss Rejection |
| :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for th, met in threshold_results.items():
        report_md += f"| {th:.2f} | {met['accuracy'] * 100:.1f}% | {met['precision'] * 100:.1f}% | {met['recall'] * 100:.1f}% | {met['f1'] * 100:.1f}% | {met['near_miss_rejection_rate'] * 100:.1f}% |\n"

    report_md += """
## 4. Token Vocabulary Comparison (Yes/No vs 1/0)

| Token Set | Accuracy | F1 Score | Near-Miss Rejection | Mean \u0394z (Positive) | Mean \u0394z (Near-Miss) |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for t_name, met in token_comparison.items():
        report_md += f"| `{t_name}` | {met['accuracy'] * 100:.1f}% | {met['f1'] * 100:.1f}% | {met['near_miss_rejection_rate'] * 100:.1f}% | {met['mean_margin_positive']:+.2f} | {met['mean_margin_near_miss']:+.2f} |\n"

    report_md += f"""
## 5. Binary Entropy & Uncertainty Distribution

| Sample Category | Mean Logit Margin \u0394z | Mean Binary Entropy $H_{{binary}}$ | Interpretation |
| :--- | :---: | :---: | :--- |
| **Positive ($N=18$)** | `{best_metrics["mean_margin_positive"]:+.2f}` | `{best_metrics["mean_entropy_positive"]:.3f}` | High confidence sufficiency |
| **Near-Miss ($N=18$)** | `{best_metrics["mean_margin_near_miss"]:+.2f}` | `{best_metrics["mean_entropy_near_miss"]:.3f}` | Decisive rejection of topical non-evidence |
| **Unanswerable ($N=18$)** | `{best_metrics["mean_margin_unanswerable"]:+.2f}` | `{best_metrics["mean_entropy_unanswerable"]:.3f}` | Decisive rejection of off-topic context |
"""

    if output_report_path:
        with open(output_report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"\nReport saved to: {output_report_path}")

    return {
        "best_metrics": best_metrics,
        "beta_results": beta_results,
        "threshold_results": threshold_results,
        "token_comparison": token_comparison,
        "report_md": report_md,
    }


if __name__ == "__main__":
    report_file = PROJECT_ROOT / "plan" / "benchmark_accuracy_results.md"
    run_accuracy_benchmark(output_report_path=report_file)
