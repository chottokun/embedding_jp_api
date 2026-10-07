#!/usr/bin/env python3
"""
Comprehensive 540-Item Benchmark: Cross-Encoder 310M vs Qwen3.5-0.8B vs Qwen2.5-1.5B
Evaluates on CPU and GPU across 6 specialized domains.
"""

import argparse
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402

from app.models import get_model, unload_model  # noqa: E402
from app.services.ascii_matcher import AsciiMatcher  # noqa: E402
from app.services.logit_gate import LogitGateService, _sigmoid  # noqa: E402


def evaluate_predictions(
    scores: list[float], labels: list[int], types: list[str]
) -> dict[str, Any]:
    best_th = 0.5
    best_f1 = 0.0
    best_acc = 0.0
    best_prec = 0.0
    best_rec = 0.0
    best_near_rej = 0.0
    best_unans_rej = 0.0

    pos_scores = [s for s, t in zip(scores, types) if t == "positive"]
    near_scores = [s for s, t in zip(scores, types) if t == "near_miss"]
    unans_scores = [s for s, t in zip(scores, types) if t == "unanswerable"]

    for th in np.arange(0.01, 0.99, 0.01):
        preds = [1 if s >= th else 0 for s in scores]
        tp = sum(1 for p, label in zip(preds, labels) if p == 1 and label == 1)
        fp = sum(1 for p, label in zip(preds, labels) if p == 1 and label == 0)
        fn = sum(1 for p, label in zip(preds, labels) if p == 0 and label == 1)
        tn = sum(1 for p, label in zip(preds, labels) if p == 0 and label == 0)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / len(labels)

        near_rej = (
            sum(1 for p, t in zip(preds, types) if t == "near_miss" and p == 0)
            / len(near_scores)
            if near_scores
            else 0.0
        )
        unans_rej = (
            sum(1 for p, t in zip(preds, types) if t == "unanswerable" and p == 0)
            / len(unans_scores)
            if unans_scores
            else 0.0
        )

        if f1 > best_f1 or (f1 == best_f1 and acc > best_acc):
            best_f1 = f1
            best_th = th
            best_acc = acc
            best_prec = prec
            best_rec = rec
            best_near_rej = near_rej
            best_unans_rej = unans_rej

    mean_pos = float(np.mean(pos_scores)) if pos_scores else 0.0
    mean_near = float(np.mean(near_scores)) if near_scores else 0.0
    mean_unans = float(np.mean(unans_scores)) if unans_scores else 0.0

    return {
        "best_threshold": round(float(best_th), 2),
        "f1": round(float(best_f1), 4),
        "accuracy": round(float(best_acc), 4),
        "precision": round(float(best_prec), 4),
        "recall": round(float(best_rec), 4),
        "near_miss_rejection_rate": round(float(best_near_rej), 4),
        "unanswerable_rejection_rate": round(float(best_unans_rej), 4),
        "mean_positive": round(mean_pos, 4),
        "mean_near_miss": round(mean_near, 4),
        "mean_unanswerable": round(mean_unans, 4),
        "discrimination_gap": round(mean_pos - mean_near, 4),
    }


def run_benchmark(
    device: str = "cuda",
    dataset_path: Path = Path("benchmarks/datasets/sufficiency_eval_v2_540.json"),
    output_json: Path = None,
):
    with open(dataset_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    print(f"📊 Starting 540-item Benchmark on Device: {device.upper()}")
    print(f"Dataset items: {len(items)}")

    labels = [item["label"] for item in items]
    types = [item["type"] for item in items]
    domains = [item.get("domain", "unspecified") for item in items]

    results = {"device": device, "total_items": len(items), "models": {}}

    # Model list
    models_to_test = [
        ("cross_encoder_310m", "cl-nagoya/ruri-v3-reranker-310m", "ce"),
        ("qwen35_08b", "Takenoko12345678/Qwen3.5-0.8B-Japanese-SFT-v2", "logit_gate"),
        ("qwen25_15b", "Qwen/Qwen2.5-1.5B-Instruct", "logit_gate"),
    ]

    for model_key, model_name, model_type in models_to_test:
        print("\n==========================================")
        print(f"Testing Model: {model_key} ({model_name}) on {device.upper()}")
        print("==========================================")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        scores = []

        if model_type == "ce":
            ce = CrossEncoder(model_name, device=device)
            # Warmup
            ce.predict([("test", "test")])
            t0 = time.perf_counter()

            pairs = [(item["query"], item["document"]) for item in items]
            raw_scores = ce.predict(pairs, batch_size=16, show_progress_bar=True)
            scores = [float(_sigmoid(s)) for s in raw_scores]
            inference_duration = time.perf_counter() - t0

            del ce
        else:
            matcher = AsciiMatcher()
            model_info = get_model(model_name, device=device)
            gate_service = LogitGateService(model_info)
            # Warmup
            gate_service.predict_margins("テスト", ["テスト文書"])
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            for idx, item in enumerate(items):
                res = gate_service.predict_margins(item["query"], [item["document"]])[0]
                c = matcher.score_documents(item["query"], [item["document"]])[0]
                final_z = res["logit_margin"] + 1.2 * c
                scores.append(float(_sigmoid(final_z)))
                if (idx + 1) % 50 == 0 or idx == len(items) - 1:
                    print(
                        f"  Processed {idx + 1}/{len(items)} items...",
                        end="\r",
                        flush=True,
                    )
            print()
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_duration = time.perf_counter() - t0
            unload_model(model_name)

        gc.collect()
        peak_vram_mb = 0.0
        if torch.cuda.is_available() and device == "cuda":
            peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)

        latency_per_item_ms = round((inference_duration / len(items)) * 1000, 2)
        overall_metrics = evaluate_predictions(scores, labels, types)

        # Domain breakdown
        domain_breakdown = {}
        unique_domains = sorted(list(set(domains)))
        for dom in unique_domains:
            dom_indices = [i for i, d in enumerate(domains) if d == dom]
            dom_scores = [scores[i] for i in dom_indices]
            dom_labels = [labels[i] for i in dom_indices]
            dom_types = [types[i] for i in dom_indices]
            domain_breakdown[dom] = evaluate_predictions(
                dom_scores, dom_labels, dom_types
            )

        results["models"][model_key] = {
            "model_name": model_name,
            "latency_ms": latency_per_item_ms,
            "total_duration_sec": round(inference_duration, 2),
            "peak_vram_mb": peak_vram_mb,
            "overall": overall_metrics,
            "domains": domain_breakdown,
        }

        print(f"\n[Results for {model_key}]")
        print(
            f"  Latency: {latency_per_item_ms} ms/item (Total: {inference_duration:.2f}s)"
        )
        if peak_vram_mb > 0:
            print(f"  Peak VRAM: {peak_vram_mb} MB")
        print(
            f"  Accuracy: {overall_metrics['accuracy']:.4f} | F1: {overall_metrics['f1']:.4f}"
        )
        print(f"  Discrimination Gap: {overall_metrics['discrimination_gap']:.4f}")
        print(
            f"  Near-Miss Rejection: {overall_metrics['near_miss_rejection_rate']:.4f}"
        )

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Saved benchmark results to: {output_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 540-item comparative benchmark.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--output", type=Path, default=Path("scratch/benchmark_540_results.json")
    )
    args = parser.parse_args()

    run_benchmark(device=args.device, output_json=args.output)
