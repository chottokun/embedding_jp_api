#!/usr/bin/env python3
"""Cascade Benchmark: Dense Retrieval (Stage 1) + Reranking / Logit Gate (Stage 2)

Critical & Constructive Comparative Analysis:
- Stage 1 Models:
    1. google/embeddinggemma-2 (768d Full)
    2. google/embeddinggemma-2 (256d MRL)
    3. cl-nagoya/ruri-v3-310m
    4. cl-nagoya/ruri-v3-30m
- Stage 2 Models:
    1. None (Stage 1 Baseline)
    2. cl-nagoya/ruri-v3-reranker-310m (Cross-Encoder)
    3. Takenoko12345678/Qwen3.5-0.8B-Japanese-SFT-v2 (Logit Gate)
- Dataset:
    benchmarks/datasets/sufficiency_eval_v2_540.json (N=540, 180 unique queries x 3 doc types across 6 domains)
"""

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.models import get_model, unload_model  # noqa: E402
from src.app.services.logit_gate import LogitGateService  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402

DATASET_PATH = PROJECT_ROOT / "benchmarks" / "datasets" / "sufficiency_eval_v2_540.json"
RESULTS_JSON_PATH = PROJECT_ROOT / "scratch" / "actual_cascade_benchmark_results.json"
REPORT_MD_PATH = (
    PROJECT_ROOT
    / "docs"
    / "infrastructure"
    / "benchmark_cascade_ruri_vs_embeddinggemma.md"
)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    norm_a = a / np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(norm_a, norm_b.T)


def load_dataset() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Loads the 540-item dataset and structures it into queries, ground truth, and document corpus."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    # 540 items, each document is unique in the corpus
    corpus_docs = [item["document"] for item in data]
    doc_id_to_idx = {item["id"]: i for i, item in enumerate(data)}

    # Group by query: 180 unique queries
    # Each query has: pos_id, near_id, unans_id, domain
    query_map: dict[str, dict[str, Any]] = {}
    for item in data:
        q = item["query"]
        if q not in query_map:
            query_map[q] = {
                "query": q,
                "domain": item["domain"],
                "positive_idx": None,
                "near_miss_idx": None,
                "unanswerable_idx": None,
            }
        idx = doc_id_to_idx[item["id"]]
        if item["type"] == "positive":
            query_map[q]["positive_idx"] = idx
        elif item["type"] == "near_miss":
            query_map[q]["near_miss_idx"] = idx
        elif item["type"] == "unanswerable":
            query_map[q]["unanswerable_idx"] = idx

    queries_list = list(query_map.values())
    return data, queries_list, corpus_docs


def run_stage1_dense_retrieval(
    model_name: str,
    dim: int | None,
    queries_list: list[dict[str, Any]],
    corpus_docs: list[str],
    device: str = "cpu",
) -> tuple[dict[str, Any], np.ndarray, float]:
    """Encodes corpus and queries, measures Stage 1 latency and produces similarity rankings."""
    print(f"\n--- [Stage 1] Loading {model_name} (dim={dim or 'full'}) on {device} ---")
    gc.collect()
    model = get_model(model_name, device=device)

    # Prefix handling for ruri models
    is_ruri = "ruri" in model_name
    query_prefix = "クエリ: " if is_ruri else ""
    doc_prefix = "文章: " if is_ruri else ""

    formatted_corpus = [doc_prefix + d for d in corpus_docs]
    formatted_queries = [query_prefix + q["query"] for q in queries_list]

    # Warmup
    print("Warming up Stage 1 model...")
    model.encode(["ウォームアップ"])

    # Encode Corpus
    print(f"Encoding {len(corpus_docs)} corpus documents...")
    t0 = time.perf_counter()
    corpus_embeds_raw = model.encode(formatted_corpus, batch_size=32)
    corpus_time = time.perf_counter() - t0
    corpus_embeds = np.asarray(corpus_embeds_raw)
    if dim is not None:
        corpus_embeds = corpus_embeds[:, :dim]
    print(
        f"Corpus encoded in {corpus_time:.2f}s ({len(corpus_docs) / corpus_time:.1f} docs/s)"
    )

    # Encode Queries & Measure Latency
    print(f"Encoding {len(queries_list)} queries and measuring latency...")
    query_latencies = []
    query_embeds_list = []
    for q_text in formatted_queries:
        t_q = time.perf_counter()
        q_emb_raw = model.encode([q_text])
        lat = (time.perf_counter() - t_q) * 1000.0  # ms
        query_latencies.append(lat)
        query_embeds_list.append(np.asarray(q_emb_raw)[0])

    mean_query_lat = float(np.mean(query_latencies))
    p95_query_lat = float(np.percentile(query_latencies, 95))
    print(
        f"Stage 1 Query Latency: Mean={mean_query_lat:.2f}ms, P95={p95_query_lat:.2f}ms"
    )

    query_embeds = np.asarray(query_embeds_list)
    if dim is not None:
        query_embeds = query_embeds[:, :dim]

    # Compute similarity matrix (180 queries x 540 documents)
    t_sim = time.perf_counter()
    sim_matrix = cosine_similarity_matrix(query_embeds, corpus_embeds)
    sim_calc_lat = (time.perf_counter() - t_sim) * 1000.0 / len(queries_list)

    # Stage 1 standalone metrics
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    near_miss_at_1 = 0
    unanswerable_at_1 = 0

    top_k_indices_list = []

    for i, q_info in enumerate(queries_list):
        ranked_indices = np.argsort(sim_matrix[i])[::-1]
        top_k_indices_list.append(ranked_indices[:10].tolist())

        pos_idx = q_info["positive_idx"]
        near_idx = q_info["near_miss_idx"]
        unans_idx = q_info["unanswerable_idx"]

        top1 = ranked_indices[0]
        if top1 == pos_idx:
            recall_at_1 += 1
        elif top1 == near_idx:
            near_miss_at_1 += 1
        elif top1 == unans_idx:
            unanswerable_at_1 += 1

        top5 = set(ranked_indices[:5])
        if pos_idx in top5:
            recall_at_5 += 1

        top10 = set(ranked_indices[:10])
        if pos_idx in top10:
            recall_at_10 += 1

    n_q = len(queries_list)
    stage1_stats = {
        "model_name": model_name,
        "dim": dim or corpus_embeds.shape[1],
        "query_latency_mean_ms": round(mean_query_lat, 2),
        "query_latency_p95_ms": round(p95_query_lat, 2),
        "sim_calc_latency_ms": round(sim_calc_lat, 2),
        "total_stage1_ms": round(mean_query_lat + sim_calc_lat, 2),
        "recall_at_1": round(recall_at_1 / n_q, 4),
        "recall_at_5": round(recall_at_5 / n_q, 4),
        "recall_at_10": round(recall_at_10 / n_q, 4),
        "top1_near_miss_rate": round(near_miss_at_1 / n_q, 4),
        "top1_unans_rate": round(unanswerable_at_1 / n_q, 4),
    }

    # Free memory
    del model
    del corpus_embeds
    del query_embeds
    unload_model(model_name)
    gc.collect()

    return stage1_stats, np.array(top_k_indices_list), mean_query_lat + sim_calc_lat


def main():
    print("=" * 70)
    print(" Cascade Benchmark: EmbeddingGemma-2 vs ruri-v3 + Reranker / Logit Gate")
    print("=" * 70)

    data, queries_list, corpus_docs = load_dataset()
    print(
        f"Dataset loaded: {len(data)} items, {len(queries_list)} unique queries, {len(corpus_docs)} corpus docs."
    )

    device = "cpu"
    # Note: Using cpu since current environment is built for cpu

    # 1. Evaluate Stage 1 Models
    stage1_configs = [
        ("google/embeddinggemma-2", 768, "EmbeddingGemma-2 (768d)"),
        ("google/embeddinggemma-2", 256, "EmbeddingGemma-2 (256d MRL)"),
        ("cl-nagoya/ruri-v3-310m", None, "ruri-v3-310m (Full)"),
        ("cl-nagoya/ruri-v3-30m", None, "ruri-v3-30m (Full)"),
    ]

    stage1_results = {}
    topk_rankings = {}

    for model_name, dim, label in stage1_configs:
        stats, topk, lat = run_stage1_dense_retrieval(
            model_name=model_name,
            dim=dim,
            queries_list=queries_list,
            corpus_docs=corpus_docs,
            device=device,
        )
        stage1_results[label] = stats
        topk_rankings[label] = (topk, lat)

    # 2. Evaluate Stage 2 Cascades for Key Combinations:
    # A. ruri-v3-310m -> ruri-v3-reranker-310m
    # B. ruri-v3-310m -> Logit Gate (Qwen3.5-0.8B)
    # C. EmbeddingGemma-2 (256d) -> ruri-v3-reranker-310m
    # D. EmbeddingGemma-2 (256d) -> Logit Gate (Qwen3.5-0.8B)
    # E. EmbeddingGemma-2 (768d) -> ruri-v3-reranker-310m
    # F. ruri-v3-30m -> ruri-v3-reranker-310m
    cascade_combinations = [
        ("ruri-v3-310m (Full)", "Cross-Encoder", 5),
        ("ruri-v3-310m (Full)", "Logit Gate", 5),
        ("EmbeddingGemma-2 (256d MRL)", "Cross-Encoder", 5),
        ("EmbeddingGemma-2 (256d MRL)", "Logit Gate", 5),
        ("EmbeddingGemma-2 (768d)", "Cross-Encoder", 5),
        ("ruri-v3-30m (Full)", "Cross-Encoder", 5),
    ]

    cascade_results = {}

    # We evaluate Cross-Encoder once across all needed combinations, then Logit Gate once
    # to avoid repeated model loading and unloading overhead.
    # Group by Stage 2 model:
    print("\n" + "=" * 70)
    print(" Running Stage 2 Evaluations (Top-5 Re-ranking)")
    print("=" * 70)

    # 2.1 Cross-Encoder evaluations
    ce_model_name = "cl-nagoya/ruri-v3-reranker-310m"
    print(f"\n[Loading Stage 2: Cross-Encoder {ce_model_name}]")
    gc.collect()
    ce_model = CrossEncoder(ce_model_name, max_length=512, device=device)
    ce_model.predict([("ウォームアップ", "ウォームアップ")])

    for s1_label, s2_type, k in [
        c for c in cascade_combinations if c[1] == "Cross-Encoder"
    ]:
        topk, s1_lat = topk_rankings[s1_label]
        print(f"\nEvaluating Cascade: [{s1_label}] -> [Cross-Encoder]...")
        stage2_latencies = []
        top1_correct = 0
        top1_near_miss = 0
        top1_unans = 0
        dom_correct: dict[str, int] = {}
        dom_total: dict[str, int] = {}

        for i, q_info in enumerate(queries_list):
            q_text = q_info["query"]
            cands = topk[i][:k]
            pairs = [(q_text, corpus_docs[idx]) for idx in cands]

            t0 = time.perf_counter()
            scores = ce_model.predict(pairs)
            lat = (time.perf_counter() - t0) * 1000.0
            stage2_latencies.append(lat)

            best_cand = cands[int(np.argmax(scores))]
            dom = q_info["domain"]
            dom_total[dom] = dom_total.get(dom, 0) + 1

            if best_cand == q_info["positive_idx"]:
                top1_correct += 1
                dom_correct[dom] = dom_correct.get(dom, 0) + 1
            elif best_cand == q_info["near_miss_idx"]:
                top1_near_miss += 1
            elif best_cand == q_info["unanswerable_idx"]:
                top1_unans += 1

        n_q = len(queries_list)
        s2_mean_lat = float(np.mean(stage2_latencies))
        cascade_key = f"{s1_label} + Cross-Encoder"
        cascade_results[cascade_key] = {
            "stage1_label": s1_label,
            "stage2_label": "ruri-v3-reranker-310m",
            "stage1_latency_ms": round(s1_lat, 2),
            "stage2_latency_ms": round(s2_mean_lat, 2),
            "e2e_latency_ms": round(s1_lat + s2_mean_lat, 2),
            "top1_accuracy": round(top1_correct / n_q, 4),
            "top1_near_miss_rate": round(top1_near_miss / n_q, 4),
            "top1_unans_rate": round(top1_unans / n_q, 4),
            "domain_accuracy": {
                d: round(dom_correct.get(d, 0) / dom_total[d], 4) for d in dom_total
            },
        }
        print(
            f"Result for {cascade_key}: Accuracy={top1_correct / n_q:.4f}, E2E={s1_lat + s2_mean_lat:.2f}ms"
        )

    del ce_model
    gc.collect()

    # 2.2 Logit Gate evaluations
    lg_model_name = "Takenoko12345678/Qwen3.5-0.8B-Japanese-SFT-v2"
    print(f"\n[Loading Stage 2: Logit Gate {lg_model_name}]")
    gc.collect()
    lg_wrapper = get_model(lg_model_name, device=device)
    lg_service = LogitGateService(lg_wrapper)
    lg_service.predict_margins("テスト", ["テスト回答"])

    for s1_label, s2_type, k in [
        c for c in cascade_combinations if c[1] == "Logit Gate"
    ]:
        topk, s1_lat = topk_rankings[s1_label]
        print(f"\nEvaluating Cascade: [{s1_label}] -> [Logit Gate] (Top-{k})...")
        stage2_latencies = []
        top1_correct = 0
        top1_near_miss = 0
        top1_unans = 0
        dom_correct: dict[str, int] = {}
        dom_total: dict[str, int] = {}

        n_q = len(queries_list)
        for i, q_info in enumerate(queries_list):
            q_text = q_info["query"]
            cands = topk[i][:k]
            cand_docs = [corpus_docs[idx] for idx in cands]

            t0 = time.perf_counter()
            res = lg_service.predict_margins(q_text, cand_docs)
            lat = (time.perf_counter() - t0) * 1000.0
            stage2_latencies.append(lat)

            scores = [r["sufficiency_prob"] for r in res]
            best_cand = cands[int(np.argmax(scores))]
            dom = q_info["domain"]
            dom_total[dom] = dom_total.get(dom, 0) + 1

            if best_cand == q_info["positive_idx"]:
                top1_correct += 1
                dom_correct[dom] = dom_correct.get(dom, 0) + 1
            elif best_cand == q_info["near_miss_idx"]:
                top1_near_miss += 1
            elif best_cand == q_info["unanswerable_idx"]:
                top1_unans += 1

            if (i + 1) % 30 == 0 or (i + 1) == n_q:
                print(
                    f"  [{i + 1}/{n_q}] queries processed (current Accuracy: {top1_correct / (i + 1):.3f})"
                )

        s2_mean_lat = float(np.mean(stage2_latencies))
        cascade_key = f"{s1_label} + Logit Gate"
        cascade_results[cascade_key] = {
            "stage1_label": s1_label,
            "stage2_label": "Qwen3.5-0.8B-Logit-Gate",
            "stage1_latency_ms": round(s1_lat, 2),
            "stage2_latency_ms": round(s2_mean_lat, 2),
            "e2e_latency_ms": round(s1_lat + s2_mean_lat, 2),
            "top1_accuracy": round(top1_correct / n_q, 4),
            "top1_near_miss_rate": round(top1_near_miss / n_q, 4),
            "top1_unans_rate": round(top1_unans / n_q, 4),
            "domain_accuracy": {
                d: round(dom_correct.get(d, 0) / dom_total[d], 4) for d in dom_total
            },
        }
        print(
            f"Result for {cascade_key}: Accuracy={top1_correct / n_q:.4f}, E2E={s1_lat + s2_mean_lat:.2f}ms"
        )

    del lg_wrapper
    del lg_service
    unload_model(lg_model_name)
    gc.collect()

    # Save JSON results
    all_output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "n_queries": len(queries_list),
        "n_corpus": len(corpus_docs),
        "stage1_results": stage1_results,
        "cascade_results": cascade_results,
    }
    RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_output, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Raw results saved to: {RESULTS_JSON_PATH}")


if __name__ == "__main__":
    main()
