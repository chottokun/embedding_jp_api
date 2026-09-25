---
type: Concept
title: Logit Gate & ハイブリッド再ランキング アーキテクチャ
description: Qwen2.5-1.5B 因果言語モデルによる単一フォワードパス十分性判定、ASCII Matcher 3-gram ブースト、およびロジット空間スコア統合
status: stable
generated:
  by: agent/antigravity
  at: 2026-09-25T14:10:00Z
tags:
  - rerank
  - logit-gate
  - ascii-matcher
  - causal-lm
  - entropy
sources:
  - resource: /src/app/services/logit_gate.py
    title: Logit Gate Service Implementation
  - resource: /src/app/services/ascii_matcher.py
    title: ASCII Matcher Service Implementation
  - resource: /src/app/services/rerank.py
    title: Rerank Service Dynamic Dispatch
  - resource: /plan/logit.md
    title: Logit Gate Implementation Plan
---

# Logit Gate & Hybrid Reranking Architecture

## 1. Overview & Problem Definition

In Retrieval-Augmented Generation (RAG) pipelines, dense embeddings (e.g., Ruri-v3, BGE-M3) frequently retrieve **"near-miss" documents**—texts that share high topical overlap with the query but lack decisive factual evidence (e.g., specific error codes, condition branches, model numbers). Moreover, standard cross-encoders (`bge-reranker`, `ruri-reranker`) assign relatively high scores to the least irrelevant document even in "complete absence" scenarios, triggering LLM hallucinations.

Furthermore, subword tokenizers inevitably fragment technical identifiers (e.g., error code `0x80070035`, file paths `docker-compose.yml`, version tags `v3`), reducing retrieval precision.

The **Logit Gate & Hybrid Rerank** engine solves both problems simultaneously inside the stateless `/v1/rerank` endpoint without requiring client-side full-text search databases.

---

## 2. Core Architectural Components

```
[Client (Dify / LangChain / LlamaIndex)]
              │  POST /v1/rerank (query, documents, model="Qwen/Qwen2.5-1.5B-Instruct")
              ▼
┌───────────────────────────────────────────────────────────────┐
│ 1. ASCII Matcher (CPU)                                        │
│    - Extracts alphanumeric & symbol identifiers from query    │
│    - Excludes stop words and pure digits                      │
│    - Computes character 3-gram Containment Ratio (0.0 to 1.0)  │
└──────────────────────────────┬────────────────────────────────┘
                               │ Containment scores
                               ▼
┌───────────────────────────────────────────────────────────────┐
│ 2. Logit Gate Engine (Qwen2.5-1.5B CausalLM)                  │
│    - Safe document truncation (LOGIT_GATE_MAX_DOC_CHARS)     │
│    - Mini-batch chunking (LOGIT_GATE_BATCH_SIZE)              │
│    - ChatML prompt template application                      │
│    - Single forward pass on next-token position               │
│    - LogSumExp aggregation across positive & negative tokens  │
│    - Normalized binary Shannon entropy calculation            │
└──────────────────────────────┬────────────────────────────────┘
                               │ Logit margins Δz & Entropy
                               ▼
┌───────────────────────────────────────────────────────────────┐
│ 3. Score Fusion & Fail-Safe Sorting                           │
│    - Logit-space fusion: Δz_final = Δz + (β * Containment)     │
│    - Sigmoid normalization: Score = σ(Δz_final)               │
│    - Fail-safe filtering: drop_failed=False preserves array   │
│      length (score: 0.0, passed: false) to prevent client OOM │
└───────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Formulation

### 3.1 LogSumExp Logit Margin
The prompt ends with `<|im_start|>assistant\n`. The unnormalized logits at the immediate next token position are evaluated over target positive tokens $\mathcal{T}_{\text{pos}}$ (`Yes`, `yes`, `はい`) and negative tokens $\mathcal{T}_{\text{neg}}$ (`No`, `no`, `いいえ`):

$$z_{\text{pos}} = \log \sum_{i \in \mathcal{T}_{\text{pos}}} e^{z_i}, \quad z_{\text{neg}} = \log \sum_{j \in \mathcal{T}_{\text{neg}}} e^{z_j}$$

$$\Delta z = z_{\text{pos}} - z_{\text{neg}} - \text{baseline\_margin}$$

### 3.2 Logit-Space Hybrid Boost (ASCII Fusion)
Instead of linear score blending (which breaks probability bounds and allows false positive leaks), the ASCII containment ratio is added in logit space:

$$\Delta z_{\text{final}} = \Delta z + (\beta \cdot \text{Containment})$$

$$S_{\text{final}} = \sigma(\Delta z_{\text{final}}) = \frac{1}{1 + e^{-\Delta z_{\text{final}}}}$$

* **Self-Defense Mechanism**: If a document is completely off-topic ($\Delta z \approx -5.0$), even a perfect identifier match ($\text{Containment} = 1.0, \beta = 1.2$) yields $\Delta z_{\text{final}} = -3.8 \implies P \approx 0.02$, decisively blocking unrelated documents.
* **Near-Miss Rescue**: If a document is on the borderline ($\Delta z \approx 0.0$), matching the exact technical identifier boosts the score decisively to $P \approx 0.77$.

### 3.3 Normalized Binary Shannon Entropy
Quantifies model hesitation/uncertainty:

$$H_{\text{binary}}(P) = - P \log_2(P) - (1 - P) \log_2(1 - P)$$

* Maximum uncertainty $H = 1.0$ at $P = 0.5$ ($\Delta z = 0.0$).
* Decisive confidence $H \to 0.0$ at $P \to 1.0$ or $P \to 0.0$.

---

## 4. Benchmark Verification Summary

Measured on $N=108$ expanded evaluation dataset (`benchmarks/datasets/sufficiency_eval.json`) on GPU (NVIDIA RTX 3060):

| Metric | Measured Value ($N=108$) | Target / Baseline |
| :--- | :---: | :---: |
| **Near-Miss Rejection Rate** | **100.0%** (36/36) | $\ge 90.0\%$ |
| **Unanswerable Rejection Rate** | **100.0%** (36/36) | $\ge 90.0\%$ |
| **Precision** | **100.0%** | $\ge 85.0\%$ |
| **Overall Accuracy ($\tau=0.30$)** | **95.4%** | $\ge 85.0\%$ |
| **F1 Score ($\tau=0.30$)** | **92.5%** | $\ge 85.0\%$ |
| **Overall Accuracy ($\tau=0.55$)** | **91.7%** | $\ge 85.0\%$ |
| **Inference Latency (GPU)** | **~30 ms / doc** | Realtime SLA |
| **VRAM Memory Leak** | **0.00 MB** | Zero leak |

See detailed benchmark reports in [docs/infrastructure/benchmarks.md](/docs/infrastructure/benchmarks.md).
