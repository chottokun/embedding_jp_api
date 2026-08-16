---
type: Concept
title: 実機ベンチマーク・負荷測定結果
description: NVIDIA GeForce RTX 3060 (12GB VRAM) / ホスト CPU 実機による推論レイテンシ、スループット、並行負荷実測値
status: stable
generated:
  by: agent/antigravity
  at: 2026-08-16T09:10:00Z
tags:
  - benchmarks
  - performance
  - latency
  - throughput
  - rtx3060
sources:
  - resource: /benchmark_suite.py
    title: Benchmark Test Suite
  - resource: /test_multimodal_suite.py
    title: Multimodal Stress Test Suite
---

# 実機ベンチマーク・負荷測定結果

## 1. 測定環境

- **GPU**: NVIDIA GeForce RTX 3060 (12GB VRAM)
- **CPU**: ホスト CPU (x86_64)
- **OS**: Linux (Ubuntu)
- **Python / Framework**: Python 3.13 (`uv`), PyTorch, FastAPI

---

## 2. テキスト埋め込み性能 (`cl-nagoya/ruri-v3-310m`)

### ① 単一テキスト推論レイテンシ (100回連続)
- **平均レイテンシ**: **22.76 ms** (±0.74 ms)
- **P50 (中央値)**: **22.70 ms**
- **P95**: **24.14 ms**
- **P99**: **25.79 ms**

### ② バッチサイズ別スループット
| バッチサイズ | レスポンス時間 | スループット (items/sec) | トークン処理速度 (tok/sec) |
| :---: | :---: | :---: | :---: |
| **1** | 24.0 ms | 41.7 items/sec | 708.1 tok/sec |
| **8** | 51.6 ms | 155.1 items/sec | 2,635.9 tok/sec |
| **32** | 168.7 ms | 189.7 items/sec | 3,355.9 tok/sec |
| **64** | 299.8 ms | **213.5 items/sec** | **3,809.2 tok/sec** |

---

## 3. リランカー性能 (`cl-nagoya/ruri-v3-reranker-310m`)

- **クエリ × 5件の文書ペア**: 平均 **593.17 ms** (P50: 592.19 ms, P95: 651.25 ms)

---

## 4. 高並行負荷耐性 (テキスト 50 並行 / マルチモーダル 20 並行)

- **テキスト 50 並行同時リクエスト**:
  - 成功率: **50/50 (100.0%)**, エラー率 0%
  - 平均レイテンシ: 1,754.2 ms (P50: 581.2 ms, P95: 5,909.7 ms)
  - スループット: **8.4 req/sec**
- **マルチモーダル (`bge-visualized-m3`) 20 並行同時リクエスト**:
  - 成功率: **20/20 (100.0%)**, エラー率 0%
  - スレッドセーフロックにより GPU メモリ破壊・テンソル競合なく完了。
