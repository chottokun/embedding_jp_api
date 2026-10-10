---
okf_version: "0.2"
title: Infrastructure Knowledge Index
description: コンテナ構成、オフライン（エアギャップ）運用、ベンチマークデータのインデックス
---

# Infrastructure Knowledge Index

## 概要

Docker / Docker Compose による GPU・CPU コンテナデプロイ、完全オフライン運用、環境変数設定、および実機ベンチマークデータに関するインフラナレッジです。

## ドキュメント一覧

* [Docker / Docker Compose デプロイメントガイド](./docker.md) - GPU / CPU マルチステージビルド、Compose 構成
* [完全オフライン（エアギャップ）運用ガイド](./offline_mode.md) - 事前ダウンロード、オフライン検証、`.env` 階層型設定
* [実機ベンチマーク測定結果](./benchmarks.md) - NVIDIA GeForce RTX 3060 (12GB VRAM) / ホスト CPU での実測データ
* [Logit Gate 対 従来型クロスエンコーダー 直接対決ベンチマーク](./comparative_benchmark_results.md) - Ruri-v3-reranker-310m と Qwen2.5-1.5B のレイテンシ・ニアミス遮断性能比較
* [Logit Gate Qwen3.5-0.8B ベンチマーク](./benchmark_qwen35_08b_results.md) - 日本語SFT 0.8Bモデルの実測性能
* [二段カスケード詳細比較ベンチマーク (EmbeddingGemma-2 vs ruri-v3 × Reranker / Logit Gate)](./benchmark_cascade_ruri_vs_embeddinggemma.md) - 540件高難度データセットによる全カスケード構成の実機測定と批判的・建設的分析
* [EmbeddingGemma-2 テキスト 540件高難度ベンチマーク](./benchmark_540_embeddinggemma_results.md) - MRL 4段階次元削減と推論レイテンシの実測
* [EmbeddingGemma-2 マルチモーダル 60件評価ベンチマーク](./benchmark_multimodal_embeddinggemma_results.md) - 4大ビジネスドメイン実画像を用いたText-to-Image / Image-to-Text実測
* [GPU実機 (RTX 3060) Locust 負荷テスト検証レポート](./benchmark_gpu_locust_load_test.md) - 4モデル同時稼働・10〜60同時ユーザー・12種エンドポイント（テキスト・マルチモーダル・リランカー）での実測データ
