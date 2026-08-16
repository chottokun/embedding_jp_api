---
type: Concept
title: サポートモデル仕様・日本語プレフィックス定義
description: Ruri-v3, BGE-M3, Visualized-BGE, Ruri Reranker の詳細仕様とタスク別プレフィックスルール
status: stable
generated:
  by: agent/antigravity
  at: 2026-08-16T09:10:00Z
tags:
  - models
  - ruri-v3
  - bge-m3
  - reranker
  - prefix
sources:
  - resource: /config/models.yml
    title: Model Configuration Registry
  - resource: /src/app/config.py
    title: Model & Prefix Constants
---

# サポートモデル仕様・日本語プレフィックス定義

## 1. 埋め込みモデル (Embedding Models)

| モデルID | タイプ | 出力次元数 | 最大コンテキスト長 | 特徴 |
| :--- | :---: | :---: | :---: | :--- |
| `cl-nagoya/ruri-v3-30m` | テキスト | 256 | 8,192 tokens | 超軽量・高速・低レイテンシ（モバイル/エッジ推奨） |
| `cl-nagoya/ruri-v3-310m` | テキスト | 768 | 8,192 tokens | 高精度・日本語特化・標準エンベディングモデル |
| `BAAI/bge-m3` | テキスト | 1024 | 8,192 tokens | 多言語対応・Dense/Sparse 検索基盤 |
| `bge-visualized-m3` | マルチモーダル | 1024 | 8,192 tokens | 画像（図面・表・チャート）＋日本語テキスト統合表現 |

## 2. リランカーモデル (Reranking Models)

| モデルID | タイプ | 最大コンテキスト長 | 特徴 |
| :--- | :---: | :---: | :--- |
| `cl-nagoya/ruri-v3-reranker-310m` | Cross-Encoder | 8,192 tokens | 質問と文書のペアを高精度にスコアリング（Top-N 絞り込み用） |

## 3. Ruri-v3 日本語タスク別プレフィックス

Ruri-v3 モデルでは、非対称検索やクラスタリングの精度を最大化するため、以下のプレフィックスを自動適用または `input_type` パラメータで指定します。

| `input_type` 値 | プレフィックス文字列 | 用途 |
| :--- | :--- | :--- |
| `query` | `"検索クエリ: "` | 検索質問文（Asymmetric Search） |
| `document` | `"検索文書: "` | 検索対象ドキュメント（ナレッジ側） |
| `classification` | `"トピック: "` | テキスト分類・クラスタリング |
| `clustering` | `"トピック: "` | 同上 |
| `sts` | `""` (なし) | 対称的な文章間類似度比較 |

- **二重付与防止**: テキストが既に `"検索クエリ: "` 等で始まっている場合は自動付与をスキップします。
