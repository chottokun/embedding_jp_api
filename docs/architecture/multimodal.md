---
type: Concept
title: マルチモーダル埋め込み推論アーキテクチャ
description: bge-visualized-m3 による画像＋テキストの統合エンコード、スキーマ正規化、およびエッジケース処理
status: stable
generated:
  by: agent/antigravity
  at: 2026-08-16T09:10:00Z
tags:
  - multimodal
  - bge-visualized-m3
  - vision
  - cross-modal
sources:
  - resource: /src/app/models.py
    title: Model Wrappers and VisualizedBGE
  - resource: /src/app/image_utils.py
    title: Safe Image Loading & SSRF Guards
---

# マルチモーダル埋め込み推論アーキテクチャ

## 1. 概要

`bge-visualized-m3` をバックエンドとし、画像（図面、グラフ、UIモック、スケッチ等）とテキスト（キャプション、注記、検索クエリ）を同一の **1024次元空間** に埋め込む機能を提供します。

```mermaid
graph LR
    InputImg["画像 (PNG/JPEG/WebP)"] --> ImgPrep["EVA-CLIP Preprocess (RGB 224x224)"]
    InputTxt["テキスト (日本語)"] --> TxtTok["BGE-M3 Tokenizer"]
    
    ImgPrep --> VisProj["Visual Projection Layer"]
    TxtTok --> Transformer["BGE-M3 Transformer Encoder"]
    VisProj --> Transformer
    
    Transformer --> DensePool["Dense Vector Pooling"]
    DensePool --> Output["1024次元 正規化ベクトル"]
```

## 2. 入力スキーマと自動正規化

API は以下の形式を受け付け、内部で `(Optional[str], Optional[PIL.Image])` に自動正規化します。

1. **フラット形式 (`FlatMultimodalItem`)**:
   ```json
   {
     "model": "bge-visualized-m3",
     "input": {
       "text": "API Gateway と DB のシステム構成図",
       "image_url": "data:image/png;base64,..."
     }
   }
   ```
2. **OpenAI ContentPart 形式**:
   ```json
   {
     "model": "bge-visualized-m3",
     "input": [
       {"type": "text", "text": "認証フローチャート"},
       {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
     ]
   }
   ```
3. **画像単体（テキスト省略 / 空文字）**:
   - `text: ""` または `text: null` の場合、テキストを `None` として扱い純粋な画像エンコードを実行。
4. **バッチ形式**:
   - 上記アイテムの配列を渡すことで、複数アイテムを一括処理。

## 3. エッジケースと堅牢化

- **透過 PNG (RGBA)**: PIL にてアルファチャンネルを破棄し `RGB` へ安全に変換。
- **超長文テキスト**: トークン切り詰めにより最大長制約を遵守。
- **テキスト専用モデルへの誤送信ガード**: `cl-nagoya/ruri-v3-310m` 等に画像が送信された場合、400 Bad Request で明示的に拒絶。
