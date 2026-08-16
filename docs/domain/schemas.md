---
type: Concept
title: API スキーマ・バリデーション仕様
description: Pydantic モデル、リクエスト・レスポンス定義、およびセキュリティ制約
status: stable
generated:
  by: agent/antigravity
  at: 2026-08-16T09:10:00Z
tags:
  - schemas
  - pydantic
  - validation
  - security
sources:
  - resource: /src/app/schemas.py
    title: Pydantic Schemas Definition
---

# API スキーマ・バリデーション仕様

## 1. 埋め込みエンドポイント (`POST /v1/embeddings`)

### リクエスト (`EmbeddingRequest`)
```json
{
  "model": "cl-nagoya/ruri-v3-310m",
  "input": "日本語テキスト",
  "input_type": "query",
  "apply_ruri_prefix": false,
  "user": "user-123"
}
```

### 制約事項 (DoS 防止)
- `MAX_INPUT_LENGTH`: 単一文字列の最大文字数 = 65,536 文字
- `MAX_INPUT_ITEMS`: バッチ配列の最大要素数 = 256 件
- `ImageSourceString`: Base64 / URL 画像の最大文字列長 = 25,000,000 文字 (~18MB Base64)

### レスポンス (`EmbeddingResponse`)
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.0123, -0.0456, ...],
      "index": 0
    }
  ],
  "model": "cl-nagoya/ruri-v3-310m",
  "usage": {
    "prompt_tokens": 12,
    "total_tokens": 12
  }
}
```

---

## 2. リランキングエンドポイント (`POST /v1/rerank`)

### リクエスト (`RerankRequest`)
```json
{
  "model": "cl-nagoya/ruri-v3-reranker-310m",
  "query": "日本語の検索クエリ",
  "documents": [
    "ドキュメント1のテキスト",
    "ドキュメント2のテキスト"
  ],
  "top_n": 3,
  "return_documents": true
}
```

### レスポンス (`RerankResponse`)
```json
{
  "query": "日本語の検索クエリ",
  "data": [
    {
      "document": 1,
      "score": 0.9421,
      "text": "ドキュメント2のテキスト"
    },
    {
      "document": 0,
      "score": 0.1205,
      "text": "ドキュメント1のテキスト"
    }
  ],
  "model": "cl-nagoya/ruri-v3-reranker-310m",
  "usage": {
    "prompt_tokens": 48,
    "total_tokens": 48
  }
}
```
