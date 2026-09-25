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
  "dimensions": 512,
  "encoding_format": "float",
  "user": "user-123"
}
```

### 制約事項 (DoS 防止)
- `MAX_INPUT_LENGTH`: 単一文字列の最大文字数 = 65,536 文字
- `MAX_INPUT_ITEMS`: バッチ配列の最大要素数 = 256 件
- `ImageSourceString`: Base64 / URL 画像の最大文字列長 = 25,000,000 文字 (~18MB Base64)
- `MAX_PAYLOAD_SIZE`: HTTP リクエストボディ最大長 = 32MB (413 Payload Too Large)

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
> ※ `encoding_format: "base64"` 指定時は、`embedding` が IEEE 754 リトルエンディアン float32 の Base64 文字列として返却されます。

---

## 2. リランキングエンドポイント (`POST /v1/rerank`)

### リクエスト (`RerankRequest`)
```json
{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "query": "日本語の検索クエリ",
  "documents": [
    "ドキュメント1のテキスト",
    "ドキュメント2のテキスト"
  ],
  "top_n": 3,
  "return_documents": true,
  "threshold": 0.55,
  "drop_failed": false,
  "use_ascii_boost": true
}
```

#### パラメータ詳細
- `threshold` (float, optional): 回答十分性判定の確率閾値（未指定時は `.env` の `LOGIT_GATE_THRESHOLD`、デフォルト: `0.55`）。
- `drop_failed` (bool, default: `false`): `true` の場合、閾値未満のドキュメントを結果から除外。`false`（デフォルト）の場合、全件保持して不合格ドキュメントを `passed: false` で末尾にソート（Dify等のクライアントクラッシュを防止）。
- `use_ascii_boost` (bool, optional): 半角英数字 3-gram ブーストの有効/無効（未指定時はサーバー設定に従う）。

### レスポンス (`RerankResponse`)
```json
{
  "query": "日本語の検索クエリ",
  "data": [
    {
      "document": 1,
      "score": 0.9421,
      "text": "ドキュメント2のテキスト",
      "passed": true,
      "logit_margin": 1.45,
      "containment_score": 0.85,
      "entropy": 0.325
    },
    {
      "document": 0,
      "score": 0.0821,
      "text": "ドキュメント1のテキスト",
      "passed": false,
      "logit_margin": -3.21,
      "containment_score": 0.0,
      "entropy": 0.182
    }
  ],
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "usage": {
    "prompt_tokens": 128,
    "total_tokens": 128
  }
}
```

#### Logit Gate 拡張メタデータ
- `passed` (bool): 閾値 $\tau$ を通過したかどうかの判定フラグ。
- `logit_margin` (float): ポジティブ/ネガティブ トークン間の未補正ロジット差分 $\Delta z$。
- `containment_score` (float): クエリ内英数字の 3-gram 包含率スコア（0.0〜1.0）。
- `entropy` (float): 二値正規化シャノンエントロピー $H_{\text{binary}} \in [0.0, 1.0]$（判定の不確実性の度合い）。

---

## 3. モデル一覧エンドポイント (`GET /v1/models`)

### レスポンス (`ModelList`)
```json
{
  "object": "list",
  "data": [
    {
      "id": "cl-nagoya/ruri-v3-30m",
      "object": "model",
      "created": 1726315200,
      "owned_by": "custom",
      "permission": []
    }
  ]
}
```

---

## 4. モデルアンロードエンドポイント (`POST /v1/models/unload`)

### リクエスト (`UnloadRequest`)
```json
{
  "model": "cl-nagoya/ruri-v3-310m"
}
```

### レスポンス (`UnloadResponse`)
```json
{
  "unloaded_models": ["cl-nagoya/ruri-v3-310m"],
  "remaining_memory": 128450560
}
```

---

## 5. ヘルス・レディネス・メトリクス

- `GET /health`, `GET /healthz`: `{"status": "ok"}`
- `GET /ready`: `{"status": "ready", "gpu_available": true, "models_loaded": [...]}`
- `GET /metrics`: Prometheus テキスト形式メトリクス

---

## 6. 標準エラーレスポンス (`ErrorResponse`)

HTTP 400, 401, 413, 429, 500, 503 時の共通レスポンス構造:
```json
{
  "detail": "エラー内容を示すメッセージ"
}
```
