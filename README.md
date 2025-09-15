# OpenAI互換 Embedding & Rerank APIサーバー

## 1. 概要

このプロジェクトは、日本語のテキスト埋め込み（Embedding）および再ランキング（Rerank）機能を提供する、OpenAI互換のFastAPIサーバーです。
名古屋大学CL研究室が開発した[Ruri v3モデル](https://huggingface.co/cl-nagoya/ruri-v3-30m)などを利用することを想定しています。

TDD（テスト駆動開発）のアプローチに基づき、モックモデルを使用してAPIのロジックを堅牢にテストしています。

## 2. API仕様

### 2.1. 埋め込み (Embeddings)

`POST /v1/embeddings`

OpenAIの[Embeddings API](https://platform.openai.com/docs/api-reference/embeddings)と互換性のあるエンドポイントです。

#### リクエスト例

```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
-H "Content-Type: application/json" \
-d '{
  "input": "今日の天気は晴れです。",
  "model": "cl-nagoya/ruri-v3-30m"
}'
```

#### 特殊な前処理：プレフィックス

`ruri-v3`モデルの性能を最大化するため、入力に応じて以下のプレフィックスが自動的に付与されます。

- **単一文字列の場合**: `"検索クエリ: "` が先頭に付与されます。
- **文字列の配列の場合**: 各要素に `"検索文書: "` が先頭に付与されます。

#### レスポンス形式

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, ...],
      "index": 0
    }
  ],
  "model": "cl-nagoya/ruri-v3-30m",
  "usage": {
    "prompt_tokens": 0,
    "total_tokens": 0
  }
}
```

---

### 2.2. 再ランキング (Rerank)

`POST /v1/rerank`

クエリに対して、提供されたドキュメントのリストを関連性の高い順に並べ替えます。

#### リクエスト例

```bash
curl -X POST "http://localhost:8000/v1/rerank" \
-H "Content-Type: application/json" \
-d '{
  "query": "AIの未来について",
  "documents": [
    "これは猫についての文章です。",
    "人工知能は今後の社会を大きく変えるでしょう。",
    "日本の首都は東京です。"
  ],
  "model": "cl-nagoya/ruri-v3-reranker-310m"
}'
```

#### レスポンス形式

```json
{
  "query": "AIの未来について",
  "data": [
    {
      "document": 1,
      "score": 0.9
    },
    {
      "document": 2,
      "score": 0.5
    },
    {
      "document": 0,
      "score": 0.1
    }
  ],
  "model": "cl-nagoya/ruri-v3-reranker-310m"
}
```

## 3. セットアップと実行

### 3.1. 必要なツール
- [Poetry](https://python-poetry.org/)
- Python 3.11+

### 3.2. 環境のセットアップ

1. リポジトリをクローンします。
2. 依存関係をインストールします。
   ```bash
   poetry install
   ```

### 3.3. 開発サーバーの実行

Uvicornを使用して開発サーバーを起動します。ポートは環境変数 `APP_PORT` で変更可能です（デフォルト: 8000）。

```bash
export APP_PORT=8080
poetry run uvicorn src.app.main:app --reload --port $APP_PORT
```

### 3.4. 本番環境での実行 (Gunicorn)

Linuxベースの環境では、GunicornとUvicornワーカーを組み合わせて実行することが推奨されます。

```bash
poetry run gunicorn --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --preload src.app.main:app
```
**注意**: `--preload` はモデルをワーカー間で共有するのに役立ちますが、GPU使用時にはCUDAコンテキストの問題を引き起こす可能性があります。GPU環境では `--preload` を外すか、単一ワーカーでの実行を検討してください。

## 4. テストの実行

プロジェクトのテストスイートはPytestを使用して実行します。

```bash
poetry run pytest
```

## 5. Dockerでの実行

GPU（NVIDIA CUDA）対応のDockerfileが用意されています。

### 5.1. Dockerイメージのビルド

```bash
docker build -t my-ruri-app .
```

### 5.2. Dockerコンテナの起動 (GPU版)

```bash
docker run --gpus all -p 8000:8000 -e APP_PORT=8000 my-ruri-app
```

### 5.3. Docker (CPU版)での実行

GPUがない環境やCPUでの実行を希望する場合は、`Dockerfile.cpu`を使用します。

#### 5.3.1. Dockerイメージのビルド (CPU版)

```bash
docker build -f Dockerfile.cpu -t my-ruri-app-cpu .
```

#### 5.3.2. Dockerコンテナの起動 (CPU版)

```bash
docker run -p 8000:8000 -e APP_PORT=8000 my-ruri-app-cpu
```

## 6. 実際のモデルへの切り替えガイド

現在の実装は、テストを容易にするための**モックモデル**を使用しています。実際のHugging Faceモデルを使用するには、以下の手順に従ってください。

### 6.1. 追加ライブラリのインストール

`sentence-transformers`と、GPUを利用する場合は`torch`が必要です。

```bash
poetry add sentence-transformers torch
```

### 6.2. モデルローダーの修正

`src/app/models.py` ファイルを以下のように修正します。

1. **実際のモデルクラスをインポートします。**
   ```python
   from sentence_transformers import SentenceTransformer, CrossEncoder
   import torch
   ```

2. **`get_model` 関数を修正し、モックの代わりに実際のモデルをロードするようにします。**

   ```python
   # 変更前
   # model = MockSentenceTransformer(model_name=model_name)
   # model = MockCrossEncoder(model_name=model_name)

   # 変更後
   device = "cuda" if torch.cuda.is_available() else "cpu"

   if model_name in EMBEDDING_MODELS:
       model = SentenceTransformer(model_name, device=device)
       _model_cache[model_name] = model
       return model

   if model_name in RERANK_MODELS:
       model = CrossEncoder(model_name, device=device)
       _model_cache[model_name] = model
       return model
   ```

これにより、APIはリクエストに応じて実際のHugging Faceモデルをダウンロードし、推論に使用するようになります。
