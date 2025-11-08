# OpenAI互換 Embedding & Rerank APIサーバー

## 1. 概要

このプロジェクトは、日本語のテキスト埋め込み（Embedding）および再ランキング（Rerank）機能を提供する、OpenAI互換のFastAPIサーバーです。
名古屋大学にて開発された[Ruri v3モデル](https://huggingface.co/cl-nagoya/ruri-v3-30m)などを利用することを想定しています。

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

#### `ruri-v3`モデルのオプショナルな前処理：プレフィックス

`ruri-v3`モデルの性能を最大化するため、`apply_ruri_prefix: true` を設定することで、入力に応じて以下のプレフィックスを付与できます。

- **単一文字列の場合**: `"検索クエリ: "` が先頭に付与されます。
- **文字列の配列の場合**: 各要素に `"検索文書: "` が先頭に付与されます。

この機能はデフォルトで無効になっており、`ruri-v3`以外のモデルではこの設定は無視されます。

#### `apply_ruri_prefix` を有効にするリクエスト例

```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
-H "Content-Type: application/json" \
-d '{
  "input": "今日の天気は晴れです。",
  "model": "cl-nagoya/ruri-v3-30m",
  "apply_ruri_prefix": true
}'
```

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
poetry run gunicorn --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 300 src.app.main:app
```
**注意**: GPU環境では `--preload` はCUDAコンテキストの問題を引き起こす可能性があるため、本番環境ではワーカー数を1に設定し、`--preload` を外すことを推奨します。また、モデルのロードに時間がかかる場合があるため、`--timeout` を適切に設定してください。

## 4. テストの実行

プロジェクトのテストスイートはPytestを使用して実行します。

```bash
poetry run pytest
```

## 5. 簡単実行 (Easy Run with Docker)

`run.sh` スクリプトを使用することで、コンテナの起動とモデルのキャッシュを簡単に行えます。

```bash
# 実行権限を付与（初回のみ）
chmod +x run.sh

# スクリプトを実行 (デフォルトはCPUモード)
./run.sh

# GPUモードで実行する場合
# ./run.sh gpu
```

このスクリプトは、ホストの `~/.cache/models` をコンテナにマウントし、モデルをキャッシュすることで2回目以降の起動を高速化します。

## 6. Dockerでの手動実行

`run.sh` を使用せず、手動でDockerを操作する手順です。

### 6.1. モデルキャッシュについて

コンテナ起動時にホストのディレクトリをコンテナのキャッシュディレクトリ (`/root/.cache`) にマウントすると、モデルのダウンロードが一度で済み、再起動が高速になります。

```bash
# キャッシュディレクトリをホスト上に作成
mkdir -p ~/.cache/models
```

### 6.2. Dockerイメージのビルド

**GPU版:** 
```bash
docker build -t embedding_jp_api-gpu .
```

**CPU版:** 
```bash
docker build -f Dockerfile.cpu -t embedding_jp_api-cpu .
```

### 6.3. Dockerコンテナの起動

**GPU版:** 
```bash
docker run --gpus all -p 8000:8000 \
  -v ~/.cache/models:/root/.cache \
  embedding_jp_api-gpu
```

**CPU版:** 
```bash
docker run -p 8000:8000 \
  -v ~/.cache/models:/root/.cache \
  embedding_jp_api-cpu
```

## 7. 負荷テストの実行


Locustを使用してAPIの負荷テストを実行できます。LocustはWebベースのUIを提供し、テストの進行状況をリアルタイムで確認できます。

1.  **Locustの起動**

    ```bash
    locust -f locustfile.py --host http://localhost:8000
    ```

    このコマンドを実行すると、LocustのWeb UIが `http://localhost:8089` で利用可能になります。

2.  **負荷テストの開始**

    Web UIにアクセスし、テストユーザー数とRamp-up期間を設定してテストを開始します。

    `locustfile.py`には、EmbeddingとRerankのエンドポイントに対するテストシナリオが定義されています。
