# OpenAI互換 Embedding & Rerank APIサーバー

## 1. 概要

このプロジェクトは、日本語のテキスト埋め込み（Embedding）および再ランキング（Rerank）機能を提供する、OpenAI互換のFastAPIサーバーです。
名古屋大学にて開発された[Ruri v3モデル](https://huggingface.co/cl-nagoya/ruri-v3-30m)などを利用することを想定しています。

## 2. API仕様

### 2.1. 埋め込み (Embeddings)

`POST /v1/embeddings`

OpenAI標準パラメータに加え、Ruri-v3等のモデル性能を最大限に引き出すための拡張パラメータをサポートしています。

#### リクエストボディ (JSON)

| フィールド名 | 型 | 必須 | 説明 |
| --- | --- | --- | --- |
| `input` | string \| array | Yes | 埋め込み対象のテキストまたはテキストのリスト。 |
| `model` | string | Yes | 使用するモデルID（例: `cl-nagoya/ruri-v3-310m`）。 |
| `input_type` | string | No | タスクの種類を指定。Ruri-v3のプレフィックスに自動マッピングされます。 |
| `instruction` | string | No | モデルへの具体的な指示文。将来的な指示ベースモデルへの対応用。 |
| `apply_ruri_prefix` | boolean | No | `true`の場合、`input_type`が未指定でも入力形式に基づき自動でプレフィックスを付与します（互換性用）。 |

#### `input_type` とプレフィックスのマッピング

`input_type`を指定すると、Ruri-v3モデルに対して以下の日本語プレフィックスが自動挿入されます。

* **`query`**: `"検索クエリ: "` （非対称検索の質問側）
* **`document`**: `"検索文書: "` （非対称検索の回答・知識ベース側）
* **`classification`**: `"トピック: "` （分類、クラスタリング用）
* **`clustering`**: `"トピック: "` （同上）
* **`sts`**: `""` (空文字) （文の類似度、対称的タスク用）

#### 処理ルール

- **プレフィックスの二重付与防止**: 入力テキストが既に指定のプレフィックスで始まっている場合、API側での重複付与は行われません。
- **トークン切り詰め (Truncation)**: 入力がモデルの最大長（Ruri-v3は8,192トークン）を超える場合、プレフィックスを優先的に保持し、入力テキストの後方を切り詰めます。

#### Python SDK 利用例

OpenAI公式クライアントの `extra_body` を利用して拡張パラメータを渡せます。

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-no-key")

# input_typeを明示的に指定して埋め込みを生成
response = client.embeddings.create(
    model="cl-nagoya/ruri-v3-310m",
    input="名古屋大学で開発されたモデルについて教えて。",
    extra_body={
        "input_type": "query"
    }
)
```

---

### 2.2. 再ランキング (Rerank)

`POST /v1/rerank`

Jina/Cohere等の標準的な再ランキングAPIに準拠したスキーマを提供します。

#### リクエストボディ (JSON)

| フィールド名 | 型 | 必須 | 説明 |
| --- | --- | --- | --- |
| `query` | string | Yes | 検索クエリ。 |
| `documents` | array | Yes | ランク付け対象の文書リスト。 |
| `model` | string | Yes | 使用するモデルID（例: `cl-nagoya/ruri-v3-reranker-310m`）。 |
| `top_n` | integer | No | 返却する上位件数（`top_k`も互換性のために受付可能）。 |
| `return_documents` | boolean | No | レスポンスに文書の本文を含めるかどうか。 |

#### レスポンスボディ (JSON)

レスポンスには、クエリとドキュメントのペアの合計トークン消費量を示す `usage` フィールドが含まれます。

| フィールド名 | 型 | 説明 |
| --- | --- | --- |
| `query` | string | 検索クエリ。 |
| `data` | array | ランク付けされた文書とそのスコア。 |
| `model` | string | 使用されたモデルID。 |
| `usage` | object | トークン使用量（`prompt_tokens`, `total_tokens`）。 |

#### リクエスト例 (curl)

```bash
curl -X POST "http://localhost:8000/v1/rerank" \
-H "Content-Type: application/json" \
-d '{
  "query": "AIの未来について",
  "documents": [
    "猫について",
    "人工知能の進化",
    "日本の首都"
  ],
  "model": "cl-nagoya/ruri-v3-reranker-310m",
  "top_n": 2,
  "return_documents": true
}'
```

## 3. セットアップと実行

### 3.1. 必要なツール
- [uv](https://docs.astral.sh/uv/) (推奨) または [Poetry](https://python-poetry.org/)
- Python 3.11+

### 3.2. 環境のセットアップ

uvを使用する場合：
```bash
uv sync
```

Poetryを使用する場合：
```bash
poetry install
```

### 3.3. 開発サーバーの実行

Uvicornを使用して開発サーバーを起動します。ポートは環境変数 `APP_PORT` で変更可能です（デフォルト: 8000）。

uvを使用する場合：
```bash
export APP_PORT=8000
uv run uvicorn src.app.main:app --reload --port $APP_PORT
```

Poetryを使用する場合：
```bash
export APP_PORT=8000
poetry run uvicorn src.app.main:app --reload --port $APP_PORT
```

### 3.4. 本番環境での実行 (Gunicorn)

Linuxベースの環境では、GunicornとUvicornワーカーを組み合わせて実行することが推奨されます。

```bash
poetry run gunicorn --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 300 src.app.main:app
```
**注意**: GPU環境では `--preload` はCUDAコンテキストの問題を引き起こす可能性があるため、本番環境ではワーカー数を1に設定し、`--preload` を外すことを推奨します。また、モデルのロードに時間がかかる場合があるため、`--timeout` を適切に設定してください。

### 3.5. パフォーマンスとスレッドセーフティ

このサーバーは、高負荷なモデル推論を効率的に処理するために以下の最適化が行われています。

- **Embeddings処理の高速化**: トークン数の計算を入力処理と同時に行うことで、冗長なトークナイズ（lengthチェック、usage計算、モデルエンコード）を削減し、O(N)パスを最小化しています。
- **スレッドプールによる並列実行**: 推論処理を行うエンドポイントを `def` (同期) で定義することで、FastAPIが内部のスレッドプールを使用して並列にリクエストを処理できるようにしています。
- **スレッドセーフなモデルロード**: `threading.Lock` を導入しており、並列リクエストが発生しても安全にモデルをロード・キャッシュできます。

## 4. テストの実行

uvを使用する場合：
```bash
uv run pytest
```

Poetryを使用する場合：
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
