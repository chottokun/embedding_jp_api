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

### 3.1. モデル設定 (`config/models.yml`)

利用可能なモデルは `config/models.yml` で管理されています。**このファイルに記載されていないモデルはAPIで使用できません。**リクエスト時に未登録のモデルを指定するとエラーが返されます。

```yaml
embedding_models:
  - "cl-nagoya/ruri-v3-30m"
  - "cl-nagoya/ruri-v3-310m"

rerank_models:
  - "cl-nagoya/ruri-v3-reranker-310m"
```

モデルを追加・変更する場合はこのファイルを編集し、サーバーを再起動してください。Docker環境では、事前に `./run.sh download` でモデルをダウンロードしておくことを推奨します。

### 3.2. 必要なツール
- [uv](https://docs.astral.sh/uv/) (開発・パッケージ管理。Poetry の代わりに全面的に採用されています)
- Python 3.11+

### 3.3. 環境のセットアップ

```bash
uv sync
```

### 3.4. 開発サーバーの実行

Uvicornを使用して開発サーバーを起動します。ポートは環境変数 `APP_PORT` で変更可能です（デフォルト: 8000）。

```bash
export APP_PORT=8000
uv run uvicorn src.app.main:app --reload --port $APP_PORT
```

### 3.5. 本番環境での実行 (Gunicorn)

Linuxベースの環境では、GunicornとUvicornワーカーを組み合わせて実行することが推奨されます。

```bash
export GUNICORN_WORKERS=2
uv run gunicorn --workers $GUNICORN_WORKERS --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 300 --worker-tmp-dir /dev/shm --keep-alive 5 src.app.main:app
```

**安定性のためのポイント**:
- `--worker-tmp-dir /dev/shm`: ワーカーハートビートファイルをメモリ上に配置し、Docker環境でのI/O遅延によるタイムアウトを防止します。
- `--keep-alive 5`: HTTP Keep-Alive接続のタイムアウト（秒）を設定し、接続の再利用を改善します。
- GPU環境では `--preload` はCUDAコンテキストの問題を引き起こす可能性があるため、`--preload` を外すことを推奨します。
- 複数ワーカーを使用する場合、モデルごとの排他的ロックが自動的に有効になり、スレッド安全性が確保されます。

### 3.6. パフォーマンスとセキュリティの最適化
 
 このサーバーは、高負荷なモデル推論を効率的に処理し、安全に運用するために以下の最適化が行われています。
 
 - **モデルロックの最適化**: 推論（`encode`/`predict`）のみをロック対象とし、トークナイズやレスポンス成形などのCPU集中的な処理をロック外で実行することで、並列リクエスト時のスループットを向上させています。
 - **セキュリティヘッダーの自動付与**: すべてのレスポンスに `X-Frame-Options: DENY`, `Content-Security-Policy` 等の標準的なセキュリティヘッダーを付与し、共通のWeb脆弱性から保護します。
 - **Embeddings処理の高速化**: トークン数の計算を入力処理と同時に行うことで、冗長なトークナイズ（lengthチェック、usage計算、モデルエンコード）を削減し、O(N)パスを最小化しています。
- **スレッドプールによる並列実行**: 推論処理を行うエンドポイントを `def` (同期) で定義することで、FastAPIが内部のスレッドプールを使用して並列にリクエストを処理できるようにしています。
- **スレッドセーフなモデル・トークナイザー保護**: `threading.Lock` によるモデル推論の保護に加え、ライブラリ内部での競合（`Already borrowed`）を完全に回避するため、トークナイズから推論までを一貫して排他制御下に置いています。これにより、100名規模の同時アクセス下でもエラー率0%の安定稼働を実現しています。
- **バッチ処理時のプレフィックス計算最適化**: Ruri-v3モデル等のプレフィックスが必要なモデルにおいて、同一リクエスト内の複数入力に対してプレフィックスのトークン計算を1回に集約し、CPU負荷を軽減しています。

### 3.7. 性能評価とキャパシティ

#### CPUモード（Locust、10同時ユーザー）

| メトリクス | 未最適化 (Baseline) | 最適化済み (Optimized) | 改善効果 |
| :--- | :--- | :--- | :--- |
| **スループット (req/s)** | 2.65 | 2.65 | (安定) |
| **Rerank 中央値応答時間 (ms)** | 210 | 180 | **14.3% 高速化** |
| **Rerank 95% 応答時間 (ms)** | 9,700 | 1,800 | **81.4% 削減 (安定化)** |
| **エラー率** | 1.3% (500 Errorなど) | 0.0% | **100% 成功** |

- **最大処理能力**: 合計 **2.5 〜 3.0 req/s** 程度が、CPUのみの構成でエラーなく安定処理できる限界値の目安です。

#### GPUモード（Locust、100同時ユーザー、NVIDIA GeForce RTX 3060）

| メトリクス | 値 | 評価 |
| :--- | :--- | :--- |
| **合計リクエスト数** | 1,478 | - |
| **スループット (req/s)** | **32.0** | GPUリソースの最大活用 |
| **中央値応答時間 (ms)** | **67** | 極限負荷でも超高速 |
| **95% 応答時間 (ms)** | 8,700 | キューイングによる遅延 |
| **エラー率** | **0.0%** (0/1,478) | **完全な安定性** |

- **極限負荷への対応**: スレッド安全性の抜本的強化（トークナイズから推論までの一貫したロック保護）により、100名の同時ユーザーという極限状況下においてもクラッシュ（`Already borrowed`）を完全に排除し、エラー率0%を達成しました。
- **スケーラビリティ**: 高負荷時でも中央値応答時間は一桁・二桁ミリ秒台を維持しており、ロック最適化による並列処理の効果が実証されています。

- **安定性**: `--worker-tmp-dir /dev/shm` と環境変数による最適化により、長時間の高負荷テストにおいても接続タイムアウトが発生せず、安定した動作を確認しています。

## 4. テストの実行

```bash
uv run pytest
```

## 5. 詳細設定 (Environment Variables)

APIサーバーの動作は以下の環境変数で調整可能です。これらは `.env` ファイルに記述するか、実行時に直接指定できます。

| 変数名 | デフォルト値 | 説明 |
| --- | --- | --- |
| `GUNICORN_WORKERS` | `2` | Gunicornのワーカープロセス数。CPUコア数やメモリに合わせて調整してください。 |
| `APP_PORT` | `8000` | APIサーバーが待機するポート番号（ローカル実行時）。 |
| `OMP_NUM_THREADS` | `1` | OpenMPのスレッド数。`1`に設定することでCPUコア競合を防止します。 |
| `MKL_NUM_THREADS` | `1` | Intel MKLのスレッド数。`1`に設定することでCPUコア競合を防止します。 |
| `TOKENIZERS_PARALLELISM` | `false` | HuggingFace Tokenizersの並列処理。`false`に設定することでGunicornワーカー内でのデッドロックを防止します。 |
| `OFFLINE_MODE` | `false` | `true`に設定すると、Hugging Face Hubへのアクセスを行いません。事前にモデルをダウンロードしておく必要があります。 |

### 5.1. .env ファイルでの設定
プロジェクト直下に `.env` ファイルを作成して設定を記述できます。
```bash
GUNICORN_WORKERS=4
```

---

## 6. Dockerによる実行

`run.sh` スクリプトを使用することで、コンテナの起動、モデルの管理、オフラインモードの設定を簡単に行えます。

### 6.1. モデルの事前ダウンロード

起動時間を短縮するため、およびオフライン環境で利用するために、あらかじめモデルをダウンロードすることができます。ダウンロードされたモデルはプロジェクトルートの `.cache/models` に保存されます。

```bash
# CPU版イメージを使用してダウンロード
./run.sh download cpu

# GPU版イメージを使用してダウンロード
./run.sh download gpu
```

### 6.2. サーバーの起動と停止

```bash
# サーバーを起動 (デフォルトはCPUモード)
./run.sh run cpu

# GPUモードで起動
./run.sh run gpu

# サーバーを停止
./run.sh stop
```

### 6.3. オフラインモード

環境変数 `OFFLINE_MODE=true` を設定して起動すると、Hugging Face Hubへのアクセスが発生しなくなります。事前に `download` コマンドでモデルを取得済みである必要があります。

```bash
export OFFLINE_MODE=true
./run.sh run cpu
```

## 7. Docker Composeによる管理

より詳細な管理（ポート変更やワーカー数調整）を行う場合は、`docker-compose.yml` を直接利用または `run.sh` と環境変数を組み合わせて使用します。

```bash
# ポート8080、ワーカー数4で起動する例
APP_PORT=8080 GUNICORN_WORKERS=4 ./run.sh run cpu
```

### サービス名
- **`api-cpu`**: CPU専用イメージ
- **`api-gpu`**: NVIDIA GPU対応イメージ

## 8. 負荷テストの実行

Locustを使用してAPIの負荷テストを実行できます。

1.  **Locustの起動**

    ```bash
    uv run locust -f locustfile.py --host http://localhost:8000
    ```

2.  **ヘッドレスモードでの実行（30秒間）**

    ```bash
    uv run locust -f locustfile.py --headless -u 5 -r 1 --run-time 30s --host http://localhost:8000
    ```

## 9. ビルドパフォーマンスの最適化

本プロジェクトのDockerイメージは `torch` などの巨大なライブラリを含むため、初回ビルドに時間がかかる場合があります。
`Dockerfile` / `Dockerfile.cpu` は**マルチステージビルド**を採用しており、以下の最適化が行われています。

- **ステージ1 (Builder)**: 依存関係のインストールのみを実行。`pyproject.toml` が変更されない限り、Dockerキャッシュが再利用されます。
- **ステージ2 (Runtime)**: 軽量なランタイムイメージ（GPU版は `cuda:12.1.1-runtime`）をベースに、ビルド済みパッケージとソースコードのみをコピーします。
- **効果**: ソースコードのみの変更時はステージ1がキャッシュヒットするため、**再ビルドが数十秒で完了**します。

---

## 10. Text Embeddings Inference (TEI) 統合 (GPU専用)

Hugging Face社が提供する高速な埋め込みベクトル推論サーバー **Text Embeddings Inference (TEI)** に推論処理をオフロードするためのプロキシ（ゲートウェイ）機能をサポートしています。

GPU環境において本プロジェクトの FastAPI をゲートウェイとして前面に置き、バックエンドで TEI コンテナを動かすことで、API Key 認証やセキュリティヘッダー、PII マスクといった付加価値を維持したまま、**推論スループットを劇的に高速化**できます。

### 10.1. パフォーマンス比較結果 (同時10ユーザー負荷テスト時)

| 評価項目 | ローカルモデル推論 (既存 GPU) | TEI プロキシ推論 (今回 GPU) | 改善効果 |
| :--- | :--- | :--- | :--- |
| **全体平均応答時間** | 1,117 ms | **21 ms** | **約 53 倍高速化** |
| **Embeddings 応答時間** | 1,219 ms | **17 ms** | **約 71 倍高速化** |
| **Rerank 応答時間** | 810 ms | **32 ms** | **約 25 倍高速化** |
| **最大スループット (req/s)** | 2.42 req/s | **3.38 req/s** | **+39.6%** |

### 10.2. 設定方法

FastAPI 起動時に以下の環境変数を設定すると、対応するエンドポイントへのリクエストが TEI へ自動的にプロキシされます。

* **`EMBEDDING_TEI_URL`**: 埋め込み用 TEI コンテナのホストURL（例: `http://localhost:8081`）
* **`RERANK_TEI_URL`**: リランク用 TEI コンテナのホストURL（例: `http://localhost:8082`）

### 10.3. バックエンドコンテナ (TEI) の起動方法例

Docker を使用して GPU 上で TEI コンテナをオフライン（ローカルキャッシュ利用）で起動するコマンドの例です。

```bash
# 埋め込みモデル (ruri-v3-30m) の TEI コンテナを起動 (ポート 8081)
docker run -d --gpus all \
  --name tei-embeddings \
  -p 8081:80 \
  -e HF_HUB_OFFLINE=1 \
  -v $(pwd)/.cache/models:/data \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id /data/hub/models--cl-nagoya--ruri-v3-30m/snapshots/24899e5de370b56d179604a007c0d727bf144504
```
