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

### 2.2. マルチモーダル埋め込み (Multimodal Embeddings)

`POST /v1/embeddings`

`bge-visualized-m3` モデルを指定することで、画像単体、またはテキスト＋画像の複合入力に対するベクトル埋め込みを生成できます。

#### 入力フォーマット

1. **フラット形式 (Flat Item)**:
   ```json
   {
     "model": "bge-visualized-m3",
     "input": {
       "text": "青い服を着た人物",
       "image_url": "data:image/png;base64,iVBORw0KG..."
     }
   }
   ```
2. **OpenAI Chat互換 コンテンツパーツ配列 (Content Parts)**:
   ```json
   {
     "model": "bge-visualized-m3",
     "input": [
       {"type": "text", "text": "赤い車"},
       {"type": "image_url", "image_url": {"url": "https://example.com/car.png"}}
     ]
   }
   ```

#### セキュリティ仕様
- **SSRF防御**: リダイレクト追従時を含め、プライベートIP（127.0.0.1, 10.x, 192.168.x）およびクラウドメタデータエンドポイント（169.254.169.254）へのアクセスはブロックされ、HTTP 400 を返却します。
- **DoS防御**: 画像デコード爆弾防御（`MAX_IMAGE_PIXELS = 20,000,000`）およびファイルサイズ制限（15MB）を適用。
- **モデル不一致ガード**: 画像入力をテキスト専用モデル（Ruri-v3等）に送信した場合は、モデル破壊や500エラーを防ぐため HTTP 400 を返却します。

---

### 2.3. ヘルスチェック (Health Checks)

マイクロサービスの死活監視（Liveness / Readiness Probes）用エンドポイントを提供します。APIキー認証は不要です。

- `GET /health`
- `GET /healthz`

**レスポンス例**:
```json
{"status": "ok"}
```

---

### 2.4. 再ランキング (Rerank)

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
 - **PII（個人情報）の自動マスクと情報漏洩防止**: システムエラー（500）発生時、グローバル例外ハンドラーがエラーログ内のメールアドレス等の個人特定情報（PII）を `[REDACTED]` に自動的にマスキングします。また、クライアントには詳細なスタックトレースを返さず汎用エラーのみを返却し、情報漏洩を防ぎます。
 - **Embeddings処理の高速化**: トークン数の計算を入力処理と同時に行うことで、冗長なトークナイズ（lengthチェック、usage計算、モデルエンコード）を削減し、O(N)パスを最小化しています。
- **スレッドプールによる並列実行**: 推論処理を行うエンドポイントを `def` (同期) で定義することで、FastAPIが内部のスレッドプールを使用して並列にリクエストを処理できるようにしています。
- **スレッドセーフなモデル・トークナイザー保護**: `threading.Lock` によるモデル推論の保護（`model.lock`）に加え、`transformers`のRust製高速トークナイザーにおける内部競合（`Already borrowed`）を完全に回避するため、個別のトークンカウント時に `model.tokenizer_lock` を使用し、さらに `model.encode`/`model.predict` による推論実行中にも `model.tokenizer_lock` を同時に保持し続ける二重ロック保護設計を採用しています。これにより、100名規模の並列高負荷アクセス下でもエラー率0%の極めて高い安定稼働を実現しています。
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

## 8. テストと負荷・ストレステストの実行

### 8.1. 単体・統合テスト (pytest)
```bash
uv run pytest
```

### 8.2. 実動コンテナ E2E 検証 (`test_e2e_live.py`)
稼働中のサーバーまたはコンテナに対して、認証・埋め込み・画像入力・SSRF防御・同時実行性を一括検証します。
```bash
# ポート8000に対して実行
uv run python test_e2e_live.py 8000
```

### 8.3. 100並行同時接続ストレステスト (`run_heavy_load_test.py`)
テキスト埋め込み、バッチ推論、リランク、画像入力拒否、ヘルスチェックを混在させた100並行の同時アクセス耐久テストを実行します。
```bash
uv run python run_heavy_load_test.py
```

### 8.4. Locustによる負荷テスト
```bash
# Web UI起動
uv run locust -f locustfile.py --host http://localhost:8000

# ヘッドレスモードでの実行（30秒間、20同時ユーザー）
uv run locust -f locustfile.py --headless -u 20 -r 5 --run-time 30s --host http://localhost:8000
```

## 9. ビルドパフォーマンスの最適化

本プロジェクトのDockerイメージは `torch` などの巨大なライブラリを含むため、初回ビルドに時間がかかる場合があります。
`Dockerfile` / `Dockerfile.cpu` は**マルチステージビルド**を採用しており、以下の最適化が行われています。

- **ステージ1 (Builder)**: 依存関係のインストールのみを実行。`pyproject.toml` が変更されない限り、Dockerキャッシュが再利用されます。
- **ステージ2 (Runtime)**: 軽量なランタイムイメージ（GPU版は `cuda:12.1.1-runtime`）をベースに、ビルド済みパッケージとソースコードのみをコピーします。
- **効果**: ソースコードのみの変更時はステージ1がキャッシュヒットするため、**再ビルドが数十秒で完了**します。

---

## 10. Text Embeddings Inference (TEI) 統合

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

### 10.4. トラブルシューティング & 重要な注意点 (罠)

TEI を用いた本番運用において、以下の重大な「技術的制約（罠）」が存在するため、必ずご確認ください。

1. **ModernBERT モデル (ruri-v3-30m等) をロードする際のバージョン制約**
   TEI 1.5 以前のコンテナイメージを使用した場合、ModernBERT アーキテクチャのモデルをロードしようとすると `unknown variant modernbert` エラーが発生してコンテナが異常終了します。
   * **対策**: 必ず最新（ModernBERT をネイティブサポートした `ghcr.io/huggingface/text-embeddings-inference:latest` または `2.0+`）のイメージを使用してください。

2. **オフラインモード (`HF_HUB_OFFLINE=1`) 時のモデルパス指定**
   コンテナをオフライン（ローカルキャッシュマウント）で動作させる際、`--model-id` にモデルのベース名（例: `cl-nagoya/ruri-v3-30m`）を指定するだけでは、TEI が内部で Hugging Face API への接続を試みて `relative URL without a base` 等のエラーで失敗します。
   * **対策**: ローカルマウントしたディレクトリ内のスナップショットコミットIDの実パス（例: `/data/hub/models--cl-nagoya--ruri-v3-30m/snapshots/<COMMIT_ID>`）を直接 `--model-id` に指定してロードさせてください。

3. **コンテナメモリとスレッド競合の最適化**
   Gunicorn を使ってマルチワーカーで動作させる際、各ワーカーが Hugging Face Tokenizers の並列スレッドを起動するとデッドロックが発生する可能性があります。
   * **対策**: 環境変数 `TOKENIZERS_PARALLELISM=false` を必ず設定した状態で起動してください（本プロジェクトの Dockerfile 内では自動設定されています）。

### 10.5. CPU環境下における TEI 導入の意義と効果

GPUを搭載していないローカルまたは本番CPU環境であっても、TEIを導入する意義と技術的なメリットは十分にあります。

#### ① Python の GIL（グローバルインタプリタロック）制約の回避
Python（FastAPI + `sentence-transformers`）のローカル推論構成では、非同期通信（async/await）を使用しても、背後の PyTorch による行列演算やトークナイズ処理中に **GIL（Global Interpreter Lock）** が発生し、並行リクエストが重なったときにスレッドがブロックされて応答速度が著しく低下します。
* **メリット**: TEI は **Rust** で動作するため GIL の制約が一切なく、FastAPI が非同期で処理を TEI にオフロードするだけで、複数の CPU コアを競合なく 100% フル稼働させて推論を並行処理できます。これにより、同時アクセス時にも FastAPI サーバーが引きずられて遅延・フリーズするのを防げます。

#### ② Dynamic Batching（動的バッチング）
TEI の最大の特徴である **Dynamic Batching**（複数の並行リクエストの入力をリアルタイムで1つのバッチにまとめて行列演算を行う機能）は、CPU環境でも動作します。
* **メリット**: 高負荷時にリクエストを1件ずつ直列に処理するのではなく、まとめて一括で推論にかけるため、多重アクセス時のスループット（秒間処理リクエスト数）が飛躍的に高まり、遅延のスパイク（悪化）を防ぎます。

#### ③ 高度な CPU ベクトル最適化とメモリ効率
* **メリット**: TEI は CPU 向けの高速なベクトル演算命令セット（AVX-512、AMX、Neon など）に高度に最適化されています。また、PyTorch や巨大な Python ライブラリ群を含む Python コンテナに比べ、Candle バックエンドと Rust で動く TEI は**メモリフットプリントが極めて小さく**、ホストサーバーの物理メモリ消費を低く抑えられます。

#### ⚠️ CPU環境での注意点と限界
* **レイテンシの限界**: 単一アクセス時における1リクエストあたりの純粋な応答速度（レイテンシ）自体は、GPUのような爆発的な高速化（70倍など）は望めず、ローカル Python 推論に対しておよそ 2〜5 倍程度の高速化にとどまります（CPUのクロック性能に縛られます）。
* **スレッド競合**: TEI コンテナが CPU 資源をフルに使い切ると、前面の FastAPI サーバーや他のプロセスと物理コアを奪い合うため、コンテナの割り当て CPU スレッド数を適切にコントロールすることを推奨します。
