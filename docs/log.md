# Knowledge Update Log

## 2026-09-12
* **Infrastructure & Kubernetes**: 本番運用向けの Kubernetes マニフェスト（`deploy/kubernetes/deployment.yaml`, `service.yaml`, `hpa.yaml`）および `docs/deployment.md` を追加。HPA による CPU/メモリ負荷に応じた水平自動スケール（1〜5 Pod）と死活／準備監視・Prometheus スクレイプ定義を標準化しました。
* **API Documentation & DX**: OpenAPI 3.x / Swagger UI (`/docs`) のスキーマ定義を大幅拡充。エンドポイント別のタグ分類（`Embeddings`, `Reranking`, `Models`, `Health`, `Metrics`）、概要・詳細説明、レスポンスコード（400, 401, 413, 429, 503）、および `EmbeddingRequest` / `RerankRequest` の実例（`json_schema_extra`）を配備しました（テスト: `src/tests/test_extended_features.py`）。
* **Observability & APM**: Prometheus メトリクスにモデル別トークン消費カウンタ（`http_prompt_tokens_total`）およびエンドポイント別バッチサイズ分布ヒストグラム（`http_request_batch_size`）を追加し、運用監視・リソース予測性能を強化しました（テスト: `src/tests/test_metrics.py`）。
* **Feature & Performance**: アプリケーション起動時に事前ロードを行う `PRELOAD_MODELS` 設定を新設。初回リクエストのコールドスタート遅延をゼロにする事前ウォームアップ機構を導入しました（テスト: `src/tests/test_config.py`, `src/tests/test_graceful_shutdown.py`）。
* **SRE & High Availability**: FastAPI Lifespan におけるグレースフルシャットダウン（`SHUTDOWN_DRAIN_TIMEOUT_SECONDS`）および In-Flight リクエスト追跡ミドルウェアを実装。シャットダウン移行中の新規リクエストに対して `503 Service Unavailable` を返却し、処理中リクエストを正常完了させるドレイン機構を導入しました（テスト: `src/tests/test_graceful_shutdown.py`）。
* **Optimization & Performance**: `TORCH_DTYPE` 環境変数による推論精度切り替え（`bfloat16`, `float16`, `float32`）および `torch.autocast` を統合。SentenceTransformer / CrossEncoder / VisualizedBGE モデルへ型安全に反映しました（テスト: `src/tests/test_torch_dtype.py`）。
* **Feature**: OpenAI 完全互換の `dimensions`（Matryoshka 次元削減 + L2 再正規化）および `encoding_format: "base64"`（IEEE 754 float32 リトルエンディアン Base64 化）を実装し、ローカル推論・TEIプロキシ・マルチモーダルの全経路に統合しました（テスト: `src/tests/test_dimensions_and_encoding.py`）。
* **Security & SRE**: API キー／クライアント IP 単位で毎分リクエスト数を制限するスライディングウィンドウ型 `RateLimiter`（`RATE_LIMIT_PER_MINUTE`、超過時 `429 Too Many Requests` + `Retry-After`）を導入し、死活監視エンドポイントの自動除外を適用しました（テスト: `src/tests/test_rate_limit.py`）。
* **Feature**: OpenAI 互換のモデル一覧取得エンドポイント (`GET /v1/models`) を新設し、テスト `src/tests/test_models_endpoint.py` を追加しました。
* **Security**: 悪意ある巨大リクエスト（32MB超）による OOM を早期防御する `PayloadLimitMiddleware`（`413 Payload Too Large`）を導入し、テスト `src/tests/test_payload_limit.py` を追加しました。
* **Feature & SRE**: メモリ／VRAM を動的に解放可能なモデルアンロードエンドポイント (`POST /v1/models/unload`) を新設し、テスト `src/tests/test_model_unload.py` を追加しました。
* **Security**: マルチモーダル画像ダウンロードにおける DNS Rebinding / TOCTOU 脆弱性対策として、SSRF 検査時に解決した安全な IP アドレスを TCP 接続先として直接固定（IP Pinning）する `SafeNetworkBackend` を実装しました。
* **Observability**: 構造化 JSON ロギングおよび `X-Request-ID` によるリクエスト追跡（ContextVars連携）を実装し、テスト `src/tests/test_logger.py` を追加しました。
* **Feature**: `/ready` エンドポイントを新設し、Liveness (`/health`, `/healthz`) と分離して GPU 状態およびロード済みモデルを監視可能にしました。
* **Observability**: `prometheus_client` を導入し、リクエスト数・レイテンシを計測する `/metrics` エンドポイントを新設しました。
* **Optimization**: `VisualizedBGEEmbeddingModel.encode_multimodal` における画像テンソル・テキストのバッチ一括処理化を実装し、マルチモーダル推論のスループットを向上させました。
* **Refactor**: TEI プロキシ処理を `httpx.AsyncClient` による完全非同期呼び出し（`async/await`）へ刷新し、I/Oブロッキングを解消しました。
* **CI & Security**: GitHub Actions CI に `uv audit`（依存関係脆弱性診断）および `gitleaks`（シークレット漏洩スキャン）を組み込み、リポジトリ全体の静的解析（`ruff check .`）を適用しました。
* **Code Health**: Jules との連携により、`scratch/` および `scripts/` に残存していた Lint エラーを完全解消しました。


## 2026-08-29
* **Creation**: `docs/architecture/services.md` を作成し、サービス層（`src/app/services/`）の抽象基底クラス、FastAPI `Depends` による依存性注入（DI）、および `MockEmbeddingService`/`MockRerankService` のモック設計を文書化しました。
* **Update**: PR #84（モジュール構成の責務分離・DI化および GitHub Actions CI の高速化）のマージに伴い、CI/テスト分離ポリシー（`not integration` による高速ユニットテスト）と実機ベンチマーク/実動統合テストの動作検証結果を反映しました。

## 2026-08-16
* **Creation**: LLM-Wiki (OKF v0.2) ナレッジベースを整備・構造化しました。
* **Update**: 実データ図面＋テキストによるマルチモーダル（`bge-visualized-m3`）エンコード検証および網羅的負荷テスト結果をナレッジに統合しました。
* **Update**: 完全オフライン（エアギャップ）モデル事前ダウンロードおよびロード検証（`--verify-offline`）、`.env` 階層型設定の仕様を追加しました。
* **Update**: 実機 NVIDIA GeForce RTX 3060 (12GB VRAM) / ホスト CPU によるベンチマーク測定結果を反映しました。
