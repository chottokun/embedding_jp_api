# Knowledge Update Log

## 2026-08-29
* **Creation**: `docs/architecture/services.md` を作成し、サービス層（`src/app/services/`）の抽象基底クラス、FastAPI `Depends` による依存性注入（DI）、および `MockEmbeddingService`/`MockRerankService` のモック設計を文書化しました。
* **Update**: PR #84（モジュール構成の責務分離・DI化および GitHub Actions CI の高速化）のマージに伴い、CI/テスト分離ポリシー（`not integration` による高速ユニットテスト）と実機ベンチマーク/実動統合テストの動作検証結果を反映しました。

## 2026-08-16
* **Creation**: LLM-Wiki (OKF v0.2) ナレッジベースを整備・構造化しました。
* **Update**: 実データ図面＋テキストによるマルチモーダル（`bge-visualized-m3`）エンコード検証および網羅的負荷テスト結果をナレッジに統合しました。
* **Update**: 完全オフライン（エアギャップ）モデル事前ダウンロードおよびロード検証（`--verify-offline`）、`.env` 階層型設定の仕様を追加しました。
* **Update**: 実機 NVIDIA GeForce RTX 3060 (12GB VRAM) / ホスト CPU によるベンチマーク測定結果を反映しました。
