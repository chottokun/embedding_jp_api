## 📝 コーディングルール

* **Python バージョン**：プロジェクトで指定した Python バージョンを使用する
* **パッケージ管理**：`uv` を利用し、依存関係は `pyproject.toml` で管理する
* **仮想環境**：`uv` が作成する仮想環境を利用する
* **コードフォーマット**：`ruff format` を使用する
* **Lint**：`ruff check` を実行し、警告・エラーを解消してからコミットする
* **型ヒント**：公開 API や主要な関数・クラスには型ヒントを付与する
* **テスト**：新機能・修正には必要に応じてテストを追加し、コミット前に実行する
* **設定管理**：機密情報はコードに含めず、環境変数や `.env` を利用する
* **依存関係の追加**：`uv add` を使用し、`pip install` による直接追加は行わない

- **Secrets**: NEVER hardcode real keys/tokens. Gitleaks scanning is enforced. Use fake placeholders (e.g., `YOUR_KEY`).