# Security Rules

## Dependency Audit

依存パッケージの脆弱性を定期的に確認する。

- Local check:
  ```bash
  uv audit
````

* CI check:
  Pull Request 時に自動実行する。

## Secrets

* API Key、Token、Password をコードへ直接記載しない
* `.env` を利用する
* Gitleaks による検査を通過すること

## Dependency Policy

* 新規依存は必要性を確認する
* 可能な限り標準ライブラリを優先する


