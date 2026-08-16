---
type: Concept
title: 完全オフライン（エアギャップ）運用ガイド
description: モデルの事前ダウンロード、HF_HUB_OFFLINE 検証、.env / config.toml による外部通信遮断
status: stable
generated:
  by: agent/antigravity
  at: 2026-08-16T09:10:00Z
tags:
  - offline
  - airgap
  - configuration
  - pre-download
sources:
  - resource: /src/app/download_models.py
    title: Model Pre-downloader & Offline Dry-Run
  - resource: /src/app/config.py
    title: Hierarchical Configuration Loader
  - resource: /.env.example
    title: Environment Variable Example Template
---

# 完全オフライン（エアギャップ）運用ガイド

## 1. 概要

閉域網（オンプレミス、金融・医療機関の隔離環境、エアギャップ環境）において、外部ネットワークへのアクセスを行わずに API サーバーを稼働させるための手順です。

## 2. モデル事前ダウンロード & オフライン自己検証

すべてのテキストモデル・マルチモーダル重み（`Visualized_m3.pth`）・リランカーモデルを一括事前ダウンロードし、即座にオフラインロードを自己検証します。

```bash
# 全モデルのダウンロードと HF_HUB_OFFLINE=1 下での Dry-Run ロード検証
PYTHONPATH=src uv run python src/app/download_models.py --verify-offline
```

## 3. 設定の優先順位と `.env` サポート

設定は以下の優先順位で自動解決されます：
1. **OS 環境変数** (最優先)
2. **`.env` ファイル**
3. **`config/config.toml`**
4. **デフォルト値**

### `.env` によるオフライン指定
```dotenv
# .env ファイルに記載
OFFLINE_MODE=true
API_KEY=your-secure-api-key
PORT=8000
```

`OFFLINE_MODE=true` を指定すると、サーバー起動時に以下の環境変数が自動的に設定され、Hugging Face Hub への外部アクセスが完全に遮断されます。
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `HF_DATASETS_OFFLINE=1`
