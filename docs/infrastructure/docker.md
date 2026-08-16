---
type: Concept
title: Docker / Docker Compose デプロイメントガイド
description: GPU / CPU 向けの Dockerfile、Docker Compose 設定、およびヘルスチェック仕様
status: stable
generated:
  by: agent/antigravity
  at: 2026-08-16T09:10:00Z
tags:
  - docker
  - docker-compose
  - deployment
  - gpu
  - cpu
sources:
  - resource: /Dockerfile
    title: GPU Dockerfile
  - resource: /Dockerfile.cpu
    title: CPU Dockerfile
  - resource: /docker-compose.yml
    title: Docker Compose Configuration
---

# Docker / Docker Compose デプロイメントガイド

## 1. 起動コマンド

### GPU 環境 (NVIDIA Container Toolkit 必須)
```bash
docker compose up -d --build api-gpu
```

### CPU 環境
```bash
docker compose up -d --build api-cpu
```

## 2. Dockerfile 構成

- **`Dockerfile` (GPU)**: `nvidia/cuda:12.4.1-runtime-ubuntu22.04` をベースに PyTorch CUDA 版をインストール。
- **`Dockerfile.cpu` (CPU)**: `python:3.12-slim` をベースに軽量 CPU 版 PyTorch をインストール。
- **マルチステージビルド**: `uv` による依存関係解決とモデル事前ダウンロードをビルド時に実行可能。

## 3. ヘルスチェック仕様

コンテナ内部から以下のエンドポイントで生存状態・準備状態を監視します。
- `GET /health`: サービス稼働状態（200 OK）
- `GET /health/ready`: モデルのロード完了・推論準備状態（200 OK）
