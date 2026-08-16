---
okf_version: "0.2"
title: Infrastructure Knowledge Index
description: コンテナ構成、オフライン（エアギャップ）運用、ベンチマークデータのインデックス
---

# Infrastructure Knowledge Index

## 概要

Docker / Docker Compose による GPU・CPU コンテナデプロイ、完全オフライン運用、環境変数設定、および実機ベンチマークデータに関するインフラナレッジです。

## ドキュメント一覧

* [Docker / Docker Compose デプロイメントガイド](./docker.md) - GPU / CPU マルチステージビルド、Compose 構成
* [完全オフライン（エアギャップ）運用ガイド](./offline_mode.md) - 事前ダウンロード、オフライン検証、`.env` 階層型設定
* [実機ベンチマーク測定結果](./benchmarks.md) - NVIDIA GeForce RTX 3060 (12GB VRAM) / ホスト CPU での実測データ
