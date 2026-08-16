---
okf_version: "0.2"
title: Architecture Knowledge Index
description: システムアーキテクチャ、推論パイプライン、並行制御モデルのインデックス
---

# Architecture Knowledge Index

## 概要

APIサーバーの全体構造、マルチモーダル推論エンジン、スレッドセーフティ、セキュリティ防御層に関する技術仕様です。

## ドキュメント一覧

* [システム概要・マイクロサービス連携](./overview.md) - 全体アーキテクチャ、API Gateway、TEI Proxy、セキュリティ多層防御
* [マルチモーダル推論アーキテクチャ](./multimodal.md) - `bge-visualized-m3` による画像＋テキスト統合エンコード、SSRF 防御
* [並行制御・スレッドセーフティモデル](./concurrency.md) - `threading.Lock` と `anyio` ワーカープールによる GPU/CPU 競合制御
