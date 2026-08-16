---
okf_version: "0.2"
title: Embedding JP API Knowledge Base
description: 日本語テキスト埋め込み・マルチモーダル埋め込み・リランキングAPIサーバーのナレッジベース
---

# Embedding JP API Knowledge Base

## 概要

本リポジトリにおけるシステム設計・アーキテクチャ・ドメイン仕様・インフラ運用の最上位ナレッジインデックスです。

## ナレッジ領域

* [Architecture](./architecture/README.md) - システム全体構造、スレッドセーフモデル、マルチモーダル推論アーキテクチャ
* [Domain](./domain/README.md) - 対応モデル仕様（Ruri-v3, BGE）、API スキーマ定義、プレフィックス仕様
* [Infrastructure](./infrastructure/README.md) - Docker / GPU・CPU Compose 構成、オフライン完全稼働（エアギャップ）、実機ベンチマークデータ

## 更新履歴

* [Knowledge Update Log](./log.md)
