---
type: Architecture Decision
title: サービス層設計と依存性注入（DI）・モック化アーキテクチャ
description: BaseEmbeddingService/BaseRerankServiceの抽象化、FastAPI Dependsによる疎結合設計、および高速CIモック
status: stable
generated:
  by: agent/antigravity
  at: 2026-08-29T13:42:00Z
tags:
  - architecture
  - dependency-injection
  - services
  - testing
  - fast-api
sources:
  - resource: /src/app/services/base.py
    title: Base Service Interfaces
  - resource: /src/app/services/embedding.py
    title: Concrete Embedding Service
  - resource: /src/app/services/rerank.py
    title: Concrete Rerank Service
  - resource: /src/app/services/mock.py
    title: Mock Services for Fast Testing
  - resource: /src/app/main.py
    title: FastAPI Routing and DI Binding
---

# サービス層設計と依存性注入（DI）・モック化アーキテクチャ

## 1. 概要

APIルーター（FastAPI エンドポイント）とモデル推論・外部プロキシ処理の密結合を解消するため、サービス層（`src/app/services/`）を導入し、**依存性注入（Dependency Injection / DI）** に基づく疎結合アーキテクチャを採用しています。

```mermaid
graph TD
    Router["FastAPI ルーター (src/app/main.py)"]
    
    subgraph "依存性注入 (FastAPI Depends)"
        get_emb["get_embedding_service()"]
        get_rr["get_rerank_service()"]
    end

    subgraph "抽象基底クラス (src/app/services/base.py)"
        BaseEmb["BaseEmbeddingService"]
        BaseRR["BaseRerankService"]
    end

    subgraph "本番具象サービス"
        EmbServ["EmbeddingService (src/app/services/embedding.py)"]
        RRServ["RerankService (src/app/services/rerank.py)"]
    end

    subgraph "テスト用高速モック"
        MockEmb["MockEmbeddingService (src/app/services/mock.py)"]
        MockRR["MockRerankService (src/app/services/mock.py)"]
    end

    Router -->|"Depends"| get_emb
    Router -->|"Depends"| get_rr

    get_emb -->|"返却"| BaseEmb
    get_rr -->|"返却"| BaseRR

    BaseEmb <|-- EmbServ
    BaseEmb <|-- MockEmb
    BaseRR <|-- RRServ
    BaseRR <|-- MockRR
```

---

## 2. サービスインターフェース定義

### 2.1. 抽象基底クラス (`src/app/services/base.py`)
- **`BaseEmbeddingService`**: `create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse` を定義。
- **`BaseRerankService`**: `create_rerank(request: RerankRequest) -> RerankResponse` を定義。

### 2.2. 本番実装 (`EmbeddingService`, `RerankService`)
- **`EmbeddingService`**:
  - テキスト・画像（Base64 / URL）のパース、SSRF検証。
  - TEI Proxy有効時の自動ルーティング。
  - Ruri-v3プレフィックス付与とトークン切り詰め（Truncation）。
  - `anyio.to_thread.run_sync` によるスレッドセーフなモデル推論実行。
- **`RerankService`**:
  - TEI Proxy / ローカル CrossEncoder の切り替え。
  - クエリ・ドキュメントペアのスコアリングとトークン数算出。
  - `top_n` による上位絞り込みとソート処理。

---

## 3. テスト高速化とモック設計 (`MockEmbeddingService`, `MockRerankService`)

### 3.1. 背景と課題
従来はテスト実行時にも Hugging Face からの実モデルダウンロードや重みロードが発生し、CI実行時間が数分以上を要していました。

### 3.2. モック差し替えによる解決
FastAPI の `app.dependency_overrides` を利用し、ユニットテスト時にモデルをロードしない軽量モックへ即座に差し替えることが可能です。

```python
from app.main import app, get_embedding_service
from app.services import MockEmbeddingService

def test_fast_embedding():
    app.dependency_overrides[get_embedding_service] = lambda: MockEmbeddingService(vector_dim=1024)
    try:
        # 重みロードなしで即座に検証可能 (ミリ秒単位)
        response = client.post("/v1/embeddings", json={"model": "bge-visualized-m3", "input": "test"})
        assert response.status_code == 200
    finally:
        app.dependency_overrides.clear()
```

### 3.3. テスト分離ポリシー
- **ユニットテスト (CI実行対象)**: `pytest -m "not integration"`
  - モックまたは軽量なスタブを用いて API スキーマ、バリデーション、エラーハンドリング、プロキシロジックを約 5〜10 秒で検証。
- **統合テスト (ローカル/リグレッション実行対象)**: `pytest -m "integration"`
  - 実モデルのロードおよび実推論精度・出力次元（`test_multimodal_real.py` など）を検証。
