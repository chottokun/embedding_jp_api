---
type: Architecture Decision
title: サービス層設計と依存性注入（DI）・モック化アーキテクチャ
description: BaseEmbeddingService/BaseRerankServiceの抽象化、FastAPI Dependsによる疎結合設計、および高速CIモック
status: stable
generated:
  by: agent/antigravity
  at: 2026-10-18T06:30:00Z
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
  - resource: /src/app/services/decisions.py
    title: Concrete Decisions Service
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
    Router["FastAPI ルーター (src/app/routers/)"]
    
    subgraph "依存性注入 (FastAPI Depends)"
        get_emb["get_embedding_service()"]
        get_rr["get_rerank_service()"]
        get_dec["get_decisions_service()"]
    end

    subgraph "抽象基底クラス (src/app/services/base.py)"
        BaseEmb["BaseEmbeddingService"]
        BaseRR["BaseRerankService"]
        BaseDec["BaseDecisionsService"]
    end

    subgraph "本番具象サービス"
        EmbServ["EmbeddingService (src/app/services/embedding.py)"]
        RRServ["RerankService (src/app/services/rerank.py)"]
        DecServ["DecisionsService (src/app/services/decisions.py)"]
    end

    subgraph "テスト用高速モック"
        MockEmb["MockEmbeddingService (src/app/services/mock.py)"]
        MockRR["MockRerankService (src/app/services/mock.py)"]
        MockDec["MockDecisionsService (src/app/services/mock.py)"]
    end

    Router -->|"Depends"| get_emb
    Router -->|"Depends"| get_rr
    Router -->|"Depends"| get_dec

    get_emb -->|"返却"| BaseEmb
    get_rr -->|"返却"| BaseRR
    get_dec -->|"返却"| BaseDec

    BaseEmb <|-- EmbServ
    BaseEmb <|-- MockEmb
    BaseRR <|-- RRServ
    BaseRR <|-- MockRR
    BaseDec <|-- DecServ
    BaseDec <|-- MockDec
```

---

## 2. サービスインターフェース定義

### 2.1. 抽象基底クラス (`src/app/services/base.py`)
- **`BaseEmbeddingService`**: `create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse` を定義。
- **`BaseRerankService`**: `create_rerank(request: RerankRequest) -> RerankResponse` を定義。
- **`BaseDecisionsService`**: `create_decisions(request: DecisionsRequest) -> DecisionsResponse` を定義。

### 2.2. 本番実装 (`EmbeddingService`, `RerankService`, `DecisionsService`)
- **`EmbeddingService`**:
  - テキスト・画像（Base64 / URL）のパース、SSRF検証。
  - TEI Proxy有効時の自動ルーティング。
  - Ruri-v3プレフィックス付与とトークン切り詰め（Truncation）。
  - `anyio.to_thread.run_sync` によるスレッドセーフなモデル推論実行。
- **`RerankService`**:
  - `create_rerank` をオーケストレーターとし、経路別プライベートヘルパー（`_process_tei_proxy`, `_process_logit_gate`, `_process_standard_rerank`）へ責務を分離。
  - TEI Proxy 応答時のドキュメントインデックス境界検証、クエリ・ドキュメントペアのスコアリングとトークン数算出、`top_n` による上位絞り込みとソート処理。
- **`DecisionsService`**:
  - OpenAI Decisions API 互換の意思決定評価サービス。
  - `LOGIT_GATE_MODELS`（`Qwen/Qwen2.5-1.5B-Instruct` 等）と `LogitGateService` + `AsciiMatcher` を用い、`anyio.to_thread.run_sync` でスレッドセーフに `predicate`（確率判定）、`choice`（Softmax 選択肢確率分布）、`score`（レベル別確率分布と期待値スコア）、および `refusal` を算出。

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
