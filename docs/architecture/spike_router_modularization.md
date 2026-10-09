# Spike C: APIRouter化とLifespan状態共有パターンの技術検証レポート

## 1. 背景と目的
現在の `src/app/main.py` は、すべてのエンドポイント（`/v1/embeddings`, `/v1/rerank`, `/v1/models`, `/healthz`など）、リクエスト処理（ミドルウェア）、ライフサイクル管理、依存関係注入（DI）などが1ファイルに集中しており、保守性や可読性の面で課題が生じつつあります。
本Spikeでは、FastAPIの `APIRouter` を用いてエンドポイントを関心事ごとにモジュール化し、`Lifespan`（ライフサイクルイベント）と `app.state` を活用した状態管理・共有リソースの受け渡しパターンについて技術検証・設計検討を行います。

---

## 2. ルーター分割案の策定
`main.py` の責務を削減し、以下のディレクトリ構造でルーターを分割します。

```text
src/app/
 ├── main.py                # FastAPIの初期化、Lifespanの定義、ミドルウェア・ルーターの登録に特化
 ├── routers/
 │    ├── __init__.py
 │    ├── embeddings.py      # /v1/embeddings エンドポイント
 │    ├── rerank.py          # /v1/rerank エンドポイント
 │    ├── models.py          # /v1/models, /v1/models/{model_id}/unload エンドポイント
 │    └── health.py          # /health, /healthz, /ready, /metrics エンドポイント
 └── dependencies.py        # 各ルーターで共有する依存性注入（DI）関数（認証、Service、State等の取得）
```

これにより、各ファイルが担当するドメインが明確になり、機能追加やテストが容易になります。

---

## 3. 共有リソースの受け渡しパターン

現状、`inference_semaphore`（同時実行数制御）などのリソースは `main.py` のグローバル変数として定義され、モデルローダーやTEIクライアントも暗黙的またはService初期化時に渡されています。ルーター分割後のリソース管理パターンとして、以下の2つを比較・検討します。

### パターンA: サービス層の直接参照（グローバル/モジュールインポート）
- **概要**: `inference_semaphore` や 共有設定を別モジュール（例: `state.py` または `dependencies.py`）にグローバル変数として定義し、各ルーター・サービスから直接 `import` して使用する。
- **メリット**: 実装がシンプルで、既存コードからの移行コストが低い。
- **デメリット**: テスト時のモック化が難しくなる。FastAPIのDIの恩恵を受けにくい。状態がプロセス内で完全にシングルトンになり、柔軟性に欠ける。

### パターンB: `request.app.state` を活用したDI設計（推奨）
- **概要**: `Lifespan` 時点（アプリケーション起動時）に生成されるリソース（`httpx.AsyncClient`、`AsyncThreadSemaphore`など）を `app.state` に格納し、ルーター側では `Request` オブジェクトまたは `Depends` を通じて取得する。
- **メリット**:
  - グローバル変数への依存を排除でき、テスト時に `app.state` にモックオブジェクトを注入しやすくなる。
  - FastAPIの依存関係注入（DI）システムと親和性が高く、リクエストスコープで明示的に依存関係を表現できる。
- **デメリット**: DI関数（`Depends`）の引数定義が増えるため、シグネチャが少し長くなる。

**【結論】**
本プロジェクトのテスト容易性と今後の拡張性を考慮し、**パターンB（`request.app.state` を活用したDI設計）** を採用します。

### 対象となるリソースの配置先
1. **`inference_semaphore`**: `Lifespan`内で初期化し、`app.state.inference_semaphore` に格納。DI経由でルーターへ渡す。
2. **TEIクライアント (`httpx.AsyncClient`)**: 既存通り `Lifespan`内で初期化し、`app.state.tei_client` に格納。DI(`get_embedding_service`等)でServiceへ渡す。
3. **モデルローダー (`get_model`)**: DI（`dependencies.py`）内で定義し、必要なルーター/サービスへ注入する。

---

## 4. Lifespanコンテキストマネージャの設計

現在のLifespanにおけるシャットダウン時のドレイン処理（`active_requests` の追跡と待機）との整合性を保ちつつ、ルーター分割に対応する設計を行います。

### ライフサイクルの流れ
1. **起動時（`yield` 前）**:
   - `TEIクライアント`の生成・`app.state.tei_client` へのセット。
   - `inference_semaphore` の生成・`app.state.inference_semaphore` へのセット。
   - `active_requests` (Taskセット) の初期化と状態へのセット。
   - （オプション）モデルのプリロード。
2. **アプリケーション実行中 (`yield`)**:
   - カスタムミドルウェアが `request.app.state.active_requests` に現在処理中のタスクを追加・削除。
3. **終了時（`yield` 後）**:
   - ドレイン処理の開始：`app.state.active_requests` に残存するタスクを `SHUTDOWN_DRAIN_TIMEOUT_SECONDS` を上限に `asyncio.wait` で待機する。
   - `TEIクライアント` の非同期クローズ (`aclose`)。

これにより、グローバル変数（`is_shutting_down`, `active_requests`）を `app.state` に閉じ込めることが可能になります（ミドルウェアは `app` インスタンスからこれらにアクセスします）。

---

## 5. 具体的なPoCコード例

以下のコードは、本Spikeに基づくプロトタイプ（PoC）です。既存のファイルを編集するものではありません。

### 5-1. スリム化された `main.py`
ルーターの登録とLifespanの定義に特化します。

```python
import asyncio
import logging
from contextlib import asynccontextmanager
from anyio import AsyncThreadSemaphore
import httpx
from fastapi import FastAPI

from .config import MAX_CONCURRENT_INFERENCES, SHUTDOWN_DRAIN_TIMEOUT_SECONDS, PRELOAD_MODELS
from .models import get_model
from .routers import embeddings, rerank, models_router, health
# ※ カスタムミドルウェアやロガー定義は簡略化のため省略または別ファイル化を想定

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 状態の初期化
    app.state.tei_client = httpx.AsyncClient(timeout=30.0)
    app.state.inference_semaphore = AsyncThreadSemaphore(MAX_CONCURRENT_INFERENCES)
    app.state.active_requests = set()
    app.state.is_shutting_down = False

    # モデルのプリロード
    if PRELOAD_MODELS:
        for model_name in PRELOAD_MODELS:
            await asyncio.to_thread(get_model, model_name)

    try:
        yield
    finally:
        app.state.is_shutting_down = True
        active_requests = app.state.active_requests
        if active_requests:
            logging.info(
                f"Graceful shutdown: Waiting for {len(active_requests)} requests "
                f"(max {SHUTDOWN_DRAIN_TIMEOUT_SECONDS}s)."
            )
            try:
                await asyncio.wait(active_requests, timeout=SHUTDOWN_DRAIN_TIMEOUT_SECONDS)
            except Exception as e:
                logging.warning(f"Error during drain: {e}")
        
        await app.state.tei_client.aclose()

app = FastAPI(title="Japanese API (Modularized)", lifespan=lifespan)

# Routerの登録
app.include_router(health.router)
app.include_router(models_router.router)
app.include_router(embeddings.router)
app.include_router(rerank.router)
```

### 5-2. `src/app/dependencies.py` (新規追加想定)
ルーターで利用する共通のDI関数を定義します。

```python
from fastapi import Request, Depends
from anyio import AsyncThreadSemaphore
from .services import BaseEmbeddingService, EmbeddingService
from .models import get_model

def get_inference_semaphore(request: Request) -> AsyncThreadSemaphore:
    """app.state からセマフォを取得する"""
    return request.app.state.inference_semaphore

def get_embedding_service(request: Request) -> BaseEmbeddingService:
    """
    Service層へのDI
    (TEIへのプロキシ関数などもここで組み立ててServiceに注入する想定)
    """
    tei_client = request.app.state.tei_client
    
    # 実際の実装では proxy_to_tei_func や model_loader を注入する
    # 例: return EmbeddingService(proxy_to_tei_func=..., model_loader=get_model)
    return EmbeddingService(model_loader=get_model)

def verify_api_key(request: Request):
    """APIキーの検証ロジック (省略)"""
    pass
```

### 5-3. `src/app/routers/embeddings.py` (ルーター定義例)
エンドポイントの処理のみに集中させます。

```python
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Request
from anyio import AsyncThreadSemaphore

from ..schemas import EmbeddingRequest, EmbeddingResponse
from ..dependencies import get_embedding_service, verify_api_key, get_inference_semaphore
from ..config import INFERENCE_SEMAPHORE_TIMEOUT_SECONDS
from ..services import BaseEmbeddingService

router = APIRouter(tags=["Embeddings"])

@router.post(
    "/v1/embeddings",
    response_model=EmbeddingResponse,
    dependencies=[Depends(verify_api_key)],
)
async def create_embeddings(
    request: EmbeddingRequest,
    service: BaseEmbeddingService = Depends(get_embedding_service),
    semaphore: AsyncThreadSemaphore = Depends(get_inference_semaphore),
):
    try:
        async with asyncio.timeout(INFERENCE_SEMAPHORE_TIMEOUT_SECONDS):
            async with semaphore:
                response = await service.create_embeddings(request)
                return response
    except TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Inference queue timeout. Server is under high load.",
        )
```

## 6. まとめ
本Spikeによる検証の結果、エンドポイントを `APIRouter` によって分割し、`Lifespan` で初期化した共有リソース（TEIクライアント、同時実行制御セマフォなど）を `app.state` と FastAPI の依存関係注入（DI）を通じて連携させるアーキテクチャが実現可能であることが確認できました。
これにより、肥大化した `main.py` をスリム化し、単体テストの容易性とコードの見通しを大幅に改善できます。
