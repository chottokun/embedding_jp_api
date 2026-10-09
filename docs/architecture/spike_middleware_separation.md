# [Spike A] FastAPIミドルウェア層分離の技術検証と設計検討

## 1. 分離対象ミドルウェアの整理

現在 `src/app/main.py` には多数のミドルウェアが `@app.middleware("http")` として定義されており、コードの肥大化と責務の混在が課題となっています。これを `src/app/middleware/` ディレクトリに分離するための設計を整理します。

対象となる主要なミドルウェアは以下の通りです。

1. **RequestLoggingMiddleware**
   - **責務**: 構造化ログ（JSON）の出力、`request_id` の生成と `ContextVar` への保存、リクエスト・レスポンス情報の記録、および必要に応じたPII（個人情報）マスク処理。
   - **分離の方針**: ログ出力とリクエストコンテキストの管理に特化させます。

2. **SecurityHeadersMiddleware**
   - **責務**: セキュリティ関連のHTTPヘッダー（`X-Content-Type-Options`, `X-Frame-Options`, `Strict-Transport-Security` など）をレスポンスに付与。
   - **分離の方針**: 現在の `request_logging_and_security_headers` からセキュリティヘッダー付与のロジックを抽出し、独立したミドルウェアとして定義します。

3. **RateLimitMiddleware**
   - **責務**: スライディングウィンドウやトークンバケットアルゴリズムを用いた、クライアントIPやAPIキーに基づくレート制限。
   - **分離の方針**: インメモリやRedisなどのバックエンドに依存しないインターフェースを持たせ、分離します。

4. **MetricsMiddleware**
   - **責務**: Prometheusなどのメトリクス収集（リクエスト数、レイテンシなど）。
   - **分離の方針**: `prometheus_metrics_middleware` をクラスベース、あるいは分離された関数として抽出し、エンドポイントのパスやステータスコードをラベルとして付与します。

5. **PayloadSizeLimitMiddleware** / **GracefulShutdownDrainMiddleware**
   - **責務**: リクエストボディサイズの制限や、シャットダウン時のリクエストドレイン処理。これらも個別のファイルへ分離します。

---

## 2. ストリーム消費問題（Request Body Caching）の安全性検証

FastAPI (Starlette) のミドルウェアで `request.body()` を読み取ると、リクエストボディのストリームが消費され、後続のハンドラやミドルウェアでボディを読み取ろうとした際にフリーズするかエラーになる問題（Stream Consumption Problem）があります。ログ記録やPIIマスク、ペイロードサイズ制限でリクエストボディを参照する場合、この問題を回避する必要があります。

### 回避パターン: カスタム `receive` を用いたボディのキャッシュと復元

リクエストボディを一度読み取った後、再度読み取れるように非同期のイテレータ（ジェネレータ）を再構築して `request._receive` を上書きするアプローチが標準的かつ安全です。

```python
async def get_body(request: Request) -> bytes:
    # 一度ボディを読み取る
    body = await request.body()
    
    # 読み取ったボディを返すダミーの receive 関数を作成
    async def receive() -> dict:
        return {"type": "http.request", "body": body, "more_body": False}
    
    # リクエストオブジェクトの receive を上書きして後続処理で読めるようにする
    request._receive = receive
    return body
```

このパターンを用いれば、`RequestLoggingMiddleware` でペイロードをログ出力・PIIマスク検証しつつ、後続のルーティング処理に影響を与えません。ただし、メモリにボディ全体がロードされるため、極端に大きなファイルのアップロード等には注意が必要です（`PayloadSizeLimitMiddleware` を先に実行して防ぐべきです）。

---

## 3. 推奨ディレクトリ構成 & インターフェース設計

### ディレクトリ構成

ミドルウェアの責務ごとにモジュールを分割し、`__init__.py` でまとめてエクスポートする構成を推奨します。

```text
src/app/
├── middleware/
│   ├── __init__.py           # setup_middlewares などの初期化関数や一括インポート
│   ├── logging.py            # RequestLoggingMiddleware
│   ├── security.py           # SecurityHeadersMiddleware
│   ├── rate_limit.py         # RateLimitMiddleware
│   ├── metrics.py            # MetricsMiddleware
│   ├── payload_limit.py      # PayloadSizeLimitMiddleware
│   └── graceful_shutdown.py  # GracefulShutdownDrainMiddleware
├── main.py                   # app.add_middleware() で登録
```

### 登録順序（逆順実行の考慮）

Starlette の `add_middleware`（または `app.add_middleware`）は、**後に追加されたものが先に実行（外側にラップ）** されます。つまり、最初に実行したいミドルウェアほど最後に `add_middleware` する必要があります。

1. **Graceful Shutdown** (一番最初にリクエストを弾くべき)
2. **Request Logging** (最初のリクエスト検知と、最後のエラーログ・レイテンシ計測)
3. **Metrics** (レイテンシやHTTPステータスの記録)
4. **Security Headers** (レスポンスヘッダー付与)
5. **Rate Limit** (過剰アクセスを早期に遮断)
6. **Payload Size Limit** (巨大なボディを早期に遮断)

したがって、`main.py` での `app.add_middleware` 呼び出しは、実行したい順序と**逆**に記述するか、`Middleware` クラスのリストとして FastAPI アプリ初期化時に渡すのが明確で推奨されます。

```python
# FastAPI初期化時に登録する場合の例（リストの上から順に外側となるため、最初に実行される）
app = FastAPI(
    middleware=[
        Middleware(GracefulShutdownDrainMiddleware),
        Middleware(RequestLoggingMiddleware),
        Middleware(MetricsMiddleware),
        Middleware(SecurityHeadersMiddleware),
        Middleware(RateLimitMiddleware),
        Middleware(PayloadSizeLimitMiddleware),
    ]
)
```

---

## 4. 具体的なPoCコード例

Python 3.12+ の型ヒントを使用し、クラスベース（`BaseHTTPMiddleware`）と純粋なASGIミドルウェアを組み合わせたPoCコードです。

### `src/app/middleware/logging.py` (Request Body 消費回避を含む)

```python
import time
import logging
from uuid import uuid4
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Message
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        start_time = time.perf_counter()
        
        # Stream消費問題の回避: Bodyのキャッシュと再構築
        body = await request.body()
        async def receive() -> Message:
            return {"type": "http.request", "body": body, "more_body": False}
        request._receive = receive

        # PIIマスクなどのロジックをここに挟むことができる
        # masked_body = mask_pii(body)
        
        logger.info(f"Request started: {request.method} {request.url.path}", extra={"request_id": request_id})

        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"Request failed: {str(e)}", extra={"request_id": request_id})
            raise
        
        process_time = time.perf_counter() - start_time
        response.headers["X-Request-ID"] = request_id
        
        logger.info(
            f"Request completed: {response.status_code}", 
            extra={
                "request_id": request_id, 
                "process_time_ms": round(process_time * 1000, 2)
            }
        )
        
        return response
```

### `src/app/middleware/security.py`

```python
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Awaitable, Callable

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response
```

### `src/app/middleware/setup.py` (登録ユーティリティ例)

```python
from fastapi import FastAPI
from starlette.middleware import Middleware

# 実装した各ミドルウェアをインポート
from .logging import RequestLoggingMiddleware
from .security import SecurityHeadersMiddleware
# from .rate_limit import RateLimitMiddleware
# from .metrics import MetricsMiddleware
# from .payload_limit import PayloadSizeLimitMiddleware
# from .graceful_shutdown import GracefulShutdownDrainMiddleware

def get_middleware_stack() -> list[Middleware]:
    """
    FastAPI(middleware=get_middleware_stack()) のように使用する。
    リストの上部にあるものほど、外側のレイヤーとして先に実行される。
    """
    return [
        # Middleware(GracefulShutdownDrainMiddleware),
        Middleware(RequestLoggingMiddleware),
        # Middleware(MetricsMiddleware),
        Middleware(SecurityHeadersMiddleware),
        # Middleware(RateLimitMiddleware),
        # Middleware(PayloadSizeLimitMiddleware),
    ]

# 使用例:
# app = FastAPI(middleware=get_middleware_stack())
```
