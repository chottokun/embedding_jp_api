# [Spike B] 認証・認可ロジックのFastAPI Dependsモジュール化検証レポート

## 1. 概要
本レポートは、現在 `src/app/main.py` にベタ書きされている認証・認可関連のロジック（`HTTPBearer`, `APIKeyHeader`, `verify_api_key` 等）を `src/app/dependencies/auth.py` に分離し、モジュール化するための事前検証（Spike）結果をまとめたものです。
この分離により、`main.py` はルーティングとアプリケーションの初期化に特化し、関心事の分離（Separation of Concerns）を実現します。

## 2. 分離対象の抽出と要件
抽出対象となる主な機能とロジックは以下の通りです。

1. **`HTTPBearer` セキュリティ定義**: `security = HTTPBearer(auto_error=False)` 
2. **`verify_api_key` デペンデンシ**:
   - 単一キー（`API_KEY`）および複数キー（`API_KEYS_MAP`）の検証をサポート。
3. **`verify_admin_key` デペンデンシ（新規追加）**:
   - `/v1/models/unload` のような特権オペレーションに対する管理者用キーの検証。
   - `ADMIN_KEY` が未設定の場合は、セキュリティの観点から標準の `verify_api_key` にフォールバックするか、アクセスを拒否する仕様を検討。

## 3. セキュリティ & パフォーマンス考慮点

### セキュリティ
- **タイミング攻撃対策**: パスワードやAPIキーの比較時に発生しうるタイミング攻撃を防ぐため、標準ライブラリの `secrets.compare_digest` を引き続き使用します。
- **管理者権限の分離**: `ADMIN_KEY` を導入することで、モデルのアンロード等システムリソースに影響を与えるエンドポイントの保護を強化します。

### パフォーマンス最適化
- **キャッシュの利用**: `API_KEYS_MAP` と `API_KEY` を結合した許可リスト（`configured_keys`）は、リクエストのたびに再計算するのではなく、モジュールのロード時（またはアプリケーション起動時）にキャッシュ（リスト化）しておくことで、計算量を削減します。
- **O(N) 検索の影響**: キーの数が膨大でない限り（通常数十〜数百程度）、`for`ループと `secrets.compare_digest` による線形検索 O(N) でもパフォーマンスに致命的な影響は与えません。ハッシュマップ（O(1)）での検索はタイミング攻撃のリスクがあるため避けます。

### 後方互換性とテスト
- 既存の `src/tests/test_auth.py` で定義されているすべてのテストケース（タイミング攻撃の検証、未設定時のパススルー、複数キーの検証など）がパスするよう、振る舞いを完全に維持します。

## 4. FastAPI Dependency Injection パターン

FastAPI のクリーンな設計を維持するため、分離先モジュールから関数をインポートし、以下のようにエンドポイントに組み込みます。

```python
# src/app/main.py
from fastapi import Depends, FastAPI
from app.dependencies.auth import verify_api_key, verify_admin_key

app = FastAPI()

@app.post("/v1/embeddings", dependencies=[Depends(verify_api_key)])
async def create_embeddings(...):
    pass

@app.post("/v1/models/unload", dependencies=[Depends(verify_admin_key)])
async def unload_models(...):
    pass
```

## 5. PoC (Proof of Concept) コード例

以下は `src/app/dependencies/auth.py` の完全な設計プロトタイプです。

```python
import secrets
from typing import Optional, List
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# 実際の実装では config モジュールから読み込む
from app.config import API_KEY, API_KEYS_MAP, ADMIN_KEY

security = HTTPBearer(auto_error=False)

def _get_configured_keys() -> List[str]:
    """設定されたAPIキーをリスト化し、キャッシュする（リクエストごとの再計算を防ぐ）"""
    keys = list(API_KEYS_MAP.keys()) if API_KEYS_MAP else []
    if API_KEY and API_KEY not in keys:
        keys.append(API_KEY)
    return keys

# アプリケーション起動時に評価・キャッシュされる
_CONFIGURED_KEYS = _get_configured_keys()

async def verify_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> Optional[HTTPAuthorizationCredentials]:
    """
    標準のAPIキーを検証するDependency。
    キーが設定されていない場合は認証なしで許可。
    """
    if _CONFIGURED_KEYS:
        if auth is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API Key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # タイミング攻撃を防ぐため定数時間で比較
        matched = False
        for valid_key in _CONFIGURED_KEYS:
            if secrets.compare_digest(auth.credentials, valid_key):
                matched = True
                break
                
        if not matched:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API Key",
                headers={"WWW-Authenticate": "Bearer"},
            )
    return auth

async def verify_admin_key(
    auth: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> Optional[HTTPAuthorizationCredentials]:
    """
    特権操作（モデルアンロード等）のための管理者用APIキーを検証するDependency。
    """
    if ADMIN_KEY:
        if auth is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing Admin API Key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not secrets.compare_digest(auth.credentials, ADMIN_KEY):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing Admin API Key",
                headers={"WWW-Authenticate": "Bearer"},
            )
    else:
        # ADMIN_KEYが設定されていない場合は、標準のAPI_KEYロジックにフォールバックさせる
        # （これにより後方互換性を維持しつつ、デフォルトの動作を保証）
        return await verify_api_key(auth)
        
    return auth
```

## 6. 今後のステップ
1. 上記設計案を基に `src/app/dependencies/auth.py` を実装。
2. `src/app/main.py` から認証ロジックを削除し、新しい dependency をインポートするようリファクタリング。
3. `src/app/config.py` に `ADMIN_KEY` の読み込み処理を追加。
4. `src/tests/test_auth.py` を実行・拡張し、`verify_admin_key` のテストを追加。
