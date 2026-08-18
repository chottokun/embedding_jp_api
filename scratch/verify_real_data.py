import os
import sys
import time
import base64
import io
import math
from pathlib import Path
from PIL import Image, ImageDraw

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from app.models import get_model, VisualizedBGEEmbeddingModel
from app.config import EMBEDDING_MODELS, RERANK_MODELS


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def create_sample_image(text: str = "システム構成図", color: tuple = (70, 130, 180)) -> Image.Image:
    """実データのテスト用画像を動的に生成"""
    img = Image.new("RGB", (256, 256), color=(240, 244, 248))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 236, 100], fill=color, outline=(30, 60, 90), width=2)
    draw.rectangle([40, 140, 216, 220], fill=(220, 230, 242), outline=(30, 60, 90), width=2)
    draw.line([(128, 100), (128, 140)], fill=(30, 60, 90), width=3)
    return img


def run_text_embedding_verification():
    print("\n========================================================")
    print("1. テキスト埋め込み実データ検証 (RURI, BGE-M3)")
    print("========================================================")
    
    test_queries = [
        "検索クエリ: 日本の首都はどこですか？",
        "検索ドキュメント: 日本の首都は東京都であり、政治・経済・文化の中心地です。",
        "検索ドキュメント: リンゴとミカンの果物栽培における土壌改良技術について。",
    ]

    for model_id in ["cl-nagoya/ruri-v3-30m", "cl-nagoya/ruri-v3-310m", "BAAI/bge-m3"]:
        print(f"\n--- モデル検証: {model_id} ---")
        t0 = time.perf_counter()
        model = get_model(model_id, device="cpu")
        load_time = time.perf_counter() - t0
        print(f"  ✓ ロード完了 ({load_time:.2f}秒)")

        t0 = time.perf_counter()
        embeddings = model.encode(test_queries, normalize_embeddings=True)
        infer_time = time.perf_counter() - t0
        
        dim = len(embeddings[0])
        sim_relevant = cosine_similarity(embeddings[0].tolist(), embeddings[1].tolist())
        sim_irrelevant = cosine_similarity(embeddings[0].tolist(), embeddings[2].tolist())
        
        print(f"  ✓ 埋め込み次元数: {dim}")
        print(f"  ✓ 推論時間 (3文): {infer_time*1000:.1f}ms")
        print(f"  ✓ 関連ドキュメントとの類似度: {sim_relevant:.4f}")
        print(f"  ✓ 無関係ドキュメントとの類似度: {sim_irrelevant:.4f}")
        
        assert dim > 0, "次元数が不正です"
        assert sim_relevant > sim_irrelevant, f"関連文書の類似度({sim_relevant})が無関係文書({sim_irrelevant})を下回っています"
        print(f"  🎯 判定: 合格 (関連度判定正常: {sim_relevant:.4f} > {sim_irrelevant:.4f})")


def run_multimodal_verification():
    print("\n========================================================")
    print("2. マルチモーダル実データ検証 (bge-visualized-m3)")
    print("========================================================")
    
    t0 = time.perf_counter()
    model = get_model("bge-visualized-m3", device="cpu")
    print(f"  ✓ ロード完了 ({time.perf_counter() - t0:.2f}秒)")

    # 実画像生成
    diagram_img = create_sample_image(text="システム構成図", color=(30, 100, 200))
    nature_img = create_sample_image(text="自然風景", color=(34, 139, 34))

    # テストケース:
    # 1. 画像のみ
    # 2. テキストのみ
    # 3. 画像 + テキスト複合
    items = [
        ("システムアーキテクチャ設計図", diagram_img),
        ("自然豊かな森林と山脈の風景", nature_img),
        ("クラウドインフラ構成図", None),
        ("青空と緑の草原", None),
    ]

    t0 = time.perf_counter()
    embeddings = model.encode_multimodal(items)
    infer_time = time.perf_counter() - t0
    
    dim = len(embeddings[0])
    print(f"  ✓ マルチモーダル埋め込み次元数: {dim}")
    print(f"  ✓ 推論時間 ({len(items)}アイテム): {infer_time*1000:.1f}ms")
    
    # 類似度評価
    # システム構成図(画像+テキスト) と クラウドインフラ構成図(テキスト)
    sim_diagram = cosine_similarity(embeddings[0], embeddings[2])
    # システム構成図(画像+テキスト) と 青空と緑の草原(テキスト)
    sim_mismatch = cosine_similarity(embeddings[0], embeddings[3])

    print(f"  ✓ アーキテクチャ図(画像+文) vs クラウドインフラ(文) 類似度: {sim_diagram:.4f}")
    print(f"  ✓ アーキテクチャ図(画像+文) vs 草原風景(文) 類似度: {sim_mismatch:.4f}")
    
    assert dim == 1024, f"bge-visualized-m3 の次元数は 1024 である必要があります (実際: {dim})"
    assert sim_diagram > sim_mismatch, f"画像-テキスト間のセマンティック類似度が期待を満たしていません ({sim_diagram} vs {sim_mismatch})"
    print(f"  🎯 判定: 合格 (マルチモーダル類似度正常: {sim_diagram:.4f} > {sim_mismatch:.4f})")


def run_reranker_verification():
    print("\n========================================================")
    print("3. リランカー実データ検証 (ruri-v3-reranker-310m)")
    print("========================================================")
    
    t0 = time.perf_counter()
    model = get_model("cl-nagoya/ruri-v3-reranker-310m", device="cpu")
    print(f"  ✓ ロード完了 ({time.perf_counter() - t0:.2f}秒)")

    query = "機械学習における過学習の防ぎ方"
    passages = [
        "過学習を防ぐ手法として、正則化（L1/L2）、ドロップアウト、データ拡張、アーリーストッピングなどがあります。",
        "日本の温泉地ランキングでは、草津温泉や別府温泉、有馬温泉などが上位に選ばれています。",
        "ニューラルネットワークの汎化性能向上のため、学習データのバリデーション分割やクロスバリデーションが推奨されます。",
    ]
    
    pairs = [[query, p] for p in passages]
    t0 = time.perf_counter()
    scores = model.predict(pairs)
    infer_time = time.perf_counter() - t0
    
    print(f"  ✓ 推論時間 ({len(pairs)}ペア): {infer_time*1000:.1f}ms")
    for i, (p, score) in enumerate(zip(passages, scores)):
        print(f"    [{i+1}] スコア: {score:+.4f} | 内容: {p[:35]}...")

    assert scores[0] > scores[1], "過学習対策ドキュメントのスコアが温泉ドキュメントを下回っています"
    assert scores[2] > scores[1], "汎化性能ドキュメントのスコアが温泉ドキュメントを下回っています"
    print("  🎯 判定: 合格 (リランキング順位スコア正常)")


def run_device_switching_verification():
    print("\n========================================================")
    print("4. デバイス切り替え・フォールバック検証 (CPU / CUDA)")
    print("========================================================")
    
    cuda_available = torch.cuda.is_available()
    print(f"  現在のCUDA利用可能性: {cuda_available}")

    # 明示的な CPU デバイス指定
    print("\n  [Case A] 明示的な device='cpu' の動作検証")
    model_cpu = get_model("cl-nagoya/ruri-v3-30m", device="cpu")
    emb_cpu = model_cpu.encode(["テスト文字列"], normalize_embeddings=True)
    assert len(emb_cpu[0]) > 0
    print("  ✓ CPUロード＆推論: 正常完了")

    # CUDA指定（CUDAが無い環境ではCPUへ安全フォールバックすることを確認）
    print("\n  [Case B] device='cuda' 指定時のフォールバック検証")
    model_cuda_req = get_model("cl-nagoya/ruri-v3-30m", device="cuda")
    emb_cuda_req = model_cuda_req.encode(["テスト文字列"], normalize_embeddings=True)
    assert len(emb_cuda_req[0]) > 0
    print("  ✓ CUDA要求時の安全実行/フォールバック: 正常完了")

    # VisualizedBGEEmbeddingModel のデバイス指定
    print("\n  [Case C] VisualizedBGEEmbeddingModel のデバイス属性整合性")
    v_model = get_model("bge-visualized-m3", device="cpu")
    print(f"    v_model.device: {v_model.device}")
    print(f"    v_model.model.device: {v_model.model.device}")
    assert str(v_model.device) == str(v_model.model.device) == "cpu"
    print("  ✓ マルチモーダルモデルのデバイス整合性: 完全一致")
    print("  🎯 判定: 合格 (デバイス切り替え・フォールバック正常)")


def run_fastapi_endpoints_real_verification():
    print("\n========================================================")
    print("5. FastAPI エンドポイント実データ結合検証 (/v1/embeddings, /v1/rerank)")
    print("========================================================")
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    # 1. /v1/embeddings (テキスト)
    print("\n  [Endpoint 1] POST /v1/embeddings (テキスト)")
    res = client.post(
        "/v1/embeddings",
        json={
            "model": "cl-nagoya/ruri-v3-30m",
            "input": ["自然言語処理技術の進化", "検索エンジンの仕組み"],
        },
    )
    assert res.status_code == 200, f"Error: {res.text}"
    data = res.json()
    assert len(data["data"]) == 2
    assert len(data["data"][0]["embedding"]) > 0
    print(f"  ✓ ステータス 200, 次元数: {len(data['data'][0]['embedding'])}, Usage: {data['usage']}")

    # 2. /v1/embeddings (マルチモーダル Base64)
    print("\n  [Endpoint 2] POST /v1/embeddings (マルチモーダル Base64画像)")
    img = create_sample_image("APIテスト画像")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    res = client.post(
        "/v1/embeddings",
        json={
            "model": "bge-visualized-m3",
            "input": {
                "text": "クラウドシステム設計",
                "image_url": b64_str,
            },
        },
    )
    assert res.status_code == 200, f"Error: {res.text}"
    data = res.json()
    assert len(data["data"]) == 1
    assert len(data["data"][0]["embedding"]) == 1024
    print(f"  ✓ ステータス 200, マルチモーダル埋め込み次元数: {len(data['data'][0]['embedding'])}")

    # 3. /v1/rerank (リランキング)
    print("\n  [Endpoint 3] POST /v1/rerank (テキストリランキング)")
    res = client.post(
        "/v1/rerank",
        json={
            "model": "cl-nagoya/ruri-v3-reranker-310m",
            "query": "日本の世界遺産",
            "documents": [
                "姫路城や屋久島、古都京都の文化財などが日本の代表的な世界遺産です。",
                "最新のスマートフォンに搭載されたAI機能の解説記事です。",
                "富士山は2013年に世界文化遺産に登録されました。",
            ],
            "top_n": 2,
            "return_documents": True,
        },
    )
    assert res.status_code == 200, f"Error: {res.text}"
    data = res.json()
    results = data["data"]
    assert len(results) == 2
    print(f"  ✓ ステータス 200, Top-{len(results)} 返却:")
    for r in results:
        print(f"    - Doc {r['document']}: score={r['score']:+.4f} | text={r.get('text', '')[:35]}...")
    assert results[0]["document"] in (0, 2)
    print("  🎯 判定: 合格 (全APIエンドポイント実データ推論正常)")


if __name__ == "__main__":
    t_start = time.perf_counter()
    print("🚀 実データ・デバイス動作検証テストを開始します")
    
    run_text_embedding_verification()
    run_multimodal_verification()
    run_reranker_verification()
    run_device_switching_verification()
    run_fastapi_endpoints_real_verification()
    
    total_sec = time.perf_counter() - t_start
    print(f"\n========================================================")
    print(f"🎉 全ての実データ・デバイス検証テストに合格しました！ (総所要時間: {total_sec:.2f}秒)")
    print(f"========================================================")
