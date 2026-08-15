import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
import yaml

# プロジェクトルートを取得
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_FILE = CONFIG_DIR / "models.yml"


def load_model_ids() -> list[str]:
    """config/models.yml から全モデルIDを取得します。"""
    if not MODELS_FILE.exists():
        print(f"Error: {MODELS_FILE} が見つかりません。")
        sys.exit(1)

    with open(MODELS_FILE, "r") as f:
        data = yaml.safe_load(f)
        if not data:
            print(f"Error: {MODELS_FILE} が空です。")
            sys.exit(1)

    model_ids = []
    if "embedding_models" in data:
        model_ids.extend(data["embedding_models"])
    if "rerank_models" in data:
        model_ids.extend(data["rerank_models"])
    return model_ids


def download_models(verify_offline: bool = False) -> None:
    """
    config/models.yml に記述された全モデルおよび追加重みをダウンロードします。
    """
    model_ids = load_model_ids()
    if not model_ids:
        print("ダウンロードするモデルが設定されていません。")
        return

    print(f"📦 {len(model_ids)} 個のモデルをダウンロードします...")

    for model_id in model_ids:
        print(f"\n[ダウンロード中] {model_id}")
        try:
            if model_id == "bge-visualized-m3":
                print("  • BAAI/bge-m3 (ベースモデル) をダウンロード中...")
                snapshot_download(repo_id="BAAI/bge-m3")
                print("  • BAAI/bge-visualized (Visualized_m3.pth) をダウンロード中...")
                hf_hub_download(
                    repo_id="BAAI/bge-visualized", filename="Visualized_m3.pth"
                )
            else:
                snapshot_download(repo_id=model_id)
            print(f"  ✓ 完了: {model_id}")
        except Exception as e:
            print(f"  ❌ エラー: {model_id} のダウンロードに失敗しました: {e}")
            sys.exit(1)

    print("\n✨ すべてのモデルダウンロードが完了しました。")

    if verify_offline:
        verify_offline_loading(model_ids)


def verify_offline_loading(model_ids: list[str]) -> None:
    """
    HF_HUB_OFFLINE=1 を強制した状態で全モデルがロード可能かを検証（Dry-Run）します。
    """
    print("\n" + "=" * 60)
    print("🔍 オフラインロード自己検証（HF_HUB_OFFLINE=1 Dry-Run）開始")
    print("=" * 60)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    # app.models のインポート
    try:
        from app.models import get_model
    except ImportError:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from app.models import get_model

    for model_id in model_ids:
        print(f"• 検証中: {model_id} ...", end=" ", flush=True)
        try:
            model = get_model(model_id, device="cpu")
            assert model is not None
            print("OK ✓")
        except Exception as e:
            print(f"FAILED ❌\nエラー: {e}")
            sys.exit(1)

    print("\n🎉 全モデルのオフライン完全ロード検証に成功しました！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download models and verify offline loading."
    )
    parser.add_argument(
        "--verify-offline",
        action="store_true",
        help="ダウンロード完了後に HF_HUB_OFFLINE=1 で全モデルのロードを検証します",
    )
    args = parser.parse_args()

    download_models(verify_offline=args.verify_offline)
