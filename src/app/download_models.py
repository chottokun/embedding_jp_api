import sys
import yaml
from pathlib import Path
from huggingface_hub import snapshot_download

# プロジェクトルートを取得
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_FILE = CONFIG_DIR / "models.yml"


def download_models():
    """
    config/models.yml に記述された全モデルをダウンロードします。
    """
    if not MODELS_FILE.exists():
        print(f"Error: {MODELS_FILE} が見つかりません。")
        sys.exit(1)

    with open(MODELS_FILE, "r") as f:
        data = yaml.safe_load(f)
        if not data:
            print(f"Error: {MODELS_FILE} が空です。")
            sys.exit(1)

    # 全てのモデルIDを抽出
    model_ids = []
    if "embedding_models" in data:
        model_ids.extend(data["embedding_models"])
    if "rerank_models" in data:
        model_ids.extend(data["rerank_models"])

    if not model_ids:
        print("ダウンロードするモデルが設定されていません。")
        return

    print(f"{len(model_ids)} 個のモデルをダウンロードします...")

    for model_id in model_ids:
        print(f"\nモデルをダウンロード中: {model_id}")
        try:
            if model_id == "bge-visualized-m3":
                # Download BAAI/bge-m3 base model and Visualized_m3.pth weights
                snapshot_download(repo_id="BAAI/bge-m3")
                try:
                    from huggingface_hub import hf_hub_download

                    hf_hub_download(
                        repo_id="BAAI/bge-visualized-m3", filename="Visualized_m3.pth"
                    )
                except Exception:
                    snapshot_download(repo_id="BAAI/bge-visualized-m3")
            else:
                snapshot_download(repo_id=model_id)
            print(f"完了: {model_id}")
        except Exception as e:
            print(f"エラー: {model_id} のダウンロードに失敗しました: {e}")


if __name__ == "__main__":
    download_models()
