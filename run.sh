#!/bin/bash

# このスクリプトは、Dockerコンテナを起動してAPIサーバーを実行します。
#
# 機能:
# - ホストPCの~/.cache/modelsをコンテナ内の/root/.cacheにマウントします。
#   これにより、モデルファイルがキャッシュされ、2回目以降の起動が高速になります。
# - ポート8000をホストのポート80にマッピングします。
#
# 使用法:
# ./run.sh [cpu|gpu]
#
# 引数:
#   cpu (デフォルト): CPU版のDockerイメージ (embedding_jp_api-cpu) を使用します。
#   gpu:             GPU版のDockerイメージ (embedding_jp_api-gpu) を使用します。

# デフォルトはCPUモード
MODE=${1:-cpu}

# Dockerイメージ名
CPU_IMAGE="embedding_jp_api-cpu"
GPU_IMAGE="embedding_jp_api-gpu"

# モデルキャッシュ用のホスト側ディレクトリ
# 存在しない場合は作成する
CACHE_DIR="$HOME/.cache/models"
mkdir -p "$CACHE_DIR"

echo "モデルキャッシュディレクトリ: $CACHE_DIR"
echo "コンテナ内のポート8000をホストのポート8000にマッピングします。"

if [ "$MODE" = "gpu" ]; then
    echo "GPUモードでコンテナを起動します (イメージ: $GPU_IMAGE)..."
    docker run --gpus all -p 8000:8000 \
      -v "$CACHE_DIR:/root/.cache" \
      "$GPU_IMAGE"
else
    echo "CPUモードでコンテナを起動します (イメージ: $CPU_IMAGE)..."
    docker run -p 8000:8000 \
      -v "$CACHE_DIR:/root/.cache" \
      "$CPU_IMAGE"
fi

