#!/bin/bash

# このスクリプトは、Dockerを使用してAPIサーバーの管理（起動、停止、モデルのダウンロード）を行います。

# デフォルト設定
MODE=${2:-cpu} # cpu or gpu
COMMAND=${1:-run} # run, stop, download
CACHE_DIR="$(pwd)/.cache/models"
mkdir -p "$CACHE_DIR"

# Dockerイメージ/サービス名
SERVICE_NAME="api-$MODE"

function usage() {
    echo "使用法: $0 [run|stop|download] [cpu|gpu]"
    echo ""
    echo "コマンド:"
    echo "  run      : サーバーを起動します（デフォルト）"
    echo "  stop     : サーバーを停止します"
    echo "  download : config/models.yml に記載されたモデルを事前にダウンロードします"
    echo ""
    echo "引数:"
    echo "  cpu     : CPUモードを使用します（デフォルト）"
    echo "  gpu     : GPUモードを使用します"
    exit 1
}

if [[ "$COMMAND" == "help" || "$COMMAND" == "-h" ]]; then
    usage
fi

case "$COMMAND" in
    run)
        echo "$MODE モードでサーバーを起動します..."
        # オフラインモードの設定を確認（環境変数がセットされていれば引き継ぐ）
        export OFFLINE_MODE=${OFFLINE_MODE:-false}
        docker compose up -d "$SERVICE_NAME"
        echo "サーバーが起動しました。ログを確認するには 'docker compose logs -f $SERVICE_NAME' を実行してください。"
        ;;
    stop)
        echo "サーバーを停止します..."
        docker compose stop
        ;;
    download)
        echo "モデルをダウンロードします..."
        # ダウンロードは常にCPU版のイメージを使用して実行
        docker compose run --rm -e HF_HUB_OFFLINE=0 "$SERVICE_NAME" python -m src.app.download_models
        echo "モデルのダウンロードが完了しました。"
        ;;
    *)
        usage
        ;;
esac
