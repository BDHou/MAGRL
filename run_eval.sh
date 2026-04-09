#!/bin/bash
# MAGRL MAPPO 评估脚本
# 用法: bash run_eval.sh <checkpoint_path> [--save output.png] [--no-baseline]
#
# 示例:
#   bash run_eval.sh saved_models/run_20260409_113201/best
#   bash run_eval.sh saved_models/run_20260409_113201/best --save eval_result.png
#   bash run_eval.sh saved_models/run_20260409_113201/best --no-baseline

set -e

if [ -z "$1" ]; then
    echo "用法: bash run_eval.sh <checkpoint_path> [--save output.png] [--no-baseline]"
    echo ""
    echo "可用的 checkpoint:"
    find saved_models checkpoints -name "rllib_checkpoint.json" -maxdepth 6 2>/dev/null | sed 's|/rllib_checkpoint.json||' | while read d; do echo "  $d"; done
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate yc_mamaskabledppo_env

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

cd "$PROJECT_ROOT"
python -u -m src.eval_ma "$@"
