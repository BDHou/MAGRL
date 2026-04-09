#!/bin/bash
# MAGRL MAPPO 训练启动脚本
# 用法: bash run_train.sh [iterations] [workers] [lr] [checkpoint_freq] [version_tag] [train_batch_size] [rollout_fragment_length]
#   iterations:      训练迭代数 (默认 200)
#   workers:         并行 worker 数 (默认 2)
#   lr:              学习率 (默认 5e-5)
#   checkpoint_freq: 保存频率 (默认 10)
#   version_tag:     实验版本标记 (默认: 当前分支-短commit)
#   train_batch_size:每轮采样 batch 大小 (默认 4000)
#   rollout_fragment_length: 每个 worker 单次采样片段长度 (默认 32)

set -e

ITERATIONS=${1:-200}
WORKERS=${2:-2}
LR=${3:-5e-5}
CHECKPOINT_FREQ=${4:-10}
VERSION_TAG_INPUT=${5:-}
TRAIN_BATCH_SIZE=${6:-4000}
ROLLOUT_FRAGMENT_LENGTH=${7:-32}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

if [ -n "$VERSION_TAG_INPUT" ]; then
    VERSION_TAG="$VERSION_TAG_INPUT"
else
    BRANCH_NAME="$(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown-branch")"
    COMMIT_SHORT="$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")"
    VERSION_TAG="${BRANCH_NAME}-${COMMIT_SHORT}"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RL_TAG="mappo-shared-policy-rllib2"
PARAM_TAG="i${ITERATIONS}_w${WORKERS}_lr${LR}_ckpt${CHECKPOINT_FREQ}_tb${TRAIN_BATCH_SIZE}_rf${ROLLOUT_FRAGMENT_LENGTH}"
SAVE_DIR="$PROJECT_ROOT/saved_models/${RL_TAG}/${VERSION_TAG}/${PARAM_TAG}_${TIMESTAMP}"

echo "=========================================="
echo "  MAGRL MAPPO Training"
echo "  Iterations: $ITERATIONS | Workers: $WORKERS"
echo "  LR: $LR | Checkpoint Freq: $CHECKPOINT_FREQ"
echo "  Train Batch Size: $TRAIN_BATCH_SIZE"
echo "  Rollout Fragment Length: $ROLLOUT_FRAGMENT_LENGTH"
echo "  Version: $VERSION_TAG"
echo "  Save Dir: $SAVE_DIR"
echo "=========================================="

eval "$(conda shell.bash hook)"
conda activate yc_mamaskabledppo_env

cd "$PROJECT_ROOT"
python -u -m src.train_ma \
    --iterations "$ITERATIONS" \
    --workers "$WORKERS" \
    --lr "$LR" \
    --train-batch-size "$TRAIN_BATCH_SIZE" \
    --rollout-fragment-length "$ROLLOUT_FRAGMENT_LENGTH" \
    --checkpoint-freq "$CHECKPOINT_FREQ" \
    --save-dir "$SAVE_DIR"
