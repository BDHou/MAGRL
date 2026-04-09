#!/bin/bash
# MAGRL MAPPO 训练启动脚本
# 用法: bash run_train.sh [iterations] [workers]
#   iterations: 训练迭代数 (默认 50)
#   workers:    并行 worker 数 (默认 2)

set -e

ITERATIONS=${1:-200}
WORKERS=${2:-2}

echo "=========================================="
echo "  MAGRL MAPPO Training"
echo "  Iterations: $ITERATIONS | Workers: $WORKERS"
echo "=========================================="

eval "$(conda shell.bash hook)"
conda activate yc_mamaskabledppo_env

cd /Users/steven/Documents/GridProjects/MAGRL
python -m src.train_ma --iterations "$ITERATIONS" --workers "$WORKERS"
