#!/bin/bash

# 初始化 conda（根据你的 shell 类型选择）
eval "$(conda shell.bash hook)"

# 激活 conda 环境
conda activate yc_mamaskabledppo_env

# 运行 TensorBoard，自行修改logdir为实际的logdir
python -m tensorboard.main --logdir=saved_models --port=6009
