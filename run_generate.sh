#!/bin/bash
# MAGRL 环境数据生成脚本
# 向 IEEE 标准配电网中注入多种可调资源，生成 RL 环境所需的拓扑与时序数据。
#
# 用法: bash run_generate.sh [feeder] [days] [interval_min] [n_bess] [n_pv] [n_gen] [n_dr] [seed]
#   feeder:       IEEE 标准网络名 (默认 case33bw)
#   days:         模拟天数 (默认 30)
#   interval_min: 时间间隔/分钟 (默认 15, 即每天 96 步)
#   n_bess:       BESS 储能数量 (默认 3)
#   n_pv:         Inverter/PV 数量 (默认 3)
#   n_gen:        Generator 数量 (默认 0)
#   n_dr:         Demand Response 数量 (默认 0)
#   seed:         随机种子 (默认 42)
#
# 输出目录: data/generated/{feeder}_b{n_bess}p{n_pv}g{n_gen}d{n_dr}/
# 输出文件:
#   topology.p         - pandapower 网络 (含所有注入资源)
#   resource_table.csv - 资源清单 (type_id, bus, P_max, S_max, E_max, eta)
#   load_p.csv         - 负荷有功时序 (T, num_loads)
#   load_q.csv         - 负荷无功时序 (T, num_loads)
#   pv_p.csv           - 光伏发电时序 (T, num_pv)
#   total_power_curve.png - 时序分析图

set -e

FEEDER=${1:-case33bw}
DAYS=${2:-30}
INTERVAL_MIN=${3:-15}
N_BESS=${4:-3}
N_PV=${5:-3}
N_GEN=${6:-0}
N_DR=${7:-0}
SEED=${8:-42}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
OUT_DIR="$PROJECT_ROOT/data/generated"

echo "=========================================="
echo "  MAGRL Environment Data Generation"
echo "  Feeder: $FEEDER"
echo "  Days: $DAYS | Interval: ${INTERVAL_MIN}min"
echo "  Resources: BESS=$N_BESS PV=$N_PV Gen=$N_GEN DR=$N_DR"
echo "  Seed: $SEED"
echo "  Output: $OUT_DIR/${FEEDER}_b${N_BESS}p${N_PV}g${N_GEN}d${N_DR}/"
echo "=========================================="

eval "$(conda shell.bash hook)"
conda activate yc_mamaskabledppo_env

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
python -u scripts/generate_env_data.py \
    --feeder "$FEEDER" \
    --out_dir "$OUT_DIR" \
    --days "$DAYS" \
    --interval_min "$INTERVAL_MIN" \
    --n_bess "$N_BESS" \
    --n_pv "$N_PV" \
    --n_gen "$N_GEN" \
    --n_dr "$N_DR" \
    --seed "$SEED"
