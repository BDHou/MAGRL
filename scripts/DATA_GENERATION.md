# MAGRL 环境数据生成系统

## 概述

本系统用于生成 MARL 强化学习环境所需的配电网拓扑与时序数据。核心能力是向 IEEE 标准配电网中注入多种可调分布式能源（DER），生成带有 TypeID 标签的网络拓扑和时间序列数据。

## 快速开始

```bash
bash run_generate.sh case33bw 30 15 3 3 1 2 42
#                     |       |  |  | | | | |
#                     feeder  |  |  | | | | seed
#                         days   |  | | | n_dr
#                         interval  | | n_gen
#                                   | n_pv
#                                   n_bess
```

## 输出目录结构

```
data/generated/case33bw_b3p3g1d2/
  topology.p            pandapower 网络对象 (含所有注入的资源)
  resource_table.csv    资源清单
  load_p.csv            负荷有功时序 (T, num_loads)
  load_q.csv            负荷无功时序 (T, num_loads)
  pv_p.csv              光伏发电时序 (T, num_pv)，正值
  total_power_curve.png 时序功率分析图
  topology.png          网络拓扑图（标注资源位置）
```

目录名格式: `{feeder}_b{n_bess}p{n_pv}g{n_gen}d{n_dr}`

## 资源类型与 Pandapower 映射

| TypeID | 类型名 | Pandapower 元件 | PP 表 | 说明 |
|--------|--------|----------------|-------|------|
| 0 | BESS (储能) | `pp.create_storage` | `net.storage` | 配储一体化, 默认绑定 PV 母线 |
| 1 | Generator (发电机) | `pp.create_sgen(type="Generator")` | `net.sgen` | 小型柴油/燃气机组 |
| 2 | Inverter/PV (逆变器) | `pp.create_sgen(type="PV")` | `net.sgen` | 分布式光伏 |
| 3 | DemandResponse (可调负荷) | `pp.create_load(controllable=True)` | `net.load` | 可调节负荷 |

每个元件的 `name` 字段编码 TypeID: `BESS_T0_bus17`, `Gen_T1_bus26`, `PV_T2_bus15`, `DR_T3_bus19`

## 标幺值系统

所有资源容量参数 `P_max` 以标幺值 (p.u.) 表示:

```
S_base = sum(net.load.p_mw)   # 馈线总有功负荷
P_max_mw = P_max_pu * S_base
```

选用总负荷作为基准的理由:
- DER 渗透率的标准定义即 P_DER / P_load_total
- 自动适配不同规模的网络 (33-bus ~3.7MW vs 69-bus ~3.8MW vs mv_oberrhein ~30MW)

BESS 的容量 E_max 通过储能时长 `e_duration_h` 定义:

```
E_max_mwh = P_max_mw * e_duration_h
C-rate = 1 / e_duration_h
```

## 默认参数与工程依据

以 IEEE 33-bus (12.66kV, S_base = 3.715 MW) 为例:

### PV/Inverter
- `P_max`: 0.08-0.30 p.u. (0.30-1.11 MW/台)
- `s_over_p`: 1.10 (IEEE 1547-2018 Cat B)
- 3 台总渗透率: 24-90%
- 依据: IEC 61727, IEEE 1547

### BESS
- `P_max`: 0.03-0.13 p.u. (0.11-0.48 MW/台)
- `e_duration_h`: 1.5-3.0 小时 (C-rate 0.33-0.67C)
- `eta`: 0.92-0.95 (锂电单程效率, round-trip 85-90%)
- 依据: NREL "Cost Projections for Utility-Scale Battery Storage"

### Generator
- `P_max`: 0.03-0.13 p.u. (0.11-0.48 MW/台)
- `s_over_p`: 1.18 (功率因数 pf=0.85)
- 依据: IEEE 1547 Category III

### Demand Response
- `P_max`: 0.005-0.03 p.u. (0.019-0.11 MW/台)
- 物理含义: 单节点负荷的 10-30% 可调
- 依据: FERC Order 2222

## resource_table.csv 字段说明

| 列名 | 类型 | 说明 |
|------|------|------|
| resource_id | int | 全局资源编号 (0, 1, 2, ...) |
| type_id | int | TypeID (0/1/2/3) |
| type_name | str | 类型名 (bess/generator/inverter/demand_response) |
| bus | int | 母线编号 |
| pp_element | str | pandapower 表名 (storage/sgen/load) |
| pp_index | int | 在对应 PP 表中的行索引 |
| P_max_pu | float | 额定有功功率 (p.u. of S_base) |
| P_max_mw | float | 额定有功功率 (MW) |
| S_max_mw | float | 额定视在功率 (MVA) |
| E_max_mwh | float | 额定电量 (MWh)，仅 BESS 有效 |
| eta | float | 效率 (单程), 仅 BESS 有效 |

## 配置覆盖

`ResourceInjector` 采用深度合并: 只需写想覆盖的部分, 其余用默认值。

```python
from scripts.resource_injector import ResourceInjector

config = {
    "inverter": {
        "count": 5,
        "params": {"P_max": (0.15, 0.25)},  # 覆盖 P_max, 其余用默认
    },
    "bess": {"count": 5},                     # 只改数量
}
injector = ResourceInjector(config=config, seed=42)
net, table = injector.inject(net)
```

### 母线选择策略

每种资源类型可独立配置:

| 策略 | 说明 |
|------|------|
| `farthest` | 离 slack bus 拓扑距离最远的母线优先 (默认) |
| `random` | 从非 slack 母线中随机选择 |
| `manual` | 使用 `bus_list` 指定母线, 如 `[5, 12, 18]` |

注入顺序固定为 inverter -> bess -> generator -> demand_response。
BESS 在 `farthest` 策略下会自动绑定 PV 母线 (配储一体化)。

## 文件依赖关系

```
run_generate.sh
  -> scripts/generate_env_data.py      主入口
       -> scripts/resource_injector.py  资源注入 (ResourceInjector)
       -> scenario/base_scenario.py     时序曲线 (负荷/PV/云量形状函数)
```
