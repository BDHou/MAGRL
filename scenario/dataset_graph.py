# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandapower as pp

#找“slack bus”（平衡母线）
#pandapower 用 net.ext_grid 表表示“外部电网（上级电网）”连接点。
#slack bus 通常就是 ext_grid 连接的那个 bus。
def get_slack_bus(net: pp.pandapowerNet) -> int:
    if len(net.ext_grid) == 0:
        raise ValueError("No ext_grid found. This script expects a slack/ext_grid bus.")
    return int(net.ext_grid.bus.iloc[0])

#如果 ext_grid 为空 → 说明这个网没有外部电网连接点 → 没法定义 slack → 直接报错。
#否则取 net.ext_grid.bus.iloc[0]：拿第一条 ext_grid 的 bus id 当 slack。

#核心函数（把电网变成图张量）
def build_graph_tensors(
    net: pp.pandapowerNet, #pandapower 的电网对象（包含 bus/line/load/sgen 等表）
    cfg, #ScenarioConfig，里面存了比如 v_min/v_max 等配置（没写类型，应该是 dataclass）
    load_forecast_mult: float, #负荷预测倍率（在 generate_dataset_for_feeder 里算出来的 load_fc[t]）
    pv_forecast_mult: float, #光伏预测倍率（pv_fc[t]）
    last_vm_obs: Optional[np.ndarray], #上一个时间步的电压观测（可能 None），用于给模型一点“历史信息”
    rng: np.random.Generator #随机数生成器，用于加噪声/扰动（为了模拟测量误差等）
) -> Dict[str, np.ndarray]:
    """
    Build graph inputs:
      x: node features [N, F]
      edge_index: [2, 2E]
      edge_attr: [2E, A]
    Node order is net.bus.index order.
    Edge order is net.line order.
    """
    # 节点顺序：按 net.bus.index 的顺序
	# 边顺序：按 net.line.index 的顺序
    buses = net.bus.index.to_numpy().astype(int) 
    #电网里所有 bus 的 id（pandapower 里 bus index 可能是 0,1,2…也可能不是连续的）
    nb = len(buses) #bus 数量
    bus_id_map = {b: i for i, b in enumerate(buses)} #做一个字典，把 bus id 映射成 0..N-1 的“张量索引”
    # 为什么需要 bus_id_map？
    # 因为 edge_index 要用“连续编号”的节点索引（0..N-1）才方便。
    # 但 pandapower 的 bus index 不一定连续、也不一定从 0 开始。
    slack = get_slack_bus(net) #找到 slack bus 的 id

    is_slack = np.array([1.0 if b == slack else 0.0 for b in buses], dtype=np.float32)
    #给每个节点加一个“是不是 slack”的特征
    vn_kv = net.bus.loc[buses, "vn_kv"].to_numpy(dtype=np.float32)
    #vn_kv 表示每个 bus 的 “nominal voltage”（额定电压等级）。
    #低压网可能是 0.4 kV，中压可能是 10kV、20kV
    #这个信息很关键，因为不同电压等级对应不同阻抗尺度、电压波动范围等。也是 node feature 的一部分。

    # stable base load by bus
    #每个 bus 的“固定基准负荷”
    #net._base_load_bus_p 是你在 build_net_with_pv() 里提前算好的：
    # 把所有 load 元件按 bus 汇总后的有功负荷 P（MW）。
    #意思是：这是这个 feeder 的“静态结构属性”，不随时刻变化（时刻变化通过 multiplier 来体现）。
    base_load_p = net._base_load_bus_p.astype(np.float32)

    # forecast load by bus
    #load_p_fc / load_q_fc：这一时刻的“负荷预测值”
    #load_forecast_mult 是你在 generate_dataset_for_feeder() 里提前算的 load_fc[t]。
    #oad_p_fc = base_load_p * load_forecast_mult 
    #就表示：预测这一时刻，负荷会比基准大多少倍/小多少倍。
    #无功负荷预测 load_q_fc
    #电力系统里负荷通常有：有功 P：真正做功的功率，无功 Q：维持电磁场（例如电机、变压器）的“循环功率”
    load_p_fc = base_load_p * float(load_forecast_mult)
    base_load_q = net._base_load_bus_q.astype(np.float32)
    # 如果网络里原本 base_load_q 全是 0（np.allclose 判断）
    # → 说明这个 feeder 数据没提供 Q
    # → 那就用一个比例 cfg.load_q_over_p 来“补一个合理的 Q”：
    if np.allclose(base_load_q, 0.0):
        load_q_fc = load_p_fc * float(cfg.load_q_over_p)
    #如果 feeder 本身就有 Q → 那就同样按 multiplier 缩放：
    else:
        load_q_fc = base_load_q * float(load_forecast_mult)

    # PV forecast by bus
    #pv_p_fc 每个 bus 上预测的 PV 有功注入（MW）
    # pv_q_fc：每个 bus 上预测的 PV 无功注入（这里先全 0）
    pv_p_fc = np.zeros(nb, dtype=np.float32)
    pv_q_fc = np.zeros(nb, dtype=np.float32)
    for i, sid in enumerate(net._pv_ids): 
        #net._pv_ids前面创建 PV 时存的 PV 元件 id 列表（每个 PV 是一个 sgen）
        b = int(net.sgen.at[sid, "bus"]) #找它挂在哪个 bus
        bi = bus_id_map[b] #把这个 bus id 映射到张量索引
        pv_p_fc[bi] += float(net._pv_pmax[i] * pv_forecast_mult) #把这个 PV 的预测出力加到对应 bus
    #  这里没有给 pv_q_fc 赋值，所以它一直为 0。
    # 这代表 输入给 GNN 的 PV 无功预测 = 0（先简化），实际无功会在控制策略/潮流求解里产生。

    # last Vm observation 上一时刻电压观测（含测量噪声）
    #last_vm_obs 是上一时刻的电压 vm（每个 bus 一个值，单位 p.u.）
    #如果没有上一时刻（比如 t=0），就默认全 1.0 p.u.（电压标幺值 1.0 = 正常电压）
	#然后根据 cfg.meas_noise_std_vm 加一点高斯噪声，模拟测量误差
    if last_vm_obs is None:  
        last_vm = np.ones(nb, dtype=np.float32)
    else:
        last_vm = last_vm_obs.astype(np.float32)

    if cfg.meas_noise_std_vm > 0:
        last_vm = last_vm + rng.normal(0, cfg.meas_noise_std_vm, size=last_vm.shape).astype(np.float32)

    # Node feature matrix
    #很关键：拼成节点特征矩阵 x
    #把这些长度为 nb 的向量，按列拼在一起。所以 x 的形状是：x.shape == (nb, 8)
    #每个 bus 一行，8 个特征分别是：
    #is_slack：是不是 slack bus（0/1），vn_kv：额定电压等级，base_load_p：基准有功负荷
    #load_p_fc：预测有功负荷，load_q_fc：预测无功负荷，v_p_fc：预测 PV 有功注入
    #pv_q_fc：预测 PV 无功注入（目前全 0），last_vm：上一时刻电压观测（含噪声）
    x = np.stack(
        [is_slack, vn_kv, base_load_p, load_p_fc, load_q_fc, pv_p_fc, pv_q_fc, last_vm],
        axis=1
    ).astype(np.float32)


    #接下来两段，从 pandapower 的 line 表里，把电网连线翻译成 GNN 需要的 edge_index 和 edge_attr。

    # edges (lines)

    # 取出每条线路的两端 bus
    # net.line 是 pandapower 里“线路表”（每一行是一条 line）
	# from_bus / to_bus 就是每条线连接的两个 bus：
	# 第 i 条线：from_bus[i] -> to_bus[i]
	# nl 是线路条数（number of lines）
    from_bus = net.line.from_bus.to_numpy(dtype=int)
    to_bus = net.line.to_bus.to_numpy(dtype=int)
    nl = len(net.line)

    # 把 bus ID 映射成连续的节点编号
    #为什么要 map？pandapower 的 bus index 可能不是 0..N-1 连续的（有的网络 bus 编号很大、跳号）
    #但 GNN 通常希望节点编号是紧凑的 0..N-1
    #之前建了：bus_id_map = {bus_id -> 0..N-1}
    #这里做的就是：
	# src[i] = 第 i 条线的起点 bus 在 GNN 里的节点编号
	# dst[i] = 第 i 条线的终点 bus 在 GNN 里的节点编号
    src = np.array([bus_id_map[int(b)] for b in from_bus], dtype=np.int64)
    dst = np.array([bus_id_map[int(b)] for b in to_bus], dtype=np.int64)
    
    #构造 edge_index：把每条线变成 双向边，非常关键
        #GNN 常用的边表示：edge_index，形状是 [2, E]，第一行：所有边的起点节点编号，第二行：所有边的终点节点编号
    #为什么要 concatenate([src, dst]) + concatenate([dst, src])？
    #因为电网的线本质上是“无向连接”，但很多 GNN 实现用有向边来表达信息传递：
    #有 nl 条线：src[i] -> dst[i]，再加一份反向：dst[i] -> src[i]，最终边数变成 2 * nl。
    #前 nl 条边：正向，后 nl 条边：反向（复制一份），
    #直觉：电线两头都能互相影响，所以消息要能从 A 传到 B，也要从 B 传到 A。
    edge_index = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0)

    # edge attrs (static only; no leakage)
    # 这些是电力线路最常见的“静态参数”：
    # r_ohm_per_km：电阻（每公里多少欧姆）越大 → 线路损耗越大、电压降越明显
    #电抗（每公里），影响无功/相角/潮流分布
    #线路长度（公里），越长通常阻抗越大，影响更大
    r = net.line.r_ohm_per_km.to_numpy(dtype=np.float32)
    x_ohm = net.line.x_ohm_per_km.to_numpy(dtype=np.float32)
    length = net.line.length_km.to_numpy(dtype=np.float32)

    # max_i_ka：线路载流上限（如果有）
    if "max_i_ka" in net.line.columns:
        max_i = net.line.max_i_ka.to_numpy(dtype=np.float32)
        max_i = np.nan_to_num(max_i, nan=0.0) #有些值是 NaN → 用 0 替代，避免训练时出 NaN
    else:
        max_i = np.zeros(nl, dtype=np.float32) #有些 feeder 的 line 表没有这个字段 → 就填 0

    # 拼成单条边的特征向量 [r, x, length, max_i]
    edge_attr_fwd = np.stack([r, x_ohm, length, max_i], axis=1)
    # 把边特征也复制一份，匹配“双向边”
    edge_attr = np.concatenate([edge_attr_fwd, edge_attr_fwd], axis=0).astype(np.float32)

    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr} #返回图输入
    #总结：最终给 GNN 的图就是三件套：x: 节点特征 (N, F)（前面做的 8 维特征）
    #edge_index: 边连接关系 (2, 2*nl)，edge_attr: 边特征 (2*nl, 4)
    #研究意义：没有用潮流结果（没有用 net.res_*），所以不会泄露未来答案（no leakage）
    #它只用“网络拓扑 + 静态线路参数 + 预测/观测输入”构建图→ 这非常适合后面用 GNN 做预测/分类/控制策略学习

#把潮流计算的“线路结果（p/q/loading）”变成和 edge_index 完全对齐的“有向边监督标签”（2×nl 条边）
#前面在 build_graph_tensors() 里把每条线路复制成了 双向边（forward + backward），
#所以这里也要把标签复制成 2×nl，并且要处理方向。
def build_edge_targets_from_res(net: pp.pandapowerNet, res: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    #电网（已经 build 好 + PV + load 等）res：你在 run_powerflow_with_controls() 里收集的结果字典
    #注意：这些 p_from/q_from 的方向定义是 pandapower 的：按照 line 表的 from_bus -> to_bus 方向。
    """
    Build directed-edge targets aligned with edge_index (2*nl).
    Forward is net.line direction; backward is negated.
    """
    p = res["p_from"]
    q = res["q_from"]
    loading = res["loading"]

    # 为什么反向要 -p、-q？
	# 假设某条 line 在 forward 方向（from→to）测得 p = +2 MW意味着功率从 from 流向 to。
	# 那么在 backward 方向（to→from）看同一条线，功率应该是 -2 MW。（方向反了，符号也反）
    #跟前面构造 edge_index 的方式完全一致：edge_index 也是先放 [src->dst]，再放 [dst->src]
    #同一根线，换个方向看，潮流符号必须反过来。
    y_edge_p = np.concatenate([p, -p], axis=0).astype(np.float32)
    y_edge_q = np.concatenate([q, -q], axis=0).astype(np.float32)
    #loading：反向边不需要取负号
    #loading_percent 是“这根线的负载/拥塞程度”，它不是方向量，只有大小：
    y_edge_loading = np.concatenate([loading, loading], axis=0).astype(np.float32)
    #反送/反向潮流 rpf：用 p < 0 判断
    # rpf 这里其实就是“这条有向边上的有功功率是不是为负”。
	# 对 forward 边：p_from < 0 → 表示功率其实从 to→from 流（反着流）
	# 对 backward 边：因为你用了 -p，它的符号也同步反过来，所以也一致。
    #输出是 0/1 标签（int64）：1：该有向边是“负功率方向”（反向），0：该有向边是“正功率方向”（与边方向一致）
    y_edge_rpf = (y_edge_p < 0).astype(np.int64)

    #返回的标签字典
    return {
        "y_edge_p": y_edge_p, #回归目标（预测边功率）
        "y_edge_q": y_edge_q, #回归目标
        "y_edge_loading": y_edge_loading, #回归目标（可选）
        "y_edge_rpf": y_edge_rpf, #分类目标（线是否反向）
    }