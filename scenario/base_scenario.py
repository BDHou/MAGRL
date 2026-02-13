# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import pickle
import zlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import pandapower as pp
import pandapower.networks as pn
from pandapower.auxiliary import LoadflowNotConverged

from .dataset_graph import build_graph_tensors, build_edge_targets_from_res


# ----------------------------
# Config (base scenario) 基础场景，为后面的风险事件作准备
# ----------------------------
@dataclass
class ScenarioConfig:
    # ----- dataset -----
    out_dir: str = "out_dataset"
    seed: int = 1
    days: int = 30
    steps_per_day: int = 24
    horizon: int = 1

    # ----- feeders 馈线 (try in order; missing ones will be skipped) -----
    feeder_candidates: Tuple[str, ...] = (
        "case33bw",
        "case69",
        #not available in my pandapower version.
        "mv_oberrhein",
        "mv_urban",
       # not available in my pandapower version.
    #    "lv_schutterwald",
    ## v_schutterwald 的 结果不对 ， 超大地图nbus≈2940, nline≈3000
    )

    # ----- PV placement & sizing -----光伏放哪里，装多大
    n_pv: int = 3  #在网络里放 3 个 PV 点（用 pp.create_sgen 创建 3 个发电机）
    pv_bus_strategy: str = "farthest"   # 决定PV放在哪些，
    #“farthest”：优先放在离变电站/外部电网（slack bus）最远的地方| "high_index"选 编号最大的几个 bus，当拓扑距离算不出来时就用这个兜底
    pv_pmax_mw_range: Tuple[float, float] = (1.5, 4.0) #每个 PV 的额定最大有功功率 Pmax（单位 MW）从 1.5 到 4.0 随机抽。
    pv_daily_scale_range: Tuple[float, float] = (0.6, 1.8) #每天给PV一个daily scale，今天功率强不强，0.6-今天整体偏弱，1.8-今天整体强（阳光明媚）
    pv_pf: float = 1.0  # 功率因数，currently Q=0 unless volt-var enabled，只有开启 Volt-Var 控制时，它才会根据电压调 Q（吸/发无功）

    # ----- load scaling ----- 负荷怎么变
    load_daily_scale_range: Tuple[float, float] = (0.8, 1.2) #每天给负荷一个倍率：0.8的话这天整体负荷低一些，•	1.2的话：这天整体负荷高一些
    load_q_over_p: float = 0.35 #如果原始网络里负荷没有 无功Q（很多测试网可能 Q=0），那就用这个规则补 什么事有功P，什么是无功Q？

    # ----- stochasticity ----- 随机性，让场景更像真实世界
    meas_noise_std_vm: float = 0.002 #电压测量噪声标准差 0.002 pu（大概 0.2%）
    meas_noise_std_pq: float = 0.01 #对 P/Q 这类功率测量加噪声，标准差 1%。（也就是 PV 输出有 1% 左右随机误差。）
    cloud_ar: float = 0.92 #越接近 1：云的状态越“连续”，不会一小时晴一小时暴雨那样跳变
    cloud_sigma: float = 0.12 #随机扰动幅度
    cloud_drop_prob: float = 0.10 #每个时刻有 10% 概率发生一次“突然变阴”
    cloud_drop_mag: Tuple[float, float] = (0.2, 0.6) #把 cloud 因子乘一个 0.2~0.6 的随机数.直觉：模拟“云突然遮住太阳，PV 瞬间掉下去”
    forecast_error_std: float = 0.06 #预测值（forecast）会有 6% 标准差误差.  

    # ----- operational limits ----- 运行限制：用来打标签
    # 电压允许范围（per-unit）：
	# •	低于 0.95：欠压
	# •	高于 1.05：过压
    v_min: float = 0.95
    v_max: float = 1.05

    # ----- solver ----- 潮流求解器设置
    runpp_algorithm: str = "bfsw" #默认潮流算法用 bfsw（Backward/Forward Sweep，常用于配电网）。
    # 实际调用的是 runpp_robust()，它会尝试多个算法，这里 runpp_algorithm 在你代码里“默认值/保留”。
    robust_algos: Tuple[str, ...] = ("bfsw", "nr") #如果潮流不收敛（算不出来），就换算法重试
    # bfsw：配电网常用、通常稳，nr：Newton-Raphson（经典潮流算法），对某些网络更快/更准，但有时更容易不收敛
    robust_init_modes: Tuple[str, ...] = ("auto", "flat") #潮流迭代需要一个“初始猜测”，这里提供两种
    # auto：pandapower 自己选一个较合理的初值（可能用上一次结果或内部策略），flat：平坦启动（电压都先从 1.0 pu、角度 0 开始猜）
    robust_tol_mva: Tuple[float, ...] = (1e-8, 1e-6) #收敛容差（越小越严格）：
    robust_max_iter: Tuple[int, ...] = (100, 200) #每次求解最多迭代多少步，100 次不行，就允许到 200 次再试一次
    robust_tries: int = 2 #这上面那一套（轮数（tries）× 算法（algos）× 初值（init）× 容差（tol）× 最大迭代（max_iter））总共试几次才失败，返回pf_ok=0

    # ----- control (application) ----- 控制策略设置，让光伏逆变器做点事。什么是光伏逆变器？
    enable_volt_var: bool = True #允许逆变器根据电压自动调无功 Q。
    # 电压太高（>1.04~1.06）：逆变器吸收无功（Q变负），帮助降压
	# 电压太低（<0.98~0.94）：逆变器提供无功（Q变正），帮助升压
    enable_curtailment: bool = True #削减光伏有功，允许削减 PV 的 P（直接少发电）。
    # Volt-Var 有时不够（比如电压还是超、或者反向潮流太严重），就进入“削发电”策略
    volt_var_iters: int = 4 #每个时刻会做：解潮流 → 看电压 → 调 Q → 再解潮流 → … 重复 4 次
    #     为什么要迭代多次？
    # 因为你调了 Q 以后，电压会变；电压变了，新的 Q 目标又会变。

    volt_var_damping: float = 0.6 #“阻尼/平滑系数”，防止一下子调太猛引起震荡。
    volt_var_q_step_frac: float = 0.25 #再加一层“每次最多只允许走 25% 的 Q 变化步长”，
    # 就算目标 Q 很大，也别一下子跳过去，每次最多走 1/4，避免数值震荡和不收敛。
    inverter_s_over_p: float = 1.10 #逆变器容量限制：视在功率 Smax = 1.10 × Pmax
    # 逆变器不能同时无限制发 P 和 Q，它受制于：S² = P² + Q²
    # 如果你把 S/P 设得大一些（比如 1.2），逆变器就有更多“无功余量”来调电压
	# 设得小，电压控制能力就弱
    curtail_on_export: bool = True #如果出现“对上级电网反向送电”（net.res_ext_grid.p_mw < 0），就考虑削减 PV。
    #我们可能不希望向上级反送太多功率？（或者这代表 RPF 风险）。？
    curtail_on_overvoltage: bool = True #如果出现过压（最大电压 > v_max），也触发削减 PV。
    curtail_step: float = 0.08 #每次削减 8%（逐步削，不是一下削光）。
    curtail_max_frac: float = 0.50 #单个 PV 最多削到 原来的 50%（给一个底线，避免削得太夸张）。
    curtail_max_rounds: int = 4 #每个时刻最多做 4 轮削减尝试。检查是否需要削 → 选一个 PV 削 8% → 重跑潮流 → 还不行再来…



# ----------------------------
# Utilities
# ----------------------------
def stable_int_hash(s: str) -> int:
    return int(zlib.crc32(s.encode("utf-8")))

#加载标准测试电网（feeder），不行就返回None
def load_feeder_by_name(name: str) -> Optional[pp.pandapowerNet]:
    if not hasattr(pn, name):
        return None
    fn = getattr(pn, name)
    try:
        return fn() #真正创建这个电网对象（pandapowerNet）,pandapowerNet：这个 feeder 的完整数据结构(bus（节点),line（线路）, load（负荷）,ext_grid（外部电网/上级电网),等等)
    except Exception:
        return None

#什么是 slack bus / ext_grid？潮流计算必须有一个“参考点/电源端”，它负责：给整个系统提供平衡功率（系统总有功/无功缺口由它补），固定电压幅值和相角参考（否则方程没法唯一解）
# 在 pandapower 里，通常用 ext_grid 来表示外部电网（上级电网/变电站等），它连到某一个 bus，上面那个 bus 就是 slack bus。
def get_slack_bus(net: pp.pandapowerNet) -> int:
    if len(net.ext_grid) == 0:
        raise ValueError("No ext_grid found. This script expects a slack/ext_grid bus.")
        #如果这个网络没有 ext_grid，就没“电源参考点”，后面很多地方（比如选“离 slack 最远的 PV bus”）都依赖 slack，所以直接报错提醒：这个网络不符合你脚本假设
    return int(net.ext_grid.bus.iloc[0]) #net.ext_grid 是一个表（DataFrame，里面有一列 bus 表示 ext_grid 接在哪个 bus 上，iloc[0] 取第一行（通常只有一个 ext_grid），转成 int 返回

#下面这个函数：
# 输入：
# 	net：pandapower 的配电网模型（里面有很多 bus、line、ext_grid 等）
# 	n_pv：要放几个 PV（上面默认 3 个）
# 	strategy：怎么挑 bus（上面默认 "farthest"，后面代码也支持 "high_index",if strategy == "high_index":）

# 输出：
# 	一个列表：被选中的 bus 编号（比如 [18, 30, 25]）
def choose_pv_buses(net: pp.pandapowerNet, n_pv: int, strategy: str = "farthest") -> List[int]:
    buses = list(net.bus.index.astype(int)) #拿到所有 bus 的编号（像 0,1,2,…）
    slack = get_slack_bus(net) #上面定义的函数，找到 slack
    candidates = [b for b in buses if b != slack] #候选 bus = 所有 bus 去掉 slack，因为一般我们不会把 PV 放在 slack（放那没意思，也不符合“远端挑战”）
    #	slack 就像“市中心总站”
	# candidates 是“除了总站以外的所有小区站点”

    #特殊情况：要放的 PV 数太多，那就直接返回能放的全部（最多就这些）。
    if n_pv >= len(candidates):
        return candidates[:n_pv]
    #策略 1：按编号挑（high_index）把 bus 编号从大到小排，取前 n_pv 个
    if strategy == "high_index":
        return sorted(candidates, reverse=True)[:n_pv]
    #策略 2（默认）：挑“拓扑最远”的 bus（farthest），以下这段：计算每个 bus 到 slack 的“走路步数”，然后挑最远的。
    try:
        import pandapower.topology as top
        import networkx as nx
        G = top.create_nxgraph(net, respect_switches=True) 
        #上面这句：把电网变成图（graph），bus 是“节点”，line 是“边”，respect_switches=True：如果某些开关断开，就不算连通（更真实）。
        dist = nx.single_source_shortest_path_length(G, slack)
        #计算 slack 到所有点的最短距离，从 slack 出发，算到每个 bus 的最短路径长度，这里的距离是“走过多少条线/边”
        ranked = sorted(candidates, key=lambda b: (dist.get(b, -1), b), reverse=True)
        #给候选 bus 排序：先按距离，再按编号
        #dist.get(b, -1) 的意思是：如果某个 bus 在 dist 里找不到（可能不连通），就给它距离 -1（很近/很差），避免报错。

        return ranked[:n_pv]
        #取前 n_pv 个作为 PV bus
    except Exception:
        return sorted(candidates, reverse=True)[:n_pv]
    #如果 networkx / topology 有问题、或者某些网模型导致建图失败，就退化成最简单的策略：按 bus 编号从大到小挑。


# pp.runpp() 是 pandapower 的潮流计算函数：
# 给定负荷、发电、网络拓扑，求每个 bus 的电压（vm_pu）、相角（va_degree）、
# 每条线的潮流（p_from_mw、q_from_mvar）等等。
#runpp_robust()：“想办法把潮流算出来”的保险丝，尽量别让整个流程因为某一次不收敛就直接死掉。

def runpp_robust(
    net: pp.pandapowerNet, #电网对象（里面有 bus/line/load/sgen/ext_grid 等）
    cfg: ScenarioConfig, #里面放了“要尝试哪些求解配置”
    calculate_voltage_angles: bool = True  #是否计算电压相角（True 就输出 va_degree）
) -> Tuple[bool, Dict]: #输出：bool: 成功 True / 失败 False，Dict: 成功时告诉我们用了哪些参数；失败时告诉最后一次错误是什么
    last_err = None #记录最后一次失败的原因。
    for _ in range(cfg.robust_tries): #最外层，把整套参数组合再整体跑几轮。
        #四重循环：枚举所有尝试组合
        for algo in cfg.robust_algos:  #algorithm e.g. ("auto", "flat")
            # "nr"：Newton-Raphson（经典潮流算法，很多系统收敛快，但对初值/条件敏感）
            #"bfsw"：Backward/Forward Sweep（配电网径向结构很常用，很多时候更稳）
            for init in cfg.robust_init_modes: #初始值
            #"flat"：平坦初值（所有 bus 电压从 1.0 pu 开始）
    	    #"auto"：pandapower 自动选（可能用以前的结果或更聪明的初始化）
                for tol in cfg.robust_tol_mva: # tolerance_mva
                # 1e-8 更严格
                # 1e-6 更宽松（更容易判定收敛）
                    for it in cfg.robust_max_iter:#max_iteration
                    #收敛慢时给它更多次数（100 不够就 200）
                        #跑潮流：
                        try:
                            pp.runpp(
                                net,
                                algorithm=algo,
                                init=init,
                                tolerance_mva=tol,
                                max_iteration=it,
                                calculate_voltage_angles=calculate_voltage_angles,
                            )
                            return True, {"algorithm": algo, "init": init, "tol": tol, "max_iter": it} #成功
                        #except：失败就记下错误，继续试下一组
                        except LoadflowNotConverged as e:
                            last_err = e
                        # LoadflowNotConverged：最常见，意思就是“迭代到了 max_iter 也没满足收敛条件”    
                        except Exception as e:
                            last_err = e
    return False, {"error": repr(last_err)} #全部都失败：返回 False + 最后错误

# 总尝试次数 = 2×2×2×2×2 = 32 次（每一步潮流最多试 32 次）
# 如果网络很大、某些点很难收敛，就会变慢


# ----------------------------
# Profiles 生成“时间序列曲线”，可以把它当成“合成数据的天气与用电行为模型”
# ----------------------------
# 一天 24 小时的“用电曲线”
def base_load_shape(hour: int) -> float:
    h = hour % 24 #确保 hour 不管是多少，都折算成 0~23 点。
    #用三个“钟形峰”（高斯形状）叠加：
    morning = math.exp(-((h - 8) / 3.0) ** 2) # 早高峰，中心在 8 点
    evening = math.exp(-((h - 19) / 3.0) ** 2) # 晚高峰，中心在 19 点
    midday = 0.2 * math.exp(-((h - 13) / 5.0) ** 2) # 午间小峰，中心 13 点，幅度更小、更宽
    return 0.65 + 0.35 * (0.6 * morning + 1.0 * evening + midday) #把这些峰合成并加上一个基础底座
    # 输出一个倍率，大概在 0.65 ~ 接近 1.0 之间波动。
    # 后面会用它乘上 load_daily_scale[day] 来得到当天的实际负荷水平。
# 晴天 PV 的“日照曲线”
# 夜里是 0，早上 6 点开始爬升，中午最大发电，下午下降，晚上 18 点归零
def clear_sky_pv_shape(hour: int) -> float:
    h = hour % 24
    x = (h - 6) / 12.0
    if x <= 0 or x >= 1:
        return 0.0
    return float(math.sin(math.pi * x))
    # 输出一个 0~1 的倍率（晴空形状）。
    # 后面会做：pv_shape * pv_daily_scale[day] * cloud[t]
    # 也就是 晴天形状 × 当天 PV 强度 × 云层影响。

# 云层随机扰动（让 PV 更像真的）
# 这个函数就是生成一个长度为 T 的数组 cf[t]，表示每个时刻 PV 的“天气系数”。
# T: 总步数（比如 30天×24小时 = 720），ar: 自回归系数（类似“惯性/记忆”），越接近 1：云层状态变化越慢（更平滑）
# 我们设置 0.92：说明云层有强相关性，上一小时云多，这小时也可能云多
# sigma: 随机噪声强度（抖动幅度），drop_prob: “突发遮挡”的概率（每一步有多大概率突然掉）
# drop_mag: 掉落的幅度范围（比如乘上 0.2~0.6），rng: 随机数生成器（保证可复现）
def simulate_cloud_factor(
    T: int, ar: float, sigma: float, drop_prob: float,
    drop_mag: Tuple[float, float], rng: np.random.Generator
) -> np.ndarray:
    cf = np.zeros(T, dtype=float)
    cf[0] = 1.0
    for t in range(1, T):
        cf[t] = ar * cf[t - 1] + (1 - ar) * 1.0 + rng.normal(0, sigma)
        # 突发云遮挡事件，以 drop_prob 概率，突然把 cf 乘一个 0.2~0.6 的系数，PV 会瞬间掉到原来的 20%~60%。
        if rng.random() < drop_prob:
            cf[t] *= rng.uniform(drop_mag[0], drop_mag[1])
        # 裁剪：中间允许到 1.2（可能是噪声造成短暂超过 1），但最后输出强行压到 0~1。
        cf[t] = float(np.clip(cf[t], 0.0, 1.2))
    return np.clip(cf, 0.0, 1.0)
    # 输出：一个数组 cf[t]，每个元素在 0~1：1：晴天，0.7：有云，PV 少 30%，0.3：很厚云遮挡，PV 掉 70%


# ----------------------------
# Build net + PV + base load cache
# 把 pandapower 的电网 net “预处理”一下：先把原始负荷保存成基准（base），
# 再把负荷按 bus 汇总保存起来，最后在选定的 bus 上加上 PV（sgen），并把 PV 的关键参数也缓存到 net 里。
# ----------------------------
def build_net_with_pv(net: pp.pandapowerNet, cfg: ScenarioConfig, rng: np.random.Generator) -> pp.pandapowerNet:
    # 保存“原始负荷”作为基准（base load），net.load 是一个表，里面每一行是一条负荷（load）。
	# p_mw 是有功功率（MW），也就是“真正消耗电的部分”。这里把它拷贝出来存在 net._base_load_p 里。
    #作用是以后要做“日负荷倍率 load_mult”，就用 base * load_mult，不用担心把原始数据弄丢。
    net._base_load_p = net.load["p_mw"].to_numpy().copy()
    # 处理无功负荷 q_mvar（没有就补 0）
    # 有的网络数据里可能没有 q_mvar，那就：，给 net.load 表新增一列 q_mvar=0，同时保存基准 _base_load_q 全零
    if "q_mvar" in net.load.columns:
        net._base_load_q = net.load["q_mvar"].to_numpy().copy()
    else:
        net.load["q_mvar"] = 0.0
        net._base_load_q = np.zeros(len(net.load), dtype=float)
    # 建立 bus 的索引映射（bus_id_map）
    # net.bus.index 是所有 bus 的编号（比如 0,1,2,…）。
	# bus_id_map 是一个字典：bus编号 -> 在数组中的位置。
    buses = net.bus.index.to_numpy().astype(int)
    bus_id_map = {b: i for i, b in enumerate(buses)}
    # 计算“每个 bus 上的总负荷”（把多个 load 汇总到 bus）
    base_bus_p = np.zeros(len(buses), dtype=float)
    base_bus_q = np.zeros(len(buses), dtype=float)

    for pos, row in enumerate(net.load.itertuples()):
        b = int(row.bus) #row.bus：这个负荷挂在哪个 bus
        bi = bus_id_map[b]
        # pos：这是第几个 load（用它去取 _base_load_p[pos]）
        # base_bus_p[i]：第 i 个 bus（按 buses 数组顺序）上的总有功负荷
	    # base_bus_q[i]：同理总无功负荷
        base_bus_p[bi] += float(net._base_load_p[pos]) #把这个 load 的 P 加到对应 bus 的总 P 上
        base_bus_q[bi] += float(net._base_load_q[pos]) #把这个 load 的 Q 加到对应 bus 的总 Q 上
    # 然后保存到 net 里
    net._base_load_bus_p = base_bus_p 
    net._base_load_bus_q = base_bus_q 
    # 作用：后面构图（GNN 输入）或者做特征时，很可能需要“每个 bus 上的负荷”，直接用缓存即可，不用每一步重新遍历 loads。

    # 选哪些 bus 放 PV（调用前面写的策略）
    # cfg.n_pv=3：放 3 个 PV
	# cfg.pv_bus_strategy="farthest"：默认选“离 slack 最远”的 bus
    pv_buses = choose_pv_buses(net, cfg.n_pv, cfg.pv_bus_strategy)

    # 在这些 bus 上创建 PV（sgen），并给每个 PV 随机一个最大容量
    # sgen 是什么？pandapower 里 sgen 是 “static generator”，常用来表示：光伏 PV，风电，其他分布式发电（不需要复杂动态模型）
    pv_ids, pv_pmax, pv_smax = [], [], []
    for b in pv_buses:
        # pmax 是 PV 最大有功（MW），随机一个范围 (1.5, 4.0) MW，表示每个 PV 的装机容量不一样。
        pmax = float(rng.uniform(cfg.pv_pmax_mw_range[0], cfg.pv_pmax_mw_range[1])) 
        # 逆变器视在功率上限（MVA）
        # 逆变器通常允许 Smax 比 Pmax 稍大一点（比如 1.1 倍），这样才有余量输出无功 Q 做 Volt-Var 控制。
        smax = float(cfg.inverter_s_over_p * pmax)

        sid = pp.create_sgen(
            net, bus=int(b),
            p_mw=0.0, q_mvar=0.0, #为什么是0？这是“搭建场景阶段”。后面每个时刻会根据 PV 曲线把 p_mw、q_mvar 更新成当下值。
            name=f"PV_bus{b}",
            type="PV", controllable=False #表示不使用 pandapower 内置控制器对象（比如 pp.control 模块）来自动优化。
        )
        #创建完以后把 PV 的 id、pmax、smax 记下来：
        pv_ids.append(int(sid))
        pv_pmax.append(pmax)
        pv_smax.append(smax)
    #把 PV 的关键信息“缓存”到 net 里，方便后续快速更新
    net._pv_ids = pv_ids
    net._pv_pmax = np.array(pv_pmax, dtype=float)
    net._pv_smax = np.array(pv_smax, dtype=float)
    net._pv_buses = np.array(pv_buses, dtype=int)

    return net

    # 这函数返回的 net “多了什么”？
    # 基准负荷缓存：net._base_load_p, net._base_load_q，net._base_load_bus_p, net._base_load_bus_q
    #PV 资产信息：net._pv_ids（PV 在 net.sgen 表里的行号），net._pv_pmax, net._pv_smax，net._pv_buses
    #以上为了是为了后面“快速循环 T=720 个小时”不重复做重活。

# ----------------------------
# Volt-Var helpers
# 为了实现：
# 电压高了就吸无功（-Q）把电压压下来；电压低了就送无功（+Q）把电压抬上去。
# 但无功 Q 不是想给多少就给多少，还要受逆变器容量限制（Smax）。
# 在配电网里，给 +Q（送无功） → 电压倾向于 升高，给 -Q（吸无功） → 电压倾向于 降低
# ----------------------------
# v_pu = per-unit 电压。1.0 pu 就是“额定电压”（正常值）。>1.0 说明电压偏高，<1.0 说明电压偏低。
# 我们代码里 v_min=0.95, v_max=1.05 就是允许范围。
# qmax 表示“当前这一刻逆变器最多能输出/吸收的无功大小”（上下限）。
# qmax 会随 p 变化（后面 inverter_q_limit 计算）。
def volt_var_q(v_pu: float, qmax: float) -> float:
    if v_pu >= 1.06: #电压非常高 → 吸无功到最大（-qmax）
        return -qmax
    if v_pu >= 1.04: #电压有点高，1.04 到 1.06 之间，Q 从 0 平滑地变到 -qmax。
        return -qmax * (v_pu - 1.04) / 0.02
    if v_pu <= 0.94: #电压非常低 → 送无功到最大（+qmax）
        return +qmax
    if v_pu <= 0.98: #电压有点低。同样是线性渐变
        return +qmax * (0.98 - v_pu) / 0.04
    return 0.0 #电压正常范围，不出无功（Q=0），避免乱调。

#解决：逆变器在当前 P 下最多还能提供多少 Q？
# S² = P² + Q²
# S：视在功率（容量上限），逆变器铭牌能力（这里是 smax）
# P：有功输出（光伏发的有功）
# Q：无功输出（电压调节用）
def inverter_q_limit(smax: float, p: float) -> float:
    p = abs(float(p))
    if p >= smax: #如果 p >= smax：说明逆变器容量全被有功占满了 → 不能再出无功（qmax=0）
        return 0.0
    return float(math.sqrt(max(smax * smax - p * p, 0.0)))

# 控制流程（在后面 run_powerflow_with_controls 里）大概是这样：
# 	1.	已知当前 PV 输出 p_now
# 	2.	算 qlim = inverter_q_limit(smax, p_now)  （这一刻的 Q 上限）
# 	3.	读当前电压 v_pu
# 	4.	算目标 qtar = volt_var_q(v_pu, qlim)（电压对应的 Q 目标）
# 	5.	把 PV 的 q_mvar 往 qtar 调（后面还有 damping 和 step 限制）

# ----------------------------
# PF （潮流） + Controls
# run_powerflow_with_controls()：“把这一时刻的场景写进电网 + 跑潮流 +（可选）做控制”
#给定这一时刻的负荷倍率、光伏倍率、云量等 → 把 net 里的负荷和 PV 数值更新好 → 然后跑一次潮流（PF），
#如果开了 control 再做 Volt-Var / Curtailment。
# ----------------------------
def run_powerflow_with_controls(
    net: pp.pandapowerNet, #pandapower 的电网对象（里面有 bus、line、load、sgen 等表）
    cfg: ScenarioConfig, #配置（参数大全）
    load_mult: float, #这一时刻负荷倍率
    pv_mult: float, #这一时刻光伏倍率
    cloud_factor: float, #云影响（0~1），云多就更小
    noise_std: float, #测量/扰动噪声强度
    control: bool, #是否开启控制
    rng: np.random.Generator #随机数生成器（保证可复现）
) -> Tuple[Optional[Dict[str, np.ndarray]], Dict]:
# 返回值：
# res：如果潮流成功，返回各种数组（电压、潮流、标签等）；失败返回 None
# info：字典，记录成功/失败原因、用了哪个算法等
    # loads 更新负荷
    # net._base_load_p，这是在 build_net_with_pv() 里缓存的基准负荷有功（每个 load 元件的 p_mw）。
    p_load = net._base_load_p * load_mult 
    # 如果基准 Q 全是 0：说明没有可靠的无功数据
    # → 就用一个固定比例 load_q_over_p 来“估算”负荷的无功：
    # 如果基准 Q 不是 0：说明 feeder 本来就给了 Q
    # → 那就跟着一起按倍率缩放：
    if np.allclose(net._base_load_q, 0.0):
        q_load = p_load * cfg.load_q_over_p
    else:
        q_load = net._base_load_q * load_mult

    net.load.loc[:, "p_mw"] = p_load
    net.load.loc[:, "q_mvar"] = q_load
    # 这就是把这一时刻的负荷写回 pandapower 网络里。
    # pandapower 的潮流计算就会读取这些表格里的值。

    # PV 更新光伏 PV（在 pandapower 里用 sgen 表示）
    pv_actual_mult = pv_mult * cloud_factor
    # 在 build_net_with_pv() 里创建 PV 的时候，给每个 PV 随机了一个 pmax，并缓存到了 net._pv_pmax。
    pv_p = net._pv_pmax * pv_actual_mult
    # 先让 PV 的无功为 0：如果后面开启了 volt-var 控制，才会把 q_mvar 改成非零。如果不开控制，就一直 Q=0。
    pv_q = np.zeros_like(pv_p)

    #加噪声：让数据更像真实测量
    if noise_std > 0:
        pv_p = pv_p * (1.0 + rng.normal(0, noise_std, size=pv_p.shape))

    #把 PV 写回到 net.sgen（关键）
    # 对每个 PV（sgen）：
	# •	把它当前时刻的发电有功 p_mw 写进去
	# •	把它当前无功 q_mvar 写进去
    for i, sid in enumerate(net._pv_ids):
        net.sgen.at[sid, "p_mw"] = float(max(pv_p[i], 0.0)) #避免因为噪声导致 pv_p 变成负数（PV 不会“负发电”）
        net.sgen.at[sid, "q_mvar"] = float(pv_q[i])

    # base PF 先跑一次“没有控制动作的潮流”
    #runpp_robust(...)：尝试用不同算法/初始化/容差/迭代次数去跑潮流（之前写的“鲁棒版 runpp”）
	# •	ok=True：表示潮流收敛，net.res_bus / net.res_line 等结果表会被填好
	# •	ok=False：表示各种组合都没收敛，→ 直接返回 None，并在 info 里写上 "stage": "base_pf"，意思是“在基础潮流阶段失败”
    ok, info0 = runpp_robust(net, cfg, calculate_voltage_angles=True)
    if not ok:
        return None, {"stage": "base_pf", **info0}
    # 初始化两个记录数组
    # curtail_frac[i]：记录第 i 个 PV 被削减（curtail）了多少（这一段还没用到，后面 Curtailment 会用）
	# q_set[i]：记录最后每个 PV 的无功设定值（后面会填）
    curtail_frac = np.zeros(len(net._pv_ids), dtype=float)
    q_set = np.zeros(len(net._pv_ids), dtype=float)
    # 如果开了 control：进入控制逻辑
    # 两个开关：
	# control：这次生成的数据是否“带控制”
	# enable_volt_var：控制里是否启用 Volt-Var（无功调压）
    if control:
        # Volt-Var
        if cfg.enable_volt_var:
            for _ in range(cfg.volt_var_iters):
                vm_series = net.res_bus.vm_pu
                for i, sid in enumerate(net._pv_ids):
                    b = int(net.sgen.at[sid, "bus"]) #找它连接在哪个 bus
                    v = float(vm_series.at[b]) #取这个 bus 的电压
                    #计算这个 PV 当前允许的无功上限（Q 限制）
                    p_now = float(net.sgen.at[sid, "p_mw"]) #这个 PV 现在发了多少有功（MW）
                    #这个 PV 的逆变器视在功率上限 Smax
                    #（之前在 build_net_with_pv 里设置的 smax = inverter_s_over_p * pmax
                    #inverter_q_limit(smax, p）意思是，已经发了 P，就只能剩下一部分能力给 Q，所以可用的最大 |Q| 是： \sqrt{S^2 - P^2}
                    qlim = inverter_q_limit(net._pv_smax[i], p_now)
                    qtar = volt_var_q(v, qlim) #根据电压 v 计算目标无功 qtar（目标Q）（Volt-Var 曲线）

                    #不直接跳到 qtar：做 damping（缓一点）
                    #为什么要 damping？如果每一步都“立刻跳到 qtar”，可能会导致：
                    #电压来回振荡（过冲）,潮流更难收敛
                    #新的 Q 取 60% 目标 + 40% 旧值,让控制更平滑、更稳定。
                    qold = float(net.sgen.at[sid, "q_mvar"])
                    qnew = (1 - cfg.volt_var_damping) * qold + cfg.volt_var_damping * qtar
                    qnew = float(np.clip(qnew, -qlim, +qlim))
                    #再加一个“每步最大变化量”限制（防止跳太大）
                    #volt_var_q_step_frac=0.25 表示：
                    #每一轮迭代，Q 最多变化到 0.25 * qlim
                    dq_max = cfg.volt_var_q_step_frac * qlim
                    if dq_max > 0:
                        qnew = float(np.clip(qnew, qold - dq_max, qold + dq_max))
                    # 把新的无功设定写回 PV 元件
                    # 这一步就是“下发控制动作”。   
                    net.sgen.at[sid, "q_mvar"] = qnew
                #每一轮改完所有 PV 的 Q，就重新跑一次潮流
                #因为你改了 Q，整个系统的电压、潮流都会变，所以必须再跑潮流更新结果。
                #如果此时不收敛，就返回失败，并标记阶段是 "volt_var_pf"，这样你就能统计“失败发生在 Volt-Var 迭代阶段”。
                ok, info_vv = runpp_robust(net, cfg, calculate_voltage_angles=True)
                # 上面代码中Volt-Var 是一个闭环：1.	先看当前电压 V
	            #  2.	根据 V 计算要注入的 Q
	            # 3.	注入 Q 后电压会变
	            # 4.	再根据新电压修正 Q
                # 所以要重复几轮，直到“V 和 Q 比较稳定”。
                if not ok:
                    return None, {"stage": "volt_var_pf", **info_vv}

        # Curtailment （限发/削减）：PV 本来能发 3MW，但为了不让电压过高/不让电力倒灌，就让它只发 2.7MW。
        #每一轮只挑 1 个 PV 来砍一点点 → 重新跑潮流 → 看是否还需要继续砍 → 最多砍 curtail_max_rounds 轮。
        if cfg.enable_curtailment:
            for _round in range(cfg.curtail_max_rounds): #最多尝试 4 次（每次可能砍一个 PV 一点点）
                #先判断“要不要砍”（触发条件）
                vm_series = net.res_bus.vm_pu #取所有 bus 的电压（潮流结果）
                # ext_grid 是“主电网/外部电源”（slack 连接点）
	            # net.res_ext_grid.p_mw 是主电网那边测到的功率流
	            # 如果它 < 0：说明系统在向主电网“送电”
                export = (net.res_ext_grid.p_mw.values[0] < 0)
                #如果某个 bus 电压超过上限 → 过压 overvoltage
                overV = (float(vm_series.max()) > cfg.v_max) if cfg.curtail_on_overvoltage else False
                # 如果满足以下任意一个，就要削减：
                # 有 export 且允许用 export 触发削减：export and cfg.curtail_on_export
                # 有过压：overV
                do_curtail = (export and cfg.curtail_on_export) or overV
                if not do_curtail: #如果都没有，就 break：说明不用再砍了，提前结束（很好，少损失发电）。
                    break

                #收集每个 PV 的“位置电压”和“当前发电功率”
                pv_bus_v, pv_p_now = [], []
                for i, sid in enumerate(net._pv_ids): #之前创建的 PV 列表（sgen id），对每个pv:
                    b = int(net.sgen.at[sid, "bus"]) #找它连在哪个 bus
                    pv_bus_v.append(float(vm_series.at[b])) #记录该 bus 电压
                    pv_p_now.append(float(net.sgen.at[sid, "p_mw"])) #记录该 PV 当前有功
                #然后转成 numpy
                pv_bus_v = np.array(pv_bus_v)
                pv_p_now = np.array(pv_p_now)
                #决定“优先砍谁”（选择策略）
                # 触发原因不同，砍人的标准也不同。
                # 如果是过压 overV=True：砍电压最高处的 PV，最能直接降过压。
                #如果不是过压，而是 export：砍发电最大的 PV，最能减少向外送电。
                order = np.argsort(-pv_bus_v) if overV else np.argsort(-pv_p_now) #按 PV 所在节点电压从高到低排序;按 PV 发电功率从大到小排序

                i = int(order[0])
                sid = net._pv_ids[i]
                # 对选中的 PV 做一次“小幅削减”
                p_now = float(net.sgen.at[sid, "p_mw"])
                if p_now <= 1e-9: #如果它本来就几乎没发电，就没必要砍它，换下一轮。
                    continue
                #然后计算削减后的新功率 p_new：
                p_new_floor = p_now * (1.0 - cfg.curtail_max_frac) #curtail_max_frac=0.50：最多削减到 50%
                #底线是保留 50% 发电,防止“砍过头”
                p_new = max(p_now * (1.0 - cfg.curtail_step), p_new_floor) #curtail_step=0.08：每一轮削减 8%
                # curtail_frac[i]：累计记录“砍了多少”（用于日志/训练标签）
	            # 把这个 PV 的 p_mw 直接改成削减后的功率
                curtail_frac[i] += float(1.0 - (p_new / max(p_now, 1e-9)))
                net.sgen.at[sid, "p_mw"] = p_new
                #砍完必须重新跑潮流（看系统是不是稳定了）
                # 因为改了发电功率 P，整个系统电压/潮流会变
            	# 如果这一步潮流不收敛，就返回失败，并标记 "stage": "curtail_pf"
                ok, info_ct = runpp_robust(net, cfg, calculate_voltage_angles=True)
                if not ok:
                    return None, {"stage": "curtail_pf", **info_ct}

                #为什么削减后还要再跑一次 Volt-Var（半轮版本）？
                # 原因：削减 PV 的 P 以后：
	            # •	PV 的有功变了 → 逆变器能提供的无功裕度 qlim = sqrt(S^2-P^2) 也变了
	            # •	电压也变了 → 原来的 Volt-Var Q 设定可能不再合适
                # 所以用 volt_var_iters//2（比如 4//2=2）做“短一点的 Volt-Var 再调整”，让系统更一致、更稳定。
                #如果这一步不收敛，阶段记为 "post_curtail_voltvar_pf"。
                if cfg.enable_volt_var:
                    for _ in range(max(1, cfg.volt_var_iters // 2)):
                        vm_series = net.res_bus.vm_pu
                        for j, sid2 in enumerate(net._pv_ids):
                            b2 = int(net.sgen.at[sid2, "bus"])
                            v2 = float(vm_series.at[b2])

                            p2 = float(net.sgen.at[sid2, "p_mw"])
                            qlim2 = inverter_q_limit(net._pv_smax[j], p2)
                            qtar2 = volt_var_q(v2, qlim2)

                            qold2 = float(net.sgen.at[sid2, "q_mvar"])
                            qnew2 = (1 - cfg.volt_var_damping) * qold2 + cfg.volt_var_damping * qtar2
                            qnew2 = float(np.clip(qnew2, -qlim2, +qlim2))

                            dq_max2 = cfg.volt_var_q_step_frac * qlim2
                            if dq_max2 > 0:
                                qnew2 = float(np.clip(qnew2, qold2 - dq_max2, qold2 + dq_max2))

                            net.sgen.at[sid2, "q_mvar"] = qnew2

                        ok, info_vv2 = runpp_robust(net, cfg, calculate_voltage_angles=True)
                        if not ok:
                            return None, {"stage": "post_curtail_voltvar_pf", **info_vv2}
        #最后把每个 PV 的最终 Q 保存下来
        for i, sid in enumerate(net._pv_ids):
            q_set[i] = float(net.sgen.at[sid, "q_mvar"])

    # collect aligned results
    #把 pandapower 潮流计算的结果（DataFrame）整理成“对齐好的 numpy 向量”，
    #方便后面做 dataset / GNN 张量。

    # 先拿到 bus 和 line 的“索引顺序”，index 是它们的编号（bus id / line id）
    buses = net.bus.index.to_numpy() #每行一个 bus（母线/节点）
    lines = net.line.index.to_numpy() #每行一条 line（线路/边）
    #必须保证 vm[0] 对应 buses[0]，vm[1] 对应 buses[1]……否则训练时会“节点编号对不上”，模型学的就乱了。

    #从 net.res_bus 里抽 bus 级别结果
    # net.res_bus 是“潮流结果表”，每行对应一个 bus。
    #电压幅值 vm（per-unit），vm_pu：电压幅值（标幺值），例如 1.02 表示比额定高 2%
    vm = net.res_bus.loc[buses, "vm_pu"].to_numpy(dtype=np.float32) 
    #电压相角 va（如果有），有些算法/设置可能不算相角（或者不提供列）
	#如果有就取出来；没有就全置 0
    if "va_degree" in net.res_bus.columns:
        va = net.res_bus.loc[buses, "va_degree"].to_numpy(dtype=np.float32)
    else:
        va = np.zeros(len(buses), dtype=np.float32)
    #节点注入功率 pinj / qinj，p_mw：有功注入，q_mvar：无功注入
    pinj = net.res_bus.loc[buses, "p_mw"].to_numpy(dtype=np.float32)
    qinj = net.res_bus.loc[buses, "q_mvar"].to_numpy(dtype=np.float32)
    # 这个“注入”通常是“发电 - 负荷”，所以可能正也可能负。
	# •	正：该 bus 总体在“往网里送功率”（可能有 PV）
	# •	负：该 bus 总体在“从网里取功率”（负荷为主）

    #从 net.res_line 里抽 line 级别结果
    #net.res_line 是每条 line 的潮流结果表（每行对应一条线路）。
    # (1) 从端功率 p_from / q_from
    # p_from_mw 是“从 from 端看过去”的有功潮流
	# •	如果 p_from_mw 是负数，说明功率方向跟默认方向反过来了（后面就用来判断 RPF）
    p_from = net.res_line.loc[lines, "p_from_mw"].to_numpy(dtype=np.float32)
    q_from = net.res_line.loc[lines, "q_from_mvar"].to_numpy(dtype=np.float32)
    #(2) 线路负载率 loading_percent（如果有）
    if "loading_percent" in net.res_line.columns:
        loading = net.res_line.loc[lines, "loading_percent"].to_numpy(dtype=np.float32)
    else:
        loading = np.zeros(len(lines), dtype=np.float32)
    # 	loading_percent：线路负载率（类似“用了容量的百分之多少”）
	# •	不一定每个网络/模型都有这个列，所以没有就填 0，保持维度一致
    
    # 把一堆数组打包成统一的 res 字典，这些都是 回归目标或特征候选（连续值）。
    res: Dict[str, np.ndarray] = {
        "vm": vm, "va": va, "pinj": pinj, "qinj": qinj,
        "p_from": p_from, "q_from": q_from, "loading": loading,
        # rpf_line：反向潮流线（Reverse Power Flow）
        #若 p_from < 0，说明这条线路上的潮流方向“反了”（相对 from→to）
        # 转成 0/1 标签（int64），这是要预测的一个重要事件标签：哪些线发生了 RPF
        "rpf_line": (p_from < 0).astype(np.int64),
        # v_viol：电压越限的节点（Voltage violation）
        #     若 vm 小于 0.95 或大于 1.05 → 记为 1，否则 0
        # 这是 node-level 的分类标签：哪些节点电压不合规

        "v_viol": ((vm < cfg.v_min) | (vm > cfg.v_max)).astype(np.int64),
        #  export：是否向主网反送，<0 代表系统向外送电（反送到主网）
        # 这里输出是 shape=(1,) 的数组（不是标量）
        "export": np.array([int(net.res_ext_grid.p_mw.values[0] < 0)], dtype=np.int64),
        #记录 PV 的动作与控制结果（用于分析/监督）
        # PV 的最终有功/无功，每个 PV 一个值
        #     pv_p：最终发了多少 MW（可能被削减了）
        # •	pv_q：最终无功（Volt-Var 调出来的）
        "pv_p": np.array([float(net.sgen.at[sid, "p_mw"]) for sid in net._pv_ids], dtype=np.float32),
        "pv_q": np.array([float(net.sgen.at[sid, "q_mvar"]) for sid in net._pv_ids], dtype=np.float32),
        #curtail_frac：每个 PV 被削了多少（累计）
        # 可以分析控制动作有多激进
        "curtail_frac": curtail_frac.astype(np.float32),
        #q_set：最终 Q 设定（刚才那段最后收集的）
        # 一般跟 pv_q 差不多，都是最后的 Q 值
        # 为了后面分析（例如区分“控制策略输出的目标” vs “实际结果”）
        "q_set": q_set.astype(np.float32),
    }

    return res, {"stage": "ok", **info0}
    # res：这一时刻的“观测/结果/标签”打包完毕
    # 第二个 dict 是“状态信息”：
	# stage="ok"：这一步成功
	# info0：来自 runpp_robust 的 algo/init/tol/max_iter 等信息（用来 debug/统计）


# ----------------------------
# Dataset generation (per feeder)
# 基本就是“把一个 feeder（配电网）在很多时间步上跑潮流 +（可选）控制 +（可选）风险事件，
# 然后把每个时间步打包成一条训练样本（给 GNN 用）”。
# ----------------------------

#这个函数“没做任何形状变换”，只是把 pv_act_value 原样返回。
# 以后可以把它扩展成：比如根据 hour 做更复杂映射，或加入某种调度逻辑。
# 目前作用：让代码结构清晰（留接口），但行为是 identity。
def pv_shape_from_mult(pv_act_value: float, hour: int) -> float:
    return float(pv_act_value)


def generate_dataset_for_feeder(
    net: pp.pandapowerNet,
    feeder_name: str,
    cfg: ScenarioConfig,
    control: bool,
    risk_mgr=None,   # RiskManager or None
    scenario_seed: Optional[int] = None,   # ✅ 新增
) -> Tuple[List[Dict], pd.DataFrame]:

    #随机数种子：保证 control / nocontrol 走“同一条天气+负荷轨迹”
    # ✅ 保证 no-control / control 用同一条随机轨迹
    # 之前加的 stable_int_hash(feeder_name) 就是为了 不同 feeder 不同随机轨迹，但每次重跑保持一致。
    #	同一个 feeder 的 nocontrol / control 如果用同样的 scenario_seed，它们的：
	# 云量序列 cloud，PV daily scale，load daily scale，forecast noise，都是一样的。
    # ✅ 这样差异主要来自“控制有没有启用”。
    if scenario_seed is None:
        scenario_seed = int(cfg.seed + stable_int_hash(feeder_name))
    rng = np.random.default_rng(int(scenario_seed))
    # 在网络里“放 PV” + 缓存 base load
    #这句 print 是 debug：告诉这个 feeder 的规模。
    net = build_net_with_pv(net, cfg, rng)
    print(feeder_name, "nbus=", len(net.bus), "nline=", len(net.line), "nload=", len(net.load))

    #时间长度：一共多少步，每一步相当于 1 小时
    # 现在 days=30, steps_per_day=24 → T_total=720

    T_total = cfg.days * cfg.steps_per_day

    # 生成“每天的 PV 和负荷强度”（慢变化）
    # 这一天整体 PV 强一点还是弱一点（0.6~1.8）
    # 这一天整体负荷偏高还是偏低（0.8~1.2）
    # 这是“日级别”的慢变化（类似季节/当天状况）
    pv_daily_scale = rng.uniform(cfg.pv_daily_scale_range[0], cfg.pv_daily_scale_range[1], size=cfg.days)
    load_daily_scale = rng.uniform(cfg.load_daily_scale_range[0], cfg.load_daily_scale_range[1], size=cfg.days)
    # 生成“云量序列 cloud”（快变化 + 随机骤降）
    cloud = simulate_cloud_factor(
        T_total, cfg.cloud_ar, cfg.cloud_sigma, cfg.cloud_drop_prob, cfg.cloud_drop_mag, rng
    )
    # 生成 4 条时间序列：实际 vs 预测

    pv_fc = np.zeros(T_total, dtype=float)
    load_fc = np.zeros(T_total, dtype=float)
    pv_act = np.zeros(T_total, dtype=float)
    load_act = np.zeros(T_total, dtype=float)

    for t in range(T_total): #在 loop 里对每个时间步 t：
        day = t // cfg.steps_per_day #得到当前 day/hour
        hour = t % cfg.steps_per_day
        # 得到当天“形状”
        pv_shape = clear_sky_pv_shape(hour) # 白天像正弦波
        ld_shape = base_load_shape(hour) # 早晚峰
        # 实际值（act）
        # PV 实际 = 日照形状 × 当天强度 × 云量影响
        # 负荷实际 = 负荷形状 × 当天强度（这里没有给负荷加“云”那种快随机）
        pv_act[t] = pv_shape * pv_daily_scale[day] * cloud[t]
        load_act[t] = ld_shape * load_daily_scale[day]
        # 预测值（fc）
        pv_fc[t] = pv_shape * pv_daily_scale[day] * (1.0 + rng.normal(0, cfg.forecast_error_std))
        load_fc[t] = ld_shape * load_daily_scale[day] * (1.0 + rng.normal(0, cfg.forecast_error_std))
        #     预测 = 理想值 × (1 + 预测误差)
        # 预测误差标准差 forecast_error_std=0.06，就是 ±6% 那种量级
        # clip截断到合理范围，防止噪声导致非常离谱的值。
        pv_fc[t] = float(np.clip(pv_fc[t], 0.0, 2.5))
        load_fc[t] = float(np.clip(load_fc[t], 0.2, 2.0))

    #用数组存每个时间步潮流结果/信息（先跑完再组样本）
    solved_res: List[Optional[Dict[str, np.ndarray]]] = [None] * T_total # 每步的 res（如果PF成功）
    solved_info: List[Dict] = [{} for _ in range(T_total)]  # 每步的 info（算法、stage等）
    risk_meta_list: List[Optional[Dict]] = [None] * T_total # 每步的 risk 事件信息

    logs: List[Dict] = [] # 用于 DataFrame summary
    fail = 0
    # 这里是设计上的重点：先把所有时间步潮流算完，再从中抽样本。
	# 	因为要用 last_vm_list[t]（上一时刻电压）作为特征，必须先有上一时刻结果。


    # 主循环：每个时间步都跑潮流（可选风险 + 可选控制）
    for t in range(T_total):
        # ---- risk injection step， risk 注入（如果启用） ----
        #风险管理器可能会：，触发 N-1（断一条线），线路降额，overload 之类
	    #  它可能让负荷临时变大：extra_load > 1
        if risk_mgr is not None:
            risk_meta = risk_mgr.step(net, t)
            risk_meta_list[t] = risk_meta
            extra_load = float(risk_mgr.extra_load_multiplier())
        else:
            risk_meta = None
            extra_load = 1.0
        # 跑潮流+控制
        res, info = run_powerflow_with_controls(
            net=net,
            cfg=cfg,
            load_mult=float(load_act[t]) * extra_load, #load_act[t] 是“这一小时负荷倍率”
            pv_mult=float(pv_shape_from_mult(pv_act[t], hour=t % 24)), #pv_act[t] 是“这一小时 PV 出力倍率”
            cloud_factor=1.0,
            noise_std=cfg.meas_noise_std_pq,
            control=control, #control 决定有没有 volt-var + curtailment
            rng=rng
        )
        # 如果潮流失败
        if res is None:
            fail += 1
            solved_res[t] = None
            solved_info[t] = info
            logs.append({
                "t": t,
                "day": t // cfg.steps_per_day,
                "hour": t % cfg.steps_per_day,
                "pf_ok": 0,
                "stage": info.get("stage", "fail"),
                "error": info.get("error", "")[:240],
                "risk_active": int(risk_mgr is not None and len(risk_mgr.active) > 0) if risk_mgr else 0,
            })
            continue
        # 如果成功
        solved_res[t] = res
        solved_info[t] = info

        logs.append({
            "t": t,
            "day": t // cfg.steps_per_day,
            "hour": t % cfg.steps_per_day,
            "pf_ok": 1,
            "error": "", #这是为了保证 df 里总有 error 列，不会KeyError。
            "stage": info.get("stage", "ok"),
            "pf_algo": info.get("algorithm", ""),
            "pf_init": info.get("init", ""),
            "pf_tol": info.get("tol", np.nan),
            "pf_iter": info.get("max_iter", np.nan),
            "export": int(res["export"][0]),
            "num_rpf_lines": int(res["rpf_line"].sum()),
            "num_v_viol": int(res["v_viol"].sum()),
            "max_vm": float(res["vm"].max()),
            "min_vm": float(res["vm"].min()),
            "pv_total_p": float(res["pv_p"].sum()),
            "pv_total_q": float(res["pv_q"].sum()),
            "curtail_total": float(res["curtail_frac"].sum()),
            "risk_active": int(risk_mgr is not None and len(risk_mgr.active) > 0) if risk_mgr else 0,
        })
    # 统计失败率
    #比如 lv_schutterwald 当时 control=1 全失败，就是这里统计出来的。
    print(f"[{feeder_name} | control={control}] PF failures: {fail}/{T_total} = {fail/max(T_total,1):.3f}")

    # last vm feature：：把上一时刻电压作为特征
    last_vm_list: List[Optional[np.ndarray]] = [None] * T_total
    last_vm_list[0] = None
    for t in range(1, T_total):
        last_vm_list[t] = None if solved_res[t - 1] is None else solved_res[t - 1]["vm"].copy()
        #  last_vm_list[t] = 上一小时的 bus 电压向量
        # 如果上一小时潮流失败，就置 None

        # 这就是一个“时序信息”：告诉模型“刚刚的电压是什么样”，这对预测下一步很有帮助。


    # build samples，真正“生成训练样本 samples”（GNN 在这里登场）
    samples: List[Dict] = []
    for t in range(cfg.horizon, T_total): 
        #从 t = cfg.horizon 开始采样。
        #   如果 horizon=1：从 t=1 开始（因为 t=0 没有 last step）
        # 如果 horizon=k：你可能是在为“预测未来 k 步”留接口（现在还没完全用起来）
        res = solved_res[t]
        #  如果这一刻潮流都没算出来（res=None），直接跳过
        if res is None:
            continue
        # 组 GNN 输入张量 g，GNN 就是在 build_graph_tensors() 这一步正式出现的。
        # 把 pandapower net 变成图学习需要的东西，一般包括：
        # •	x_node: 每个节点的 feature（例如 forecast load、forecast PV、上一时刻电压等）
        # •	edge_index: 图的边连接（from/to bus）
        # •	edge_attr: 每条线的特征（阻抗、额定容量等）
        # •	可能还有一些 mask / mapping
        g = build_graph_tensors(
            net=net,
            cfg=cfg,
            load_forecast_mult=float(load_fc[t]),
            pv_forecast_mult=float(pv_fc[t]),
            last_vm_obs=last_vm_list[t],
            rng=rng
        )
        # 组 edge 的监督目标
        # 这通常会把 line-level 的真实结果（p_from、q_from、loading 等）整理成监督标签。
        y_edge = build_edge_targets_from_res(net, res)
        # 把一切打包成 sample dict
        sample = {
            "meta": {  #meta：用来追溯、调试、复现
                "feeder": feeder_name,
                "t": int(t),
                "day": int(t // cfg.steps_per_day),
                "hour": int(t % cfg.steps_per_day),
                "control": int(control),  #0/1
                "scenario_seed": int(scenario_seed),  #保证可复现
                "cfg": asdict(cfg), #把配置全存下来（后面复现实验很方便）
                "pf_info": solved_info[t], #潮流用了哪个算法/收敛参数
                "risk": risk_meta_list[t],  # None or dict，这一刻有没有风险事件、是什么
            },
            **g,  #GNN 的输入，这是模型要吃的 X（节点/边特征+拓扑）。
            # regression targets
            "y_node_vm": res["vm"].astype(np.float32), #回归（预测电压）
            "y_node_va": res["va"].astype(np.float32), #回归（预测电压）
            **y_edge,
            # derived labels，y_...：监督学习标签（真值）
            "y_export": res["export"].astype(np.int64), #分类（是否向主网反送）
            "y_node_vviol": res["v_viol"].astype(np.int64), #分类（哪些节点越限）
            "y_line_rpf": res["rpf_line"].astype(np.int64), #分类（哪些线反向潮流）
            # actions (analysis) 控制动作记录（分析用）
            # 这些不是必须拿来训练，但你后面做解释性分析、对比 control vs nocontrol 会很有用。
            "pv_p": res["pv_p"],
            "pv_q": res["pv_q"],
            "curtail_frac": res["curtail_frac"],
        }
        samples.append(sample)

    #summary 表：日志 DataFrame
    # summary 就是保存成 summary_xxx.csv 的东西
	# asmples 保存成 pkl，里面是模型训练用的样本列表
    summary = pd.DataFrame(logs)
    return samples, summary


# ----------------------------
# Metrics helpers
# ----------------------------
# summarize_metrics(df)：把一堆逐时日志汇总成“几行总体指标”
def summarize_metrics(df: pd.DataFrame) -> Dict[str, float]:
    # 先只保留潮流成功的行
    # 如果 df 里有 pf_ok 这一列，就只统计 pf_ok==1 的行（成功的潮流结果）
	# 如果没有 pf_ok（极少情况），就直接用 df
    df_ok = df[df["pf_ok"] == 1].copy() if "pf_ok" in df.columns else df
    if len(df_ok) == 0: #如果全部都失败：返回一堆 NaN，没有成功结果，就无法汇总。
        return {
            "export_rate": float("nan"),
            "avg_num_rpf_lines": float("nan"),
            "avg_num_v_viol": float("nan"),
            "max_vm_mean": float("nan"),
            "min_vm_mean": float("nan"),
            "pv_total_p_mean": float("nan"),
            "curtail_total_mean": float("nan"),
            "pv_energy_sum_mwh": float("nan"),
        }

    # 逐个指标算平均或总和
    #注意：
    #现在的写法是 sum(pv_total_p)，这在 每步 = 1 小时 的前提下，
    #数值上等同于 MWh（因为 MWh = MW × 1h）。
    #如果以后改成 15分钟一步（96 steps/day），那这里就必须乘以 Δt（小时）：
    #pv_energy_sum_mwh = df_ok["pv_total_p"].sum() * (24/steps_per_day)
    out: Dict[str, float] = {}
    out["export_rate"] = float(df_ok["export"].mean()) if "export" in df_ok.columns else float("nan")
    out["avg_num_rpf_lines"] = float(df_ok["num_rpf_lines"].mean()) if "num_rpf_lines" in df_ok.columns else float("nan")
    out["avg_num_v_viol"] = float(df_ok["num_v_viol"].mean()) if "num_v_viol" in df_ok.columns else float("nan")
    out["max_vm_mean"] = float(df_ok["max_vm"].mean()) if "max_vm" in df_ok.columns else float("nan")
    out["min_vm_mean"] = float(df_ok["min_vm"].mean()) if "min_vm" in df_ok.columns else float("nan")
    out["pv_total_p_mean"] = float(df_ok["pv_total_p"].mean()) if "pv_total_p" in df_ok.columns else float("nan")
    out["curtail_total_mean"] = float(df_ok["curtail_total"].mean()) if "curtail_total" in df_ok.columns else float("nan")
    out["pv_energy_sum_mwh"] = float(df_ok["pv_total_p"].sum()) if "pv_total_p" in df_ok.columns else float("nan")
    return out
    #“如果列不存在就给 NaN”的防御写法
    #某些 feeder / 某些模式没有某列，也不会报错
    #metrics 仍然能输出，只是缺的那项为 NaN

#把对象序列化保存到磁盘
def save_pickle(obj, path: Path):
    #确保输出文件夹存在（不存在就创建）
    #parents=True：多层目录也一起创建
    #exist_ok=True：目录已经存在也不报错
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f: #以二进制写入方式打开文件（pickle 是二进制格式）
        pickle.dump(obj, f)
        #   把 Python 对象 obj（比如你的 samples 列表）直接“打包”存进去
        # 以后用 pickle.load(open(...,"rb")) 就能原样读取