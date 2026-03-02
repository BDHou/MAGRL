# -*- coding: utf-8 -*-

"""
“在线电网仿真引擎（一步一步走的环境内核）”
负责：给定当前时间 t + 动作 action → 修改电网 → 跑潮流 → 输出图结构观测 obs + 真值 targets + 指标 metrics。

它的作用非常“底层”和关键：后面所有 GNN、多智能体、训练/评估，都要靠它提供数据与交互接口。
你现在项目里大概有三层（从底到顶）：
A. 物理仿真层（你这个 OnlineBackend）
	•	输入：action
	•	内部：写入负荷/光伏P，写入动作Q（可选curtail），跑 pandapower 潮流
	•	输出：
	•	obs_graph：给模型看的输入（图：x, edge_index, edge_attr…）
	•	targets：真值（vm/va/line flow/violations/export…）用于监督学习或评估
	•	metrics：统计指标（max_vm、num_v_viol、export…）用于 reward/报表
	•	meta：时间、seed、risk等调试信息
✅ 所以它是数据生成器 + 在线交互环境。

B. 环境封装层（未来你会有一个 env.py）
	•	把 OnlineBackend.step() 包一层，变成标准 RL / multi-agent 接口：
	•	reset() -> obs
	•	step(action) -> obs, reward, done, info
	•	这里会定义 reward、done 条件、动作裁剪等。

C. 策略/模型层（GNN + Multi-agent）
	•	根据 obs_graph 做推理，输出 action
	•	训练时（RL 或 imitation/SL）更新参数

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import pandapower as pp

from .base_scenario import (
    ScenarioConfig,  #配置类（有 days、steps_per_day、seed、v_min/v_max、噪声、pv规模范围等）
    load_feeder_by_name, #通过名字加载一个 feeder（电网案例）
    stable_int_hash, #把字符串 feeder 名字变成稳定整数（用于 seed）
    build_net_with_pv, #把原始 feeder net0 复制/加工，加上 PV、并保存一些缓存字段（比如 base_load_p、pv_ids…）
    runpp_robust, #更稳定地跑潮流（失败了能给出 info）
    base_load_shape, #给出一天中某个时刻的负荷曲线形状（比如中午高、晚上高）
    clear_sky_pv_shape, #晴天 PV 输出曲线形状（早上上升、中午高、晚上0）
    simulate_cloud_factor, #生成云量/遮挡序列（让 PV 实际输出抖动、偶尔骤降）
)
from .dataset_graph import build_graph_tensors, build_edge_targets_from_res
from .risk_events import RiskConfig, RiskManager
# build_graph_tensors：把 pandapower 的 net 转成 GNN 输入：x / edge_index / edge_attr
# build_edge_targets_from_res：从潮流结果里提取你需要监督学习的标签（比如线路潮流、rpf 等）

@dataclass
#StepResult：每一步返回一个“统一包”
class StepResult:
    ok: bool # bool：潮流算没算成功（True/False）
    info: Dict[str, Any] #任何信息，比如 pf solver 输出、错误原因
    obs_graph: Optional[Dict[str, np.ndarray]]
    #成功：里面有 x / edge_index / edge_attr
	# 失败：给 None（因为没结果，无法构建 obs）
    targets: Optional[Dict[str, np.ndarray]]
    # 成功：线路/节点标签
	# 失败：None
    metrics: Optional[Dict[str, Any]]
    # 成功：你打印的 max_vm、export 等
	# 失败：None
    meta: Dict[str, Any]
    #meta: Dict[str, Any]：一定有，包含 t/day/hour/seed/risk_active 等，方便 debug 和复现实验


class OnlineBackend:
    """
    完整 online backend：
      - 固定 feeder + cfg
      - 内置可复现的时间序列：pv_act/load_act + pv_fc/load_fc + cloud，
      先生成一整条 episode 的时间序列（实际值+预测值）
      - step(action) -> (可选风险) -> 写 net -> run PF -> 输出 obs/targets/metrics
      每一步接收 action → 写入 PV 的 Q（可能还削减 P） → 跑潮流 → 输出
    """

    def __init__(
        self,
        feeder_name: str, #你要用哪个 feeder（比如 "case33bw"）
        cfg: ScenarioConfig, #配置对象，里面有 days、steps_per_day、电压上下限、随机种子、噪声参数等。
        *,#非常重要，它的意思是：后面的参数必须用“关键字方式”传。# 例如你必须写 mode="q_frac"，不能写成位置参数。
        control_enabled: bool = False,  # True: 允许你未来把 cfg.enable_volt_var/cutailment 那套也接上
        #现在基本没用，未来你要接 volt-var 或 curtail 逻辑可以用。
        mode: str = "q_frac",           # action 解释方式："q_frac"，action是[-1,1]的比例，乘以q限幅得到q_mvar
        # | "q_mvar"action 直接就是 q_mvar
        enable_curtail_action: bool = False,  # action 是否包含 curtail 分量action 是否还包含“削减PV有功P”的分量（curtail）
        #False：action 只有 Q,True：action = [Q…, curt…]
        risk_cfg: Optional[RiskConfig] = None, #风险事件配置，如果是 None 就不启用风险
        scenario_seed: Optional[int] = None, #场景随机种子（影响 PV/负荷序列、噪声等）
        risk_seed: Optional[int] = None, #风险随机种子（影响风险事件何时发生）
    ):
        self.feeder_name = feeder_name
        self.cfg = cfg
        self.control_enabled = bool(control_enabled)
        #把你传进来的东西存进对象，后面 step/reset 都要用。
        #bool(control_enabled)：防止你传进来奇怪类型（比如 0/1 或 numpy bool），强制变成 Python 的 True/False。
        self.mode = str(mode) #self.mode = str(mode)：确保 mode 是字符串
        if self.mode not in ("q_frac", "q_mvar"): #if ... not in ...：如果你传错比如 "qfrc"，立刻报错，不让程序“糊里糊涂继续跑”
            raise ValueError("mode must be 'q_frac' or 'q_mvar'")#主动抛异常提醒你参数错了
        self.enable_curtail_action = bool(enable_curtail_action) 
        #同样强制转换成 True/False。,这个会影响 action_dim()：决定 action 有多长。

        # ---- load feeder ----
        net0 = load_feeder_by_name(feeder_name) #net0 = load_feeder_by_name(feeder_name)
        # 调你之前 base_scenario.py 里写的函数，返回一个 pandapower 网络对象（拓扑）。
        # 你可以把 net0 理解为：原始电网蓝图（还没加 PV，还没加你自己的缓存字段）。
        if net0 is None: #如果找不到这个 feeder，就直接报错，不继续。
            raise RuntimeError(f"Feeder '{feeder_name}' not available in your pandapower version.")
        if len(net0.ext_grid) == 0 or len(net0.line) == 0: 
        #     xt_grid 是外部电网/平衡母线，没有它潮流没法解
        # line 是线路，没有线路就不是电网
        # 所以做一个基本 sanity check，防止后面跑潮流才发现错。
            raise RuntimeError(f"Feeder '{feeder_name}' missing ext_grid or line.")

        # ---- seeds (stable & reproducible) ----这段非常关键，决定你每次跑是不是“可复现”。
        if scenario_seed is None: #•	如果你没手动指定 seed，就自动生成一个。
            scenario_seed = int(cfg.seed + stable_int_hash(feeder_name)) 
            # cfg.seed：你 config 里给的基准种子
            # stable_int_hash(feeder_name)：把 feeder 名字（比如 case33bw）稳定地变成一个整数
            # 两个加起来 -> 不同 feeder 也会有不同 seed，这样不同 feeder 的随机序列不会完全一样。
        self.scenario_seed = int(scenario_seed) #存下来，方便你打印 debug 或复现
        self.rng = np.random.default_rng(self.scenario_seed) 
        #创建一个 numpy 的随机数生成器（推荐用法）后面生成 PV/负荷、噪声、cloud 都用它。

        # risk
        self.risk_cfg = risk_cfg #保存风险配置
        if risk_cfg is not None: #如果你给了风险配置，才启用风险模块
            if risk_seed is None: #如果你没给 risk_seed，就自动生成一个
                risk_seed = int(cfg.seed + 999 + stable_int_hash(feeder_name))
                #和 scenario_seed 类似，但额外加 999，目的是：让 risk 的随机序列和 scenario 的随机序列分开（不互相干扰）
            self.risk_seed = int(risk_seed) #保存
            self.risk_mgr = RiskManager(risk_cfg, np.random.default_rng(self.risk_seed))
            #创建一个风险管理器，它内部会按时间 step 决定要不要触发某些风险
        else: #你不启用风险时，后面 step() 会走无风险路径
            self.risk_seed = None
            self.risk_mgr = None

        # ---- build net with PV + caches ----真正构建 online 要用的 net（加 PV + 缓存字段）
        self.net: pp.pandapowerNet = build_net_with_pv(net0, cfg, self.rng)
        #  用 “原始电网蓝图 net0”
        # 按 cfg 的设置把 PV（sgen）加进去
        # 并且在 net 上挂一些你自己加的缓存字段（例如 _pv_ids, _pv_pmax, _base_load_p 等）
        # 这一步很关键：OnlineBackend 需要一个“可修改、可持续 step 的活 net”
        self.n_pv = len(self.net._pv_ids)
        #   _pv_ids 是你在 build_net_with_pv 里自己加的：代表 PV 对应的 sgen 行索引
        #	有几个 PV，就决定 action_dim 是多少

        # ---- episode time axis ----episode 时间轴相关
        self.T_total = int(cfg.days * cfg.steps_per_day)
        # cfg.days：episode 有几天,cfg.steps_per_day：一天多少步（比如 48 表示半小时一格）
        #T_total：总步数,
        self.t = 0  # current timestep index,self.t = 0：当前时间步从 0 开始

        # ---- pre-generate episode profiles (act + forecast) ----预生成整条 episode 的时间序列（关键）
        self._build_profiles()
        #    这行调用下面的 _build_profiles()。意思：你一创建 backend，就把这一整条 episode 的：
        # 	PV 实际倍率 pv_act，负荷实际倍率 load_act，PV 预测倍率 pv_fc，负荷预测倍率 load_fc，cloud 云遮挡
        # 全部生成好，存在对象里。
        # 这样 step 时，只要 t 取对应位置就行，逻辑非常干净。

        # ---- last vm ----
        self.last_vm: Optional[np.ndarray] = None
        #last_vm：上一时刻的电压（vm_pu）。
        # 你在 obs 里可能要给模型“上一时刻电压”，所以保存着。
        #     初始为 None，因为 reset 之前没有上一时刻。


        # ---- last action (debug) ----
        self.last_action: Optional[np.ndarray] = None
                # •	last_action：只是 debug 用，你可以看看上一步 action 是啥。
    # -----------------------
    # Episode profiles
    # -----------------------
    def _build_profiles(self):
        cfg = self.cfg
        T_total = self.T_total
        rng = self.rng
        # cfg、T_total、rng 只是写个短变量名，后面代码更好读。不然每行都写 self.cfg.xxx 很长。

        pv_daily_scale = rng.uniform(cfg.pv_daily_scale_range[0], cfg.pv_daily_scale_range[1], size=cfg.days)
        load_daily_scale = rng.uniform(cfg.load_daily_scale_range[0], cfg.load_daily_scale_range[1], size=cfg.days)
        #   每一天 PV 的“总体强度”随机取一个（比如今天太阳强一点/弱一点）
        # 每一天负荷的“总体强度”随机取一个（比如今天整体用电高/低）
        cloud = simulate_cloud_factor(
            T_total, cfg.cloud_ar, cfg.cloud_sigma, cfg.cloud_drop_prob, cfg.cloud_drop_mag, rng
        )
        #   cloud 是长度 T_total 的数组，每个时刻一个值（例如 0.6~1.0）
        # 它会让 PV 实际值下降：pv_act[t] = pv_shape * daily_scale * cloud[t]
        # 参数 cloud_ar / cloud_sigma / drop_prob / drop_mag
        # 是你在 config 里定义的云变化特性。

        #准备四个数组（全部先填 0）
        pv_fc = np.zeros(T_total, dtype=float) #PV forecast（预测）
        load_fc = np.zeros(T_total, dtype=float) #load forecast
        pv_act = np.zeros(T_total, dtype=float) #PV actual（真实）
        load_act = np.zeros(T_total, dtype=float) #load actual
        #因为后面要逐个 t 填进去

        # 循环每一个时间步，填值
        for t in range(T_total): #t 从 0 到 T_total-1
            day = t // cfg.steps_per_day #整数除法，算出这是第几天
            hour = t % cfg.steps_per_day #取余数，算出这是一天里的第几个时间格（你叫 hour，但其实是“时间槽 index”）

            # 得到“日内形状”
            pv_shape = clear_sky_pv_shape(hour) #晴天 PV 曲线形状，见base_scenario
            #夜晚接近 0,中午最高
            ld_shape = base_load_shape(hour) #负荷曲线形状
            #早晚用电高,白天可能低一些（取决于你定义）

            pv_act[t] = pv_shape * pv_daily_scale[day] * cloud[t] #PV 实际值 = 晴天曲线形状 × 当天PV强度 × 云遮挡
            load_act[t] = ld_shape * load_daily_scale[day] #负荷实际值 = 负荷曲线形状 × 当天负荷强度
            #（负荷这里没有乘 cloud，因为云主要影响太阳，不影响负荷）

            #计算“预测值 forecast”（加入预测误差）
            pv_fc[t] = pv_shape * pv_daily_scale[day] * (1.0 + rng.normal(0, cfg.forecast_error_std))
            load_fc[t] = ld_shape * load_daily_scale[day] * (1.0 + rng.normal(0, cfg.forecast_error_std))
            #    (1.0 + rng.normal(0, std))：乘上一个“接近 1 的随机数”
            # •	比如 1.05 表示预测偏高 5%
            # •	0.95 表示预测偏低 5%
            # •	注意：预测值没有乘 cloud（因为预测不可能知道真实云遮挡），这是合理的。
            #pv_shape * pv_daily_scale[day]乘起来就是：如果今天是晴天（没有云遮挡），你预计这个时刻 PV 大概有多强
            #ld_shape * load_daily_scale[day]，乘起来就是：你预计这个时刻负荷大概有多大
            pv_fc[t] = float(np.clip(pv_fc[t], 0.0, 2.5))
            load_fc[t] = float(np.clip(load_fc[t], 0.2, 2.0))
            #         np.clip(x, a, b)：把 x 限制在 [a,b] 内
            # •	这样避免 forecast 出现负数、或者过大导致潮流跑崩
            #为什么 PV 上限 2.5、负荷下限 0.2？
            #这是你人为设的“合理倍率范围”,防止模型训练/online 测试出现极端值
            #forecast = 晴天曲线 × 当天强度 × (1 ± 预测噪声)，再用 clip 做安全保护。

        #保存到 self 上
        self.cloud = cloud
        self.pv_act = pv_act
        self.load_act = load_act
        self.pv_fc = pv_fc
        self.load_fc = load_fc
        #     以后 step 只需要：
        # •	pv_mult = self.pv_act[t]
        # •	load_mult = self.load_act[t]
        # •	obs 里用 pv_fc[t]、load_fc[t]

    def get_episode_profiles(self) -> Dict[str, np.ndarray]:
        return {
            "cloud": self.cloud.copy(),
            "pv_act": self.pv_act.copy(),
            "load_act": self.load_act.copy(),
            "pv_fc": self.pv_fc.copy(),
            "load_fc": self.load_fc.copy(),
        }

    # -----------------------
    # Reset & Step
    # -----------------------
    #reset()把环境回到某个时间点 t0，清空历史状态（比如 last_vm），并返回一个“初始观测 obs”
    def reset(self, t0: int = 0) -> StepResult:
        """
        重置到 t=t0，并清空 last_vm。风险 manager 的 active 也清空（重新来一条 episode）。
        """
        t0 = int(t0) #把 t0 强制变成整数
        if not (0 <= t0 < self.T_total): 
            #检查 t0 合法不合法
            # self.T_total 是总步数，比如 2 天 × 每天 48 步 = 96 步
            # 所以 t0 必须在 [0, T_total-1] 之间
            raise ValueError(f"t0 must be in [0, {self.T_total-1}]")

        self.t = t0 #把当前时间指针移到 t0
        self.last_vm = None #清空“上一时刻电压”的缓存
        #因为你后面 obs 会用到 last_vm（作为历史信息），reset 后不应该带着上一条 episode 的 last_vm
        self.last_action = None #清空上一步 action，仅仅是 debug 用


        #     我们有一个风险系统 RiskManager，里面可能会有“正在发生的风险事件”列表 active：
        # reset 的时候必须清空，否则上一条 episode 的风险会“穿越”到下一条 episode
        # 	active.clear() 就是把列表清空
        if self.risk_mgr is not None:
            self.risk_mgr.active.clear()

        # 直接做一次“无动作”step，生成初始 obs（相当于 env.reset() 给 obs）
        # 这里默认 action=0
        action_dim = self.action_dim()
        action0 = np.zeros(action_dim, dtype=float)
        return self.step(action0, treat_as_reset=True) 
        #treat_as_reset=True 是干嘛？它是为了告诉 step：,
        #“这次 step 是 reset 触发的，不要把时间推进到 t0+1”。
        #         你后面的 step() 里有一句：
        # 	•	如果不是 reset 才 self.t += 1

        # 这样 reset 返回的那一步结果的 meta["t"] 还是 t0，不会跳过去。

    #action_dim()：告诉你 action 长度是多少（你要传给 step 的 action 数组多长）
    #重点来了：为什么 reset 里要 step 一次？
    # 这几行做什么？
	# •	先算 action 需要多长
	# •	构造一个“全 0 的 action”：
	# •	如果 action 只含 Q：那就是所有 PV 的 Q 都设为 0
	# •	如果 action 还含 curtail：那就是 Q=0 且 curtail=0（不削减）
	# •	然后直接执行一次 step()，把 t0 时刻的潮流跑出来，生成 obs/targets/metrics
    #     为什么要这么做？

    # 因为你的 reset 要返回一个“完整的初始观测 obs”。
    # 但你这个系统的 obs 不是“静态就有”，它依赖于：
    # 	•	写入 load/pv 的状态
    # 	•	（可选）风险注入
    # 	•	跑一次潮流 runpp
    # 	•	然后才能从 net.res_bus / net.res_line 得到电压、潮流等
    # 	•	才能 build obs_graph/targets/metrics

    # 所以：不跑一次 step，你拿不到初始 obs。

    def action_dim(self) -> int:
        """
        如果 enable_curtail_action=False:
            action = [pv0_q, pv1_q, ...]  (n_pv)
        否则：
            action = [pv0_q,...,pv_{n-1}_q, pv0_curt,...,pv_{n-1}_curt] (2*n_pv)
            self.n_pv = PV 的数量，比如 3 个 PV
        •	如果 enable_curtail_action=False：
        •	action 只控制 Q
        •	action 长度 = n_pv
        •	例如：[q0, q1, q2]
        •	如果 enable_curtail_action=True：
        •	action 既控制 Q，又控制 curtail（削减 P）
        •	action 长度 = 2*n_pv
        •	例如：[q0,q1,q2, curt0,curt1,curt2]

        最后return那行就是用一个很短的表达式实现这个逻辑。
        """
        return self.n_pv * (2 if self.enable_curtail_action else 1)

    #_split_action()：如果 action 里包含两部分（Q + curtail），把它切开
    def _split_action(self, action: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        a = np.asarray(action, dtype=float).reshape(-1)
        #  把你传进来的 action：,强制变成 numpy 数组,强制变成 float,.reshape(-1)：拉平成一维向量
        # 无论你传 (3,) 还是 (1,3) 都变成 (3,)
        if a.shape[0] != self.action_dim():        # 这行是“防呆”
            raise ValueError(f"action dim mismatch: got {a.shape[0]}, expect {self.action_dim()}")
        # 如果你传的 action 长度不对，就立刻报错。
        # 比如你有 3 个 PV，但你只传了 2 个数，那后面写 PV Q 时就会错，所以提前拦截。
        if not self.enable_curtail_action:
            return a, None
            # 这行意思:如果你没启用 curtail action，那 action 就全是 Q：
            # 返回 (q_action, curt_action),这里 curt_action 不存在，所以返回 None
        q = a[: self.n_pv] #这三行是“切 action”
        curt = a[self.n_pv :]
        return q, curt

# _write_state()。这一段非常关键：它负责在当前时间步 t，把“真实的负荷/光伏有功功率 P
#”写进 pandapower 的网络 net 里。无功 Q 不在这里写，Q 由后面的 _apply_action() 根据 action 写进去。
    def _write_state(self, load_mult: float, pv_mult: float, extra_load: float):
        """
        写入该时刻的 load & pv P（Q由 action 写）
        # load_mult：负荷倍率（来自时间序列 self.load_act[t]）
        pv_mult：光伏倍率（来自 self.pv_act[t]，里面已经乘了 cloud）
        extra_load：风险系统带来的额外负荷倍率（来自风险注入）
        """
        net = self.net
        cfg = self.cfg
        #为了少写 self.net self.cfg，方便后面代码更短、更清晰。

        # ---- load ----
        p_load = net._base_load_p * float(load_mult) * float(extra_load)
        # net._base_load_p：你在 build_net_with_pv() 时缓存的“每个负荷的基准 P 值”（一个数组，长度等于 load 数量）
        # 例如每个 load 基准是 [0.05, 0.08, 0.03, ...] MW
        # 乘上 load_mult：表示一天内不同时间的用电变化
        # 乘上 extra_load：表示风险事件造成的额外负荷
        # 所以 p_load 是当前时刻每个负荷应该用多少有功功率 P。
        """
        这段在算负荷的无功 Q（q_mvar）
        这里有两种情况：
        情况 A：你的 feeder 基准里 没有给负荷 Q（全是 0）
        np.allclose(net._base_load_q, 0.0) 为 True
        那就用一个简单的规则“按比例生成 Q”：
            •	q_load = p_load * cfg.load_q_over_p
        cfg.load_q_over_p 就是你设定的 Q/P 比例（比如 0.3）
            •	如果某个 load 的 P=1.0 MW
        那 Q=0.3 Mvar
        这很常见：很多测试 feeder 只有 P，没有 Q，就用一个固定功率因数的近似。

        情况 B：你的 feeder 基准里 本来就有负荷 Q
        那就按同样的倍率缩放基准 Q：
            •	q_load = base_q * load_mult * extra_load    
        """
        if np.allclose(net._base_load_q, 0.0):
            q_load = p_load * float(cfg.load_q_over_p)
        else:
            q_load = net._base_load_q * float(load_mult) * float(extra_load)


        # 这两行是“真正写回 pandapower 网络”
        net.load.loc[:, "p_mw"] = p_load
        net.load.loc[:, "q_mvar"] = q_load

        # ---- pv P ----
        pv_p = net._pv_pmax * float(pv_mult) #每个 PV 的“最大有功功率”（一个数组，比如 [1.0, 0.8, 1.2] MW）
        #这一时刻 PV 的倍率（来自 pv_act[t]，包含白天形状 + 日尺度 + cloud）
        #   所以这里是在算：
        # 当前时刻每个 PV 实际注入的有功 P：P_now = Pmax * pv_mult

        #加入测量噪声（可选）
        if cfg.meas_noise_std_pq > 0:
            pv_p = pv_p * (1.0 + self.rng.normal(0, cfg.meas_noise_std_pq, size=pv_p.shape))

        # 写回 写回 pandapower 的 net.sgen
        for i, sid in enumerate(net._pv_ids):
            net.sgen.at[sid, "p_mw"] = float(max(pv_p[i], 0.0))
            net.sgen.at[sid, "q_mvar"] = 0.0
        # 你“为什么这里要清零？不是后面要写 Q 吗？”
        # 答案：因为 _write_state() 的职责是“把环境状态写好（P、负荷等）”，它先把 Q 置成一个“干净的默认值”。
        # 随后 _apply_action() 才根据 action 写真正的 Q。


    #这里就是“Q 由 action 写进去”的地方
    # q_action：每个 PV 一个值，用来控制 Q，curt_action：可选，每个 PV 一个值，用来削减 P（你现在通常没用）
    #q_mvar：action 直接就是 q_mvar，q_frac：action 是比例（-1 到 1），乘以可用的最大 Q（qlim）得到 q_mvar
    def _apply_action(self, q_action: np.ndarray, curt_action: Optional[np.ndarray]):
        """
        把 action 写入 PV 的 q_mvar（并可选 curtail PV 的 p_mw）
        - mode="q_mvar": 直接认为 action 是 Mvar
        - mode="q_frac": action ∈ [-1,1]，映射到 qlim * frac
        """

        #拿 net / cfg
        net = self.net
        cfg = self.cfg

        # 确保 q_action 是一维，并且长度 = PV 数量
        q_action = np.asarray(q_action, dtype=float).reshape(-1)
        if q_action.shape[0] != self.n_pv:
            raise ValueError("q_action size mismatch")

        #如果 curtail 不为空，也检查长度
        if curt_action is not None:
            curt_action = np.asarray(curt_action, dtype=float).reshape(-1)
            if curt_action.shape[0] != self.n_pv:
                raise ValueError("curt_action size mismatch")

        # 逐 PV 写 Q (and maybe curtail P)开始逐个 PV 写 Q（重点在这一段）
        #
        for i, sid in enumerate(net._pv_ids):
            p_now = float(net.sgen.at[sid, "p_mw"])
            #读取这个 PV 当前时刻的有功 P，注意意：这个 P 刚刚在 _write_state() 写进去的
            smax = float(net._pv_smax[i])
            #这个 PV 的额定视在功率上限 Smax（单位 MVA）
            #net._pv_smax 是你在 build_net_with_pv() 的时候存进去的“缓存数组”
            qlim = float(np.sqrt(max(smax * smax - p_now * p_now, 0.0)))
            #PV 当前已经输出了 P，那么它剩余的“无功能力”就被限制了。# 直觉：P 越大，可用 Q 越小；P=0 时，可用 Q 最大。
            if self.mode == "q_mvar": #如果 mode="q_mvar"
                qset = float(np.clip(q_action[i], -qlim, +qlim))
                #你给的 action 就是想设的 Q（单位 Mvar）
	            # 但不能超过 [-qlim, +qlim]，所以 clip 
            else:
                #如果 mode="q_frac"
                # q_frac: clamp [-1,1] then multiply qlim，action 先限制在 [-1, 1]
                #然后乘以 qlim 变成真正的 Q
                #action=+1 → 用满正向无功能力，action=-1 → 用满负向无功能力，action=0 → Q=0
                #
                frac = float(np.clip(q_action[i], -1.0, +1.0))
                qset = frac * qlim

            net.sgen.at[sid, "q_mvar"] = qset

            # optional curtailment action on P
            if curt_action is not None:
                # curt_action ∈ [0,1] 解释为 “削减比例”
                # 例如 0.2 -> 把 p_now 变成 p_now*(1-0.2)
                cf = float(np.clip(curt_action[i], 0.0, 1.0))
                # 给个底线（和 cfg.curtail_max_frac 对齐）：最多削到 50%
                floor_frac = float(1.0 - cfg.curtail_max_frac)
                p_new = max(p_now * (1.0 - cf), p_now * floor_frac)
                net.sgen.at[sid, "p_mw"] = float(p_new) #真正写入 Q 的那一行
                #sid：PV 在 sgen 表的行，"q_mvar"：pandapower 里无功字段，
                #qset：我们刚计算好的 Q（保证不超过物理上限）
    """
    _read_metrics() 的作用
    它把潮流（PF）跑完后，net.res_* 里面的结果读出来，算成我们关心的指标
    （电压越界、反向潮流、是否外送、PV 总 P/Q 等），打包成一个 dict 返回。
    _read_metrics：内部函数（下划线开头），只给 OnlineBackend 自己用
    返回 Dict[str, Any]：一个字典，key 是字符串，value 可以是任何类型（float/int/array 等）
    """
    def _read_metrics(self) -> Dict[str, Any]:
        net = self.net
        #net：pandapower 网络（包含 bus、line、load、sgen，以及潮流结果 res_bus/res_line/res_ext_grid）
        cfg = self.cfg
        #cfg：你定义的 ScenarioConfig，里面有 v_min, v_max 等阈值

        #拿 bus 和 line 的索引列表
        buses = net.bus.index.to_numpy()
        lines = net.line.index.to_numpy()

        #读 bus 电压 vm_pu 并计算越界
        vm = net.res_bus.loc[buses, "vm_pu"].to_numpy(dtype=np.float32)
        v_viol = ((vm < cfg.v_min) | (vm > cfg.v_max)).astype(np.int64)

        p_from = net.res_line.loc[lines, "p_from_mw"].to_numpy(dtype=np.float32)
        rpf_line = (p_from < 0).astype(np.int64)

        export = int(net.res_ext_grid.p_mw.values[0] < 0)

        pv_p = np.array([float(net.sgen.at[sid, "p_mw"]) for sid in net._pv_ids], dtype=np.float32)
        pv_q = np.array([float(net.sgen.at[sid, "q_mvar"]) for sid in net._pv_ids], dtype=np.float32)

        out = {
            "max_vm": float(vm.max()),
            "min_vm": float(vm.min()),
            "num_v_viol": int(v_viol.sum()),
            "export": int(export),
            "num_rpf_lines": int(rpf_line.sum()),
            "pv_total_p": float(pv_p.sum()),
            "pv_total_q": float(pv_q.sum()),
            "pv_p": pv_p,
            "pv_q": pv_q,
        }
        return out

    def _build_obs_targets(self, t: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        obs_graph = “模型能看到的输入”（用 forecast + 上一步电压）
        targets = “这一步潮流跑出来的真相/答案”（训练监督用）
        obs 用 forecast + last_vm
        targets 用本次 PF res（直接从 net.res_* 读，复用 build_edge_targets_from_res）
        """
        cfg = self.cfg

        # 构建 obs_graph（给模型看的输入）
        obs_graph = build_graph_tensors(
            net=self.net, #当前电网（pandapower net），里面有 bus/line/load/sgen。
            cfg=cfg, #配置（比如是否加噪声、用哪些特征等）。
            load_forecast_mult=float(self.load_fc[t]),
            #负荷预测倍率（forecast）。注意：它不是实际 load，是 “预测的负荷大小”。
            pv_forecast_mult=float(self.pv_fc[t]),
            #光伏预测倍率（forecast）。
            last_vm_obs=self.last_vm,
            #上一时刻的电压（last voltage magnitude）。
            # 这很常见：你让模型“记住一点历史信息”，相当于给它一个很轻量的 memory。
            rng=self.rng, #如果 build_graph_tensors 会加噪声或随机扰动，用同一个 rng 保证可复现。
        )
        """
        这一步得到的 obs_graph 通常包含：
        •	obs_graph["x"]：每个 bus 的特征矩阵（节点特征）
        •	obs_graph["edge_index"]：图的边连接关系
        •	obs_graph["edge_attr"]：每条线的特征
        前面打印的的shape：x=(33, 8), edge_index=(2, 74), edge_attr=(74, 4)
        """

        # 用“和 base_scenario 里 res 同键名”的结构喂给 build_edge_targets_from_res
        #准备 targets（用潮流结果当“标准答案”）
        #先拿 bus/line 的 index 列表
        buses = self.net.bus.index.to_numpy()
        lines = self.net.line.index.to_numpy()

        #从 res_bus 读取节点电压 vm
        vm = self.net.res_bus.loc[buses, "vm_pu"].to_numpy(dtype=np.float32)
        # 读取电压相角 va（可能没有）
        #有些 pandapower 设置/模型可能不输出相角（或者你没开 calculate_voltage_angles）。
        if "va_degree" in self.net.res_bus.columns:
            va = self.net.res_bus.loc[buses, "va_degree"].to_numpy(dtype=np.float32)
        else:
            va = np.zeros(len(buses), dtype=np.float32)

        # 读取每个 bus 的注入功率 pinj/qinj
        # p_mw/q_mvar 在 res_bus 里代表“该母线的净注入”（发电-负荷）。
        pinj = self.net.res_bus.loc[buses, "p_mw"].to_numpy(dtype=np.float32)
        qinj = self.net.res_bus.loc[buses, "q_mvar"].to_numpy(dtype=np.float32)

        # 读取每条线的潮流 p_from/q_from
        #     res_line.p_from_mw：线路从 “from bus” 端流出的有功功率。
        # •	如果 p_from < 0，说明方向反了（你后面就用它判 RPF）。
        p_from = self.net.res_line.loc[lines, "p_from_mw"].to_numpy(dtype=np.float32)
        q_from = self.net.res_line.loc[lines, "q_from_mvar"].to_numpy(dtype=np.float32)

        # 读取 loading_percent（可能没有）
        #     	•	有些模型/版本会算线路负载率（loading%），有些没有。
        # •	没有就补 0（保持维度一致）。
        if "loading_percent" in self.net.res_line.columns:
            loading = self.net.res_line.loc[lines, "loading_percent"].to_numpy(dtype=np.float32)
        else:
            loading = np.zeros(len(lines), dtype=np.float32)

        # 把这些结果打包成 res 字典
        res = {
            "vm": vm, "va": va, "pinj": pinj, "qinj": qinj,
            "p_from": p_from, "q_from": q_from, "loading": loading,
        }
        # 复用 build_edge_targets_from_res
        targets = build_edge_targets_from_res(self.net, res)
        # 你也可以顺手把 node-level 真值一起挂在 targets 里（训练/监督更方便）
        # 再额外挂一些 node-level / event-level 标签（更方便训练）
        targets["y_node_vm"] = vm.astype(np.float32)
        targets["y_node_va"] = va.astype(np.float32)
        # 电压越界标签 y_node_vviol
        targets["y_node_vviol"] = ((vm < cfg.v_min) | (vm > cfg.v_max)).astype(np.int64)
        # 是否外送 y_export
        targets["y_export"] = np.array([int(self.net.res_ext_grid.p_mw.values[0] < 0)], dtype=np.int64)
        # 线路反向潮流 y_line_rpf
        targets["y_line_rpf"] = (p_from < 0).astype(np.int64)

        return obs_graph, targets
        #最后返回
        #  obs_graph：给你的 GNN / policy 输入
        # targets：给你监督学习、评估、debug 用

    """
    拿风险（可选）→ 写入真实负荷/光伏P 
    → 写入动作Q（可选削减P）→ 跑潮流 → 读结果 → 组装 obs/targets → 更新时间
    """
    def step(self, action: np.ndarray, treat_as_reset: bool = False) -> StepResult:
        """
        执行一步：
          - 风险注入（可选）
          - 写入负荷/PV P
          - 写 action -> PV Q (+可选 curtail P)
          - 跑 PF（robust）
          - 输出 obs/targets/metrics
        """
        cfg = self.cfg
        t = int(self.t)

        # meta
        meta: Dict[str, Any] = {
            "feeder": self.feeder_name,
            "t": int(t),
            "day": int(t // cfg.steps_per_day),
            "hour": int(t % cfg.steps_per_day),
            "scenario_seed": int(self.scenario_seed),
            "risk_seed": int(self.risk_seed) if self.risk_seed is not None else None,
            "mode": self.mode,
            "enable_curtail_action": int(self.enable_curtail_action),
        }

        # 1) risk injection
        if self.risk_mgr is not None:
            risk_meta = self.risk_mgr.step(self.net, t)
            extra_load = float(self.risk_mgr.extra_load_multiplier())
            meta["risk"] = risk_meta
            meta["risk_active"] = int(len(self.risk_mgr.active) > 0)
            meta["extra_load"] = float(extra_load)
        else:
            extra_load = 1.0
            meta["risk"] = None
            meta["risk_active"] = 0
            meta["extra_load"] = 1.0

        # 2) write base state (act profiles)
        load_mult = float(self.load_act[t])
        pv_mult = float(self.pv_act[t])  # cloud already in pv_act
        self._write_state(load_mult=load_mult, pv_mult=pv_mult, extra_load=extra_load)

        # 3) apply action
        q_action, curt_action = self._split_action(action)
        self._apply_action(q_action=q_action, curt_action=curt_action)
        self.last_action = np.asarray(action, dtype=float).copy()

        # 4) run PF
        ok, pf_info = runpp_robust(self.net, cfg, calculate_voltage_angles=True)
        info = {"stage": "ok" if ok else "pf_fail", **pf_info}

        if not ok:
            # 失败：不更新 last_vm，obs/targets/metrics 都返回 None（由 env 决定怎么处理）
            if not treat_as_reset:
                self.t = min(self.t + 1, self.T_total)  # 仍推进时间，避免死循环
            return StepResult(
                ok=False,
                info=info,
                obs_graph=None,
                targets=None,
                metrics=None,
                meta=meta,
            )

        # 5) read metrics
        metrics = self._read_metrics()

        # 6) build obs & targets
        obs_graph, targets = self._build_obs_targets(t)

        # 7) update last_vm (用于下一步 obs)
        self.last_vm = targets["y_node_vm"].copy()

        # 8) advance time
        if not treat_as_reset:
            self.t += 1

        return StepResult(
            ok=True,
            info=info,
            obs_graph=obs_graph,
            targets=targets,
            metrics=metrics,
            meta=meta,
        )