# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandapower as pp


@dataclass
class RiskConfig:
    #第一类风险风险：contingency（故障/跳闸事件）
    enable_contingency: bool = False  #False：不启用这类风险，True：启用“随机跳闸/故障”事件
    contingency_type: str = "n-1"          # "n-1" | "n-k"
    # N-1：系统里有 N 个关键设备（比如线路），随机掉 1 个（模拟单点故障）
	# N-k：随机掉 k 个（更严重、更极端）

    contingency_k: int = 1
    contingency_prob_per_step: float = 0.02 #每一个时间步触发故障的概率
    #例如 0.02：每小时有 2% 概率发生一次故障，注意：这不是“每天 2%”，而是“每一步 2%”。
    contingency_duration_steps: int = 6 #故障持续多久（多少步之后恢复）
    #例如 6：持续 6 个时间步
    contingency_elements: Tuple[str, ...] = ("line",)  # 表示“故障发生在哪类设备上”：("line","trafo","sgen")
    #("line"线路跳闸（最常见）,"trafo"（变压器故障）,"sgen"（分布式电源掉线）)
    # 现在默认只有 ("line",)，就是只让线路随机跳闸，
    avoid_islanding: bool = False  # 可先False；True需要做更复杂连通性筛选
    #avoid_islanding=False 的意思是：暂时不管会不会造成孤岛，随机挑就行（实现简单，但容易让潮流失败）
    #True 的话要额外做图连通性检查（更复杂），确保选的线路不会把网络切断成孤岛。
    
    #第二类风险：line_derating（线路降额）
    # 这类风险模拟现实里：
	# 天气热、设备老化、维修限制等原因
    # 导致某些线路“可承受容量下降”（更容易过载）
    enable_line_derating: bool = False #是否启用线路降额
    derate_prob_per_step: float = 0.02 #每个时间步触发“降额事件”的概率
    derate_duration_steps: int = 6 #降额持续多少步后恢复
    derate_range: Tuple[float, float] = (0.7, 1.0)
    # 降额倍数范围：
	# 比如随机抽到 0.75：线路容量变成原来的 75%
	# 1.0 表示不降额（上界通常是 1.0）


    #第三类风险：overload（负荷突增/过载压力）
    # 突发的大负荷（工厂启动、EV 快充潮、极端天气导致空调暴增）
	# 使得系统突然“更重载”
    enable_overload: bool = False #是否启用负荷突增风险
    overload_prob_per_step: float = 0.02 #每步触发概率
    overload_duration_steps: int = 6 #持续多久
    overload_mult_range: Tuple[float, float] = (1.2, 1.8) #负荷乘以一个倍数：
    #1.2：负荷增加 20%，1.8：负荷增加 80%


class RiskManager:
    """
    状态机：
      - 每个 step 可能触发一个 contingency / derating / overload 事件
      - 事件持续 duration_steps，然后自动恢复
      内部记住现在有什么事件正在生效（active），并且会在持续时间结束后自动恢复。
    """

    def __init__(self, cfg: RiskConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

        self.active: Dict[str, Dict] = {}  # event_name -> event_state dict

    # --------------------------
    # Public API
    # --------------------------
    def step(self, net: pp.pandapowerNet, t: int) -> Dict:
        """
        Apply risk events for this timestep; restore expired; maybe trigger new.
        Returns a risk_meta dict you can store into sample['meta'].
        核心函数，注意：它会直接修改 net（比如断线、降额、加负荷倍率），会直接对仿真世界动手
        """
        # 1) restore expired events
        #Step 里第一件事：恢复过期事件
        # 看看 active 里有没有事件已经过了结束时间（比如 end_t <= t）
	    # 如果过期了就把 net 改回原样（比如把断掉的线路重新 in_service=True）
		# 并把该事件从 active 里删掉
        self._restore_if_expired(net, t)

        # 2) maybe trigger new events (if not already active)
        # 按概率触发新事件（但一次只允许一个同类型事件，因为它用了 "contingency" not in self.active））
        # 如果配置里启用了 contingency
	    # 并且当前没有正在生效的 contingency（"contingency" not in self.active）
        # 那就掷骰子一次 rng.random()（0~1之间）
        # 若小于 contingency_prob_per_step，就触发一次跳闸事件
        if self.cfg.enable_contingency and ("contingency" not in self.active):
            if self.rng.random() < self.cfg.contingency_prob_per_step:
                self._trigger_contingency(net, t)

        #同理，derating（线路降额）
        if self.cfg.enable_line_derating and ("derating" not in self.active):
            if self.rng.random() < self.cfg.derate_prob_per_step:
                self._trigger_derating(net, t)
        #同理，overload（负荷突增）
        if self.cfg.enable_overload and ("overload" not in self.active):
            if self.rng.random() < self.cfg.overload_prob_per_step:
                self._trigger_overload(net, t)
        #每种事件同一时间只允许一个，contingency 只能同时有 1 个，derating 只能同时有 1 个，overload 只能同时有 1 个
        #三种事件之间是允许同时存在的（可能叠加），


        # 3) build meta：：把“本步风险状态”打包返回（用于写进数据集）
        meta = {
            "t": int(t),
            "risk_cfg": asdict(self.cfg),
            "active_events": {k: self._compact_event(v) for k, v in self.active.items()},
        }
        return meta


    #extra_load_multiplier()：告诉主仿真“要不要把负荷额外放大”
    #如果当前有 overload 事件正在生效，就返回一个 >1 的倍率；否则返回 1.0。
    #
    def extra_load_multiplier(self) -> float:
        """
        If overload active, return >1 load multiplier, else 1.0.
        """
        st = self.active.get("overload")
        if st is None:
            return 1.0
        return float(st.get("mult", 1.0))

    # --------------------------
    # Internal helpers
    # --------------------------
    #遍历当前所有“正在生效”的风险事件，如果某个事件已经到期（t >= t_end），
    #就把电网恢复回原样，然后把这个事件从 active 里删掉。
    def _restore_if_expired(self, net: pp.pandapowerNet, t: int):
        to_del = []
        for name, st in self.active.items():
            if t >= st["t_end"]:
                # restore
                if name == "contingency":
                    self._restore_in_service(net, st)
                elif name == "derating":
                    self._restore_derating(net, st)
                elif name == "overload":
                    pass
                to_del.append(name)
        for k in to_del:
            del self.active[k]

    def _trigger_contingency(self, net: pp.pandapowerNet, t: int):
        k = 1 if self.cfg.contingency_type == "n-1" else max(1, int(self.cfg.contingency_k))

        # collect candidates
        #根据配置允许的元素类型（line/trafo/sgen），
        #把电网里所有这些元素的“编号”收集起来，作为故障候选集合 cand。
        cand = []
        for elem in self.cfg.contingency_elements:
            if elem == "line" and len(net.line) > 0:
                cand += [("line", int(i)) for i in net.line.index.to_list()]
                #把所有线路编号都加入候选，例如 ("line", 12) 表示第 12 条线路可以被切掉
            elif elem == "trafo" and hasattr(net, "trafo") and len(net.trafo) > 0:
                cand += [("trafo", int(i)) for i in net.trafo.index.to_list()]
                #如果电网里有变压器表 net.trafo，就把它们也加进候选
            elif elem == "sgen" and len(net.sgen) > 0:
                cand += [("sgen", int(i)) for i in net.sgen.index.to_list()]
                # net.sgen 是“静态发电机”（这里你 PV 就是用 sgen 创建的）
                # 如果允许 sgen contingency，那就等于“让某个 PV 退出运行”也算一种风险事件

        if len(cand) == 0:
            return
            #如果电网里根本没有你允许切的元件（比如没 line、没 trafo），就直接返回，不触发


        #从候选元件列表里随机选出 k 个，把它们设成 in_service=False（相当于断开/故障退出运行），
        #并把“被断开的元件 + 它们原来的状态 + 事件持续时间”记录到 self.active["contingency"] 里。
        # sample k distinct
        k = min(k, len(cand))
        chosen_idx = self.rng.choice(len(cand), size=k, replace=False)
        chosen = [cand[int(i)] for i in chosen_idx]

        # snapshot original in_service
        original = []
        for elem, idx in chosen:
            df = getattr(net, elem)
            if "in_service" in df.columns:
                original.append((elem, idx, bool(df.at[idx, "in_service"])))
            else:
                # some tables may not have in_service, skip
                original.append((elem, idx, True))

        # apply: set in_service False
        for elem, idx, _orig in original:
            df = getattr(net, elem)
            if "in_service" in df.columns:
                df.at[idx, "in_service"] = False

        self.active["contingency"] = {
            "t_start": int(t),
            "t_end": int(t + self.cfg.contingency_duration_steps),
            "type": self.cfg.contingency_type,
            "chosen": original,  # (elem, idx, orig_in_service)
        }

    #
    #之前在 _trigger_contingency() 里，把一些元件设置成 in_service=False（故障退出运行），
    #并记录了它们原来的状态 orig。
    #现在_restore_in_service(...)：把被切掉的元件恢复回原来的状态
    def _restore_in_service(self, net: pp.pandapowerNet, st: Dict):
        for elem, idx, orig in st.get("chosen", []):
            df = getattr(net, elem)
            if "in_service" in df.columns:
                df.at[idx, "in_service"] = bool(orig)

    #  _trigger_derating(...)：随机挑一条线路，把它“降额”（可承载电流变小）

    def _trigger_derating(self, net: pp.pandapowerNet, t: int):
        if len(net.line) == 0: #没有线路，就没法做线路降额 → 直接返回
            return
        if "max_i_ka" not in net.line.columns: #这条网络的线路表里没有最大电流额定值字段 → 也没法降额 → 返回
            return

        line_id = int(self.rng.choice(net.line.index.to_numpy())) #从所有线路的 ID 里随机抽一条
        orig = float(net.line.at[line_id, "max_i_ka"]) #保存原来的最大电流额定值，方便之后恢复
        der = float(self.rng.uniform(self.cfg.derate_range[0], self.cfg.derate_range[1]))
        #随机生成一个“降额系数”，例如 derate_range=(0.7, 1.0)
        newv = orig * der #得到新上限，比如，原来 0.4kA，der=0.8 → 新上限 0.32kA

        net.line.at[line_id, "max_i_ka"] = newv
        #这一句就是“生效”：线路的最大允许电流变小了。


        #记录事件状态（状态机记账），把“降额事件”存进 self.active，以后到期了就能恢复：
        self.active["derating"] = {
            "t_start": int(t), #持续时间
            "t_end": int(t + self.cfg.derate_duration_steps), #持续时间
            "line_id": int(line_id), #哪条线被降额
            "orig_max_i_ka": float(orig), #原本上限（恢复用）
            "new_max_i_ka": float(newv), #降额后的上限（调试/记录用）
            "derate_factor": float(der), #降了多少比例
        }

    #把“降额”恢复回原来的线路容量
    def _restore_derating(self, net: pp.pandapowerNet, st: Dict):
        line_id = int(st["line_id"]) 
        if "max_i_ka" in net.line.columns:
            net.line.at[line_id, "max_i_ka"] = float(st["orig_max_i_ka"])

    #_trigger_overload(...)：触发“过载/负荷突增”事件（只记录一个倍率）
    def _trigger_overload(self, net: pp.pandapowerNet, t: int):
        mult = float(self.rng.uniform(self.cfg.overload_mult_range[0], self.cfg.overload_mult_range[1]))
        self.active["overload"] = {
            "t_start": int(t),
            "t_end": int(t + self.cfg.overload_duration_steps),
            "mult": float(mult),
        }
    #_compact_event(...)：把事件状态压缩成“好存、好看”的小字典
    def _compact_event(self, st: Dict) -> Dict:
        # avoid dumping huge things; keep only readable fields
        out = {k: v for k, v in st.items() if k not in ("chosen",)}
        if "chosen" in st:
            out["chosen"] = [(e, int(i)) for (e, i, _orig) in st["chosen"]]
        return out
