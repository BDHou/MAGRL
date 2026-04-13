# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.topology as top
import networkx as nx


class ResourceInjector:
    """
    配置驱动的可调资源注入器（标幺值版本）。
    支持向 Pandapower 网架中注入 BESS、Generator、Inverter、DemandResponse 四类资源，
    并为每个资源元件打上 TypeID 标签。

    标幺值基准 S_base:
      默认取自网络的总有功负荷 sum(net.load.p_mw)，也可通过构造参数手动指定。
      这个基准的物理含义：配置 P_max=0.10 即表示单台设备额定功率为
      馈线总负荷的 10%，直接对应 DER 渗透率的定义。

    TypeID 定义:
      0 - BESS (储能)
      1 - Generator (发电机)
      2 - Inverter (逆变器/PV)
      3 - DemandResponse (可调负荷)

    Args:
        config dict: 资源配置字典，键为资源类型名，值为该类资源的配置。
                     未提供的类型将使用 DEFAULT_CONFIG 中的默认值；
                     若某类 count=0 则跳过。
        seed int: 随机种子，用于母线选择和参数采样的可复现性。
        base_mva float | None: 手动指定标幺基准 (MW)；None 则自动取 sum(net.load.p_mw)。
    """

    TYPE_BESS = 0
    TYPE_GENERATOR = 1
    TYPE_INVERTER = 2
    TYPE_DEMAND_RESPONSE = 3

    _TYPE_MAP = {
        "bess": TYPE_BESS,
        "generator": TYPE_GENERATOR,
        "inverter": TYPE_INVERTER,
        "demand_response": TYPE_DEMAND_RESPONSE,
    }

    _NAME_PREFIX = {
        "bess": "BESS",
        "generator": "Gen",
        "inverter": "PV",
        "demand_response": "DR",
    }

    # ----------------------------------------------------------------
    # 默认参数 — 所有 P_max 均为标幺值 (p.u. of S_base)
    #
    # S_base = sum(net.load.p_mw)，即馈线总有功负荷。
    # 选用总负荷作为基准的理由：
    #   1. DER 渗透率的标准定义即为 P_DER / P_load_total
    #   2. 自动适配不同规模的网络（33-bus ~3.7MW vs 69-bus ~3.8MW
    #      vs mv_oberrhein ~30MW）
    #   3. 与规划文献中的 scenario 定义直接对应
    #
    # 各参数的工程依据：
    #
    # [PV/Inverter]
    #   P_max = 0.08-0.30 p.u.  每台 PV 占总负荷 8-30%
    #     3 台 * 0.20 = 0.60 p.u. -> 60% 渗透率（中等偏高）
    #   s_over_p = 1.10         IEEE 1547-2018 Cat B 逆变器过载比
    #
    # [BESS]
    #   P_max = 0.03-0.13 p.u.  配储比约 PV 的 30-50%
    #   e_duration_h = 1.5-3.0  储能时长 (小时)
    #     C-rate = 1/e_duration_h, 即 0.33C-0.67C (商用锂电典型区间)
    #     E_max_MWh = P_max_MW * e_duration_h
    #   eta = 0.92-0.95         锂电单程效率 (round-trip 85-90%)
    #     来源: NREL "Cost Projections for Utility-Scale Battery Storage"
    #
    # [Generator]
    #   P_max = 0.03-0.13 p.u.  小型分布式柴油/燃气机组
    #   s_over_p = 1.18          功率因数 pf=0.85 -> S/P = 1/0.85
    #
    # [Demand Response]
    #   P_max = 0.005-0.03 p.u. 单台 DR 可调量占总负荷 0.5-3%
    #     物理含义: 单节点负荷的 10-30% 可调 (FERC Order 2222)
    # ----------------------------------------------------------------
    DEFAULT_CONFIG: Dict[str, dict] = {
        "bess": {
            "count": 3,
            "bus_strategy": "farthest",
            "bus_list": None,
            "params": {
                "P_max": (0.03, 0.13),
                "e_duration_h": (1.5, 3.0),
                "s_over_p": 1.0,
                "eta": (0.92, 0.95),
            },
        },
        "generator": {
            "count": 0,
            "bus_strategy": "random",
            "bus_list": None,
            "params": {
                "P_max": (0.03, 0.13),
                "s_over_p": 1.18,
                "eta": 1.0,
            },
        },
        "inverter": {
            "count": 3,
            "bus_strategy": "farthest",
            "bus_list": None,
            "params": {
                "P_max": (0.08, 0.30),
                "s_over_p": 1.10,
                "eta": 1.0,
            },
        },
        "demand_response": {
            "count": 0,
            "bus_strategy": "random",
            "bus_list": None,
            "params": {
                "P_max": (0.005, 0.03),
                "eta": 1.0,
            },
        },
    }

    def __init__(self, config: dict = None, seed: int = 42,
                 base_mva: float = None):
        self.rng = np.random.default_rng(seed)
        self.config = self._merge_config(config or {})
        self._base_mva_override = base_mva
        self.base_mva = None  # 在 inject() 时确定

    # ================================================================
    # 公共接口
    # ================================================================

    def inject(self, net: pp.pandapowerNet) -> Tuple[pp.pandapowerNet, pd.DataFrame]:
        """
        主入口：向网络注入所有配置的资源。
        注入顺序固定为 inverter -> bess -> generator -> demand_response，
        其中 inverter 优先占位远端母线，bess 默认绑定 inverter 所在母线。

        Args:
            net pp.pandapowerNet: 待注入资源的 pandapower 网络

        Returns:
            net pp.pandapowerNet: 注入资源后的网络（原地修改）
            resource_table pd.DataFrame: 资源清单，列包含
                [resource_id, type_id, type_name, bus, pp_element, pp_index,
                 P_max_pu, P_max_mw, S_max_mw, E_max_mwh, eta]
        """
        # 确定标幺基准
        if self._base_mva_override is not None:
            self.base_mva = float(self._base_mva_override)
        else:
            self.base_mva = float(net.load["p_mw"].sum())
        assert self.base_mva > 0, (
            f"S_base = {self.base_mva:.4f} MW <= 0，网络可能没有负荷"
        )

        all_records: List[dict] = []
        occupied_buses: List[int] = []

        inject_order = ["inverter", "bess", "generator", "demand_response"]

        for rtype in inject_order:
            cfg = self.config[rtype]
            if cfg["count"] <= 0:
                continue

            inject_fn = {
                "bess": self._inject_bess,
                "generator": self._inject_generator,
                "inverter": self._inject_inverter,
                "demand_response": self._inject_demand_response,
            }[rtype]

            records = inject_fn(net, cfg, occupied_buses)
            for r in records:
                occupied_buses.append(r["bus"])
            all_records.extend(records)

        if all_records:
            resource_table = pd.DataFrame(all_records)
            resource_table.insert(0, "resource_id", range(len(resource_table)))
        else:
            resource_table = pd.DataFrame(
                columns=["resource_id", "type_id", "type_name", "bus",
                         "pp_element", "pp_index", "P_max_pu",
                         "P_max_mw", "S_max_mw", "E_max_mwh", "eta"]
            )

        net._resource_table = resource_table
        net._base_mva = self.base_mva

        self._print_summary(net, resource_table)
        return net, resource_table

    # ================================================================
    # 各类型资源的注入实现
    # ================================================================

    def _inject_bess(self, net, cfg, occupied_buses) -> List[dict]:
        """
        注入储能 (BESS)，映射为 pp.create_storage。
        特殊逻辑：如果 bus_strategy 为 "farthest" 且已有 inverter 注入，
        则优先绑定 inverter 所在母线（配储一体化）。

        Args:
            net pp.pandapowerNet: 网络
            cfg dict: 该资源类型的配置
            occupied_buses list[int]: 已被其他资源占用的母线列表

        Returns:
            records list[dict]: 注入记录列表
        """
        count = cfg["count"]
        params = cfg["params"]

        inverter_buses = self._get_existing_inverter_buses(net)
        if cfg["bus_strategy"] == "farthest" and len(inverter_buses) >= count:
            buses = inverter_buses[:count]
        else:
            buses = self._select_buses(
                net, count, cfg["bus_strategy"], cfg.get("bus_list"),
                exclude=occupied_buses
            )

        records = []
        type_id = self.TYPE_BESS
        s_over_p = params.get("s_over_p", 1.0)
        for i, b in enumerate(buses):
            p_max_pu = self._sample_param(params["P_max"])
            p_max_mw = p_max_pu * self.base_mva
            s_max_mw = p_max_mw * s_over_p

            e_dur = self._sample_param(params.get("e_duration_h"))
            e_dur = e_dur if e_dur is not None else 2.0
            e_max_mwh = p_max_mw * e_dur

            eta_val = self._sample_param(params.get("eta"))
            eta = eta_val if eta_val is not None else 0.93

            name = f"BESS_T{type_id}_bus{b}"
            pp_idx = pp.create_storage(
                net, bus=int(b), p_mw=0.0,
                max_e_mwh=e_max_mwh, max_p_mw=p_max_mw,
                name=name,
            )
            records.append({
                "type_id": type_id,
                "type_name": "bess",
                "bus": int(b),
                "pp_element": "storage",
                "pp_index": int(pp_idx),
                "P_max_pu": p_max_pu,
                "P_max_mw": p_max_mw,
                "S_max_mw": s_max_mw,
                "E_max_mwh": e_max_mwh,
                "eta": eta,
            })
        return records

    def _inject_generator(self, net, cfg, occupied_buses) -> List[dict]:
        """
        注入发电机 (Generator)，映射为 pp.create_sgen，type="Generator"。

        Args:
            net pp.pandapowerNet: 网络
            cfg dict: 该资源类型的配置
            occupied_buses list[int]: 已被其他资源占用的母线列表

        Returns:
            records list[dict]: 注入记录列表
        """
        count = cfg["count"]
        params = cfg["params"]
        buses = self._select_buses(
            net, count, cfg["bus_strategy"], cfg.get("bus_list"),
            exclude=occupied_buses,
        )

        records = []
        type_id = self.TYPE_GENERATOR
        s_over_p = params.get("s_over_p", 1.18)
        for i, b in enumerate(buses):
            p_max_pu = self._sample_param(params["P_max"])
            p_max_mw = p_max_pu * self.base_mva
            s_max_mw = p_max_mw * s_over_p

            eta_val = self._sample_param(params.get("eta"))
            eta = eta_val if eta_val is not None else 1.0

            name = f"Gen_T{type_id}_bus{b}"
            pp_idx = pp.create_sgen(
                net, bus=int(b), p_mw=0.0, q_mvar=0.0,
                sn_mva=s_max_mw, name=name,
                type="Generator", controllable=False,
            )
            records.append({
                "type_id": type_id,
                "type_name": "generator",
                "bus": int(b),
                "pp_element": "sgen",
                "pp_index": int(pp_idx),
                "P_max_pu": p_max_pu,
                "P_max_mw": p_max_mw,
                "S_max_mw": s_max_mw,
                "E_max_mwh": None,
                "eta": eta,
            })
        return records

    def _inject_inverter(self, net, cfg, occupied_buses) -> List[dict]:
        """
        注入逆变器/PV (Inverter)，映射为 pp.create_sgen，type="PV"。
        S_max 默认由 P_max * s_over_p 自动计算。

        Args:
            net pp.pandapowerNet: 网络
            cfg dict: 该资源类型的配置
            occupied_buses list[int]: 已被其他资源占用的母线列表

        Returns:
            records list[dict]: 注入记录列表
        """
        count = cfg["count"]
        params = cfg["params"]
        buses = self._select_buses(
            net, count, cfg["bus_strategy"], cfg.get("bus_list"),
            exclude=occupied_buses,
        )

        records = []
        type_id = self.TYPE_INVERTER
        s_over_p = params.get("s_over_p", 1.10)
        for i, b in enumerate(buses):
            p_max_pu = self._sample_param(params["P_max"])
            p_max_mw = p_max_pu * self.base_mva
            s_max_mw = p_max_mw * s_over_p

            eta_val = self._sample_param(params.get("eta"))
            eta = eta_val if eta_val is not None else 1.0

            name = f"PV_T{type_id}_bus{b}"
            pp_idx = pp.create_sgen(
                net, bus=int(b), p_mw=0.0, q_mvar=0.0,
                sn_mva=s_max_mw, name=name,
                type="PV", controllable=False,
            )
            records.append({
                "type_id": type_id,
                "type_name": "inverter",
                "bus": int(b),
                "pp_element": "sgen",
                "pp_index": int(pp_idx),
                "P_max_pu": p_max_pu,
                "P_max_mw": p_max_mw,
                "S_max_mw": s_max_mw,
                "E_max_mwh": None,
                "eta": eta,
            })
        return records

    def _inject_demand_response(self, net, cfg, occupied_buses) -> List[dict]:
        """
        注入可调负荷 (Demand Response)，映射为 pp.create_load，controllable=True。

        Args:
            net pp.pandapowerNet: 网络
            cfg dict: 该资源类型的配置
            occupied_buses list[int]: 已被其他资源占用的母线列表

        Returns:
            records list[dict]: 注入记录列表
        """
        count = cfg["count"]
        params = cfg["params"]
        buses = self._select_buses(
            net, count, cfg["bus_strategy"], cfg.get("bus_list"),
            exclude=occupied_buses,
        )

        records = []
        type_id = self.TYPE_DEMAND_RESPONSE
        for i, b in enumerate(buses):
            p_max_pu = self._sample_param(params["P_max"])
            p_max_mw = p_max_pu * self.base_mva
            s_max_mw = p_max_mw

            eta_val = self._sample_param(params.get("eta"))
            eta = eta_val if eta_val is not None else 1.0

            name = f"DR_T{type_id}_bus{b}"
            pp_idx = pp.create_load(
                net, bus=int(b), p_mw=0.0, q_mvar=0.0,
                name=name, controllable=True,
            )
            records.append({
                "type_id": type_id,
                "type_name": "demand_response",
                "bus": int(b),
                "pp_element": "load",
                "pp_index": int(pp_idx),
                "P_max_pu": p_max_pu,
                "P_max_mw": p_max_mw,
                "S_max_mw": s_max_mw,
                "E_max_mwh": None,
                "eta": eta,
            })
        return records

    # ================================================================
    # 母线选择
    # ================================================================

    def _select_buses(
        self,
        net: pp.pandapowerNet,
        count: int,
        strategy: str,
        bus_list: Optional[List[int]],
        exclude: List[int],
    ) -> List[int]:
        """
        根据策略从网络中选择放置资源的母线。

        Args:
            net pp.pandapowerNet: 网络
            count int: 需要选择的母线数量
            strategy str: 选择策略，"farthest" / "random" / "manual"
            bus_list list[int] | None: manual 策略下的指定母线列表
            exclude list[int]: 需要排除的已占用母线

        Returns:
            selected list[int]: 被选中的母线编号列表
        """
        if strategy == "manual":
            assert bus_list is not None and len(bus_list) >= count, (
                f"manual 策略要求 bus_list 至少包含 {count} 个母线"
            )
            return [int(b) for b in bus_list[:count]]

        slack_bus = int(net.ext_grid.bus.iloc[0])
        all_buses = list(net.bus.index.astype(int))
        candidates = [b for b in all_buses if b != slack_bus and b not in exclude]

        if count >= len(candidates):
            return candidates[:count]

        if strategy == "farthest":
            return self._buses_by_distance(net, candidates, count, slack_bus)
        elif strategy == "random":
            chosen = self.rng.choice(candidates, size=count, replace=False)
            return [int(b) for b in chosen]
        else:
            raise ValueError(f"未知的母线选择策略: {strategy}")

    def _buses_by_distance(
        self, net, candidates: List[int], count: int, slack_bus: int,
    ) -> List[int]:
        """
        按到 slack bus 的拓扑距离从远到近排序，选择前 count 个母线。

        Args:
            net pp.pandapowerNet: 网络
            candidates list[int]: 候选母线
            count int: 需要选择的数量
            slack_bus int: 松弛母线编号

        Returns:
            selected list[int]: 被选中的母线编号列表
        """
        G = top.create_nxgraph(net, respect_switches=True)
        dist = nx.single_source_shortest_path_length(G, slack_bus)
        ranked = sorted(candidates, key=lambda b: (dist.get(b, -1), b), reverse=True)
        return ranked[:count]

    # ================================================================
    # 辅助方法
    # ================================================================

    def _sample_param(self, spec) -> Optional[float]:
        """
        根据参数规格采样一个值。

        Args:
            spec: 三种形式之一：
                  - (min, max) tuple: 均匀采样
                  - float/int 标量: 直接返回
                  - None: 返回 None（不适用）

        Returns:
            value float | None: 采样值
        """
        if spec is None:
            return None
        if isinstance(spec, (tuple, list)) and len(spec) == 2:
            return float(self.rng.uniform(spec[0], spec[1]))
        return float(spec)

    def _merge_config(self, user_config: dict) -> dict:
        """
        将用户配置与默认配置合并。
        用户配置中存在的键会覆盖默认值，params 级别也做深度合并。

        Args:
            user_config dict: 用户传入的配置字典

        Returns:
            merged dict: 合并后的完整配置
        """
        merged = copy.deepcopy(self.DEFAULT_CONFIG)
        for rtype, user_type_cfg in user_config.items():
            if rtype not in merged:
                raise ValueError(
                    f"未知的资源类型: '{rtype}'，"
                    f"支持的类型: {list(merged.keys())}"
                )
            for key, val in user_type_cfg.items():
                if key == "params" and isinstance(val, dict):
                    merged[rtype]["params"].update(val)
                else:
                    merged[rtype][key] = val
        return merged

    def _get_existing_inverter_buses(self, net) -> List[int]:
        """
        获取网络中已注入的 inverter (PV) 所在的母线列表。

        Args:
            net pp.pandapowerNet: 网络

        Returns:
            buses list[int]: inverter 所在的母线列表
        """
        if len(net.sgen) == 0:
            return []
        pv_mask = net.sgen["name"].str.startswith("PV_T2_", na=False)
        return net.sgen.loc[pv_mask, "bus"].astype(int).tolist()

    def _print_summary(self, net, resource_table: pd.DataFrame) -> None:
        """
        打印注入结果摘要。

        Args:
            net pp.pandapowerNet: 网络
            resource_table pd.DataFrame: 资源清单
        """
        print(f"\n=== ResourceInjector 注入完成 ===")
        print(f"  S_base = {self.base_mva:.3f} MW (总负荷有功)")
        print(f"  网络规模: {len(net.bus)} buses, {len(net.line)} lines")
        print(f"  总注入资源数: {len(resource_table)}")
        if len(resource_table) > 0:
            for type_name, group in resource_table.groupby("type_name"):
                type_id = group["type_id"].iloc[0]
                buses = group["bus"].tolist()
                p_total_pu = group["P_max_pu"].sum()
                p_total_mw = group["P_max_mw"].sum()
                print(f"    [{type_name}] TypeID={type_id}, "
                      f"count={len(group)}, buses={buses}, "
                      f"P_total={p_total_pu:.3f} p.u. ({p_total_mw:.3f} MW)")
        print(f"  net.storage: {len(net.storage)} 行")
        print(f"  net.sgen:    {len(net.sgen)} 行")
        print(f"  net.load:    {len(net.load)} 行")
        print("=" * 40)


# ================================================================
# Demo: 独立运行示例
# ================================================================
if __name__ == "__main__":
    import pandapower.networks as pn

    print("--- ResourceInjector Demo (p.u.) ---\n")

    # 1. 加载标准 IEEE 33 节点网络
    net = pn.case33bw()
    total_load = net.load["p_mw"].sum()
    print(f"原始网络: {len(net.bus)} buses, {len(net.line)} lines, "
          f"{len(net.load)} loads, 总负荷 = {total_load:.3f} MW\n")

    # 2. 定义资源配置（P_max 均为标幺值 p.u. of S_base）
    config = {
        "bess": {
            "count": 3,
            "bus_strategy": "farthest",
            "params": {
                "P_max": (0.04, 0.10),       # p.u., 每台 BESS 4-10% 总负荷
                "e_duration_h": (1.5, 2.5),   # 1.5-2.5 小时储能时长
                "eta": 0.93,
            },
        },
        "generator": {
            "count": 1,
            "bus_strategy": "random",
            "params": {
                "P_max": (0.04, 0.08),        # p.u., 4-8% 总负荷
            },
        },
        "inverter": {
            "count": 3,
            "bus_strategy": "farthest",
            "params": {
                "P_max": (0.12, 0.25),        # p.u., 每台 PV 12-25% 总负荷
            },
        },
        "demand_response": {
            "count": 2,
            "bus_strategy": "random",
            "params": {
                "P_max": (0.008, 0.025),      # p.u., 0.8-2.5% 总负荷
            },
        },
    }

    # 3. 创建注入器并执行注入（base_mva 自动从网络计算）
    injector = ResourceInjector(config=config, seed=42)
    net, table = injector.inject(net)

    # 4. 查看资源清单
    print("\n资源清单 (resource_table):")
    cols = ["resource_id", "type_name", "bus", "P_max_pu",
            "P_max_mw", "S_max_mw", "E_max_mwh", "eta"]
    print(table[cols].to_string(index=False))

    # 5. 验证潮流
    pp.runpp(net)
    print(f"\n潮流验证通过! 母线电压范围: "
          f"[{net.res_bus.vm_pu.min():.4f}, {net.res_bus.vm_pu.max():.4f}] p.u.")
