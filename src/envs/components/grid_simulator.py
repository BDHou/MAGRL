import copy
import numpy as np
import pandapower as pp
import pandapower.topology as top
import networkx as nx


class GridSimulator:
    """
    封装 pandapower 电网物理仿真的所有调用。
    环境主类不直接 import pandapower，所有与仿真器的交互都经过此类。

    v2: 新增静态拓扑特征预计算（累积阻抗）和局部电气特征提取方法，
        用于为多智能体提供异构观测，打破 Shared Policy 下的动作克隆。
    """

    def __init__(self, net_path: str):
        self.base_net = pp.from_pickle(net_path)
        self.net = None

        # ------ 预计算静态拓扑特征 ------
        self._precompute_topology_features()

    # ================================================================
    # 静态拓扑特征预计算
    # ================================================================

    def _precompute_topology_features(self) -> None:
        """
        利用 pandapower.topology + networkx 计算每个储能节点
        到 Slack Bus 的累积阻抗 (R, X)，并归一化到 [0, 1]。

        原理：配电网是树状辐射拓扑，各储能距离并网点 (ext_grid) 的
        电气距离不同。离根节点越远的储能，其动作对 PCC 功率的影响越小、
        对本地电压的影响越大。这些静态特征可以帮助 Shared Policy
        区分不同位置的智能体。
        """
        net = self.base_net

        # --- Step 1: 确定 Slack Bus（ext_grid 所在母线） ---
        slack_bus = int(net.ext_grid.bus.iloc[0])

        # --- Step 2: 构建 networkx 拓扑图 ---
        # create_nxgraph 返回的图以 bus 为节点，以 line/trafo 为边
        mg = top.create_nxgraph(net, respect_switches=True)

        # --- Step 3: 预构建边到 line/trafo 参数的查找表 ---
        # pandapower 的 nxgraph 边上有 key=(element_type, element_idx) 的信息
        # 我们需要遍历线路和变压器，建立 (from_bus, to_bus) -> (R, X) 的映射
        edge_impedance = {}  # key: frozenset({bus_a, bus_b}), value: (R_ohm, X_ohm)

        # 线路阻抗：R = r_ohm_per_km * length_km, X = x_ohm_per_km * length_km
        for idx in net.line.index:
            fb = int(net.line.at[idx, 'from_bus'])
            tb = int(net.line.at[idx, 'to_bus'])
            r = net.line.at[idx, 'r_ohm_per_km'] * net.line.at[idx, 'length_km']
            x = net.line.at[idx, 'x_ohm_per_km'] * net.line.at[idx, 'length_km']
            key = frozenset({fb, tb})
            # 如果同一对母线之间有多条线路，取阻抗较小的（并联等效近似）
            if key in edge_impedance:
                old_r, old_x = edge_impedance[key]
                edge_impedance[key] = (min(old_r, r), min(old_x, x))
            else:
                edge_impedance[key] = (r, x)

        # 变压器阻抗：从 vk_percent 和 vkr_percent 近似折算
        # Z_pu = vk% / 100, R_pu = vkr% / 100, X_pu = sqrt(Z^2 - R^2)
        # 转换为 ohm: Z_ohm = Z_pu * (V_rated^2 / S_rated)
        for idx in net.trafo.index:
            hv_bus = int(net.trafo.at[idx, 'hv_bus'])
            lv_bus = int(net.trafo.at[idx, 'lv_bus'])
            vk = net.trafo.at[idx, 'vk_percent'] / 100.0
            vkr = net.trafo.at[idx, 'vkr_percent'] / 100.0
            sn_mva = net.trafo.at[idx, 'sn_mva']
            vn_hv_kv = net.trafo.at[idx, 'vn_hv_kv']
            z_base = (vn_hv_kv ** 2) / sn_mva  # 阻抗基准值 (ohm)
            r_ohm = vkr * z_base
            x_ohm = np.sqrt(max(vk ** 2 - vkr ** 2, 0.0)) * z_base
            key = frozenset({hv_bus, lv_bus})
            edge_impedance[key] = (r_ohm, x_ohm)

        # --- Step 4: 对每个储能节点计算到 Slack Bus 的累积阻抗 ---
        storage_buses = net.storage.bus.values.astype(int)
        self.storage_buses = storage_buses

        num_storages = len(storage_buses)
        r_eq_raw = np.zeros(num_storages, dtype=np.float64)
        x_eq_raw = np.zeros(num_storages, dtype=np.float64)

        for i, s_bus in enumerate(storage_buses):
            try:
                path = nx.shortest_path(mg, source=slack_bus, target=s_bus)
            except nx.NetworkXNoPath:
                # 如果找不到路径（理论上不应发生），给一个大的默认值
                r_eq_raw[i] = 1e6
                x_eq_raw[i] = 1e6
                continue

            # 沿路径累加每条边的阻抗
            for j in range(len(path) - 1):
                key = frozenset({path[j], path[j + 1]})
                if key in edge_impedance:
                    r, x = edge_impedance[key]
                    r_eq_raw[i] += r
                    x_eq_raw[i] += x
                # 如果边不在查找表中（极端情况），跳过

        # --- Step 5: 归一化到 [0, 1] ---
        r_max = r_eq_raw.max() if r_eq_raw.max() > 0 else 1.0
        x_max = x_eq_raw.max() if x_eq_raw.max() > 0 else 1.0
        self.storage_r_eq = (r_eq_raw / r_max).astype(np.float32)
        self.storage_x_eq = (x_eq_raw / x_max).astype(np.float32)

        # 打印拓扑特征供调试
        print(f"[GridSimulator] 拓扑特征预计算完成：")
        print(f"  Slack Bus: {slack_bus}")
        for i, s_bus in enumerate(storage_buses):
            print(f"  Storage {i} @ Bus {s_bus}: "
                  f"R_eq_raw={r_eq_raw[i]:.4f} Ω, X_eq_raw={x_eq_raw[i]:.4f} Ω → "
                  f"r_eq={self.storage_r_eq[i]:.4f}, x_eq={self.storage_x_eq[i]:.4f}")

    # ================================================================
    # 基础信息（从 base_net 中提取，只读）
    # ================================================================

    @property
    def storage_ids(self) -> list:
        return self.base_net.storage.index.tolist()

    @property
    def num_storages(self) -> int:
        return len(self.base_net.storage)

    @property
    def num_buses(self) -> int:
        return len(self.base_net.bus)

    @property
    def num_loads(self) -> int:
        return len(self.base_net.load)

    @property
    def storage_max_e(self) -> np.ndarray:
        if 'max_e_mwh' in self.base_net.storage:
            return self.base_net.storage.max_e_mwh.values
        return np.full(self.num_storages, 2.0)

    @property
    def storage_max_p(self) -> np.ndarray:
        if 'max_p_mw' in self.base_net.storage:
            return self.base_net.storage.max_p_mw.values
        return np.full(self.num_storages, 1.0)

    # ================================================================
    # 仿真操作
    # ================================================================

    def reset(self) -> None:
        """深拷贝底本电网，准备新的 episode"""
        self.net = copy.deepcopy(self.base_net)

    def run_powerflow(self) -> bool:
        """
        运行潮流计算。
        Returns:
            True 如果收敛，False 如果发散
        """
        try:
            pp.runpp(self.net)
            return True
        except pp.LoadflowNotConverged:
            return False

    def set_storage_power(self, storage_id: int, p_mw: float) -> None:
        """设置指定储能的有功出力"""
        self.net.storage.at[storage_id, 'p_mw'] = p_mw

    def get_bus_voltages(self) -> np.ndarray:
        """获取所有节点电压标幺值"""
        return self.net.res_bus.vm_pu.values.astype(np.float32)

    def get_ext_grid_power(self) -> float:
        """获取变电站主网关口有功功率 (bus 0)"""  # 负值代表向主网送电，这里将其视为load了。
        return -self.net.res_bus.at[0, 'p_mw']

    # ================================================================
    # 局部电气特征提取（每步潮流后调用）
    # ================================================================

    def get_bus_voltage_single(self, bus_idx: int) -> float:
        """
        获取单个母线的电压标幺值。
        用于构建各智能体的局部观测 v_self_i。
        """
        return float(self.net.res_bus.at[bus_idx, 'vm_pu'])

    def get_local_net_load(self, bus_idx: int) -> float:
        """
        计算指定母线的本地净负荷 (MW)。
        p_net_local = Σ(该母线上的 load.p_mw) - Σ(该母线上的 sgen.p_mw)

        正值 = 该母线净消耗（有潮流需求）
        负值 = 该母线净发电（有潮流倒送风险）

        如果该母线没有 load 或 sgen，对应项为 0。
        使用原始 MW 值，不做额外归一化。
        """
        net = self.net

        # 该母线上所有负荷的有功之和
        load_mask = net.load.bus == bus_idx
        total_load_p = float(net.load.loc[load_mask, 'p_mw'].sum()) if load_mask.any() else 0.0

        # 该母线上所有分布式电源 (sgen) 的有功之和
        total_sgen_p = 0.0
        if len(net.sgen) > 0:
            sgen_mask = net.sgen.bus == bus_idx
            total_sgen_p = float(net.sgen.loc[sgen_mask, 'p_mw'].sum()) if sgen_mask.any() else 0.0

        return total_load_p - total_sgen_p
