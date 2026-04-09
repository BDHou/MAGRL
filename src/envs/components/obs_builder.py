import numpy as np
from gymnasium.spaces import Box


class ObservationBuilder:
    """
    构建各智能体的异构观测向量。

    v2: 观测从 [全局所有节点电压 + p_ext + soc + time_enc] 重构为
        固定 8 维的局部异构特征向量，打破 Shared Policy 下的对称性。

    观测向量结构 (dim=8):
    ┌────────────────────────────────┐
    │ [全局共享上下文]                │
    │  0: p_grid        — PCC 净功率 │
    │  1: time_sin      — 时间正弦   │
    │  2: time_cos      — 时间余弦   │
    │ [局部异构上下文]                │
    │  3: soc_i         — 自身 SOC   │
    │  4: v_self_i      — 本地电压   │
    │  5: r_eq_i        — 累积电阻   │
    │  6: x_eq_i        — 累积电抗   │
    │  7: p_net_local_i — 本地净负荷 │
    └────────────────────────────────┘
    """

    OBS_DIM = 8
    STEPS_PER_DAY = 96  # 15分钟步长, 24h × 4 = 96

    def __init__(self):
        """
        无需外部参数，观测维度固定为 8。
        """
        self.obs_dim = self.OBS_DIM

    def get_space(self) -> Box:
        """返回单个智能体的观测空间"""
        return Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

    def build(
        self,
        simulator,
        agent_idx: int,
        soc_value: float,
        current_step: int,
        r_eq: float,
        x_eq: float,
    ) -> np.ndarray:
        """
        构建单个智能体的异构观测向量。

        Args:
            simulator:    GridSimulator 实例（用于提取电气量）
            agent_idx:    该智能体对应的储能索引
            soc_value:    该智能体的当前 SOC
            current_step: 当前时步 (用于时间编码)
            r_eq:         该储能的归一化累积电阻 (静态特征, 0~1)
            x_eq:         该储能的归一化累积电抗 (静态特征, 0~1)

        Returns:
            观测向量 np.ndarray, shape = (8,)
        """
        # --- 全局共享特征 ---
        p_grid = simulator.get_ext_grid_power()

        t = current_step % self.STEPS_PER_DAY
        angle = 2 * np.pi * t / self.STEPS_PER_DAY
        time_sin = np.sin(angle)
        time_cos = np.cos(angle)

        # --- 局部异构特征 ---
        bus_idx = simulator.storage_buses[agent_idx]
        v_self = simulator.get_bus_voltage_single(bus_idx)
        p_net_local = simulator.get_local_net_load(bus_idx)

        return np.array(
            [p_grid, time_sin, time_cos, soc_value, v_self, r_eq, x_eq, p_net_local],
            dtype=np.float32,
        )

    def build_zero(self) -> np.ndarray:
        """返回全零观测（用于潮流发散时的兜底）"""
        return np.zeros(self.obs_dim, dtype=np.float32)
