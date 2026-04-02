import numpy as np
from gymnasium.spaces import Box


class ObservationBuilder:
    """
    构建各智能体的观测向量。
    未来如需加入邻居信息、历史窗口等，在此扩展。
    """

    STEPS_PER_DAY = 96  # 15分钟步长, 24h × 4 = 96 # CONF

    def __init__(self, num_buses: int):
        """
        Args:
            num_buses: 电网节点数量
        """
        self.num_buses = num_buses
        # 观测: [vm_pu...] + [p_ext_grid] + [soc] + [time_sin, time_cos]
        self.obs_dim = num_buses + 4

    def get_space(self) -> Box:
        """返回单个智能体的观测空间"""
        return Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

    def build(self, simulator, soc_value: float, current_step: int) -> np.ndarray:
        """
        构建一个智能体的观测向量。

        Args:
            simulator: GridSimulator 实例
            soc_value: 该智能体的当前 SOC
            current_step: 当前时步 (用于时间编码)

        Returns:
            观测向量 np.ndarray, shape = (num_buses + 4,)
        """
        vm = simulator.get_bus_voltages()
        p_ext = simulator.get_ext_grid_power()

        # 正余弦时间编码: 将一天映射到单位圆上
        t = current_step % self.STEPS_PER_DAY
        angle = 2 * np.pi * t / self.STEPS_PER_DAY
        time_sin = np.sin(angle)
        time_cos = np.cos(angle)

        return np.concatenate([vm, [p_ext, soc_value, time_sin, time_cos]])

    def build_zero(self) -> np.ndarray:
        """返回全零观测（用于潮流发散时的兜底）"""
        return np.zeros(self.obs_dim, dtype=np.float32)
