import numpy as np


class ActionProcessor:
    """
    将 RL 智能体输出的动作转换为物理量，并维护储能 SOC 状态。
    未来如果增加更多可控设备类型（如可调变压器、开关），在此扩展。
    """

    def __init__(self, storage_ids: list, max_p: np.ndarray, max_e: np.ndarray, dt: float = 0.25):
        """
        Args:
            storage_ids: pandapower storage 的索引列表
            max_p: 各储能额定功率 (MW)
            max_e: 各储能额定容量 (MWh)
            dt: 时步长度 (小时)
        """
        self.storage_ids = storage_ids
        self.num_storages = len(storage_ids)
        self.max_p = max_p.copy()
        self.max_e = max_e.copy()
        self.dt = dt
        self.soc = None

    def reset(self) -> None:
        """重置所有储能 SOC 到 50%"""
        self.soc = np.full(self.num_storages, 0.5, dtype=np.float32)

    def apply(self, agent_idx: int, action_val: float, simulator) -> float:
        """
        将单个智能体的动作下发到电网并更新 SOC。

        Args:
            agent_idx: 该智能体对应的储能索引
            action_val: RL 输出的动作值 [-1, 1]
            simulator: GridSimulator 实例

        Returns:
            实际下发的有功出力 (MW)
        """
        sid = self.storage_ids[agent_idx]

        # 将 [-1, 1] 映射到 [-max_p, +max_p]
        desired_p = action_val * self.max_p[agent_idx]

        # 物理兜底：电池满了不再充，空了不再放
        if self.soc[agent_idx] >= 0.95 and desired_p > 0:
            desired_p = 0.0
        elif self.soc[agent_idx] <= 0.05 and desired_p < 0:
            desired_p = 0.0

        # 下发到电网
        simulator.set_storage_power(sid, desired_p)

        # 更新 SOC
        energy_change = desired_p * self.dt
        self.soc[agent_idx] += energy_change / self.max_e[agent_idx]
        self.soc[agent_idx] = np.clip(self.soc[agent_idx], 0.0, 1.0)

        return desired_p

    def get_soc(self, agent_idx: int) -> float:
        """获取指定智能体的当前 SOC"""
        return self.soc[agent_idx]
