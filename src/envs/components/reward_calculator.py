import numpy as np


class RewardCalculator:
    """
    Multi-objective reward calculator with per-agent decomposed sub-rewards.

    v2: 奖励从单一全局标量改为每个智能体独立的总奖励。
        全局分量 (R_reverse, R_buy) 所有智能体共享，
        局部分量 (R_soc_i, R_action_i) 针对各智能体独立计算。

    Sub-rewards:
      R_reverse: 倒送惩罚（二次） — 核心目标            (全局)
      R_buy:     购电惩罚（线性） — 鼓励放电            (全局)
      R_soc_i:   SOC 越界软惩罚  — 安全约束             (局部)
      R_action_i:动作平滑惩罚    — 避免激进控制          (局部)

    符号约定 (与 GridSimulator.get_ext_grid_power 一致):
      p_grid > 0 → 配网从主网买电
      p_grid < 0 → 配网向主网倒送
    """

    DEFAULT_WEIGHTS = {
        "w_reverse": 2.0,
        "w_buy": 1.0,
        "w_soc": 5.0,
        "w_action": 0.01,
    }

    def __init__(self, config: dict = None):
        config = config or {}
        self.w1 = config.get("w_reverse", self.DEFAULT_WEIGHTS["w_reverse"])
        self.w2 = config.get("w_buy", self.DEFAULT_WEIGHTS["w_buy"])
        self.w3 = config.get("w_soc", self.DEFAULT_WEIGHTS["w_soc"])
        self.w4 = config.get("w_action", self.DEFAULT_WEIGHTS["w_action"])

    def calculate(self, simulator, soc_values: np.ndarray, p_bat_values: list[float]) -> dict:
        """
        计算每个智能体的独立奖励。

        Args:
            simulator:    GridSimulator 实例
            soc_values:   array of current SOC for all storages
            p_bat_values: list of actual battery power (MW) applied this step

        Returns:
            dict with keys:
              - per_agent_rewards: list[float], 每个智能体各自的总奖励
              - reward_reverse:    float, 全局倒送惩罚 (共享)
              - reward_buy:        float, 全局购电惩罚 (共享)
              - reward_soc:        list[float], 各智能体的 SOC 惩罚
              - reward_action:     list[float], 各智能体的动作惩罚
              - p_grid_actual:     float, PCC 实际功率
        """
        p_grid = simulator.get_ext_grid_power()
        num_agents = len(soc_values)

        # ============================================================
        # 全局共享分量
        # ============================================================

        # R_reverse: p_grid < 0 说明倒送，取绝对值的二次惩罚
        r_reverse = -(max(0.0, -p_grid) ** 2)

        # R_buy: p_grid > 0 说明买电，线性惩罚
        r_buy = -(max(0.0, p_grid))

        # 全局部分的加权和（所有智能体共享同一个值）
        global_reward = self.w1 * r_reverse + self.w2 * r_buy

        # ============================================================
        # 局部独立分量（每个智能体各自计算）
        # ============================================================
        per_agent_rewards = []
        reward_soc_list = []
        reward_action_list = []

        for i in range(num_agents):
            # R_soc_i: 仅当该智能体的 SOC 超出 [0.1, 0.9] 时给予线性惩罚
            r_soc_i = 0.0
            soc = soc_values[i]
            if soc < 0.1:
                r_soc_i = -((0.1 - soc) * 10.0)
            elif soc > 0.9:
                r_soc_i = -((soc - 0.9) * 10.0)

            # R_action_i: 针对该智能体的动作平滑惩罚
            r_action_i = -abs(p_bat_values[i])

            # 该智能体的总奖励 = 全局共享 + 局部独立
            total_i = global_reward + self.w3 * r_soc_i + self.w4 * r_action_i

            per_agent_rewards.append(total_i)
            reward_soc_list.append(r_soc_i)
            reward_action_list.append(r_action_i)

        return {
            "per_agent_rewards": per_agent_rewards,
            "reward_reverse": r_reverse,
            "reward_buy": r_buy,
            "reward_soc": reward_soc_list,
            "reward_action": reward_action_list,
            "p_grid_actual": p_grid,
        }
