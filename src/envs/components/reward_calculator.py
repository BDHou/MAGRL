import numpy as np


class RewardCalculator:
    """
    Multi-objective reward calculator with decomposed sub-rewards.

    Sub-rewards:
      R_reverse: Reverse power penalty (quadratic) — core objective
      R_buy:     Grid purchase penalty (linear) — incentivize discharge
      R_soc:     SOC out-of-range soft penalty — safety constraint
      R_action:  Action smoothness penalty — avoid erratic control

    Total = w1*R_reverse + w2*R_buy + w3*R_soc + w4*R_action
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
        Compute decomposed reward.

        Args:
            simulator: GridSimulator instance
            soc_values: array of current SOC for all storages
            p_bat_values: list of actual battery power (MW) applied this step

        Returns:
            dict with keys: total, reward_reverse, reward_buy, reward_soc,
                            reward_action, p_grid_actual
        """
        p_grid = simulator.get_ext_grid_power()

        # R_reverse: quadratic penalty for reverse power flow
        r_reverse = -(max(0.0, -p_grid) ** 2)

        # R_buy: linear penalty for purchasing from grid
        r_buy = -(max(0.0, p_grid))

        # R_soc: soft penalty outside [0.1, 0.9] safe zone
        r_soc = 0.0
        for soc in soc_values:
            if soc < 0.1:
                r_soc += -((0.1 - soc) * 10.0)
            elif soc > 0.9:
                r_soc += -((soc - 0.9) * 10.0)

        # R_action: penalize large actions for smoothness
        r_action = 0.0
        for p in p_bat_values:
            r_action += -abs(p)

        total = (
            self.w1 * r_reverse
            + self.w2 * r_buy
            + self.w3 * r_soc
            + self.w4 * r_action
        )

        return {
            "total": total,
            "reward_reverse": r_reverse,
            "reward_buy": r_buy,
            "reward_soc": r_soc,
            "reward_action": r_action,
            "p_grid_actual": p_grid,
        }
