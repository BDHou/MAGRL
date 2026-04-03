import os
import numpy as np
import pandas as pd


class TimeSeriesDataManager:
    """
    集中管理所有时序数据的加载、校验和注入。
    未来新增数据源（光伏、电价、温度等）只需在此类中扩展。
    """

    def __init__(self, data_path: str, num_loads_in_grid: int):
        """
        Args:
            data_path: 数据根目录
            num_loads_in_grid: 电网模型中的负荷数量，用于维度校验
        """
        # 加载负荷时序矩阵
        self.load_p = (
            pd.read_csv(os.path.join(data_path, 'load_p.csv'))
            .drop(columns=['time'], errors='ignore')
            .values
        )
        self.load_q = (
            pd.read_csv(os.path.join(data_path, 'load_q.csv'))
            .drop(columns=['time'], errors='ignore')
            .values
        )

        # 维度校验
        self._validate(num_loads_in_grid)

    def _validate(self, num_loads_in_grid: int) -> None:
        """校验时序数据与电网模型的维度一致性"""
        num_cols = self.load_p.shape[1]
        assert num_loads_in_grid == num_cols, (
            f"维度致命错误：电网有 {num_loads_in_grid} 个负荷，"
            f"但 load_p 有 {num_cols} 列！"
        )
        assert self.load_p.shape == self.load_q.shape, (
            "load_p 和 load_q 的矩阵形状不一致！"
        )

    @property
    def max_steps(self) -> int:
        """episode 最大步数（等于时序数据行数）"""
        return len(self.load_p)

    def apply_to_net(self, net, step: int) -> None:
        """
        将第 step 步的所有时序数据注入电网模型。
        Args:
            net: pandapower 电网对象
            step: 当前时步
        """
        safe_step = min(step, self.max_steps - 1)
        net.load.p_mw = self.load_p[safe_step]
        net.load.q_mvar = self.load_q[safe_step]
