import copy
import numpy as np
import pandapower as pp


class GridSimulator:
    """
    封装 pandapower 电网物理仿真的所有调用。
    环境主类不直接 import pandapower，所有与仿真器的交互都经过此类。
    未来如果需要替换仿真器（如 PowerModels.jl），只需修改此类。
    """

    def __init__(self, net_path: str):
        self.base_net = pp.from_pickle(net_path)
        self.net = None

    # ------ 基础信息（从 base_net 中提取，只读） ------

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

    # ------ 仿真操作 ------

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
