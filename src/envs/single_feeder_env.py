import gymnasium as gym
import os
from gymnasium.spaces import Box
import pandapower as pp
import pandas as pd
import numpy as np
import copy

class SingleFeederStorageEnv(gym.Env):
    def __init__(self, env_config=None):
        super().__init__()
        
        # ==========================================
        # 1. 加载底本与提取设备信息
        # ==========================================
        # 路径更新：根据你的截图目录调整
        data_path = env_config['data_path']
        self.base_net = pp.from_pickle(os.path.join(data_path, 'generated', 'topology', '20260320_test_data.p'))
        
        self.storage_ids = self.base_net.storage.index.tolist()
        self.num_storages = len(self.storage_ids)
        
        # 提取储能物理参数 (假设步长为 15分钟 = 0.25小时)
        self.dt = 0.25 
        # 如果你的测试数据里还没配 max_e_mwh，这里给个默认值兜底 (2MWh, 1MW)
        self.storage_max_e = self.base_net.storage.max_e_mwh.values if 'max_e_mwh' in self.base_net.storage else np.full(self.num_storages, 2.0)
        self.storage_max_p = self.base_net.storage.max_p_mw.values if 'max_p_mw' in self.base_net.storage else np.full(self.num_storages, 1.0)
        
        # ==========================================
        # 2. 极速读取时序矩阵 (P和Q同时读取)
        # ==========================================
        self.load_p = pd.read_csv(os.path.join(data_path, 'generated', 'load', 'load_p.csv')).drop(columns=['time'], errors='ignore').values
        self.load_q = pd.read_csv(os.path.join(data_path, 'generated', 'load', 'load_q.csv')).drop(columns=['time'], errors='ignore').values
        
        # 如果你也有光伏(sgen)的数据，取消下面的注释并改好路径即可：
        # self.sgen_p = pd.read_csv("data/generated/gen/gen_p.csv").drop(columns=['time'], errors='ignore').values
        # self.sgen_q = pd.read_csv("data/generated/gen/gen_q.csv").drop(columns=['time'], errors='ignore').values
        
        self.max_steps = len(self.load_p)
        
        # ==========================================
        # 3. ⚠️ 硬核维度校验 (Sanity Check)
        # ==========================================
        num_loads_in_grid = len(self.base_net.load)
        num_loads_in_matrix = self.load_p.shape[1]
        
        assert num_loads_in_grid == num_loads_in_matrix, \
            f"❌ 维度致命错误：电网有 {num_loads_in_grid} 个负荷，但 load_p 有 {num_loads_in_matrix} 列！"
        assert self.load_p.shape == self.load_q.shape, "❌ load_p 和 load_q 的矩阵形状不一致！"
        
        # 初始化状态变量
        self.current_step = 0
        self.net = None 
        self.soc = np.full(self.num_storages, 0.5, dtype=np.float32) # 所有储能初始电量 50%
        
        # 定义动作空间 (每个储能 -1.0满充 到 1.0满放)
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.num_storages,), dtype=np.float32)
        # 简单定义观测空间 (这里假设传出所有节点电压 + 储能SOC)
        num_buses = len(self.base_net.bus)
        self.observation_space = Box(low=0.0, high=2.0, shape=(num_buses + self.num_storages,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):    
        self.current_step = 0
        self.soc = np.full(self.num_storages, 0.5, dtype=np.float32) # 重置电量
        
        self.net = copy.deepcopy(self.base_net)
        self._update_background_power()
        
        # 初始跑一次潮流，获得真实的初始观测
        pp.runpp(self.net)
        return self._get_obs(), {}

    def step(self, action):
        self._update_background_power()
        
        # ==========================================
        # 动作下发与物理兜底逻辑
        # ==========================================
        for i, sid in enumerate(self.storage_ids):
            # 将 RL 输出的 [-1, 1] 转换为 MW
            desired_p = action[i] * self.storage_max_p[i]
            
            # 物理兜底：如果电池满了还想充(动作<0)，或空了还想放(动作>0)，强制出力归零
            if self.soc[i] >= 0.95 and desired_p > 0:
                desired_p = 0.0
            elif self.soc[i] <= 0.05 and desired_p < 0:
                desired_p = 0.0
                
            self.net.storage.at[sid, 'p_mw'] = desired_p
            
            # 更新 SOC
            energy_change = desired_p * self.dt # 充电(-p)使能量增加
            self.soc[i] += energy_change / self.storage_max_e[i]
            self.soc[i] = np.clip(self.soc[i], 0.0, 1.0)
            
        # ==========================================
        # 运行潮流与奖励结算
        # ==========================================
        try:
            pp.runpp(self.net)
            reward = self._calculate_reward()
        except pp.LoadflowNotConverged:
            reward = -100.0 # 潮流发散直接给重罚
            return self._get_obs(), reward, True, False, {"error": "Loadflow diverged"}

        self.current_step += 1
        done = bool(self.current_step >= self.max_steps)
        
        return self._get_obs(), reward, done, False, {}

    def _update_background_power(self):
        safe_step = min(self.current_step, self.max_steps - 1)
        
        # P和Q双管齐下极速赋值
        self.net.load.p_mw = self.load_p[safe_step]
        self.net.load.q_mvar = self.load_q[safe_step]
        
        # 如果有光伏：
        # self.net.sgen.p_mw = self.sgen_p[safe_step]
        # self.net.sgen.q_mvar = self.sgen_q[safe_step]
        
    def _get_obs(self):
        vm = self.net.res_bus.vm_pu.values.astype(np.float32)
        # 将电压和当前SOC拼接成一个一维数组给 AI
        return np.concatenate([vm, self.soc])
        
    def _calculate_reward(self):
        """
        核心业务目标：降低主网向上的功率倒送
        """
        # 获取变电站主网关口有功功率
        p_ext_grid = self.net.res_bus.at[0, 'p_mw']
        # print(self.net.res_bus)
        # exit()
        reward = 0.0
        # 如果 p_ext_grid < 0，说明配网正在向主网倒送电
        if p_ext_grid < 0:
            # 倒送越多，惩罚越大 (乘以10放大权重)
            reward += p_ext_grid * 10.0 
            
        # 辅助约束：轻微惩罚电压越限，保证电网安全
        vm = self.net.res_bus.vm_pu.values
        v_violations = np.sum(np.maximum(vm - 1.05, 0)) + np.sum(np.maximum(0.95 - vm, 0))
        reward -= v_violations * 20.0
        
        return reward