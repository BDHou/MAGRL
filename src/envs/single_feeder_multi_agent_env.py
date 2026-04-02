import os
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import Box, Dict

from src.envs.components import (
    GridSimulator,
    TimeSeriesDataManager,
    ActionProcessor,
    ObservationBuilder,
    RewardCalculator,
)


class MultiFeederStorageEnv(MultiAgentEnv):
    """
    兼容 MAPPO 架构的配电网多储能控制环境。

    本类仅作为「胶水层」，编排各 Component 的调用顺序。
    具体逻辑分布在 src/envs/components/ 下的各模块中：
      - GridSimulator:          电网物理仿真
      - TimeSeriesDataManager:  时序数据加载与管理
      - ActionProcessor:        动作解码 + SOC 维护
      - ObservationBuilder:     观测空间构建
      - RewardCalculator:       奖励计算
    """

    def __init__(self, env_config=None):
        super().__init__()
        env_config = env_config or {}
        data_path = env_config.get('data_path', '/Users/steven/Documents/GridProjects/YC2/data')

        # ==========================================
        # 1. 组装各组件
        # ==========================================
        net_path = os.path.join(data_path, 'generated', 'topology', '20260320_test_data.p')
        self.simulator = GridSimulator(net_path)
        self.data_mgr = TimeSeriesDataManager(data_path, self.simulator.num_loads)
        self.action_proc = ActionProcessor(
            storage_ids=self.simulator.storage_ids,
            max_p=self.simulator.storage_max_p,
            max_e=self.simulator.storage_max_e,
            dt=0.25,
        )
        self.obs_builder = ObservationBuilder(self.simulator.num_buses)
        self.reward_calc = RewardCalculator(env_config.get('reward_config'))

        # ==========================================
        # 2. 多智能体 ID 与映射
        # ==========================================
        num_storages = self.simulator.num_storages
        self.possible_agents = [f"storage_{i}" for i in range(num_storages)]
        self.agents = self.possible_agents[:]
        self._agent_ids = set(self.agents)
        self._agent_to_idx = {aid: i for i, aid in enumerate(self.possible_agents)}

        # ==========================================
        # 3. 空间定义
        # ==========================================
        self.single_action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.single_obs_space = self.obs_builder.get_space()

        self.observation_space = Dict({
            aid: self.single_obs_space for aid in self.possible_agents
        })
        self.action_space = Dict({
            aid: self.single_action_space for aid in self.possible_agents
        })

        # ==========================================
        # 4. Episode 状态
        # ==========================================
        self.max_steps = self.data_mgr.max_steps
        self.current_step = 0

    # ------------------------------------------------------------------
    # Gymnasium / RLlib 接口
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.agents = self.possible_agents[:]

        self.simulator.reset()
        self.action_proc.reset()
        self.data_mgr.apply_to_net(self.simulator.net, 0)
        self.simulator.run_powerflow()

        return self._build_obs_dict(), {}

    def step(self, action_dict):
        # 1. Inject background load for current timestep
        self.data_mgr.apply_to_net(self.simulator.net, self.current_step)

        # 2. Apply agent actions and collect actual battery power
        p_bat_values = []
        for agent_id, action in action_dict.items():
            idx = self._agent_to_idx[agent_id]
            actual_p = self.action_proc.apply(idx, action[0], self.simulator)
            p_bat_values.append(actual_p)

        # 3. Run power flow
        converged = self.simulator.run_powerflow()

        # 4. Compute reward
        self.current_step += 1
        is_done = (self.current_step >= self.max_steps) or (not converged)

        if converged:
            reward_info = self.reward_calc.calculate(
                self.simulator, self.action_proc.soc, p_bat_values
            )
            global_reward = reward_info["total"]
        else:
            reward_info = {
                "total": -100.0,
                "reward_reverse": 0.0, "reward_buy": 0.0,
                "reward_soc": 0.0, "reward_action": 0.0,
                "p_grid_actual": 0.0,
            }
            global_reward = -100.0

        # 5. Build return dicts
        obs_dict = {}
        reward_dict = {}
        terminated_dict = {}
        truncated_dict = {"__all__": False}
        info_dict = {}

        for agent_id in self.agents:
            idx = self._agent_to_idx[agent_id]
            obs_dict[agent_id] = (
                self.obs_builder.build(self.simulator, self.action_proc.get_soc(idx), self.current_step)
                if converged
                else self.obs_builder.build_zero()
            )
            reward_dict[agent_id] = global_reward
            terminated_dict[agent_id] = is_done
            info_dict[agent_id] = {
                **reward_info, "current_soc": float(self.action_proc.get_soc(idx)),
            }

        terminated_dict["__all__"] = is_done

        if is_done:
            self.agents = []

        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _build_obs_dict(self) -> dict:
        return {
            aid: self.obs_builder.build(self.simulator, self.action_proc.get_soc(self._agent_to_idx[aid]), self.current_step)
            for aid in self.agents
        }