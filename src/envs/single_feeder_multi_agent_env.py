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

    v2: 引入局部异构特征（拓扑阻抗 + 本地电压 + 本地净负荷）和随机 SOC
        初始化，打破 Shared Policy 下的动作克隆/对称性问题。

    本类仅作为「胶水层」，编排各 Component 的调用顺序。
    具体逻辑分布在 src/envs/components/ 下的各模块中：
      - GridSimulator:          电网物理仿真 + 拓扑特征预计算
      - TimeSeriesDataManager:  时序数据加载与管理
      - ActionProcessor:        动作解码 + SOC 维护（随机初始化）
      - ObservationBuilder:     8 维异构观测空间构建
      - RewardCalculator:       per-agent 奖励计算
    """

    def __init__(self, env_config=None):
        super().__init__()
        env_config = env_config or {}
        data_path = env_config['data_path']

        # ==========================================
        # 1. 组装各组件
        # ==========================================
        net_path = os.path.join(data_path, 'topology.p')
        self.simulator = GridSimulator(net_path)
        self.data_mgr = TimeSeriesDataManager(data_path, self.simulator.num_loads)
        self.action_proc = ActionProcessor(
            storage_ids=self.simulator.storage_ids,
            max_p=self.simulator.storage_max_p,
            max_e=self.simulator.storage_max_e,
            dt=0.25,
        )
        # v2: ObservationBuilder 不再需要 num_buses，观测维度固定为 8
        self.obs_builder = ObservationBuilder()
        self.reward_calc = RewardCalculator(env_config.get('reward_config'))

        # ==========================================
        # 2. 缓存静态拓扑特征（从 GridSimulator 获取）
        # ==========================================
        # storage_r_eq / storage_x_eq 是在 GridSimulator.__init__ 中
        # 通过 networkx 最短路径算法预计算的归一化累积阻抗
        self._r_eq = self.simulator.storage_r_eq  # shape: (num_storages,)
        self._x_eq = self.simulator.storage_x_eq  # shape: (num_storages,)

        # ==========================================
        # 3. 多智能体 ID 与映射
        # ==========================================
        num_storages = self.simulator.num_storages
        self.possible_agents = [f"storage_{i}" for i in range(num_storages)]
        self.agents = self.possible_agents[:]
        self._agent_ids = set(self.agents)
        self._agent_to_idx = {aid: i for i, aid in enumerate(self.possible_agents)}

        # ==========================================
        # 4. 空间定义
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
        # 5. Episode 状态
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
        # v2: action_proc.reset() 现在为每个储能随机采样初始 SOC ∈ [0.2, 0.8]
        self.action_proc.reset()
        self.data_mgr.apply_to_net(self.simulator.net, 0)
        self.simulator.run_powerflow()

        return self._build_obs_dict(), {}

    def step(self, action_dict):
        # 1. 注入当前时步的背景负荷
        self.data_mgr.apply_to_net(self.simulator.net, self.current_step)

        # 2. 下发智能体动作并收集实际出力
        p_bat_values = []
        for agent_id, action in action_dict.items():
            idx = self._agent_to_idx[agent_id]
            actual_p = self.action_proc.apply(idx, action[0], self.simulator)
            p_bat_values.append(actual_p)

        # 3. 运行潮流计算
        converged = self.simulator.run_powerflow()

        # 4. 计算奖励
        self.current_step += 1
        is_done = (self.current_step >= self.max_steps) or (not converged)

        if converged:
            # v2: 返回 per-agent 奖励
            reward_info = self.reward_calc.calculate(
                self.simulator, self.action_proc.soc, p_bat_values
            )
        else:
            num_agents = len(self.possible_agents)
            reward_info = {
                "per_agent_rewards": [-100.0] * num_agents,
                "reward_reverse": 0.0, "reward_buy": 0.0,
                "reward_soc": [0.0] * num_agents,
                "reward_action": [0.0] * num_agents,
                "p_grid_actual": 0.0,
            }

        # 5. 构建返回字典
        obs_dict = {}
        reward_dict = {}
        terminated_dict = {}
        truncated_dict = {"__all__": False}
        info_dict = {}

        for agent_id in self.agents:
            idx = self._agent_to_idx[agent_id]

            # v2: 观测构建传入 agent_idx 以提取局部异构特征
            obs_dict[agent_id] = (
                self.obs_builder.build(
                    self.simulator,
                    agent_idx=idx,
                    soc_value=self.action_proc.get_soc(idx),
                    current_step=self.current_step,
                    r_eq=self._r_eq[idx],
                    x_eq=self._x_eq[idx],
                )
                if converged
                else self.obs_builder.build_zero()
            )

            # v2: 每个智能体获得各自独立的奖励
            reward_dict[agent_id] = reward_info["per_agent_rewards"][idx]

            terminated_dict[agent_id] = is_done

            # info 包含全局 + 该智能体的局部奖励分量
            info_dict[agent_id] = {
                "reward_reverse": reward_info["reward_reverse"],
                "reward_buy": reward_info["reward_buy"],
                "reward_soc": reward_info["reward_soc"][idx],
                "reward_action": reward_info["reward_action"][idx],
                "p_grid_actual": reward_info["p_grid_actual"],
                "current_soc": float(self.action_proc.get_soc(idx)),
            }

        terminated_dict["__all__"] = is_done

        if is_done:
            self.agents = []

        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _build_obs_dict(self) -> dict:
        """为每个 agent 构建异构观测（在 reset 和发散恢复时调用）"""
        return {
            aid: self.obs_builder.build(
                self.simulator,
                agent_idx=self._agent_to_idx[aid],
                soc_value=self.action_proc.get_soc(self._agent_to_idx[aid]),
                current_step=self.current_step,
                r_eq=self._r_eq[self._agent_to_idx[aid]],
                x_eq=self._x_eq[self._agent_to_idx[aid]],
            )
            for aid in self.agents
        }