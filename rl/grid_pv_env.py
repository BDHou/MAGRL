# -*- coding: utf-8 -*-
"""
rl/grid_pv_env.py

Gymnasium 环境：把 OnlineBackend 包成标准 RL 接口。

Observation:
  - GNN graph embedding g: (hidden_dim,) 来自预训练 backbone (由 GNNObsEncoder 提供)
  - 每个 PV 局部状态 [p_mw, q_mvar, bus_vm_pu]: (n_pv * 3,)
  - 合并成 flat vector: (hidden_dim + n_pv * 3,)

  注意：本环境 _不_ 在内部跑 GNN，obs 直接返回图原始特征 flat 化 + PV 局部状态。
  GNN encoding 交给 RLlib 自定义 model (gnn_obs_encoder.py) 处理。
  这样 obs_space 里装的是"原始电网节点特征拼接"，GNN model 读入后再出 embedding。

Action:
  Box(-1, 1, (n_pv,)) 每个 PV 的无功分数 q_frac ∈ [-1, 1]
  实际无功 = q_frac * qlim (由 OnlineBackend._apply_action 处理)

Reward:
  -w_vviol  * num_v_viol
  -w_rpf    * num_rpf_lines
  -w_export * export      (0/1)
  -w_pf_fail* pf_fail_penalty  (大惩罚)

Episode:
  每条 episode 随机采一个 scenario_seed，跑 episode_len 步（默认24步=1天）。
"""
from __future__ import annotations

import copy
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from scenario.base_scenario import ScenarioConfig
from scenario.online_backend import OnlineBackend
from scenario.risk_events import RiskConfig


# 默认奖励权重
DEFAULT_REWARD_WEIGHTS = dict(
    w_vviol=1.0,
    w_rpf=0.5,
    w_export=0.3,
    w_pf_fail=10.0,
)


class GridPVEnv(gym.Env):
    """
    配电网 PV 无功控制 Gymnasium 环境。

    参数
    ----
    feeder_name : str
        pandapower 中的 feeder 名称，如 "case33bw"
    cfg : ScenarioConfig | None
        场景配置，None 则用默认值
    risk_cfg : RiskConfig | None
        风险事件配置，None 则不启用
    episode_len : int
        每条 episode 的步数，默认 24（一天）
    reward_weights : dict | None
        奖励各项的权重，None 则用默认值
    seed : int | None
        全局随机种子（控制 episode_seed 的生成）
    max_pf_failures : int
        一条 episode 中最多允许连续潮流失败几次，超过则 truncated=True
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        feeder_name: str = "case33bw",
        cfg: Optional[ScenarioConfig] = None,
        risk_cfg: Optional[RiskConfig] = None,
        episode_len: int = 24,
        reward_weights: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
        max_pf_failures: int = 5,
    ):
        super().__init__()

        self.feeder_name = feeder_name
        self.cfg = cfg or ScenarioConfig()
        self.risk_cfg = risk_cfg
        self.episode_len = int(episode_len)
        self.rw = {**DEFAULT_REWARD_WEIGHTS, **(reward_weights or {})}
        self.max_pf_failures = int(max_pf_failures)

        # 用于在 reset 时生成不同的 episode_seed
        self._master_rng = np.random.default_rng(seed)

        # 先建一个 backend 探查维度
        _probe = OnlineBackend(
            feeder_name=self.feeder_name,
            cfg=self.cfg,
            mode="q_frac",
            enable_curtail_action=False,
            risk_cfg=None,
            scenario_seed=0,
        )
        self.n_pv: int = _probe.n_pv
        self.n_bus: int = len(_probe.net.bus)
        self.node_feat_dim: int = 8      # build_graph_tensors 输出 x shape= (n_bus, 8)

        # action space: n_pv 个连续 q_frac ∈ [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_pv,),
            dtype=np.float32,
        )

        # observation space:
        #   节点特征矩阵 flat: (n_bus * node_feat_dim,)
        #   + PV 局部状态 (n_pv * 3,): [p_mw, q_mvar, bus_vm]
        self._node_obs_dim = self.n_bus * self.node_feat_dim
        self._pv_obs_dim = self.n_pv * 3
        obs_dim = self._node_obs_dim + self._pv_obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # 运行时后端（reset 时重建）
        self._backend: Optional[OnlineBackend] = None
        self._step_count: int = 0
        self._pf_fail_count: int = 0
        self._episode_seed: Optional[int] = None

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # 每条 episode 用不同的 scenario_seed（保证多样性）
        if seed is not None:
            self._master_rng = np.random.default_rng(seed)
        ep_seed = int(self._master_rng.integers(0, 2**31 - 1))
        self._episode_seed = ep_seed

        # 重建 backend（每条 episode 重新建，保证时间序列/云量重新生成）
        self._backend = OnlineBackend(
            feeder_name=self.feeder_name,
            cfg=self.cfg,
            mode="q_frac",
            enable_curtail_action=False,
            risk_cfg=self.risk_cfg,
            scenario_seed=ep_seed,
            risk_seed=ep_seed + 999 if self.risk_cfg is not None else None,
        )

        self._step_count = 0
        self._pf_fail_count = 0

        sr = self._backend.reset(t0=0)
        obs = self._make_obs(sr)
        info = self._make_info(sr)
        return obs, info

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        sr = self._backend.step(action)
        self._step_count += 1

        # 潮流失败处理
        if not sr.ok:
            self._pf_fail_count += 1
            reward = -self.rw["w_pf_fail"]
            obs = self._zero_obs()
            terminated = False
            truncated = (
                self._step_count >= self.episode_len
                or self._pf_fail_count >= self.max_pf_failures
            )
            info = self._make_info(sr)
            return obs, float(reward), terminated, truncated, info

        self._pf_fail_count = 0  # 连续失败计数归零

        reward = self._compute_reward(sr.metrics)
        obs = self._make_obs(sr)
        info = self._make_info(sr)

        terminated = False  # 电网没有"赢/输"的终态
        truncated = self._step_count >= self.episode_len

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # obs / reward helpers
    # ------------------------------------------------------------------
    def _make_obs(self, sr) -> np.ndarray:
        """
        把 StepResult 转成 flat numpy obs vector。
        如果 sr.ok=False，返回零向量（由 _zero_obs 处理）。
        """
        if not sr.ok or sr.obs_graph is None:
            return self._zero_obs()

        # 节点特征 flat
        x: np.ndarray = sr.obs_graph["x"]             # (n_bus, 8)
        node_flat = x.reshape(-1).astype(np.float32)   # (n_bus*8,)

        # PV 局部状态 [p_mw, q_mvar, bus_vm_pu] × n_pv
        pv_state = self._get_pv_local_state()          # (n_pv * 3,)

        obs = np.concatenate([node_flat, pv_state], axis=0)
        return obs.astype(np.float32)

    def _zero_obs(self) -> np.ndarray:
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _get_pv_local_state(self) -> np.ndarray:
        """读取每个 PV 的 [p_mw, q_mvar, bus_vm_pu]。"""
        net = self._backend.net
        buses = net.bus.index.to_numpy()
        bus_id_map = {int(b): i for i, b in enumerate(buses)}

        pv_feats = []
        for i, sid in enumerate(net._pv_ids):
            p = float(net.sgen.at[sid, "p_mw"])
            q = float(net.sgen.at[sid, "q_mvar"])
            b = int(net.sgen.at[sid, "bus"])
            # 潮流收敛后 res_bus 有电压；否则用 1.0 兜底
            try:
                vm = float(net.res_bus.at[b, "vm_pu"])
            except Exception:
                vm = 1.0
            pv_feats.extend([p, q, vm])
        return np.array(pv_feats, dtype=np.float32)

    def _compute_reward(self, metrics: Dict[str, Any]) -> float:
        """
        reward = -w_vviol * num_v_viol
                 -w_rpf   * num_rpf_lines
                 -w_export* export(0/1)
        """
        r = 0.0
        r -= self.rw["w_vviol"]  * float(metrics.get("num_v_viol", 0))
        r -= self.rw["w_rpf"]    * float(metrics.get("num_rpf_lines", 0))
        r -= self.rw["w_export"] * float(metrics.get("export", 0))
        return r

    def _make_info(self, sr) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "pf_ok": bool(sr.ok),
            "t": int(sr.meta.get("t", -1)),
            "step_count": self._step_count,
            "episode_seed": self._episode_seed,
        }
        if sr.ok and sr.metrics:
            info.update({
                "num_v_viol":    sr.metrics.get("num_v_viol", 0),
                "num_rpf_lines": sr.metrics.get("num_rpf_lines", 0),
                "export":        sr.metrics.get("export", 0),
                "max_vm":        sr.metrics.get("max_vm", float("nan")),
                "min_vm":        sr.metrics.get("min_vm", float("nan")),
                "pv_total_p":    sr.metrics.get("pv_total_p", 0.0),
            })
        return info

    # ------------------------------------------------------------------
    # Properties (for external use)
    # ------------------------------------------------------------------
    @property
    def obs_dim(self) -> int:
        return int(self.observation_space.shape[0])

    @property
    def act_dim(self) -> int:
        return int(self.action_space.shape[0])
