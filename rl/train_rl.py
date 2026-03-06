# -*- coding: utf-8 -*-
"""
rl/train_rl.py

Ray RLlib PPO 训练入口。

用法（主方法，需先跑过 supervised_pretrain）：
  python -m rl.train_rl \
    --ckpt checkpoints/gnn_pretrain/best.pt \
    --save_dir checkpoints/rl_ppo \
    --num_iters 200 \
    --episode_len 24

消融对比（无预训练 GNN，用纯 MLP）：
  python -m rl.train_rl --ckpt "" --save_dir checkpoints/rl_mlp --backbone mlp

冒烟测试（快速验证能跑通）：
  python -m rl.train_rl --smoke_test

依赖：
  pip install "ray[rllib]>=2.9.0" gymnasium
"""
from __future__ import annotations

import argparse
import os
import json
import warnings
from typing import Optional

# 抦截所有 DeprecationWarning （Ray/RLlib 内部大量不需要用户处理的把弃用 warning）
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*multi_gpu_train_one_step.*")
warnings.filterwarnings("ignore", message=".*UnifiedLogger.*")
warnings.filterwarnings("ignore", message=".*Logger interface.*")

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

# RLlib imports
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.utils.annotations import override

from rl.grid_pv_env import GridPVEnv
from rl.gnn_obs_encoder import GNNObsEncoder, build_edge_index_for_env
from scenario.base_scenario import ScenarioConfig
from scenario.risk_events import RiskConfig


# ------------------------------------------------------------------
# RLlib 自定义 Model：GNN 做 feature extractor
# ------------------------------------------------------------------
class GNNPolicyModel(TorchModelV2, torch.nn.Module):
    """
    RLlib TorchModelV2 包装 GNNObsEncoder。

    obs_flat (B, obs_dim)
      → GNNObsEncoder → feat (B, gnn_out_dim)
      → policy head (logits / value)
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs,   # 接受所有额外参数，避免 catalog.py:548 warning
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        torch.nn.Module.__init__(self)

        custom_cfg = model_config.get("custom_model_config", {})
        n_bus:        int = int(custom_cfg["n_bus"])
        node_feat_dim:int = int(custom_cfg.get("node_feat_dim", 8))
        n_pv:         int = int(custom_cfg["n_pv"])
        hidden_dim:   int = int(custom_cfg.get("hidden_dim", 128))
        ckpt_path:    str = custom_cfg.get("ckpt_path", "")
        frozen:       bool= bool(custom_cfg.get("frozen", True))
        backbone:     str = custom_cfg.get("backbone", "sage")

        # edge_index 作为 buffer 注册（设备无关）
        edge_index_np = custom_cfg["edge_index"]  # numpy array (2, E)
        self.register_buffer(
            "edge_index",
            torch.tensor(edge_index_np, dtype=torch.long),
        )

        # GNN encoder
        self.encoder = GNNObsEncoder(
            n_bus=n_bus,
            node_feat_dim=node_feat_dim,
            n_pv=n_pv,
            hidden_dim=hidden_dim,
            ckpt_path=ckpt_path if ckpt_path else None,
            frozen=frozen,
            backbone=backbone,
        )

        feat_dim = self.encoder.out_dim  # hidden_dim + n_pv*3

        # policy head：输出 action distribution 参数
        # PPO 连续动作：输出 mean (num_outputs = action_dim)
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, num_outputs),
        )

        # value head：输出标量 V(s)
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 1),
        )

        self._feat: Optional[torch.Tensor] = None  # 缓存供 value_function 用

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()   # (B, obs_dim)
        feat = self.encoder(obs, self.edge_index)  # (B, feat_dim)
        self._feat = feat
        logits = self.policy_head(feat)        # (B, num_outputs)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._feat is not None
        return self.value_head(self._feat).squeeze(-1)  # (B,)


# ------------------------------------------------------------------
# 主训练函数
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="RLlib PPO for GridPV control")

    # 环境
    ap.add_argument("--feeder", type=str, default="case33bw")
    ap.add_argument("--episode_len", type=int, default=24)
    ap.add_argument("--risk", action="store_true", help="启用风险事件")

    # 模型
    ap.add_argument("--ckpt", type=str, default="checkpoints/gnn_pretrain/best.pt",
                    help="预训练 GNN ckpt 路径，空串则随机初始化（ablation）")
    ap.add_argument("--backbone", type=str, default="sage",
                    choices=["sage", "gcn", "gat", "mlp"])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--frozen", type=int, default=1,
                    help="1=冻结GNN权重, 0=端到端微调")

    # 训练
    ap.add_argument("--num_iters", type=int, default=200,
                    help="RLlib 训练迭代次数")
    ap.add_argument("--num_workers", type=int, default=4,
                    help="RLlib rollout workers 数量（并行采样 env 个数）")
    ap.add_argument("--train_batch", type=int, default=4000)
    ap.add_argument("--mini_batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)

    # 保存
    ap.add_argument("--save_dir", type=str, default="checkpoints/rl_ppo")
    ap.add_argument("--save_freq", type=int, default=20,
                    help="每多少 iter 保存一次 checkpoint")

    # 开关
    ap.add_argument("--smoke_test", action="store_true",
                    help="冒烟测试模式：只跑 3 个 iter，快速验证")

    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.smoke_test:
        args.num_iters = 3
        args.num_workers = 1           # 不能用 0：old API stack 下 0 worker 时 episode 统计不填充
        args.train_batch = 200
        args.mini_batch = 64
        print("[smoke_test] 冒烟测试模式：num_iters=3, train_batch=200, mini_batch=64, num_workers=1")

    # ------------------------------------------------------------------
    # 建一个临时 env 来探查维度 / edge_index
    # ------------------------------------------------------------------
    risk_cfg = None
    if args.risk:
        risk_cfg = RiskConfig(
            enable_contingency=True,
            contingency_type="n-1",
            contingency_prob_per_step=0.03,
            contingency_duration_steps=6,
            contingency_elements=("line",),
            enable_line_derating=True,
            derate_prob_per_step=0.02,
            derate_duration_steps=6,
            enable_overload=True,
            overload_prob_per_step=0.02,
            overload_duration_steps=6,
        )

    probe_env = GridPVEnv(
        feeder_name=args.feeder,
        risk_cfg=risk_cfg,
        episode_len=args.episode_len,
        seed=0,
    )
    probe_env.reset()
    edge_index_np = build_edge_index_for_env(probe_env).numpy()  # (2, E)

    n_bus = probe_env.n_bus
    n_pv = probe_env.n_pv
    obs_dim = probe_env.obs_dim
    act_dim = probe_env.act_dim
    probe_env.close()

    print(f"[env] feeder={args.feeder} n_bus={n_bus} n_pv={n_pv}")
    print(f"[env] obs_dim={obs_dim} act_dim={act_dim}")
    print(f"[model] backbone={args.backbone} hidden={args.hidden} frozen={bool(args.frozen)}")
    print(f"[model] ckpt={args.ckpt if args.ckpt else 'None (random init)'}")

    # ------------------------------------------------------------------
    # 注册自定义 Model 和 Env
    # ------------------------------------------------------------------
    ModelCatalog.register_custom_model("gnn_policy_model", GNNPolicyModel)

    def env_creator(env_config):
        return GridPVEnv(
            feeder_name=env_config.get("feeder_name", args.feeder),
            risk_cfg=risk_cfg,
            episode_len=env_config.get("episode_len", args.episode_len),
            seed=env_config.get("seed", None),
        )

    tune.register_env("GridPVEnv-v0", env_creator)

    # ------------------------------------------------------------------
    # RLlib PPO Config
    # ------------------------------------------------------------------
    # 抑制 Ray 加速器覆盖环境变量的 FutureWarning
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    # 让 Ray worker 子进程也继承 warning 过滤（否则 worker 里的 DeprecationWarning 仍会打印）
    os.environ.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning,ignore::FutureWarning")
    ray.init(ignore_reinit_error=True)

    config = (
        PPOConfig()
        # 关闭新 API Stack，保持与 TorchModelV2 / ModelCatalog 的兼容性
        # （新 API 需要改用 RLModule，我们先用旧 API保持简单）
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env="GridPVEnv-v0",
            env_config={
                "feeder_name": args.feeder,
                "episode_len": args.episode_len,
            },
        )
        .framework("torch")
        .training(
            model={
                "custom_model": "gnn_policy_model",
                "custom_model_config": {
                    "n_bus":         n_bus,
                    "node_feat_dim": 8,
                    "n_pv":          n_pv,
                    "hidden_dim":    args.hidden,
                    "ckpt_path":     args.ckpt,
                    "frozen":        bool(args.frozen),
                    "backbone":      args.backbone,
                    "edge_index":    edge_index_np,
                },
            },
            lr=args.lr,
            gamma=args.gamma,
            lambda_=0.95,
            clip_param=0.2,
            train_batch_size=args.train_batch,
            minibatch_size=args.mini_batch,
            num_epochs=10,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
        )
        .env_runners(
            num_env_runners=args.num_workers,
            rollout_fragment_length="auto",
        )
        .resources(num_gpus=0)
        .debugging(
            log_sys_usage=False,   # 关闭 CPU/RAM 监控（不装 gputil 也不会 warn）
            log_level="ERROR",     # 只输出真正的错误
        )
        .evaluation(
            evaluation_interval=args.save_freq,
            evaluation_num_env_runners=1,
            evaluation_duration=10,
            evaluation_duration_unit="episodes",
        )
        .reporting(
            # 使用新日志 API，避免 UnifiedLogger/JsonLogger/CSVLogger deprecated warning
            metrics_num_episodes_for_smoothing=100,
        )
    )
    # 覆盖 logger_config 为 NoopLogger，彼底关闭旧日志输出
    config.logger_config = {"type": "ray.tune.logger.NoopLogger"}

    algo = config.build_algo()

    # ------------------------------------------------------------------
    # 训练循环
    # ------------------------------------------------------------------
    best_eval_reward = -float("inf")
    best_ckpt_path = os.path.join(args.save_dir, "best")
    run_log = []

    for i in range(1, args.num_iters + 1):
        result = algo.train()

        # Ray 2.54 old API stack：episode 统计在 result["env_runners"] 下
        env_runners = result.get("env_runners", {})
        ep_reward_mean = env_runners.get("episode_reward_mean", float("nan"))
        ep_len_mean    = env_runners.get("episode_len_mean",    float("nan"))
        # counters 子字典在所有 worker 模式下最稳定
        counters = result.get("counters", {})
        timesteps = (
            counters.get("num_env_steps_sampled")
            or result.get("num_env_steps_sampled")
            or result.get("timesteps_total")
            or 0
        )

        print(
            f"[iter {i:04d}/{args.num_iters}] "
            f"reward_mean={ep_reward_mean:.3f} "
            f"ep_len={ep_len_mean:.1f} "
            f"steps={timesteps}"
        )

        log_entry = {
            "iter": i,
            "episode_reward_mean": ep_reward_mean,
            "episode_len_mean": ep_len_mean,
            "timesteps_total": timesteps,
        }

        # 评估结果（如果有）
        if "evaluation" in result:
            eval_runners = result["evaluation"].get("env_runners", result["evaluation"])
            eval_reward = eval_runners.get("episode_reward_mean", float("nan"))
            log_entry["eval_reward_mean"] = eval_reward
            print(f"  [eval] reward_mean={eval_reward:.3f}")

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                algo.save(best_ckpt_path)
                print(f"  [best] saved to {best_ckpt_path}")

        run_log.append(log_entry)

        # 定期保存
        if i % args.save_freq == 0:
            ckpt = algo.save(os.path.join(args.save_dir, f"iter_{i:04d}"))
            ckpt_path = getattr(ckpt, "path", None) or getattr(getattr(ckpt, "checkpoint", None), "path", str(ckpt))
            print(f"  [ckpt] saved: {ckpt_path}")

    # 最终保存
    final_ckpt = algo.save(os.path.join(args.save_dir, "final"))
    final_path = getattr(final_ckpt, "path", None) or getattr(getattr(final_ckpt, "checkpoint", None), "path", str(final_ckpt))
    print(f"[done] final checkpoint: {final_path}")

    # 保存训练日志
    log_path = os.path.join(args.save_dir, "train_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2, ensure_ascii=False)
    print(f"[done] train log: {log_path}")

    ray.shutdown()


if __name__ == "__main__":
    main()
