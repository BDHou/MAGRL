# -*- coding: utf-8 -*-
"""
rl/eval_rl.py

评估 & 对比实验脚本。
对比 4 个方案（在同一批测试 episode 上）：
  1. Baseline-Rule   : 规则 Volt-Var（使用 cfg.enable_volt_var=True 的内置控制）
  2. Baseline-MLP-RL : 纯 MLP encoder + PPO（无预训练 GNN）
  3. Ours-GNN-RL     : 预训练 GNN (frozen) + PPO
  4. Ours-GNN-RL-FT  : 预训练 GNN (finetune) + PPO

用法：
  # 只评测已有 ckpt
  python -m rl.eval_rl \
    --ckpt_gnn checkpoints/rl_ppo/best \
    --ckpt_mlp checkpoints/rl_mlp/best \
    --episodes 100 --episode_len 24

  # 只跑 rule baseline（无需 ckpt）
  python -m rl.eval_rl --rule_only --episodes 50
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Dict, List

import numpy as np

from rl.grid_pv_env import GridPVEnv
from scenario.base_scenario import ScenarioConfig
from scenario.online_backend import OnlineBackend


# ------------------------------------------------------------------
# Rule baseline（复用 OnlineBackend + control=True 路径）
# ------------------------------------------------------------------
def eval_rule_baseline(
    feeder: str = "case33bw",
    n_episodes: int = 100,
    episode_len: int = 24,
    base_seed: int = 9999,
) -> Dict[str, float]:
    """运行内置 Volt-Var + Curtailment 规则控制，统计指标。"""
    from scenario.base_scenario import (
        ScenarioConfig, load_feeder_by_name, build_net_with_pv,
        run_powerflow_with_controls, base_load_shape, clear_sky_pv_shape,
        simulate_cloud_factor,
    )
    import copy

    cfg = ScenarioConfig(enable_volt_var=True, enable_curtailment=True)
    rng_master = np.random.default_rng(base_seed)

    all_vviol, all_rpf, all_export = [], [], []

    for ep in range(n_episodes):
        ep_seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(ep_seed)

        net0 = load_feeder_by_name(feeder)
        net = build_net_with_pv(net0, cfg, rng)

        T = cfg.days * cfg.steps_per_day
        pv_daily = rng.uniform(*cfg.pv_daily_scale_range, size=cfg.days)
        load_daily = rng.uniform(*cfg.load_daily_scale_range, size=cfg.days)
        cloud = simulate_cloud_factor(
            T, cfg.cloud_ar, cfg.cloud_sigma,
            cfg.cloud_drop_prob, cfg.cloud_drop_mag, rng,
        )

        ep_vviol, ep_rpf, ep_export = 0, 0, 0
        step_count = 0

        for t in range(min(episode_len, T)):
            day = t // cfg.steps_per_day
            hour = t % cfg.steps_per_day
            pv_mult = clear_sky_pv_shape(hour) * pv_daily[day] * cloud[t]
            load_mult = base_load_shape(hour) * load_daily[day]

            res, info = run_powerflow_with_controls(
                net=net, cfg=cfg,
                load_mult=load_mult, pv_mult=pv_mult,
                cloud_factor=1.0, noise_std=cfg.meas_noise_std_pq,
                control=True, rng=rng,
            )
            if res is None:
                continue

            ep_vviol  += int(res["v_viol"].sum())
            ep_rpf    += int(res["rpf_line"].sum())
            ep_export += int(res["export"][0])
            step_count += 1

        if step_count > 0:
            all_vviol.append(ep_vviol / step_count)
            all_rpf.append(ep_rpf / step_count)
            all_export.append(ep_export / step_count)

    return {
        "method": "Baseline-Rule",
        "vviol_rate":    float(np.mean(all_vviol)) if all_vviol else float("nan"),
        "rpf_rate":      float(np.mean(all_rpf))   if all_rpf   else float("nan"),
        "export_rate":   float(np.mean(all_export)) if all_export else float("nan"),
        "n_episodes":    n_episodes,
    }


# ------------------------------------------------------------------
# RL policy eval（加载 RLlib checkpoint）
# ------------------------------------------------------------------
def eval_rl_policy(
    ckpt_path: str,
    method_name: str,
    feeder: str = "case33bw",
    n_episodes: int = 100,
    episode_len: int = 24,
    base_seed: int = 42,
) -> Dict[str, float]:
    """加载 RLlib checkpoint，在测试 episode 上评估。"""
    import ray
    from ray.rllib.algorithms.ppo import PPO
    from rl.train_rl import env_creator  # noqa: 复用注册逻辑

    ray.init(ignore_reinit_error=True)

    algo = PPO.from_checkpoint(ckpt_path)
    env = GridPVEnv(
        feeder_name=feeder,
        episode_len=episode_len,
        seed=base_seed,
    )

    all_rewards, all_vviol, all_rpf, all_export = [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_vviol = ep_rpf = ep_export = 0
        step_count = 0

        done = False
        while not done:
            action = algo.compute_single_action(obs, explore=False)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            if info.get("pf_ok", False):
                ep_vviol  += int(info.get("num_v_viol", 0))
                ep_rpf    += int(info.get("num_rpf_lines", 0))
                ep_export += int(info.get("export", 0))
                step_count += 1

        all_rewards.append(ep_reward)
        if step_count > 0:
            all_vviol.append(ep_vviol  / step_count)
            all_rpf.append(ep_rpf      / step_count)
            all_export.append(ep_export/ step_count)

    env.close()
    ray.shutdown()

    return {
        "method":          method_name,
        "return_mean":     float(np.mean(all_rewards)),
        "return_std":      float(np.std(all_rewards)),
        "vviol_rate":      float(np.mean(all_vviol))  if all_vviol  else float("nan"),
        "rpf_rate":        float(np.mean(all_rpf))    if all_rpf    else float("nan"),
        "export_rate":     float(np.mean(all_export)) if all_export else float("nan"),
        "n_episodes":      n_episodes,
    }


# ------------------------------------------------------------------
# 打印结果表格
# ------------------------------------------------------------------
def print_table(results: List[Dict]):
    cols = ["method", "vviol_rate", "rpf_rate", "export_rate", "return_mean"]
    header = f"{'Method':<22} {'Vviol↓':>10} {'RPF↓':>10} {'Export↓':>10} {'Return↑':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.get('method','?'):<22} "
            f"{r.get('vviol_rate', float('nan')):>10.4f} "
            f"{r.get('rpf_rate',   float('nan')):>10.4f} "
            f"{r.get('export_rate',float('nan')):>10.4f} "
            f"{r.get('return_mean',float('nan')):>10.3f} "
        )
    print("=" * len(header))


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feeder",      type=str,  default="case33bw")
    ap.add_argument("--episodes",    type=int,  default=100)
    ap.add_argument("--episode_len", type=int,  default=24)
    ap.add_argument("--seed",        type=int,  default=42)
    ap.add_argument("--out",         type=str,  default="checkpoints/eval_results.json",
                    help="保存对比结果的 JSON 路径")

    # 各方案的 checkpoint（不传则跳过该方案）
    ap.add_argument("--ckpt_gnn",    type=str,  default="",
                    help="GNN-RL (frozen) checkpoint 路径")
    ap.add_argument("--ckpt_gnn_ft", type=str,  default="",
                    help="GNN-RL (finetune) checkpoint 路径")
    ap.add_argument("--ckpt_mlp",    type=str,  default="",
                    help="MLP-RL (no pretrain) checkpoint 路径")

    ap.add_argument("--rule_only",   action="store_true",
                    help="只跑规则 baseline，不需要任何 ckpt")

    args = ap.parse_args()
    results = []

    # 1. 规则 baseline（总跑）
    print("[eval] Running Baseline-Rule ...")
    r_rule = eval_rule_baseline(
        feeder=args.feeder,
        n_episodes=args.episodes,
        episode_len=args.episode_len,
        base_seed=args.seed,
    )
    results.append(r_rule)
    print(f"  → vviol={r_rule['vviol_rate']:.4f} rpf={r_rule['rpf_rate']:.4f} export={r_rule['export_rate']:.4f}")

    if not args.rule_only:
        # 2. MLP-RL
        if args.ckpt_mlp:
            print("[eval] Running Baseline-MLP-RL ...")
            r_mlp = eval_rl_policy(
                args.ckpt_mlp, "Baseline-MLP-RL",
                args.feeder, args.episodes, args.episode_len, args.seed,
            )
            results.append(r_mlp)

        # 3. GNN-RL (frozen)
        if args.ckpt_gnn:
            print("[eval] Running Ours-GNN-RL ...")
            r_gnn = eval_rl_policy(
                args.ckpt_gnn, "Ours-GNN-RL",
                args.feeder, args.episodes, args.episode_len, args.seed,
            )
            results.append(r_gnn)

        # 4. GNN-RL (finetune)
        if args.ckpt_gnn_ft:
            print("[eval] Running Ours-GNN-RL-FT ...")
            r_gnn_ft = eval_rl_policy(
                args.ckpt_gnn_ft, "Ours-GNN-RL-FT",
                args.feeder, args.episodes, args.episode_len, args.seed,
            )
            results.append(r_gnn_ft)

    print_table(results)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
