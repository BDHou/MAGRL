"""
Multi-agent model evaluation script.

v2: 适配异构观测空间 (8维) 和 per-agent 奖励分解。

Features:
  1. Run trained MAPPO policy on a full episode
  2. Run no-action baseline for comparison
  3. Record all key physical quantities and decomposed rewards
  4. Overlay RL vs baseline curves for visual comparison

Usage (Jupyter):
    from src.eval_ma import evaluate_episode, evaluate_baseline, plot_evaluation
    baseline = evaluate_baseline(env_config={...})
    records = evaluate_episode(algo, env_config={...})
    plot_evaluation(records, baseline=baseline)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from src.envs.single_feeder_multi_agent_env import MultiFeederStorageEnv

# Reward component keys (shared across functions)
REWARD_KEYS = ["reward_reverse", "reward_buy", "reward_soc", "reward_action"]


def _make_empty_records(env):
    """Create empty records dict with correct structure."""
    return {
        "steps": [],
        "rewards": [],
        **{k: [] for k in REWARD_KEYS},
        "soc": {aid: [] for aid in env.possible_agents},
        "actions": {aid: [] for aid in env.possible_agents},
        "p_ext_grid": [],
        "vm_min": [],
        "vm_max": [],
        "storage_p": {aid: [] for aid in env.possible_agents},
    }


def _record_step(records, env, step, reward_dict, info_dict, action_dict):
    """Record one step of metrics (shared by evaluate_episode and evaluate_baseline)."""
    records["steps"].append(step)

    # v2: 记录所有智能体的平均奖励（per-agent 奖励的均值作为 episode 级指标）
    agent_rewards = [reward_dict.get(aid, 0.0) for aid in env.possible_agents]
    records["rewards"].append(float(np.mean(agent_rewards)))

    # v2: 全局分量取第一个 agent 的 info（全局部分所有 agent 相同）
    # 局部分量 (soc, action) 取所有 agent 的均值
    first_agent = env.possible_agents[0]
    first_info = info_dict.get(first_agent, {})
    records["reward_reverse"].append(first_info.get("reward_reverse", 0.0))
    records["reward_buy"].append(first_info.get("reward_buy", 0.0))

    # SOC 和 action 惩罚：取所有 agent 的均值
    soc_penalties = [info_dict.get(aid, {}).get("reward_soc", 0.0) for aid in env.possible_agents]
    action_penalties = [info_dict.get(aid, {}).get("reward_action", 0.0) for aid in env.possible_agents]
    records["reward_soc"].append(float(np.mean(soc_penalties)))
    records["reward_action"].append(float(np.mean(action_penalties)))

    for aid in env.possible_agents:
        idx = env._agent_to_idx[aid]
        records["soc"][aid].append(float(env.action_proc.get_soc(idx)))

    records["p_ext_grid"].append(first_info.get("p_grid_actual", 0.0))
    vm = env.simulator.get_bus_voltages()
    records["vm_min"].append(float(np.min(vm)))
    records["vm_max"].append(float(np.max(vm)))

    for aid in env.possible_agents:
        idx = env._agent_to_idx[aid]
        sid = env.action_proc.storage_ids[idx]
        records["storage_p"][aid].append(float(env.simulator.net.storage.at[sid, 'p_mw']))

    for aid in env.possible_agents:
        if aid in action_dict:
            records["actions"][aid].append(float(action_dict[aid][0]))


def _finalize(records, step, label=""):
    """Compute summary stats."""
    records["total_reward"] = sum(records["rewards"])
    records["total_steps"] = step
    records["avg_reward"] = records["total_reward"] / max(step, 1)
    print(f"{label}Evaluation done: {step} steps, total_return = {records['total_reward']:.2f}, "
          f"avg_reward = {records['avg_reward']:.4f}")
    return records


def evaluate_baseline(env_config: dict = None) -> dict:
    """
    Run a full episode with zero actions (no storage control) as baseline.
    This shows what the grid looks like without any RL intervention.
    """
    env_config = env_config or {}
    env = MultiFeederStorageEnv(env_config)
    records = _make_empty_records(env)

    obs_dict, _ = env.reset()
    done = False
    step = 0

    while not done:
        # All storages output zero power
        action_dict = {aid: np.array([0.0]) for aid in env.possible_agents}
        obs_dict, reward_dict, terminated_dict, _, info_dict = env.step(action_dict)
        _record_step(records, env, step, reward_dict, info_dict, action_dict)
        done = terminated_dict.get("__all__", False)
        step += 1

    return _finalize(records, step, label="[Baseline] ")


def evaluate_episode(algo, env_config: dict = None) -> dict:
    """
    Run trained policy on a full episode, recording all key metrics.
    """
    env_config = env_config or {}
    env = MultiFeederStorageEnv(env_config)
    records = _make_empty_records(env)

    obs_dict, _ = env.reset()
    done = False
    step = 0

    module = algo.get_module("shared_policy")
    module.eval()

    while not done:
        action_dict = {}
        with torch.no_grad():
            for agent_id, obs in obs_dict.items():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                output = module.forward_inference({"obs": obs_tensor})
                action_dist_inputs = output["action_dist_inputs"]
                action_mean = action_dist_inputs[:, :1].numpy().squeeze()
                action_dict[agent_id] = np.array([np.clip(action_mean, -1.0, 1.0)])

        obs_dict, reward_dict, terminated_dict, _, info_dict = env.step(action_dict)
        _record_step(records, env, step, reward_dict, info_dict, action_dict)
        done = terminated_dict.get("__all__", False)
        step += 1

    return _finalize(records, step, label="[RL Policy] ")


def plot_evaluation(records: dict, baseline: dict = None, figsize=(16, 18)):
    """
    Visualize evaluation results in 5 panels.
    If baseline is provided, overlay baseline curves (dashed gray) for comparison.
    """
    steps = records["steps"]
    hours = [s * 0.25 for s in steps]
    bl_hours = [s * 0.25 for s in baseline["steps"]] if baseline else None

    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)

    title = f"MAPPO Evaluation | Steps: {records['total_steps']}  Total Return: {records['total_reward']:.2f}"
    if baseline:
        title += f"  (Baseline: {baseline['total_reward']:.2f})"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # ---- Panel 1: Power ----
    ax1 = axes[0]
    if baseline:
        ax1.plot(bl_hours, baseline["p_ext_grid"], color='gray', linewidth=1.2,
                 linestyle='--', alpha=0.6, label='Baseline (no control)')
    ax1.plot(hours, records["p_ext_grid"], 'b-', linewidth=1.5, label='RL Ext Grid Power')
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero (Reverse Threshold)')
    for aid, p_list in records["storage_p"].items():
        ax1.plot(hours, p_list, '--', linewidth=1, alpha=0.7, label=f'{aid} Output')
    ax1.set_ylabel('Power (MW)')
    ax1.set_title('External Grid Power & Storage Output')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(hours, 0, records["p_ext_grid"],
                     where=[p > 0 for p in records["p_ext_grid"]],
                     color='red', alpha=0.15)

    # ---- Panel 2: Voltage ----
    ax2 = axes[1]
    if baseline:
        ax2.fill_between(bl_hours, baseline["vm_min"], baseline["vm_max"],
                         alpha=0.15, color='gray', label='Baseline Range')
    ax2.fill_between(hours, records["vm_min"], records["vm_max"],
                     alpha=0.3, color='green', label='RL Voltage Range')
    ax2.plot(hours, records["vm_min"], 'g-', linewidth=0.8)
    ax2.plot(hours, records["vm_max"], 'g-', linewidth=0.8)
    ax2.axhline(y=1.05, color='r', linestyle='--', alpha=0.5, label='Upper 1.05')
    ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Lower 0.95')
    ax2.set_ylabel('Voltage (p.u.)')
    ax2.set_title('Bus Voltage Range')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ---- Panel 3: SOC ----
    ax3 = axes[2]
    for aid, soc_list in records["soc"].items():
        ax3.plot(hours, soc_list, linewidth=1.5, label=aid)
    ax3.axhline(y=0.9, color='r', linestyle=':', alpha=0.5, label='SOC Safe Upper (0.9)')
    ax3.axhline(y=0.1, color='r', linestyle=':', alpha=0.5, label='SOC Safe Lower (0.1)')
    ax3.set_ylabel('SOC')
    ax3.set_title('Storage SOC')
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ---- Panel 4: Total reward ----
    ax4 = axes[3]
    if baseline:
        ax4.plot(bl_hours, baseline["rewards"], color='gray', linewidth=1,
                 linestyle='--', alpha=0.5, label='Baseline')
    ax4.plot(hours, records["rewards"], 'purple', linewidth=1, alpha=0.7, label='RL Policy')
    ax4.fill_between(hours, 0, records["rewards"], alpha=0.2, color='purple')
    ax4.set_ylabel('Reward')
    ax4.set_title('Mean Agent Step Reward')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ---- Panel 5: Decomposed rewards ----
    ax5 = axes[4]
    colors = {'reward_reverse': 'red', 'reward_buy': 'orange', 'reward_soc': 'blue', 'reward_action': 'gray'}
    labels = {'reward_reverse': 'R_reverse', 'reward_buy': 'R_buy', 'reward_soc': 'R_soc (mean)', 'reward_action': 'R_action (mean)'}
    for k in REWARD_KEYS:
        ax5.plot(hours, records[k], label=labels[k], linewidth=1.2, color=colors[k])
    ax5.set_ylabel('Reward Component')
    ax5.set_xlabel('Time (hours)')
    ax5.set_title('Decomposed Reward Components (before weighting)')
    ax5.legend(loc='lower right', fontsize=8)
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig


if __name__ == "__main__":
    import argparse
    import os
    from ray.rllib.algorithms.algorithm import Algorithm

    parser = argparse.ArgumentParser(description="Evaluate MAPPO checkpoint")
    parser.add_argument("checkpoint", type=str, help="checkpoint 目录路径 (包含 rllib_checkpoint.json)")
    parser.add_argument("--save", type=str, default=None, help="保存图表到指定路径 (如 eval_result.png)")
    parser.add_argument("--no-baseline", action="store_true", help="跳过 baseline 评估")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_PATH = os.path.join(project_root, "data", "generated", "ieee_case33bw")
    env_config = {"data_path": DATA_PATH}

    # 注册环境
    from ray.tune.registry import register_env
    register_env("ieee_case33bw_MARL_Feeder_v0",
                 lambda cfg: MultiFeederStorageEnv(cfg))

    # 加载 checkpoint
    checkpoint_path = os.path.abspath(args.checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    algo = Algorithm.from_checkpoint(checkpoint_path)
    print("Checkpoint loaded successfully.")

    # 运行评估
    if not args.no_baseline:
        print("\n--- Running baseline (no control) ---")
        baseline = evaluate_baseline(env_config)
    else:
        baseline = None

    print("\n--- Running RL policy ---")
    records = evaluate_episode(algo, env_config)

    # 绘图
    fig = plot_evaluation(records, baseline=baseline)

    if args.save:
        save_path = os.path.abspath(args.save)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    algo.stop()
