from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from src.envs.single_feeder_multi_agent_env import MultiFeederStorageEnv
import os

DATA_PATH = r'/Users/steven/Documents/GridProjects/MAGRL/data/generated'  # CONF

def env_creator(env_config):
    return MultiFeederStorageEnv(env_config)

def setup_registry(env_name="Yancheng_MARL_Feeder_v0"):
    register_env(env_name, env_creator)
    return env_name

def build_mappo_algo(env_name, learning_rate=5e-5, num_workers=2):
    # 临时实例化一个环境以获取其空间定义，供策略初始化使用
    temp_env = MultiFeederStorageEnv({"data_path": DATA_PATH})
    obs_space = temp_env.single_obs_space
    act_space = temp_env.single_action_space

    config = (
        PPOConfig()
        .environment(env=env_name, env_config={"data_path": DATA_PATH})
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        # ==========================================
        # 核心：多智能体配置模块
        # ==========================================
        .multi_agent(
            # 定义策略字典：此处只定义了一个名为 "shared_policy" 的策略
            policies={
                "shared_policy": (None, obs_space, act_space, {}),
            },
            # 策略映射函数：无论哪个智能体 (storage_0, storage_1...)，都使用 shared_policy
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .env_runners(
            num_env_runners=num_workers,
            num_envs_per_env_runner=1,
        )
        .training(
            lr=learning_rate,
            gamma=0.99,
            train_batch_size=4000,
        )
    )
    return config.build()

def run_training(algo, iterations=50, checkpoint_freq=10, save_dir="saved_models", keep_last_n=3):
    """
    Execute training loop with best-model tracking and TensorBoard logging.

    Args:
        algo: RLlib Algorithm instance
        iterations: total training iterations
        checkpoint_freq: save checkpoint every N iterations
        save_dir: root directory for checkpoints (a timestamped run folder is created inside)
        keep_last_n: keep only the last N periodic checkpoints (0 = keep all)

    Returns:
        (return_history, best_checkpoint_path, latest_checkpoint_path)
    """
    import shutil
    from datetime import datetime
    from torch.utils.tensorboard import SummaryWriter

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # TensorBoard writer (logs stored alongside checkpoints)
    tb_dir = os.path.join(run_dir, "tb_logs")
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"Run directory: {run_dir}")
    print(f"TensorBoard: tensorboard --logdir {tb_dir}")

    return_history = []
    best_return = -float("inf")
    best_checkpoint_path = None
    latest_checkpoint_path = None
    periodic_checkpoints = []

    for i in range(iterations):
        result = algo.train()
        er = result.get("env_runners", {})

        ep_return = er.get("episode_return_mean", result.get("episode_reward_mean", 0.0))
        return_history.append(ep_return)

        # ---- TensorBoard logging ----
        # Episode-level metrics
        writer.add_scalar("episode/return_mean", ep_return, i)
        if "episode_return_max" in er:
            writer.add_scalar("episode/return_max", er["episode_return_max"], i)
        if "episode_return_min" in er:
            writer.add_scalar("episode/return_min", er["episode_return_min"], i)
        if "episode_len_mean" in er:
            writer.add_scalar("episode/len_mean", er["episode_len_mean"], i)

        # Learner metrics (loss, entropy, etc.)
        learners = result.get("learners", {})
        for module_id, metrics in learners.items():
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        writer.add_scalar(f"learner/{module_id}/{k}", v, i)

        # Perf metrics
        perf = result.get("perf", {})
        for k, v in perf.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f"perf/{k}", v, i)

        writer.flush()

        print(f"Iteration {i+1:03d} | Return: {ep_return:.2f}")

        # --- Periodic checkpoint (numbered) ---
        if (i + 1) % checkpoint_freq == 0:
            ckpt_dir = os.path.join(run_dir, f"iter_{i+1:04d}")
            latest_checkpoint_path = algo.save(ckpt_dir)
            periodic_checkpoints.append(ckpt_dir)
            print(f"  Checkpoint saved: {ckpt_dir}")

            if keep_last_n > 0 and len(periodic_checkpoints) > keep_last_n:
                old_dir = periodic_checkpoints.pop(0)
                if os.path.exists(old_dir):
                    shutil.rmtree(old_dir)
                    print(f"  Removed old checkpoint: {old_dir}")

        # --- Best model tracking (by episode return) ---
        if ep_return > best_return:
            best_return = ep_return
            best_dir = os.path.join(run_dir, "best")
            if os.path.exists(best_dir):
                shutil.rmtree(best_dir)
            best_checkpoint_path = algo.save(best_dir)
            print(f"  ★ New best! Return: {ep_return:.2f} -> {best_dir}")

    writer.close()

    print(f"\nTraining complete. Best return: {best_return:.2f}")
    print(f"  Best checkpoint: {best_checkpoint_path}")
    print(f"  Latest checkpoint: {latest_checkpoint_path}")
    print(f"  Run directory: {run_dir}")
    print(f"  TensorBoard: tensorboard --logdir {tb_dir}")

    return return_history, best_checkpoint_path, latest_checkpoint_path