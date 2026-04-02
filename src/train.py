import os
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

# 导入你的环境和刚刚编写的最新架构模型
from src.envs.single_feeder_env import SingleFeederStorageEnv
from src.models.custom_model import CustomStorageRLModule

DATA_PATH = r'/Users/steven/Documents/GridProjects/YC2/data'  # CONF

def env_creator(env_config):
    """环境实例化的工厂函数"""
    return SingleFeederStorageEnv(env_config)

def setup_registry(env_name="Yancheng_Feeder_v0"):
    """
    在新架构下，仅需要注册环境即可
    """
    register_env(env_name, env_creator)
    return env_name

def build_ppo_algo(env_name, learning_rate=5e-5, num_workers=2):
    """
    构建并返回适配新 API 栈的 PPO 算法实例
    """
    # 配置模型规约 (Spec)
    module_spec = RLModuleSpec(
        module_class=CustomStorageRLModule,
        model_config={
            "fcnet_hiddens": [256, 256],
        }
    )

    config = (
        PPOConfig()
        .environment(env=env_name, env_config={"data_path": DATA_PATH})
        .framework("torch")
        # 启用最新 API 栈 (虽然较新版本是默认开启的，但显式声明更加严谨)
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        # 将我们自定义的 RLModuleSpec 注入配置
        .rl_module(rl_module_spec=module_spec)
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

def run_training(algo, iterations=50, checkpoint_freq=10, save_dir="saved_models"):
    """
    执行训练大循环，并返回记录数据
    """
    reward_history = []
    checkpoint_path = None
    os.makedirs(save_dir, exist_ok=True)

    for i in range(iterations):
        result = algo.train()
        reward = result.get("env_runners", {}).get("episode_return_mean", result.get("episode_reward_mean", 0.0))
        reward_history.append(reward)
        
        print(f"Iteration {i+1:03d} | Reward: {reward:.2f}")

        if (i + 1) % checkpoint_freq == 0:
            checkpoint_path = algo.save(save_dir)
            print(f"Checkpoint saved at: {checkpoint_path}")

    return reward_history, checkpoint_path