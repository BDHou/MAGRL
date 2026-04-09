"""
完整集成测试：在 RLlib 中实例化环境 + 短训练 3 个 iteration。
验证 v2 环境与 MAPPO 训练管线的兼容性。
"""
import sys
sys.path.insert(0, '/Users/steven/Documents/GridProjects/MAGRL')

from src.train_ma import setup_registry, build_mappo_algo

DATA_PATH = '/Users/steven/Documents/GridProjects/MAGRL/data/generated/ieee_case33bw'

print("=" * 60)
print("集成测试：MAPPO + v2 异构观测环境")
print("=" * 60)

# 1. 注册环境
env_name = setup_registry()
print(f"  环境已注册: {env_name}")

# 2. 构建 MAPPO 算法（会实例化 temp_env 检查空间）
print("  正在构建 MAPPO 算法...")
algo = build_mappo_algo(env_name, learning_rate=5e-5, num_workers=0)
print("  ✅ 算法构建成功")

# 3. 短训练 3 个 iteration
print("\n  开始短训练 (3 iterations)...")
for i in range(3):
    result = algo.train()
    er = result.get("env_runners", {})
    ep_return = er.get("episode_return_mean", result.get("episode_reward_mean", 0.0))
    ep_len = er.get("episode_len_mean", 0)
    print(f"    Iter {i+1}: return={ep_return:.2f}, ep_len={ep_len:.0f}")

print("\n✅ 集成测试通过！v2 环境 + MAPPO 训练管线正常工作")
algo.stop()
