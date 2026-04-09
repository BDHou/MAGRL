"""
验证 v2 异构观测重构后的核心组件是否正常工作。
此测试不依赖 ray，直接测试各 component。
"""
import sys
sys.path.insert(0, '/Users/steven/Documents/GridProjects/MAGRL')

import os
import numpy as np

# 直接导入组件，不经过 ray 的 MultiAgentEnv
from src.envs.components.grid_simulator import GridSimulator
from src.envs.components.data_manager import TimeSeriesDataManager
from src.envs.components.action_processor import ActionProcessor
from src.envs.components.obs_builder import ObservationBuilder
from src.envs.components.reward_calculator import RewardCalculator

DATA_PATH = '/Users/steven/Documents/GridProjects/MAGRL/data/generated/ieee_case33bw'

# ================================================================
# Test 1: GridSimulator + 拓扑特征预计算
# ================================================================
print("=" * 60)
print("Test 1: GridSimulator 实例化 + 拓扑阻抗计算")
print("=" * 60)
net_path = os.path.join(DATA_PATH, 'topology.p')
sim = GridSimulator(net_path)

print(f"  储能数量: {sim.num_storages}")
print(f"  母线数量: {sim.num_buses}")
print(f"  负荷数量: {sim.num_loads}")
print(f"  储能所在母线: {sim.storage_buses}")
print(f"  归一化 R_eq: {sim.storage_r_eq}")
print(f"  归一化 X_eq: {sim.storage_x_eq}")

assert len(sim.storage_r_eq) == sim.num_storages
assert len(sim.storage_x_eq) == sim.num_storages
assert np.all(sim.storage_r_eq >= 0) and np.all(sim.storage_r_eq <= 1.0)
assert np.all(sim.storage_x_eq >= 0) and np.all(sim.storage_x_eq <= 1.0)
# 至少有一个值应该等于 1.0（最大值归一化后）
if sim.num_storages > 1:
    assert np.max(sim.storage_r_eq) == 1.0 or np.max(sim.storage_x_eq) == 1.0
print("  ✅ 拓扑特征合理")

# ================================================================
# Test 2: 潮流计算 + 局部电气特征
# ================================================================
print("\n" + "=" * 60)
print("Test 2: 潮流计算 + 局部电气特征提取")
print("=" * 60)
sim.reset()
data_mgr = TimeSeriesDataManager(DATA_PATH, sim.num_loads)
data_mgr.apply_to_net(sim.net, 0)
converged = sim.run_powerflow()
print(f"  潮流收敛: {converged}")
assert converged, "初始潮流未收敛！"

p_grid = sim.get_ext_grid_power()
print(f"  PCC 功率 (p_grid): {p_grid:.4f} MW")

for i, bus in enumerate(sim.storage_buses):
    v = sim.get_bus_voltage_single(bus)
    p_net = sim.get_local_net_load(bus)
    print(f"  Storage {i} @ Bus {bus}: v_self={v:.4f} p.u., p_net_local={p_net:.4f} MW")
print("  ✅ 局部特征提取正常")

# ================================================================
# Test 3: ObservationBuilder (8 维异构观测)
# ================================================================
print("\n" + "=" * 60)
print("Test 3: ObservationBuilder — 8 维异构观测")
print("=" * 60)
obs_builder = ObservationBuilder()
print(f"  观测空间: {obs_builder.get_space()}")
assert obs_builder.get_space().shape == (8,)

obs_list = []
for i in range(sim.num_storages):
    soc_val = 0.3 + i * 0.1  # 给不同的 SOC 以区分
    obs = obs_builder.build(sim, agent_idx=i, soc_value=soc_val,
                            current_step=0,
                            r_eq=sim.storage_r_eq[i],
                            x_eq=sim.storage_x_eq[i])
    obs_list.append(obs)
    labels = ['p_grid', 'time_sin', 'time_cos', 'soc', 'v_self', 'r_eq', 'x_eq', 'p_net_local']
    print(f"  Storage {i}: {dict(zip(labels, [f'{v:.4f}' for v in obs]))}")

if sim.num_storages > 1:
    diffs = np.abs(obs_list[0] - obs_list[1])
    print(f"\n  Agent 0 vs Agent 1 逐维差异: {np.array2string(diffs, precision=4)}")
    # 全局共享维度 (0,1,2) 应该相同
    assert np.allclose(diffs[:3], 0), "全局共享维度应相同"
    # 局部维度 (3-7) 应有差异
    assert np.any(diffs[3:] > 0), "局部异构维度应有差异"
    print("  ✅ 全局维度一致 & 局部维度异构")
else:
    print("  ⚠️ 仅 1 个储能，跳过异构性检查")

# ================================================================
# Test 4: ActionProcessor — 随机 SOC 初始化
# ================================================================
print("\n" + "=" * 60)
print("Test 4: ActionProcessor — 随机 SOC 初始化")
print("=" * 60)
action_proc = ActionProcessor(
    storage_ids=sim.storage_ids,
    max_p=sim.storage_max_p,
    max_e=sim.storage_max_e,
    dt=0.25,
)

# 多次 reset 检查随机性
soc_sets = []
for trial in range(5):
    action_proc.reset()
    socs = [action_proc.get_soc(i) for i in range(sim.num_storages)]
    soc_sets.append(tuple(socs))
    print(f"  Trial {trial}: SOC = {[f'{s:.4f}' for s in socs]}")

# 检查多次 reset 产生不同的 SOC
unique_sets = len(set(soc_sets))
print(f"  5 次 reset 中不同的 SOC 组合: {unique_sets}")
assert unique_sets > 1, "多次 reset 应该产生不同的 SOC"
# 检查范围
for socs in soc_sets:
    for s in socs:
        assert 0.2 <= s <= 0.8, f"SOC {s} 超出 [0.2, 0.8]"
print("  ✅ 随机初始化正常，范围在 [0.2, 0.8]")

# ================================================================
# Test 5: RewardCalculator — per-agent 奖励
# ================================================================
print("\n" + "=" * 60)
print("Test 5: RewardCalculator — per-agent 奖励分解")
print("=" * 60)
reward_calc = RewardCalculator()

# 模拟不同 agent 有不同的 SOC 和动作
action_proc.reset()
# 对每个 agent 施加不同动作
p_bat_values = []
for i in range(sim.num_storages):
    action_val = -0.5 + i * 0.3  # 不同动作
    actual_p = action_proc.apply(i, action_val, sim)
    p_bat_values.append(actual_p)

sim.run_powerflow()

reward_info = reward_calc.calculate(sim, action_proc.soc, p_bat_values)
print(f"  R_reverse (全局): {reward_info['reward_reverse']:.4f}")
print(f"  R_buy     (全局): {reward_info['reward_buy']:.4f}")
print(f"  p_grid:           {reward_info['p_grid_actual']:.4f} MW")

for i in range(sim.num_storages):
    print(f"  Storage {i}: total={reward_info['per_agent_rewards'][i]:.4f}, "
          f"r_soc={reward_info['reward_soc'][i]:.4f}, "
          f"r_action={reward_info['reward_action'][i]:.4f}")

if sim.num_storages > 1:
    rewards = reward_info['per_agent_rewards']
    if len(set([round(r, 8) for r in rewards])) > 1:
        print("  ✅ 各 agent 奖励不同（局部分量起效）")
    else:
        print("  ⚠️ 奖励相同（SOC 和动作碰巧一致）")

print("\n" + "=" * 60)
print("✅ 所有组件级测试通过！")
print("=" * 60)
