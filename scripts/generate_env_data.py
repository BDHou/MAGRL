import os
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
from src.envs.components.data_manager import TimeSeriesDataManager
from scenario.base_scenario import (
    ScenarioConfig,
    base_load_shape,
    clear_sky_pv_shape,
    simulate_cloud_factor,
    choose_pv_buses
)

def analyze_and_plot(load_p_matrix, original_load_count, out_dir, steps_per_day, T_total):
    import matplotlib.pyplot as plt
    print("\n--- 📊 数据分析摘要 ---")
    
    # 常规负荷部分 (用电)
    loads_p = load_p_matrix[:, :original_load_count]
    total_load_series = loads_p.sum(axis=1) # 时序总负荷
    
    # 光伏部分 (发电，由于存储为负值，我们取绝对值)
    if load_p_matrix.shape[1] > original_load_count:
        pvs_p = np.abs(load_p_matrix[:, original_load_count:])
        total_gen_series = pvs_p.sum(axis=1) # 时序总发电
    else:
        total_gen_series = np.zeros(T_total)
    
    # 打印一些关键统计量
    print(f">> 总用电负荷: 最大 = {total_load_series.max():.2f} MW, 最小 = {total_load_series.min():.2f} MW, 平均 = {total_load_series.mean():.2f} MW")
    print(f">> 总光伏发电: 最大 = {total_gen_series.max():.2f} MW, 最小 = {total_gen_series.min():.2f} MW, 平均 = {total_gen_series.mean():.2f} MW")
    
    net_load = total_load_series - total_gen_series
    print(f">> 净负荷 (Load - Gen): 最大 = {net_load.max():.2f} MW, 最小 = {net_load.min():.2f} MW (如果是负值说明出现功率向主网红反灌倒送)")
    
    # 开始画图
    plt.figure(figsize=(15, 6))
    
    # 图1：全周期观测
    time_axis = np.arange(T_total)
    plt.plot(time_axis, total_load_series, label="Total Load (MW)", color='blue', alpha=0.7)
    plt.plot(time_axis, total_gen_series, label="Total PV Generation (MW)", color='orange', alpha=0.7)
    plt.plot(time_axis, net_load, label="Net Load (MW)", color='red', linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.8)
    
    interval_min = int((24.0 / steps_per_day) * 60)
    plt.title(f"Global Time-Series Power Profile ({T_total // steps_per_day} Days, Interval: {interval_min} mins)")
    plt.xlabel("Time Step")
    plt.ylabel("Power (MW)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "total_power_curve.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n✅ 数据时序分析图已保存至: {plot_path}")

def generate_ieee_env_data(feeder_name: str, out_dir: str, top_out_path: str, cfg: ScenarioConfig, seed: int = 42):
    """
    使用 pandapower 提供的 IEEE 标准网络，生成适用强化学习环境的拓扑和时序数据。
    完全不依赖私有的 .p 文件。
    """
    rng = np.random.default_rng(seed)
    
    # 1. 加载标准电网拓扑
    print(f"Loading IEEE standard case: {feeder_name}")
    if hasattr(pn, feeder_name):
        net = getattr(pn, feeder_name)()
    else:
        raise ValueError(f"Feeder {feeder_name} not found in pandapower.networks")
        
    # 2. 为环境添加 PV（通过新建负向 Load 实现，因为环境代码依赖 net.load 长度与负荷矩阵完全一致）
    pv_buses = choose_pv_buses(net, cfg.n_pv, cfg.pv_bus_strategy)
    pv_pmax_list = []
    
    for b in pv_buses:
        pmax = float(rng.uniform(cfg.pv_pmax_mw_range[0], cfg.pv_pmax_mw_range[1]))
        pv_pmax_list.append(pmax)
        # 将 PV 注入为 load，名字带有 PV 标识，功率基准我们记录下来，这里初始化为0
        pp.create_load(net, bus=int(b), p_mw=0.0, q_mvar=0.0, name=f"PV_bus{b}")
        
    # 3. 为环境添加储能模型 (环境强依赖 storage 表和 storage_ids)
    for i, b in enumerate(pv_buses):
        # 储能位置可选：比如放置在有光伏的节点附近
        pp.create_storage(net, bus=int(b), p_mw=0.0, max_e_mwh=4.0, max_p_mw=1.0, name=f"Storage_bus{b}")
        
    num_loads = len(net.load)
    print(f"Topology configured: {len(net.bus)} buses, {num_loads} loads (including {len(pv_buses)} PVs as negative loads), {len(net.storage)} storages.")
    
    # 保存专用的环境拓扑文件
    os.makedirs(os.path.dirname(top_out_path), exist_ok=True)
    pp.to_pickle(net, top_out_path)
    print(f"Environment logic topology successfully explicitly saved to: {top_out_path}")
    
    # 4. 生成时序负荷与光伏数据
    T_total = cfg.days * cfg.steps_per_day
    
    # 原用电负荷数量 (不含我们刚刚创建的光伏源)
    original_load_count = num_loads - len(pv_buses)
    base_p = net.load["p_mw"].iloc[:original_load_count].to_numpy().copy()
    
    if "q_mvar" in net.load.columns:
        base_q = net.load["q_mvar"].iloc[:original_load_count].to_numpy().copy()
    else:
        base_q = np.zeros(original_load_count)
        
    # 生成时序随机扰动因子
    load_daily_scale = rng.uniform(cfg.load_daily_scale_range[0], cfg.load_daily_scale_range[1], size=cfg.days)
    pv_daily_scale = rng.uniform(cfg.pv_daily_scale_range[0], cfg.pv_daily_scale_range[1], size=cfg.days)
    cloud = simulate_cloud_factor(T_total, cfg.cloud_ar, cfg.cloud_sigma, cfg.cloud_drop_prob, cfg.cloud_drop_mag, rng)
    
    load_p_matrix = np.zeros((T_total, num_loads))
    load_q_matrix = np.zeros((T_total, num_loads))
    
    for t in range(T_total):
        day = t // cfg.steps_per_day
        step_in_day = t % cfg.steps_per_day
        
        # 将离散 step 比例换算成小时刻度 (0.0 ~ 23.99)
        hour = step_in_day * (24.0 / cfg.steps_per_day)
        
        # 时序基准形状
        ld_shape = base_load_shape(hour)
        pv_shape = clear_sky_pv_shape(hour)
        
        # 倍率
        load_mult = ld_shape * load_daily_scale[day]
        pv_mult = pv_shape * pv_daily_scale[day] * cloud[t]
        
        p_t = np.zeros(num_loads)
        q_t = np.zeros(num_loads)
        
        # 左侧放常规负荷
        p_t[:original_load_count] = base_p * load_mult
        q_t[:original_load_count] = base_q * load_mult
        
        # 右侧放光伏 (用负值代理馈入电网)
        for i, pval in enumerate(pv_pmax_list):
            idx = original_load_count + i
            p_t[idx] = - (pval * pv_mult)
            q_t[idx] = 0.0  # 环境的 Volt-Var如果后续要自己算则 Q 会动态变，这里基态默认0
            
        # Optional 测量高斯噪声
        if cfg.meas_noise_std_pq > 0:
            p_t = p_t * (1.0 + rng.normal(0, cfg.meas_noise_std_pq, size=p_t.shape))
            q_t = q_t * (1.0 + rng.normal(0, cfg.meas_noise_std_pq, size=q_t.shape))
            
        load_p_matrix[t] = p_t
        load_q_matrix[t] = q_t

    # 5. 写入 csv
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(load_p_matrix).to_csv(os.path.join(out_dir, "load_p.csv"), index=False)
    pd.DataFrame(load_q_matrix).to_csv(os.path.join(out_dir, "load_q.csv"), index=False)
    
    print(f"Data generation complete! Saved loads to: {out_dir}")
    
    # 6. 分析报告与时序曲线
    analyze_and_plot(load_p_matrix, original_load_count, out_dir, cfg.steps_per_day, T_total)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--feeder", type=str, default="case33bw", help="IEEE 标准反馈线名字 (如 case33bw, case69, mv_oberrhein)")
    parser.add_argument("--out_dir", type=str, default="data/generated/load")
    parser.add_argument("--top_out_path", type=str, default="data/generated/topology/ieee_env_topology.p")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--interval_min", type=int, default=15, help="生成数据的时间间隔 (分钟)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    
    steps_per_day = int((24 * 60) / args.interval_min)
    cfg = ScenarioConfig(days=args.days, steps_per_day=steps_per_day)
    
    generate_ieee_env_data(args.feeder, args.out_dir, args.top_out_path, cfg, seed=args.seed)
