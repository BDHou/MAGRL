import os
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
from scenario.base_scenario import (
    ScenarioConfig,
    base_load_shape,
    clear_sky_pv_shape,
    simulate_cloud_factor,
)
from scripts.resource_injector import ResourceInjector


def render_topology(net, resource_table, out_dir):
    """
    渲染网络拓扑图，用 networkx 布局 + matplotlib 绘制，标注各类资源位置。

    Args:
        net pp.pandapowerNet: 注入资源后的 pandapower 网络
        resource_table pd.DataFrame: 资源清单
        out_dir str: 输出目录
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import pandapower.topology as top
    import networkx as nx

    type_colors = {
        0: "#e74c3c",   # BESS - 红
        1: "#2ecc71",   # Generator - 绿
        2: "#f39c12",   # Inverter/PV - 橙
        3: "#3498db",   # DR - 蓝
    }
    type_labels = {
        0: "BESS",
        1: "Generator",
        2: "Inverter/PV",
        3: "Demand Response",
    }

    # 构建 networkx 图并计算布局
    G = top.create_nxgraph(net, respect_switches=True)
    slack_bus = int(net.ext_grid.bus.iloc[0])

    # Kamada-Kawai 布局，以 slack bus 为中心效果更好（树状辐射结构）
    pos = nx.kamada_kawai_layout(G)

    # 按资源类型分组母线
    resource_buses_by_type = {}
    if len(resource_table) > 0:
        for type_id, group in resource_table.groupby("type_id"):
            resource_buses_by_type[int(type_id)] = group["bus"].tolist()

    all_resource_buses = set()
    for buses in resource_buses_by_type.values():
        all_resource_buses.update(buses)
    plain_buses = [b for b in G.nodes() if b not in all_resource_buses
                   and b != slack_bus]

    fig, ax = plt.subplots(figsize=(14, 10))

    # 画边
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#bdc3c7",
                           width=1.2, alpha=0.7)

    # 普通母线
    if plain_buses:
        nx.draw_networkx_nodes(G, pos, nodelist=plain_buses, ax=ax,
                               node_color="#95a5a6", node_size=80, alpha=0.8)

    # Slack bus
    nx.draw_networkx_nodes(G, pos, nodelist=[slack_bus], ax=ax,
                           node_color="#2c3e50", node_size=300,
                           node_shape="s", edgecolors="black", linewidths=1.5)
    ax.annotate("Slack", pos[slack_bus],
                textcoords="offset points", xytext=(0, 12),
                fontsize=8, ha="center", fontweight="bold", color="#2c3e50")

    # 资源母线（按类型颜色区分）
    for type_id, buses in resource_buses_by_type.items():
        valid_buses = [b for b in buses if b in pos]
        if not valid_buses:
            continue
        nx.draw_networkx_nodes(G, pos, nodelist=valid_buses, ax=ax,
                               node_color=type_colors.get(type_id, "#7f8c8d"),
                               node_size=200, edgecolors="black",
                               linewidths=0.8)
        for b in valid_buses:
            label = type_labels.get(type_id, "?")
            ax.annotate(f"{label}\nbus {b}", pos[b],
                        textcoords="offset points", xytext=(10, 8),
                        fontsize=7, color=type_colors.get(type_id, "black"),
                        fontweight="bold")

    # 母线编号标注（小字）
    bus_labels = {b: str(b) for b in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=bus_labels, ax=ax,
                            font_size=6, font_color="#555555")

    # 图例
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2c3e50',
               markersize=10, label='Slack Bus'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#95a5a6',
               markersize=8, label='Bus (no resource)'),
    ]
    for type_id in sorted(resource_buses_by_type.keys()):
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=type_colors[type_id],
                   markersize=10, label=type_labels[type_id])
        )
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
              framealpha=0.9)

    ax.set_title(f"Network Topology ({len(net.bus)} buses, "
                 f"{len(net.line)} lines, "
                 f"{len(resource_table)} resources)",
                 fontsize=13)
    ax.axis("off")

    plt.tight_layout()
    topo_path = os.path.join(out_dir, "topology.png")
    plt.savefig(topo_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"拓扑图已保存至: {topo_path}")


def analyze_and_plot(load_p_matrix, pv_p_matrix, out_dir, steps_per_day, T_total):
    """
    对生成的负荷和光伏时序数据进行统计分析并画图。

    Args:
        load_p_matrix np.ndarray: 负荷有功矩阵 (T_total, num_loads)
        pv_p_matrix np.ndarray: 光伏有功矩阵 (T_total, num_pv)，值为正（发电量）
        out_dir str: 输出目录
        steps_per_day int: 每天时步数
        T_total int: 总时步数
    """
    import matplotlib.pyplot as plt
    print("\n--- 数据分析摘要 ---")

    total_load_series = load_p_matrix.sum(axis=1)

    if pv_p_matrix.shape[1] > 0:
        total_gen_series = pv_p_matrix.sum(axis=1)
    else:
        total_gen_series = np.zeros(T_total)

    print(f">> 总用电负荷: 最大 = {total_load_series.max():.2f} MW, "
          f"最小 = {total_load_series.min():.2f} MW, "
          f"平均 = {total_load_series.mean():.2f} MW")
    print(f">> 总光伏发电: 最大 = {total_gen_series.max():.2f} MW, "
          f"最小 = {total_gen_series.min():.2f} MW, "
          f"平均 = {total_gen_series.mean():.2f} MW")

    net_load = total_load_series - total_gen_series
    print(f">> 净负荷 (Load - Gen): 最大 = {net_load.max():.2f} MW, "
          f"最小 = {net_load.min():.2f} MW "
          f"(负值说明出现功率向主网反灌倒送)")

    plt.figure(figsize=(15, 6))
    time_axis = np.arange(T_total)
    plt.plot(time_axis, total_load_series, label="Total Load (MW)",
             color='blue', alpha=0.7)
    plt.plot(time_axis, total_gen_series, label="Total PV Generation (MW)",
             color='orange', alpha=0.7)
    plt.plot(time_axis, net_load, label="Net Load (MW)",
             color='red', linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.8)

    interval_min = int((24.0 / steps_per_day) * 60)
    plt.title(f"Global Time-Series Power Profile "
              f"({T_total // steps_per_day} Days, Interval: {interval_min} mins)")
    plt.xlabel("Time Step")
    plt.ylabel("Power (MW)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "total_power_curve.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n数据时序分析图已保存至: {plot_path}")


def generate_ieee_env_data(
    feeder_name: str,
    out_dir: str,
    cfg: ScenarioConfig,
    resource_config: dict = None,
    seed: int = 42,
):
    """
    使用 pandapower 提供的 IEEE 标准网络，生成适用强化学习环境的拓扑和时序数据。
    通过 ResourceInjector 注入多种可调资源（BESS、Generator、Inverter/PV、DR）。

    Args:
        feeder_name str: IEEE 标准网络名，如 "case33bw"
        out_dir str: 输出目录
        cfg ScenarioConfig: 场景配置
        resource_config dict: 资源注入配置，传给 ResourceInjector；None 则使用默认配置
        seed int: 随机种子
    """
    rng = np.random.default_rng(seed)

    # ================================================================
    # 1. 加载标准电网拓扑
    # ================================================================
    print(f"Loading IEEE standard case: {feeder_name}")
    if hasattr(pn, feeder_name):
        net = getattr(pn, feeder_name)()
    else:
        raise ValueError(f"Feeder {feeder_name} not found in pandapower.networks")

    # 记录原始负荷数量（注入 DR 前）
    original_load_count = len(net.load)

    # ================================================================
    # 2. 使用 ResourceInjector 注入所有可调资源
    # ================================================================
    injector = ResourceInjector(config=resource_config, seed=seed)
    net, resource_table = injector.inject(net)

    # 从 resource_table 提取 PV/inverter 信息，用于生成光伏时序
    pv_rows = resource_table[resource_table["type_id"] == ResourceInjector.TYPE_INVERTER]
    pv_pmax_list = pv_rows["P_max_mw"].tolist()
    pv_sgen_indices = pv_rows["pp_index"].tolist()
    num_pv = len(pv_pmax_list)

    # 当前 net.load 总数（含 DR 新增的负荷）
    num_loads = len(net.load)
    print(f"Topology configured: {len(net.bus)} buses, "
          f"{num_loads} loads (original={original_load_count}, "
          f"DR={num_loads - original_load_count}), "
          f"{len(net.storage)} storages, {len(net.sgen)} sgens, "
          f"{num_pv} PV inverters.")

    # ================================================================
    # 3. 保存网络拓扑和资源清单
    # ================================================================
    os.makedirs(out_dir, exist_ok=True)
    pp.to_pickle(net, os.path.join(out_dir, 'topology.p'))
    resource_table.to_csv(os.path.join(out_dir, "resource_table.csv"), index=False)
    print(f"Topology saved to: {os.path.join(out_dir, 'topology.p')}")
    print(f"Resource table saved to: {os.path.join(out_dir, 'resource_table.csv')}")

    # ================================================================
    # 4. 生成时序负荷数据（load_p / load_q）
    #    列数 = num_loads（原始负荷 + DR 负荷）
    #    DR 负荷的基准功率为 0，时序上保持 0（由 RL agent 控制）
    # ================================================================
    T_total = cfg.days * cfg.steps_per_day

    base_p = net.load["p_mw"].to_numpy().copy()
    if "q_mvar" in net.load.columns:
        base_q = net.load["q_mvar"].to_numpy().copy()
    else:
        base_q = np.zeros(num_loads)

    load_daily_scale = rng.uniform(
        cfg.load_daily_scale_range[0], cfg.load_daily_scale_range[1], size=cfg.days
    )
    pv_daily_scale = rng.uniform(
        cfg.pv_daily_scale_range[0], cfg.pv_daily_scale_range[1], size=cfg.days
    )
    cloud = simulate_cloud_factor(
        T_total, cfg.cloud_ar, cfg.cloud_sigma,
        cfg.cloud_drop_prob, cfg.cloud_drop_mag, rng
    )

    load_p_matrix = np.zeros((T_total, num_loads))
    load_q_matrix = np.zeros((T_total, num_loads))
    pv_p_matrix = np.zeros((T_total, num_pv))

    for t in range(T_total):
        day = t // cfg.steps_per_day
        step_in_day = t % cfg.steps_per_day
        hour = step_in_day * (24.0 / cfg.steps_per_day)

        ld_shape = base_load_shape(hour)
        pv_shape = clear_sky_pv_shape(hour)

        load_mult = ld_shape * load_daily_scale[day]
        pv_mult = pv_shape * pv_daily_scale[day] * cloud[t]

        # 负荷时序：base_p * load_mult（DR 负荷的 base_p=0，结果也为 0）
        p_t = base_p * load_mult
        q_t = base_q * load_mult

        # 光伏时序（正值，表示发电量；写入独立的 pv_p_matrix）
        for i, pval in enumerate(pv_pmax_list):
            pv_p_matrix[t, i] = pval * pv_mult

        if cfg.meas_noise_std_pq > 0:
            p_t = p_t * (1.0 + rng.normal(0, cfg.meas_noise_std_pq, size=p_t.shape))
            q_t = q_t * (1.0 + rng.normal(0, cfg.meas_noise_std_pq, size=q_t.shape))
            noise_pv = 1.0 + rng.normal(0, cfg.meas_noise_std_pq, size=num_pv)
            pv_p_matrix[t] = pv_p_matrix[t] * noise_pv

        load_p_matrix[t] = p_t
        load_q_matrix[t] = q_t

    # ================================================================
    # 5. 写入 CSV
    # ================================================================
    pd.DataFrame(load_p_matrix).to_csv(
        os.path.join(out_dir, "load_p.csv"), index=False
    )
    pd.DataFrame(load_q_matrix).to_csv(
        os.path.join(out_dir, "load_q.csv"), index=False
    )
    if num_pv > 0:
        pd.DataFrame(pv_p_matrix).to_csv(
            os.path.join(out_dir, "pv_p.csv"), index=False
        )

    print(f"\nData generation complete! Output directory: {out_dir}")
    print(f"  load_p.csv: ({T_total}, {num_loads})")
    print(f"  load_q.csv: ({T_total}, {num_loads})")
    if num_pv > 0:
        print(f"  pv_p.csv:   ({T_total}, {num_pv})")

    # ================================================================
    # 6. 分析报告与时序曲线
    # ================================================================
    analyze_and_plot(load_p_matrix, pv_p_matrix, out_dir, cfg.steps_per_day, T_total)

    # ================================================================
    # 7. 渲染拓扑图
    # ================================================================
    render_topology(net, resource_table, out_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="生成 RL 环境所需的拓扑与时序数据（支持多种可调资源注入）"
    )
    parser.add_argument("--feeder", type=str, default="case33bw",
                        help="IEEE 标准网络名 (如 case33bw, case69)")
    parser.add_argument("--out_dir", type=str, default="data/generated",
                        help="输出根目录，实际路径会追加 {feeder}_{resource_tag}")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--interval_min", type=int, default=15,
                        help="时间间隔 (分钟)")
    parser.add_argument("--seed", type=int, default=42)

    # 资源数量参数
    parser.add_argument("--n_bess", type=int, default=3, help="BESS 数量")
    parser.add_argument("--n_gen", type=int, default=0, help="Generator 数量")
    parser.add_argument("--n_pv", type=int, default=3, help="Inverter/PV 数量")
    parser.add_argument("--n_dr", type=int, default=0, help="Demand Response 数量")

    args = parser.parse_args()

    steps_per_day = int((24 * 60) / args.interval_min)
    cfg = ScenarioConfig(days=args.days, steps_per_day=steps_per_day)

    resource_config = {
        "bess":            {"count": args.n_bess},
        "generator":       {"count": args.n_gen},
        "inverter":        {"count": args.n_pv},
        "demand_response": {"count": args.n_dr},
    }

    # 自动拼接输出目录名：{base_dir}/{feeder}_b{bess}p{pv}g{gen}d{dr}
    res_tag = (f"b{args.n_bess}p{args.n_pv}"
               f"g{args.n_gen}d{args.n_dr}")
    out_dir = os.path.join(args.out_dir, f"{args.feeder}_{res_tag}")

    generate_ieee_env_data(
        args.feeder, out_dir, cfg,
        resource_config=resource_config, seed=args.seed,
    )
