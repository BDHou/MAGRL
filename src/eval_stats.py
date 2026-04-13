import numpy as np


def print_episode_summary(records: dict, baseline: dict = None, dt: float = 0.25):
    """
    打印 episode 级统计摘要，对比 RL 策略与 baseline 的关键物理指标。

    Args:
        records  dict: evaluate_episode 返回的 RL 记录
        baseline dict: evaluate_baseline 返回的无控制基线记录, 可为 None
        dt       float: 时步长度(小时), 默认 0.25 (15分钟)
    """
    print("\n" + "=" * 70)
    print("  Episode Summary")
    print("=" * 70)

    p_rl = np.array(records["p_ext_grid"])

    # 1. 倒送总功 (MWh)
    reverse_rl = np.maximum(0.0, -p_rl)
    energy_reverse_rl = float(np.trapz(reverse_rl, dx=dt))

    # 2. 购电总量 (MWh)
    buy_rl = np.maximum(0.0, p_rl)
    energy_buy_rl = float(np.trapz(buy_rl, dx=dt))

    # 3. 电压偏离度 episode 均值
    vm_dev_rl = float(np.mean(records["vm_deviation"]))

    # 4. 各储能吞吐量 (MWh)
    throughputs = {}
    for aid, p_list in records["storage_p"].items():
        throughputs[aid] = float(np.sum(np.abs(p_list)) * dt)

    # 5. 电网峰值功率 (MW)
    peak_rl = float(np.max(p_rl))

    # 6. 电压越限时步占比
    vm_min_arr = np.array(records["vm_min"])
    vm_max_arr = np.array(records["vm_max"])
    n_steps = len(vm_min_arr)
    violation_mask = (vm_min_arr < 0.95) | (vm_max_arr > 1.05)
    violation_rate_rl = float(np.sum(violation_mask)) / max(n_steps, 1)

    # 7. 总回报
    total_return_rl = records["total_reward"]

    print("\n  [RL Policy]")
    print(f"    倒送总功:           {energy_reverse_rl:10.4f} MWh")
    print(f"    购电总量:           {energy_buy_rl:10.4f} MWh")
    print(f"    电压偏离度(均方):   {vm_dev_rl:10.6f}")
    print(f"    电网峰值功率:       {peak_rl:10.4f} MW")
    print(f"    电压越限步占比:     {violation_rate_rl * 100:9.2f} %")
    print(f"    总回报:             {total_return_rl:10.2f}")
    print("    各储能吞吐量 (MWh):")
    for aid, tp in throughputs.items():
        print(f"      {aid}: {tp:.4f}")

    if baseline is not None:
        p_bl = np.array(baseline["p_ext_grid"])

        reverse_bl = np.maximum(0.0, -p_bl)
        energy_reverse_bl = float(np.trapz(reverse_bl, dx=dt))

        buy_bl = np.maximum(0.0, p_bl)
        energy_buy_bl = float(np.trapz(buy_bl, dx=dt))

        vm_dev_bl = float(np.mean(baseline["vm_deviation"]))

        peak_bl = float(np.max(p_bl))

        vm_min_bl = np.array(baseline["vm_min"])
        vm_max_bl = np.array(baseline["vm_max"])
        n_bl = len(vm_min_bl)
        viol_bl = float(np.sum((vm_min_bl < 0.95) | (vm_max_bl > 1.05))) / max(n_bl, 1)

        total_return_bl = baseline["total_reward"]

        print("\n  [Baseline (no control)]")
        print(f"    倒送总功:           {energy_reverse_bl:10.4f} MWh")
        print(f"    购电总量:           {energy_buy_bl:10.4f} MWh")
        print(f"    电压偏离度(均方):   {vm_dev_bl:10.6f}")
        print(f"    电网峰值功率:       {peak_bl:10.4f} MW")
        print(f"    电压越限步占比:     {viol_bl * 100:9.2f} %")
        print(f"    总回报:             {total_return_bl:10.2f}")

        d_reverse = energy_reverse_bl - energy_reverse_rl
        d_buy = energy_buy_bl - energy_buy_rl
        d_vm = vm_dev_bl - vm_dev_rl
        d_peak = peak_bl - peak_rl
        d_viol = viol_bl - violation_rate_rl
        d_return = total_return_rl - total_return_bl

        print("\n  [RL vs Baseline]")
        print(f"    倒送减少:           {d_reverse:+10.4f} MWh  ({_pct(d_reverse, energy_reverse_bl)})")
        print(f"    购电减少:           {d_buy:+10.4f} MWh  ({_pct(d_buy, energy_buy_bl)})")
        print(f"    电压偏离优化:       {d_vm:+10.6f}      ({_pct(d_vm, vm_dev_bl)})")
        print(f"    峰值功率削减:       {d_peak:+10.4f} MW   ({_pct(d_peak, peak_bl)})")
        print(f"    越限步减少:         {d_viol * 100:+9.2f} pp")
        print(f"    总回报提升:         {d_return:+10.2f}")

    print("=" * 70 + "\n")


def _pct(delta: float, base: float) -> str:
    """计算百分比变化的格式化字符串"""
    if abs(base) < 1e-9:
        return "N/A"
    return f"{delta / abs(base) * 100:+.1f}%"
