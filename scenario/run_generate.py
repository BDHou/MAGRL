# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from .base_scenario import (
    ScenarioConfig, load_feeder_by_name, generate_dataset_for_feeder,
    summarize_metrics, save_pickle, stable_int_hash
)
# ScenarioConfig：总配置（days、seed、pv设置、控制开关等）
# load_feeder_by_name(name)：给一个名字，比如 "case33bw"，帮你从 pandapower.networks 里加载这个电网
# generate_dataset_for_feeder(...)：真正“跑潮流 + 控制 + 生成样本/summary”的主函数
# summarize_metrics(df)：把每个 feeder 的 summary 表统计成一些指标（export_rate 等）
# save_pickle(obj, path)：把 samples 存成 .pkl
# stable_int_hash(feeder_name)：给 feeder 名字一个稳定 hash（保证不同机器/不同运行也一致）
# RiskConfig/RiskManager：风险事件注入（断线、降额、额外负载等）
from .risk_events import RiskConfig, RiskManager

##整个脚本的入口
#命令行参数：你可以在终端改哪些东西
# --mode：
# base：正常场景（无风险事件）
# risk：开启风险事件（断线/降额/overload）
# --out_dir：输出文件夹名（不写就用配置默认值）
# --days：仿真天数（不写就用配置默认 days=30）
# --seed：随机种子（保证可复现）
# python -m scenario.run_generate --mode risk --seed 1
# python -m scenario.run_generate --mode base --seed 1 
def main(): 
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="base", choices=["base", "risk"])
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    # 生成默认配置 cfg，并用命令行覆盖它
    cfg = ScenarioConfig()
    #ScenarioConfig() 提供默认值
    #在命令行传了什么，就覆盖默认
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir 
    if args.days is not None:
        cfg.days = int(args.days)
    if args.seed is not None:
        cfg.seed = int(args.seed)

    out_dir = Path(cfg.out_dir) #把字符串变成路径对象
    out_dir.mkdir(parents=True, exist_ok=True)
    # 目录不存在就创建
	# 已经存在也不报错
	# parents=True 会连同上级目录一起创建

    print("Mode:", args.mode)
    print("Config:", cfg)
    # 终端看到的那一大串 
    # ScenarioConfig(...) 就是这里打出来的。

    #找到本机可用的 feeder（关键）
    #脚本逐个尝试加载：
	# 如果你当前 pandapower 版本没有这个案例 
    #→ net is None → 打印 skip
    # 如果这个 net 不完整（没有 ext_grid 或 line）→ skip
    # （ext_grid 是“外部电网/主电源/slack”，
    #line 是线路；没有这俩潮流算不了）
    # 把可用的 (feeder_name, net) 收集进 feeder_nets
    feeder_nets = []
    for name in cfg.feeder_candidates:
        net = load_feeder_by_name(name)
        if net is None:
            print(f"[skip] feeder '{name}' not available in your pandapower version.")
            continue
        if len(net.ext_grid) == 0 or len(net.line) == 0:
            print(f"[skip] feeder '{name}' missing ext_grid or line.")
            continue
        feeder_nets.append((name, net))
    #如果一个都没找到，就直接报错
    if len(feeder_nets) == 0:
        raise RuntimeError("No valid feeders found. Update cfg.feeder_candidates based on your local pandapower.networks.")

    #结果收集箱
    #每跑完一个 feeder，你会算出一行指标（metrics），比如 export_rate、v_viol 等，
    #然后 append 到这个 list。
	#后把它变成一个 DataFrame 并写成 metrics_compare_all_feeders*.csv。
    all_metrics_rows = []

    #遍历每个 feeder
    for feeder_name, net0 in feeder_nets: 
        #feeder_nets 是前面筛出来“能加载、能跑”的电网集合。
        #feeder_name：比如 case33bw、mv_oberrhein
        #net0：pandapower 的那个电网对象（包含 bus、line、load、ext_grid 等表）
        print("\n==============================")
        print("Feeder:", feeder_name)
        print("==============================")
        print("max_i_ka in columns?", "max_i_ka" in net0.line.columns)
        if "max_i_ka" in net0.line.columns:
            s = net0.line["max_i_ka"]
            print("max_i_ka unique:", s.nunique(dropna=False))
            print("max_i_ka min/max:", float(s.min(skipna=True)), float(s.max(skipna=True)))
            print("head:", s.head().to_list())
        #非常关键：scenario_seed 保证两边随机轨迹一致
        # ✅ 同一 feeder：no-control / control 用同一条随机轨迹
        #     #同一个 feeder 下：
        # nocontrol 用 scenario_seed
        # control 也用 同一个 scenario_seed
        #stable_int_hash(feeder_name) 用来保证不同 feeder 的 seed 
        #不一样，同时又是稳定的（不会因为 Python hash 随机化而变）。
        
        scenario_seed = int(cfg.seed + stable_int_hash(feeder_name))

        # 先把 risk manager 设成 None
        risk_mgr_noc = None
        risk_mgr_ctl = None

        # 如果是 risk 模式：创建风险配置 rcfg
        if args.mode == "risk":
            rcfg = RiskConfig(
                enable_contingency=True,
                contingency_type="n-1",
                contingency_k=1,
                contingency_prob_per_step=0.03,
                contingency_duration_steps=6,
                contingency_elements=("line",),

                enable_line_derating=True,
                derate_prob_per_step=0.02,
                derate_duration_steps=6,
                derate_range=(0.7, 1.0),

                enable_overload=True,
                overload_prob_per_step=0.02,
                overload_duration_steps=6,
                overload_mult_range=(1.2, 1.8),
            )
            
            #关键：risk_seed 保证两边风险事件一致
            # +999 只是为了跟 scenario_seed 拉开距离（避免两种随机过程“碰巧相关”）
	        # 同一个 feeder，在 noc/control 两次运行中，
            # 都用同一个 risk_seed → 风险发生的时间点、类型、参数都一致
            # ✅ 风险事件也要一致：稳定 seed（不用 Python hash）
            risk_seed = int(cfg.seed + 999 + stable_int_hash(feeder_name))

            # ✅ 最稳：分别 new 两个 manager，但 RNG 用同一个 seed → 事件序列一致，状态不共享
            risk_mgr_noc = RiskManager(rcfg, np.random.default_rng(risk_seed))
            risk_mgr_ctl = RiskManager(rcfg, np.random.default_rng(risk_seed))
            #不应该让两边 share 同一个 RiskManager 对象
            # 因为 RiskManager 内部有 active 状态机，会被 step 修改。
            # 所以创建两个 RiskManager 实例 → 状态互不干扰
            # 但两者 RNG 都用同一个 seed → 触发事件序列一致

        # deepcopy：两套电网模型互不影响
        # 原因：
        # pandapower 的 net 是一个“可变对象”，里面的表会被你改：
        # load 的 p/q 会改
        # PV 的 p/q 会改
        # contingency 会把 line.in_service 改
        # derating 会把 max_i_ka 改
        # 如果你用同一个 net 对象跑两次，会互相污染。
        # 所以必须 deepcopy，保证：
        # nocontrol 和 control 是从同一个初始电网开始，但之后各自独立演化。
        net_noc = copy.deepcopy(net0)
        net_ctl = copy.deepcopy(net0)

        # 真正开始跑：nocontrol 和 control 各跑一遍
        #  生成一条时间序列（T_total = days * steps_per_day）
        # 每个 t 都跑 run_powerflow_with_controls
        # control=False：只跑 PF，不做 volt-var / curtail
        # 收集：
        # samples_noc：训练用的图数据样本（以后喂给 GNN）
        # f_noc：日志表（每个 t 一行：pf_ok、num_v_viol 等）
        samples_noc, df_noc = generate_dataset_for_feeder(
            net_noc, feeder_name, cfg,
            control=False,
            risk_mgr=risk_mgr_noc,
            scenario_seed=scenario_seed,
        )
        #然后同理跑 control=True 的一遍：
        samples_ctl, df_ctl = generate_dataset_for_feeder(
            net_ctl, feeder_name, cfg,
            control=True,
            risk_mgr=risk_mgr_ctl,
            scenario_seed=scenario_seed,
        )
        #打印 debug：control 这边到底失败在哪里
        #stage 是在 generate_dataset_for_feeder() 里记录的：
	    # ok / base_pf / volt_var_pf / curtail_pf / post_curtail_voltvar_pf …
        #value_counts 会告诉说：大多数时间都 ok？还是很多时间卡在某个阶段？
        print(f"\n[{feeder_name}] df_ctl stage counts:")
        print(df_ctl["stage"].value_counts(dropna=False))
        #把失败的步挑出来看：
        # 第几步失败（t）
        # 在哪个阶段失败（stage）
        # 错误是什么（error，比如 LoadflowNotConverged）
        #定位 lv_schutterwald 一直 nr did not converge 的关键依据。
        bad = df_ctl[df_ctl["pf_ok"] == 0][["t", "stage", "error"]].head(5)
        print(f"\n[{feeder_name}] first 5 failures in df_ctl:")
        print(bad.to_string(index=False))

        #     如果你运行的是 --mode risk，那输出文件名就带上 __risk
        # 	如果是 --mode base，suffix 就是空字符串 ""
        #避免 base 和 risk 的输出文件互相覆盖。
        suffix = "__risk" if args.mode == "risk" else ""
        #生成两个 pkl 文件路径（nocontrol 和 control）
        #这些 pkl 里装的是后面 GNN 训练真正要用的东西（图输入 x/edge_index/edge_attr + 标签 y）。
        pkl_noc = out_dir / f"feeder_{feeder_name}__nocontrol{suffix}.pkl"
        pkl_ctl = out_dir / f"feeder_{feeder_name}__control{suffix}.pkl"
        #把 samples 保存成 pickle
        #samples_noc / samples_ctl 是 Python 的 list，每个元素是一条样本 dict（含 meta、图结构、y 标签等）
        # pickle 是 Python 的“打包保存”格式
        #存下来以后，以后训练不用再重新跑 powerflow，直接 load pkl 就行
        save_pickle(samples_noc, pkl_noc)
        save_pickle(samples_ctl, pkl_ctl)

        #把日志 df 保存成 CSV
        #df_noc / df_ctl 是你在 generate_dataset_for_feeder() 里收集的 logs DataFrame
        #每一行对应一个时间步 t，比如：pf_ok 是否成功，export 是否发生，num_v_viol 电压违规数量
        #um_rpf_lines 反送线路数量，stage / error（如果失败）
        #未来可以快速画图（比如随时间的 v_viol 曲线、失败在哪个 stage）
        df_noc.to_csv(out_dir / f"summary_{feeder_name}__nocontrol{suffix}.csv", index=False)
        df_ctl.to_csv(out_dir / f"summary_{feeder_name}__control{suffix}.csv", index=False)

        # df_noc.to_csv(out_dir / f"summary_{feeder_name}__nocontrol{suffix}.csv", index=False)
        # df_ctl.to_csv(out_dir / f"summary_{feeder_name}__control{suffix}.csv", index=False)

        #汇总指标 m_noc / m_ctl
        #从 df 里提取“整体统计指标”，比如：export_rate：export 发生的比例，avg_num_rpf_lines：平均有多少条线路反送
        #avg_num_v_viol：平均电压违规数量，max_vm_mean / min_vm_mean：电压最高/最低的平均值
        #pv_total_p_mean：平均 PV 输出功率，curtail_total_mean：平均削减量，
        #pv_energy_sum_mwh：总能量（目前用的是 pv_total_p 的 sum，当作能量 proxy）
        m_noc = summarize_metrics(df_noc)
        m_ctl = summarize_metrics(df_ctl)
        #     m_noc 就是一组字典：nocontrol 的整体表现，m_ctl 就是一组字典：control 的整体表现
        #后面可以拿它们做差：delta = ctl - noc，用来衡量控制是否改善违规、是否减少反送、是否牺牲了 PV 能量等。

        #打印 risk_active 统计（只用于 debug / 验证 risk 是否真的触发）
        # risk_active 0: 480, 1: 240（说明 720 个 step 里有 240 个 step 处于风险事件影响期）
        # 并且 noc 与 ctl 的统计应该一致（因为你故意把 risk_seed 设成一样，让两条轨迹的风险事件同步发生）。
        print(f"[{feeder_name}] risk_active count (noc):")
        if "risk_active" in df_noc.columns:
            print(df_noc["risk_active"].value_counts(dropna=False))
        else:
            print("[warn] risk_active column missing in df_noc")

        print(f"[{feeder_name}] risk_active count (ctl):")
        if "risk_active" in df_ctl.columns:
            print(df_ctl["risk_active"].value_counts(dropna=False))
        else:
            print("[warn] risk_active column missing in df_ctl")

        #开始为这个 feeder 准备一行 “总指标” 的表格记录
        #先创建一个字典 row，它代表未来 CSV 表格的一行。比如先放入：{"feeder": "case33bw"}
        row = {"feeder": feeder_name}
        #把 no-control/control汇总指标写进 row，m_noc 是 summarize_metrics(df_noc) 的结果，是一个 dict，例如：
        #export_rate，avg_num_rpf_lines，avg_num_v_viol，max_vm_mean。。。pv_energy_sum_mwh
        #这段把每个 key 前面加上 noc_，变成：noc_export_rate，noc_avg_num_rpf_lines。。。同理，control 模式的指标加前缀 ctl_：
        for k, v in m_noc.items():
            row[f"noc_{k}"] = v
        for k, v in m_ctl.items():
            row[f"ctl_{k}"] = v
        
        #计算 delta：control 相比 no-control 改变了多少。这里 delta 的定义是：delta = control - no-control
        #delta_avg_num_v_viol < 0，控制把电压违规数减少了（好事）
        #delta_avg_num_rpf_lines < 0，控制把反送线路数减少了（通常是好事）
        #delta_export_rate < 0，控制让反送（ext_grid 负功率）的比例降低了（往往好事）
        #delta_pv_energy_sum_mwh < 0，控制导致 PV 输出总能量减少（一般是因为 curtailment 削减了 PV），这通常是“代价/权衡”。
        row["delta_avg_num_v_viol"] = row["ctl_avg_num_v_viol"] - row["noc_avg_num_v_viol"]
        row["delta_avg_num_rpf_lines"] = row["ctl_avg_num_rpf_lines"] - row["noc_avg_num_rpf_lines"]
        row["delta_export_rate"] = row["ctl_export_rate"] - row["noc_export_rate"]
        row["delta_pv_energy_sum_mwh"] = row["ctl_pv_energy_sum_mwh"] - row["noc_pv_energy_sum_mwh"]
        all_metrics_rows.append(row) #把这一行加入总表 all_metrics_rows

        #打印保存信息与每个 feeder 的指标，就是在 terminal 里看到的那些输出。
        print(f"[saved] {pkl_noc}")
        print(f"[saved] {pkl_ctl}")
        print("No-control metrics:", m_noc)
        print("Control metrics   :", m_ctl)

    #所有 feeder 跑完后：生成总表 metrics_df 并保存 CSV
    #这段做的是：把所有行组成一个 DataFrame（表格）根据当前模式决定文件名：base：metrics_compare_all_feeders.csv
	#risk：metrics_compare_all_feeders__risk.csv，保存 CSV，把表格打印出来给你看
    metrics_df = pd.DataFrame(all_metrics_rows)
    metrics_df.to_csv(out_dir / f"metrics_compare_all_feeders{('__risk' if args.mode=='risk' else '')}.csv", index=False)
    print("\n[saved] metrics_compare_all_feeders*.csv")
    print(metrics_df)

#最后这句：允许你用 python -m scenario.run_generate 运行
if __name__ == "__main__":
    main()