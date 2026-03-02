# -*- coding: utf-8 -*-
"""
这个 scenario/smoke_test_online.py 在你们 project 里的作用可以用一句话概括：

它是“点火测试（smoke test）/回归测试脚本”，用来快速证明 online_backend 真的能跑、接口稳定、动作确实能影响电网结果。

它不是训练代码，不是环境代码，更不是 GNN/MARL 本体；它是你每次改代码后第一时间确认：online 还活着的那把“试火机”。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base_scenario import ScenarioConfig
from .online_backend import OnlineBackend
from .risk_events import RiskConfig


def rollout(backend: OnlineBackend, policy_name: str, policy_fn, steps: int = 48, t0: int = 0):
    sr0 = backend.reset(t0=t0)
    logs = []

    # reset 返回的就是第一个 step 结果（t=t0）
    if sr0.ok:
        m = sr0.metrics
        logs.append({
            "t": sr0.meta["t"], "ok": 1,
            "max_vm": m["max_vm"], "min_vm": m["min_vm"],
            "num_v_viol": m["num_v_viol"], "export": m["export"],
            "num_rpf_lines": m["num_rpf_lines"],
            "pv_total_p": m["pv_total_p"], "pv_total_q": m["pv_total_q"],
            "risk_active": sr0.meta.get("risk_active", 0),
        })
    else:
        logs.append({"t": sr0.meta["t"], "ok": 0, "stage": sr0.info.get("stage", "")})

    # 检查 obs shape
    if sr0.ok and sr0.obs_graph is not None:
        x = sr0.obs_graph["x"]
        edge_index = sr0.obs_graph["edge_index"]
        edge_attr = sr0.obs_graph["edge_attr"]
        print(f"[{policy_name}] obs shapes: x={x.shape}, edge_index={edge_index.shape}, edge_attr={edge_attr.shape}")

    for _ in range(steps - 1):
        a = policy_fn(backend)
        sr = backend.step(a)
        if sr.ok:
            m = sr.metrics
            logs.append({
                "t": sr.meta["t"], "ok": 1,
                "max_vm": m["max_vm"], "min_vm": m["min_vm"],
                "num_v_viol": m["num_v_viol"], "export": m["export"],
                "num_rpf_lines": m["num_rpf_lines"],
                "pv_total_p": m["pv_total_p"], "pv_total_q": m["pv_total_q"],
                "risk_active": sr.meta.get("risk_active", 0),
            })
        else:
            logs.append({
                "t": sr.meta["t"], "ok": 0,
                "stage": sr.info.get("stage", ""),
                "error": sr.info.get("error", "")[:180],
                "risk_active": sr.meta.get("risk_active", 0),
            })

    df = pd.DataFrame(logs)
    ok_rate = float(df["ok"].mean()) if "ok" in df.columns else 0.0
    print(f"[{policy_name}] ok_rate = {ok_rate:.3f} ({int(df['ok'].sum())}/{len(df)})")

    # 只看成功步
    df_ok = df[df["ok"] == 1].copy()
    if len(df_ok) > 0:
        print(f"[{policy_name}] mean(num_v_viol)={df_ok['num_v_viol'].mean():.3f}, "
              f"mean(num_rpf_lines)={df_ok['num_rpf_lines'].mean():.3f}, "
              f"export_rate={df_ok['export'].mean():.3f}, "
              f"max_vm_mean={df_ok['max_vm'].mean():.4f}, min_vm_mean={df_ok['min_vm'].mean():.4f}, "
              f"pv_total_p_mean={df_ok['pv_total_p'].mean():.3f}, pv_total_q_mean={df_ok['pv_total_q'].mean():.3f}")

        if "risk_active" in df_ok.columns:
            print(f"[{policy_name}] risk_active_rate={df_ok['risk_active'].mean():.3f}")

    return df


def main():
    cfg = ScenarioConfig()
    feeder = "case33bw"

    # -------- 可选：risk 模式（你也可以先 None）--------
    # risk_cfg = None
    risk_cfg = RiskConfig(
        enable_contingency=False,
        enable_line_derating=False,
        enable_overload=False,
    )

    # -------- backend（完整 online）--------
    backendA = OnlineBackend(
        feeder_name=feeder,
        cfg=cfg,
        mode="q_frac",                 # 推荐：先用 q_frac ∈ [-1,1]
        enable_curtail_action=False,   # 先关掉 curtail action（后面你要也可以开）
        risk_cfg=risk_cfg if risk_cfg is not None else None,
        scenario_seed=None,
        risk_seed=None,
    )

    print("Feeder:", feeder)
    print("nbus:", len(backendA.net.bus), "nline:", len(backendA.net.line), "nload:", len(backendA.net.load))
    print("n_pv:", backendA.n_pv, "pv_buses:", backendA.net._pv_buses.tolist())
    print("action_dim:", backendA.action_dim(), "mode:", backendA.mode)

    # -------- policies --------
    def policy_zero(backend: OnlineBackend):
        return np.zeros(backend.action_dim(), dtype=float)

    def policy_qpos(backend: OnlineBackend):
        # q_frac = +0.7（抬电压倾向）
        return np.ones(backend.action_dim(), dtype=float) * 0.7

    def policy_qneg(backend: OnlineBackend):
        # q_frac = -0.7（压电压倾向）
        return np.ones(backend.action_dim(), dtype=float) * (-0.7)

    def policy_random(backend: OnlineBackend):
        # 小幅随机，避免太激进
        return backend.rng.uniform(-0.4, 0.4, size=(backend.action_dim(),)).astype(float)

    # -------- rollout comparisons --------
    steps = 48
    t0 = 0

    # 为了公平对比：每次 rollout 都要 new 一个 backend（否则内部 t/last_vm 会继续走）
    def new_backend():
        return OnlineBackend(
            feeder_name=feeder,
            cfg=cfg,
            mode="q_frac",
            enable_curtail_action=False,
            risk_cfg=risk_cfg if risk_cfg is not None else None,
            scenario_seed=backendA.scenario_seed,  # 固定一样
            risk_seed=backendA.risk_seed if backendA.risk_seed is not None else None,
        )

    df0 = rollout(new_backend(), "ZERO", policy_zero, steps=steps, t0=t0)
    dfp = rollout(new_backend(), "Q_POS", policy_qpos, steps=steps, t0=t0)
    dfn = rollout(new_backend(), "Q_NEG", policy_qneg, steps=steps, t0=t0)
    dfr = rollout(new_backend(), "RANDOM", policy_random, steps=steps, t0=t0)

    # 简单对比（只对成功步）
    def summarize(df: pd.DataFrame, name: str):
        d = df[df["ok"] == 1].copy()
        if len(d) == 0:
            return {"name": name, "ok_rate": float(df["ok"].mean())}
        return {
            "name": name,
            "ok_rate": float(df["ok"].mean()),
            "mean_num_v_viol": float(d["num_v_viol"].mean()),
            "mean_num_rpf_lines": float(d["num_rpf_lines"].mean()),
            "export_rate": float(d["export"].mean()),
            "max_vm_mean": float(d["max_vm"].mean()),
            "min_vm_mean": float(d["min_vm"].mean()),
            "pv_total_p_mean": float(d["pv_total_p"].mean()),
            "pv_total_q_mean": float(d["pv_total_q"].mean()),
        }

    table = pd.DataFrame([
        summarize(df0, "ZERO"),
        summarize(dfp, "Q_POS"),
        summarize(dfn, "Q_NEG"),
        summarize(dfr, "RANDOM"),
    ])
    print("\n=== Summary Compare ===")
    print(table.to_string(index=False))

    print("\n✅ 判定通过标准：Q_POS/Q_NEG 相对 ZERO 的 max_vm/min_vm/num_v_viol/export/num_rpf_lines 至少有一个明显变化。")


if __name__ == "__main__":
    main()