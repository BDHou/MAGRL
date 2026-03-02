# train/ablation_run.py
# -*- coding: utf-8 -*-
"""
Step3-D: Ablation runner

一键跑你的 ablation 并汇总成表格（CSV + JSONL）
用法：
  python -m train.ablation_run --data_root data/offline_case33bw --device mps

输出：
  checkpoints/ablation/ablation_results.csv
  checkpoints/ablation/ablation_results.jsonl

注意：
- 这个脚本会多次调用 `python -m train.supervised_pretrain ...`
- 每个 run 都会写一个独立 save_dir，然后读取其中的 run_metrics.json 汇总
"""

from __future__ import annotations

import os
import json
import argparse
import subprocess
from datetime import datetime
from typing import Dict, Any, List


def run_cmd(cmd: List[str]) -> int:
    print("\n$ " + " ".join(cmd))
    return subprocess.call(cmd)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/offline_case33bw")
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--epochs", type=int, default=20)       # ablation 轻量一点
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--out_dir", type=str, default="checkpoints/ablation")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.out_dir, f"runs_{tag}")
    os.makedirs(run_root, exist_ok=True)

    configs = []

    # -------------------------
    # Baseline (main method)
    # -------------------------
    configs.append(dict(
        name="BASE_sage_edgeattr_multitask_posw_tunedthr",
        backbone="sage",
        use_edge_attr=1,
        tasks="vm,vviol,rpf,export",
        use_pos_weight=1,
        thr_mode="tuned",
        thr_fixed=0.5,
    ))

    # -------------------------
    # Ablation 1: backbone swap
    # -------------------------
    for bb in ["mlp", "gcn", "gat", "sage"]:
        configs.append(dict(
            name=f"BB_{bb}",
            backbone=bb,
            use_edge_attr=1,
            tasks="vm,vviol,rpf,export",
            use_pos_weight=1,
            thr_mode="tuned",
            thr_fixed=0.5,
        ))

    # -------------------------
    # Ablation 2: remove edge_attr
    # -------------------------
    configs.append(dict(
        name="NO_EDGE_ATTR",
        backbone="sage",
        use_edge_attr=0,
        tasks="vm,vviol,rpf,export",
        use_pos_weight=1,
        thr_mode="tuned",
        thr_fixed=0.5,
    ))

    # -------------------------
    # Ablation 3: single-task vs multi-task
    # -------------------------
    for t in ["vviol", "rpf", "export"]:
        configs.append(dict(
            name=f"SINGLE_{t}",
            backbone="sage",
            use_edge_attr=1,
            tasks=t,
            use_pos_weight=1,
            thr_mode="tuned",
            thr_fixed=0.5,
        ))

    # -------------------------
    # Ablation 4: pos_weight / threshold contribution
    # -------------------------
    configs.append(dict(
        name="NO_POS_WEIGHT",
        backbone="sage",
        use_edge_attr=1,
        tasks="vm,vviol,rpf,export",
        use_pos_weight=0,
        thr_mode="tuned",
        thr_fixed=0.5,
    ))
    configs.append(dict(
        name="FIXED_THR_0p5",
        backbone="sage",
        use_edge_attr=1,
        tasks="vm,vviol,rpf,export",
        use_pos_weight=1,
        thr_mode="fixed",
        thr_fixed=0.5,
    ))

    results = []

    for cfg in configs:
        save_dir = os.path.join(run_root, cfg["name"])
        os.makedirs(save_dir, exist_ok=True)

        cmd = [
            "python", "-m", "train.supervised_pretrain",
            "--data_root", args.data_root,
            "--device", args.device,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--hidden", str(args.hidden),
            "--lr", str(args.lr),
            "--wd", str(args.wd),
            "--save_dir", save_dir,
            "--backbone", cfg["backbone"],
            "--use_edge_attr", str(cfg["use_edge_attr"]),
            "--tasks", cfg["tasks"],
            "--use_pos_weight", str(cfg["use_pos_weight"]),
            "--thr_mode", cfg["thr_mode"],
            "--thr_fixed", str(cfg["thr_fixed"]),
        ]

        code = run_cmd(cmd)
        if code != 0:
            print(f"[warn] run failed: {cfg['name']} (exit={code})")
            continue

        metrics_path = os.path.join(save_dir, "run_metrics.json")
        if not os.path.exists(metrics_path):
            print(f"[warn] missing run_metrics.json: {cfg['name']}")
            continue

        m = read_json(metrics_path)
        row = {
            "name": cfg["name"],
            "backbone": m.get("backbone"),
            "edge_attr": m.get("use_edge_attr"),
            "tasks": ",".join(m.get("tasks", [])),
            "pos_weight": m.get("use_pos_weight"),
            "thr_mode": m.get("thr_mode"),
            "thr_fixed": m.get("thr_fixed"),
            "thr_vviol": m["test"].get("thr_vviol"),
            "thr_rpf": m["test"].get("thr_rpf"),
            "thr_export": m["test"].get("thr_export"),
            "vm_rmse": m["test"].get("vm_rmse"),
            "vm_mae": m["test"].get("vm_mae"),
            "vviol_auc": m["test"].get("vviol_auc"),
            "vviol_f1": m["test"].get("vviol_f1"),
            "rpf_auc": m["test"].get("rpf_auc"),
            "rpf_f1": m["test"].get("rpf_f1"),
            "export_auc": m["test"].get("export_auc"),
            "export_acc": m["test"].get("export_acc"),
            "save_dir": save_dir,
        }
        results.append(row)

    # write JSONL
    jsonl_path = os.path.join(args.out_dir, "ablation_results.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # write CSV (simple, no pandas dependency)
    csv_path = os.path.join(args.out_dir, "ablation_results.csv")
    if results:
        keys = list(results[0].keys())
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for r in results:
                vals = []
                for k in keys:
                    v = r.get(k, "")
                    s = str(v)
                    # escape commas
                    if "," in s or "\n" in s:
                        s = '"' + s.replace('"', '""') + '"'
                    vals.append(s)
                f.write(",".join(vals) + "\n")

    print("\n=== Ablation Done ===")
    print("runs:", run_root)
    print("csv :", csv_path)
    print("jsonl:", jsonl_path)


if __name__ == "__main__":
    main()