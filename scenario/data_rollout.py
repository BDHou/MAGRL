# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Step3-A: 离线数据生成 & 保存 & split

生成的每个 sample = 一个时间步的图：
  - x: (N, F)
  - edge_index: (2, E)
  - edge_attr: (E, Fe)
  - labels:
      y_node_vm: (N,) float
      y_node_vviol: (N,) int {0,1}
      y_line_rpf: (E,) int {0,1}
      y_export: (1,) int {0,1}

保存为 PyG InMemoryDataset 格式：
  root/
    raw/        (空也行)
    processed/
      data.pt
      splits.pt
"""

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from .base_scenario import ScenarioConfig
from .online_backend import OnlineBackend
from .risk_events import RiskConfig


# -----------------------------
# Small helper: policies
# -----------------------------
def _policy_mix(backend: OnlineBackend, mode: str = "mix", rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    产生一个全局动作向量（backend 需要的 action_vec）
    目的：离线数据要多样，否则监督学习学不到“动作→电网变化”的关联。

    mode:
      - "zero": 全 0
      - "rand": 小幅随机 [-0.4, 0.4]
      - "pos": 全 +0.7
      - "neg": 全 -0.7
      - "mix": 按概率混合上面几种（推荐）
    """
    if rng is None:
        rng = backend.rng

    dim = backend.action_dim()
    if mode == "zero":
        return np.zeros(dim, dtype=float)
    if mode == "pos":
        return np.ones(dim, dtype=float) * 0.7
    if mode == "neg":
        return np.ones(dim, dtype=float) * (-0.7)
    if mode == "rand":
        return rng.uniform(-0.4, 0.4, size=(dim,)).astype(float)

    # mix
    p = rng.random()
    if p < 0.15:
        return np.zeros(dim, dtype=float)
    elif p < 0.35:
        return np.ones(dim, dtype=float) * 0.7
    elif p < 0.55:
        return np.ones(dim, dtype=float) * (-0.7)
    else:
        return rng.uniform(-0.4, 0.4, size=(dim,)).astype(float)


# -----------------------------
# Convert one StepResult -> PyG Data
# -----------------------------
def step_to_data(obs_graph: Dict[str, np.ndarray], targets: Dict[str, np.ndarray], meta: Dict[str, Any]) -> Data:
    """
    把 backend 的 obs_graph + targets 转成 torch_geometric.data.Data
    """
    x = torch.tensor(obs_graph["x"], dtype=torch.float32)
    edge_index = torch.tensor(obs_graph["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(obs_graph["edge_attr"], dtype=torch.float32)

    # labels
    y_node_vm = torch.tensor(targets["y_node_vm"], dtype=torch.float32).view(-1)          # (N,)
    y_node_vviol = torch.tensor(targets["y_node_vviol"], dtype=torch.long).view(-1)       # (N,)
    y_line_rpf = torch.tensor(targets["y_line_rpf"], dtype=torch.long).view(-1)           # (E,)
    y_export = torch.tensor(targets["y_export"], dtype=torch.long).view(-1)               # (1,)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_node_vm=y_node_vm,
        y_node_vviol=y_node_vviol,
        y_line_rpf=y_line_rpf,
        y_export=y_export,
    )

    # 可选：把 meta 信息附上（不影响训练，但调试很有用）
    # 注意：PyG Data 里放 python 对象会导致保存慢/不可序列化，所以只放简单数值
    data.t = int(meta.get("t", -1))
    data.day = int(meta.get("day", -1))
    data.hour = int(meta.get("hour", -1))
    data.risk_active = int(meta.get("risk_active", 0))

    return data


# -----------------------------
# PyG Dataset (InMemory)
# -----------------------------
class PowerGraphDataset(InMemoryDataset):
    """
    标准 PyG InMemoryDataset：
      - 你先用 generate_and_save(...) 把 data.pt 写好
      - 然后训练脚本里直接 root=... 读取即可
    """

    def __init__(self, root: str, transform=None, pre_transform=None):
        self.root_dir = root
        super().__init__(root=root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return []  # 我们不依赖 raw 文件

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt", "splits.pt"]

    def download(self):
        # 不需要
        pass

    def process(self):
        # 不自动 process：我们用 generate_and_save() 显式生成
        raise RuntimeError("Use generate_and_save() to create processed/data.pt first.")

    @staticmethod
    def save_splits(root: str, splits: Dict[str, List[int]]):
        os.makedirs(os.path.join(root, "processed"), exist_ok=True)
        torch.save(splits, os.path.join(root, "processed", "splits.pt"))

    @staticmethod
    def load_splits(root: str) -> Dict[str, List[int]]:
        path = os.path.join(root, "processed", "splits.pt")
        return torch.load(path)


# -----------------------------
# Main generator
# -----------------------------
def generate_and_save(
    root: str,
    *,
    feeder: str = "case33bw",
    cfg: Optional[ScenarioConfig] = None,
    risk_cfg: Optional[RiskConfig] = None,
    n_episodes: int = 200,
    horizon: int = 48,
    t0: int = 0,
    policy_mode: str = "mix",
    keep_pf_fail: bool = False,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    生成离线图数据并保存到 root/processed/data.pt
    返回一些统计信息方便你写论文/记录实验。
    """
    cfg = cfg or ScenarioConfig()

    # 统一 seed：保证可复现
    if seed is None:
        # 用 cfg.seed 做基础，避免你忘记传 seed 造成随机飘
        seed = int(cfg.seed + 12345)

    rng_master = np.random.default_rng(seed)

    all_data_list: List[Data] = []
    stat = {
        "n_episodes": int(n_episodes),
        "horizon": int(horizon),
        "t0": int(t0),
        "policy_mode": str(policy_mode),
        "keep_pf_fail": bool(keep_pf_fail),
        "seed": int(seed),
        "pf_ok_steps": 0,
        "pf_fail_steps": 0,
    }

    # 为了多样性：每个 episode 用不同的 scenario_seed
    for ep in range(n_episodes):
        ep_seed = int(rng_master.integers(0, 2**31 - 1))
        backend = OnlineBackend(
            feeder_name=feeder,
            cfg=cfg,
            mode="q_frac",
            enable_curtail_action=False,
            risk_cfg=risk_cfg,
            scenario_seed=ep_seed,    # 每条 episode 不同
            risk_seed=ep_seed + 999 if risk_cfg is not None else None,
        )

        sr0 = backend.reset(t0=t0)
        if sr0.ok and sr0.obs_graph is not None and sr0.targets is not None:
            all_data_list.append(step_to_data(sr0.obs_graph, sr0.targets, sr0.meta))
            stat["pf_ok_steps"] += 1
        else:
            stat["pf_fail_steps"] += 1
            if keep_pf_fail:
                # pf_fail 不建议当训练样本（label 不完整），这里默认不保存
                pass

        # 后续步
        for _ in range(horizon - 1):
            a = _policy_mix(backend, mode=policy_mode, rng=backend.rng)
            sr = backend.step(a)
            if sr.ok and sr.obs_graph is not None and sr.targets is not None:
                all_data_list.append(step_to_data(sr.obs_graph, sr.targets, sr.meta))
                stat["pf_ok_steps"] += 1
            else:
                stat["pf_fail_steps"] += 1
                if keep_pf_fail:
                    pass

        if verbose and (ep + 1) % max(1, n_episodes // 10) == 0:
            print(f"[data_rollout] episode {ep+1}/{n_episodes} | samples={len(all_data_list)} "
                  f"| ok={stat['pf_ok_steps']} fail={stat['pf_fail_steps']}")

    # 保存成 InMemoryDataset 的标准格式
    dataset = PowerGraphDataset.__new__(PowerGraphDataset)  # 不走 __init__
    data, slices = InMemoryDataset.collate(all_data_list)

    os.makedirs(os.path.join(root, "processed"), exist_ok=True)
    torch.save((data, slices), os.path.join(root, "processed", "data.pt"))

    # split（按样本随机划分）
    n = len(all_data_list)
    idx = np.arange(n)
    rng_master.shuffle(idx)

    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_idx = idx[:n_train].tolist()
    val_idx = idx[n_train:n_train + n_val].tolist()
    test_idx = idx[n_train + n_val:].tolist()

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    PowerGraphDataset.save_splits(root, splits)

    stat["n_samples"] = int(n)
    stat["train/val/test"] = (len(train_idx), len(val_idx), len(test_idx))

    # 也写一份 json 方便你记录实验
    with open(os.path.join(root, "processed", "rollout_stat.json"), "w", encoding="utf-8") as f:
        json.dump(stat, f, indent=2, ensure_ascii=False)

    if verbose:
        print("[data_rollout] saved:", os.path.join(root, "processed", "data.pt"))
        print("[data_rollout] splits:", stat["train/val/test"])
        print("[data_rollout] ok_rate:",
              stat["pf_ok_steps"] / max(1, (stat["pf_ok_steps"] + stat["pf_fail_steps"])))

    return stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/offline_case33bw", help="dataset root dir")
    ap.add_argument("--feeder", type=str, default="case33bw")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--policy", type=str, default="mix", choices=["mix", "zero", "rand", "pos", "neg"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--risk", action="store_true", help="enable risk manager (default off)")
    args = ap.parse_args()

    cfg = ScenarioConfig()

    risk_cfg = None
    if args.risk:
        # 先默认全关（你后面要引入风险事件再开）
        risk_cfg = RiskConfig(
            enable_contingency=False,
            enable_line_derating=False,
            enable_overload=False,
        )

    generate_and_save(
        root=args.root,
        feeder=args.feeder,
        cfg=cfg,
        risk_cfg=risk_cfg,
        n_episodes=args.episodes,
        horizon=args.horizon,
        t0=args.t0,
        policy_mode=args.policy,
        seed=args.seed,
        verbose=True,
    )


if __name__ == "__main__":
    main()