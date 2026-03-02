'''
•	主任务/主方法（Step3 的主方法）：train/supervised_pretrain.py 这套 多任务监督预训练（vm 回归 + vviol/rpf/export 分类）
•	SAGE / GCN / GAT / MLP：只是你在这个主方法里可替换的 编码器（图特征提取器），属于 ablation 里对 “backbone” 的对比
论文主方法是：
“离线 rollout 生成图数据 + 多任务监督预训练（可选 pos_weight + tuned threshold）”

'''

# train/supervised_pretrain.py
# -*- coding: utf-8 -*-
# Step3-D: 参数化监督预训练（支持 Ablation：backbone/edge_attr/tasks/pos_weight/threshold）
#
# 基本用法（主方法）：
#   python -m train.supervised_pretrain --data_root data/offline_case33bw --epochs 30 --batch_size 32 --device mps
#
# Ablation 示例：
#   1) backbone=mlp
#   python -m train.supervised_pretrain --backbone mlp ...
#
#   2) 去掉 edge_attr
#   python -m train.supervised_pretrain --use_edge_attr 0 ...
#   python -m train.supervised_pretrain --data_root data/offline_case33bw --epochs 20 --batch_size 32 --device mps --use_edge_attr 0

# backbone 对比（mlp/gcn/gat/sage）
# python -m train.supervised_pretrain --data_root data/offline_case33bw --epochs 20 --batch_size 32 --device mps --backbone mlp
# python -m train.supervised_pretrain --data_root data/offline_case33bw --epochs 20 --batch_size 32 --device mps --backbone gcn
# python -m train.supervised_pretrain --data_root data/offline_case33bw --epochs 20 --batch_size 32 --device mps --backbone gat
# sage 已经跑过
#GAT（用 CPU）
#python -m train.supervised_pretrain --data_root data/offline_case33bw \
#   --epochs 20 --batch_size 32 --device cpu --backbone gat \
#   --save_dir checkpoints/abl_gat_cpu
#   3) 单任务：只跑 vviol
#   python -m train.supervised_pretrain --tasks vviol ...
#
#   4) 关 pos_weight
#   python -m train.supervised_pretrain --use_pos_weight 0 ...
#
#   5) 固定阈值=0.5
#   python -m train.supervised_pretrain --thr_mode fixed --thr_fixed 0.5 ...
#
from __future__ import annotations

import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
)

from scenario.data_rollout import PowerGraphDataset
from models.gnn_backbone import GNNBackboneConfig, build_backbone
from models.predictor_heads import (
    HeadConfig,
    NodeVMRegressor,
    NodeVViolClassifier,
    EdgeRPFClassifier,
    ExportClassifier,
)


# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    batch_size: int = 32
    device: str = "cpu"

    # loss weights（你后面可以调）
    w_vm: float = 1.0
    w_vviol: float = 1.0
    w_rpf: float = 1.0
    w_export: float = 0.5

    # threshold search
    thr_grid_n: int = 41
    thr_min: float = 0.01
    thr_max: float = 0.99


# -----------------------------
# Utils: edge label alignment
# -----------------------------
def _match_edge_labels_to_logits(y_edge: torch.Tensor, logit_edge: torch.Tensor) -> torch.Tensor:
    """
    y_edge 可能是按“物理line”存的 (E_line,)
    logit_edge 可能是按“双向edge”预测的 (2*E_line,)
    如果发现正好差一个整数倍（常见就是2），repeat_interleave 让维度匹配。
    """
    y_edge = y_edge.view(-1)
    logit_edge = logit_edge.view(-1)

    if y_edge.numel() == logit_edge.numel():
        return y_edge

    if logit_edge.numel() % y_edge.numel() != 0:
        raise ValueError(
            f"Edge label/logit size not compatible: y={y_edge.numel()} logit={logit_edge.numel()}"
        )

    k = logit_edge.numel() // y_edge.numel()  # 通常是2
    return y_edge.repeat_interleave(k)


def math_sqrt(x: float) -> float:
    return float(np.sqrt(float(x)))


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def parse_tasks(s: str) -> List[str]:
    """
    tasks string: "vm,vviol,rpf,export" or "vviol" etc
    allowed: vm, vviol, rpf, export
    """
    s = (s or "").strip()
    if not s:
        return ["vm", "vviol", "rpf", "export"]
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    allowed = {"vm", "vviol", "rpf", "export"}
    parts = [p for p in parts if p in allowed]
    if not parts:
        return ["vm", "vviol", "rpf", "export"]
    return parts


# -----------------------------
# Step3-C: pos_weight auto
# -----------------------------
@torch.no_grad()
def compute_pos_weights_from_trainset(train_set) -> Dict[str, float]:
    vviol_pos = 0.0
    vviol_tot = 0.0
    rpf_pos = 0.0
    rpf_tot = 0.0
    exp_pos = 0.0
    exp_tot = 0.0

    for data in train_set:
        yv = data.y_node_vviol.view(-1).float()
        vviol_pos += float(yv.sum().item())
        vviol_tot += float(yv.numel())

        yr = data.y_line_rpf.view(-1).float()
        rpf_pos += float(yr.sum().item())
        rpf_tot += float(yr.numel())

        ye = data.y_export.view(-1).float()
        exp_pos += float(ye.sum().item())
        exp_tot += float(ye.numel())

    def _pos_weight(pos: float, tot: float) -> float:
        neg = max(0.0, tot - pos)
        if pos <= 1e-12:
            return 1.0
        return float(neg / pos)

    return {
        "vviol": _pos_weight(vviol_pos, vviol_tot),
        "rpf": _pos_weight(rpf_pos, rpf_tot),
        "export": _pos_weight(exp_pos, exp_tot),
        "stats": {
            "vviol_pos": vviol_pos, "vviol_tot": vviol_tot,
            "rpf_pos": rpf_pos, "rpf_tot": rpf_tot,
            "export_pos": exp_pos, "export_tot": exp_tot,
        },
    }


# -----------------------------
# threshold search
# -----------------------------
def find_best_f1_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thr_min: float,
    thr_max: float,
    thr_grid_n: int,
) -> Tuple[float, float]:
    y_true = y_true.astype(int).reshape(-1)
    y_score = y_score.astype(float).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return 0.5, float("nan")

    thrs = np.linspace(thr_min, thr_max, thr_grid_n)
    best_thr = 0.5
    best_f1 = -1.0
    for t in thrs:
        pred = (y_score >= t).astype(int)
        f1 = float(f1_score(y_true, pred, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(t)
    return best_thr, float(best_f1)


def search_thresholds_on_val(outputs_val: Dict[str, np.ndarray], cfg: TrainConfig) -> Dict[str, float]:
    thr_vviol, bestf_vviol = find_best_f1_threshold(
        outputs_val["vviol_true"], outputs_val["vviol_score"],
        cfg.thr_min, cfg.thr_max, cfg.thr_grid_n
    )
    thr_rpf, bestf_rpf = find_best_f1_threshold(
        outputs_val["rpf_true"], outputs_val["rpf_score"],
        cfg.thr_min, cfg.thr_max, cfg.thr_grid_n
    )

    # export 用 ACC 搜（更直观）
    y_true = outputs_val["export_true"].astype(int).reshape(-1)
    y_score = outputs_val["export_score"].astype(float).reshape(-1)
    if len(np.unique(y_true)) < 2:
        thr_export = 0.5
        best_acc = float("nan")
    else:
        thrs = np.linspace(cfg.thr_min, cfg.thr_max, cfg.thr_grid_n)
        best_acc = -1.0
        thr_export = 0.5
        for t in thrs:
            pred = (y_score >= t).astype(int)
            acc = float(accuracy_score(y_true, pred))
            if acc > best_acc:
                best_acc = acc
                thr_export = float(t)

    return {
        "vviol": float(thr_vviol),
        "rpf": float(thr_rpf),
        "export": float(thr_export),
        "_val_bestf_vviol": float(bestf_vviol),
        "_val_bestf_rpf": float(bestf_rpf),
        "_val_bestacc_export": float(best_acc),
    }


# -----------------------------
# Model (param backbone + optional edge_attr)
# -----------------------------
class MultiTaskModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        edge_attr_dim: int,
        hidden_dim: int,
        backbone: str = "sage",
    ):
        super().__init__()
        self.backbone_name = backbone.lower()
        self.backbone = build_backbone(GNNBackboneConfig(in_dim=in_dim, hidden_dim=hidden_dim, backbone=self.backbone_name))

        hc = HeadConfig(hidden_dim=hidden_dim, edge_attr_dim=edge_attr_dim, mlp_hidden=hidden_dim)
        self.head_vm = NodeVMRegressor(hc)
        self.head_vviol = NodeVViolClassifier(hc)
        self.head_rpf = EdgeRPFClassifier(hc)
        self.head_export = ExportClassifier(hc)

    def forward(self, batch, use_edge_attr: bool = True):
        h, g = self.backbone(batch.x, batch.edge_index, batch=batch.batch)

        pred_vm = self.head_vm(h)               # (N,)
        logit_vviol = self.head_vviol(h)        # (N,)

        edge_attr = batch.edge_attr
        if (not use_edge_attr) and (edge_attr is not None):
            edge_attr = torch.zeros_like(edge_attr)

        logit_rpf = self.head_rpf(h, batch.edge_index, edge_attr)  # (E_dir,)
        logit_export = self.head_export(g)       # (B,)
        return pred_vm, logit_vviol, logit_rpf, logit_export


# -----------------------------
# Collect outputs
# -----------------------------
@torch.no_grad()
def collect_outputs(model: nn.Module, loader: DataLoader, device: str, use_edge_attr: bool):
    model.eval()
    all_vm_true, all_vm_pred = [], []
    all_vviol_true, all_vviol_score = [], []
    all_rpf_true, all_rpf_score = [], []
    all_exp_true, all_exp_score = [], []

    for batch in loader:
        batch = batch.to(device)
        pred_vm, logit_vviol, logit_rpf, logit_export = model(batch, use_edge_attr=use_edge_attr)

        all_vm_true.append(batch.y_node_vm.detach().cpu().numpy())
        all_vm_pred.append(pred_vm.detach().cpu().numpy())

        all_vviol_true.append(batch.y_node_vviol.detach().cpu().numpy())
        all_vviol_score.append(torch.sigmoid(logit_vviol).detach().cpu().numpy())

        y_rpf = batch.y_line_rpf.float()
        y_rpf = _match_edge_labels_to_logits(y_rpf, logit_rpf)
        all_rpf_true.append(y_rpf.detach().cpu().numpy())
        all_rpf_score.append(torch.sigmoid(logit_rpf).detach().cpu().numpy())

        all_exp_true.append(batch.y_export.view(-1).detach().cpu().numpy())
        all_exp_score.append(torch.sigmoid(logit_export).detach().cpu().numpy())

    return {
        "vm_true": np.concatenate(all_vm_true, axis=0),
        "vm_pred": np.concatenate(all_vm_pred, axis=0),
        "vviol_true": np.concatenate(all_vviol_true, axis=0),
        "vviol_score": np.concatenate(all_vviol_score, axis=0),
        "rpf_true": np.concatenate(all_rpf_true, axis=0),
        "rpf_score": np.concatenate(all_rpf_score, axis=0),
        "export_true": np.concatenate(all_exp_true, axis=0),
        "export_score": np.concatenate(all_exp_score, axis=0),
    }


@torch.no_grad()
def evaluate_with_thresholds(
    outputs: Dict[str, np.ndarray],
    thresholds: Dict[str, float],
) -> Dict[str, float]:
    vm_true = outputs["vm_true"]
    vm_pred = outputs["vm_pred"]

    vviol_true = outputs["vviol_true"].astype(int)
    vviol_score = outputs["vviol_score"].astype(float)

    rpf_true = outputs["rpf_true"].astype(int)
    rpf_score = outputs["rpf_score"].astype(float)

    exp_true = outputs["export_true"].astype(int)
    exp_score = outputs["export_score"].astype(float)

    rmse = math_sqrt(mean_squared_error(vm_true, vm_pred))
    mae = float(mean_absolute_error(vm_true, vm_pred))

    vviol_auc = _safe_auc(vviol_true, vviol_score)
    rpf_auc = _safe_auc(rpf_true, rpf_score)
    exp_auc = _safe_auc(exp_true, exp_score)

    tv = float(thresholds.get("vviol", 0.5))
    tr = float(thresholds.get("rpf", 0.5))
    te = float(thresholds.get("export", 0.5))

    vviol_pred = (vviol_score >= tv).astype(int)
    vviol_f1 = float(f1_score(vviol_true, vviol_pred, zero_division=0))

    rpf_pred = (rpf_score >= tr).astype(int)
    rpf_f1 = float(f1_score(rpf_true, rpf_pred, zero_division=0))

    exp_pred = (exp_score >= te).astype(int)
    exp_acc = float(accuracy_score(exp_true, exp_pred))

    return {
        "vm_rmse": float(rmse),
        "vm_mae": float(mae),
        "vviol_auc": float(vviol_auc),
        "vviol_f1": float(vviol_f1),
        "rpf_auc": float(rpf_auc),
        "rpf_f1": float(rpf_f1),
        "export_auc": float(exp_auc),
        "export_acc": float(exp_acc),
        "thr_vviol": tv,
        "thr_rpf": tr,
        "thr_export": te,
    }


# -----------------------------
# Train
# -----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    cfg: TrainConfig,
    posw: Dict[str, torch.Tensor],
    tasks: List[str],
    use_edge_attr: bool,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    mse = nn.MSELoss()

    # pos_weight can be tensor([w]) for broadcast
    bce_vviol = nn.BCEWithLogitsLoss(pos_weight=posw["vviol"])
    bce_rpf = nn.BCEWithLogitsLoss(pos_weight=posw["rpf"])
    bce_export = nn.BCEWithLogitsLoss(pos_weight=posw["export"])

    for batch in loader:
        batch = batch.to(cfg.device)
        pred_vm, logit_vviol, logit_rpf, logit_export = model(batch, use_edge_attr=use_edge_attr)

        loss = 0.0

        if "vm" in tasks:
            y_vm = batch.y_node_vm.float()
            loss_vm = mse(pred_vm, y_vm)
            loss = loss + cfg.w_vm * loss_vm

        if "vviol" in tasks:
            y_vviol = batch.y_node_vviol.float()
            loss_vviol = bce_vviol(logit_vviol.view(-1), y_vviol.view(-1))
            loss = loss + cfg.w_vviol * loss_vviol

        if "rpf" in tasks:
            y_rpf = batch.y_line_rpf.float()
            y_rpf = _match_edge_labels_to_logits(y_rpf, logit_rpf)
            loss_rpf = bce_rpf(logit_rpf.view(-1), y_rpf.view(-1))
            loss = loss + cfg.w_rpf * loss_rpf

        if "export" in tasks:
            y_export = batch.y_export.view(-1).float()
            loss_export = bce_export(logit_export.view(-1), y_export.view(-1))
            loss = loss + cfg.w_export * loss_export

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(1, n_batches)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/offline_case33bw")

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)

    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--save_dir", type=str, default="checkpoints/gnn_pretrain")

    # Step3-D switches
    ap.add_argument("--backbone", type=str, default="sage", choices=["sage", "gcn", "gat", "mlp"])
    ap.add_argument("--use_edge_attr", type=int, default=1)  # 1=use,0=zero-out
    ap.add_argument("--tasks", type=str, default="vm,vviol,rpf,export")

    ap.add_argument("--use_pos_weight", type=int, default=1)  # 1=auto pos_weight, 0=pos_weight=1
    ap.add_argument("--thr_mode", type=str, default="tuned", choices=["tuned", "fixed"])
    ap.add_argument("--thr_fixed", type=float, default=0.5)   # used when thr_mode=fixed

    ap.add_argument("--thr_grid_n", type=int, default=41)
    ap.add_argument("--thr_min", type=float, default=0.01)
    ap.add_argument("--thr_max", type=float, default=0.99)

    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # device fallback
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("[warn] mps not available, fallback to cpu")
        device = "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] cuda not available, fallback to cpu")
        device = "cpu"

    # tasks
    tasks = parse_tasks(args.tasks)

    # dataset + splits
    ds = PowerGraphDataset(root=args.data_root)
    splits = PowerGraphDataset.load_splits(args.data_root)

    train_set = ds.index_select(splits["train"])
    val_set = ds.index_select(splits["val"])
    test_set = ds.index_select(splits["test"])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    sample0 = ds[0]
    in_dim = int(sample0.x.size(-1))
    edge_attr_dim = int(sample0.edge_attr.size(-1))

    model = MultiTaskModel(
        in_dim=in_dim,
        edge_attr_dim=edge_attr_dim,
        hidden_dim=args.hidden,
        backbone=args.backbone,
    ).to(device)

    tcfg = TrainConfig(
        lr=args.lr,
        weight_decay=args.wd,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        thr_grid_n=args.thr_grid_n,
        thr_min=args.thr_min,
        thr_max=args.thr_max,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    # pos_weight
    if args.use_pos_weight == 1:
        pw_info = compute_pos_weights_from_trainset(train_set)
        stats = pw_info["stats"]
        print(f"[pos_weight] (neg/pos)")
        print(f"  vviol : {pw_info['vviol']:.4f}  (pos={stats['vviol_pos']:.1f}/{stats['vviol_tot']:.1f})")
        print(f"  rpf   : {pw_info['rpf']:.4f}  (pos={stats['rpf_pos']:.1f}/{stats['rpf_tot']:.1f})")
        print(f"  export: {pw_info['export']:.4f}  (pos={stats['export_pos']:.1f}/{stats['export_tot']:.1f})")
        posw = {
            "vviol": torch.tensor([pw_info["vviol"]], dtype=torch.float32, device=device),
            "rpf": torch.tensor([pw_info["rpf"]], dtype=torch.float32, device=device),
            "export": torch.tensor([pw_info["export"]], dtype=torch.float32, device=device),
        }
    else:
        print("[pos_weight] disabled -> pos_weight=1 for all tasks")
        posw = {
            "vviol": torch.tensor([1.0], dtype=torch.float32, device=device),
            "rpf": torch.tensor([1.0], dtype=torch.float32, device=device),
            "export": torch.tensor([1.0], dtype=torch.float32, device=device),
        }

    use_edge_attr = bool(args.use_edge_attr == 1)

    print(f"[train] device={device} | backbone={args.backbone} | use_edge_attr={use_edge_attr} | tasks={tasks}")
    print(f"[train] train/val/test = {len(train_set)}/{len(val_set)}/{len(test_set)}")
    print(f"[train] in_dim={in_dim} edge_attr_dim={edge_attr_dim} hidden={args.hidden}")
    print(f"[thr] mode={args.thr_mode} fixed={args.thr_fixed}")

    best_val = float("inf")
    best_path = os.path.join(args.save_dir, "best.pt")
    best_thr_path = os.path.join(args.save_dir, "best_thresholds.json")
    best_metrics_path = os.path.join(args.save_dir, "run_metrics.json")

    best_thresholds = {"vviol": float(args.thr_fixed), "rpf": float(args.thr_fixed), "export": float(args.thr_fixed)}

    for ep in range(1, tcfg.epochs + 1):
        loss = train_one_epoch(model, train_loader, opt, tcfg, posw, tasks=tasks, use_edge_attr=use_edge_attr)

        out_val = collect_outputs(model, val_loader, device, use_edge_attr=use_edge_attr)

        if args.thr_mode == "tuned":
            thr_val = search_thresholds_on_val(out_val, tcfg)
            thresholds = {"vviol": thr_val["vviol"], "rpf": thr_val["rpf"], "export": thr_val["export"]}
        else:
            thr_val = {"_val_bestf_vviol": float("nan"), "_val_bestf_rpf": float("nan"), "_val_bestacc_export": float("nan")}
            thresholds = {"vviol": float(args.thr_fixed), "rpf": float(args.thr_fixed), "export": float(args.thr_fixed)}

        val_metrics = evaluate_with_thresholds(out_val, thresholds=thresholds)

        # val_key: only use metrics that exist in tasks
        # regression part:
        key = 0.0
        if "vm" in tasks:
            key += val_metrics["vm_rmse"]
        # classification part:
        if "vviol" in tasks:
            key += (1.0 - (0.0 if np.isnan(val_metrics["vviol_auc"]) else val_metrics["vviol_auc"]))
        if "rpf" in tasks:
            key += (1.0 - (0.0 if np.isnan(val_metrics["rpf_auc"]) else val_metrics["rpf_auc"]))
        if "export" in tasks:
            key += (1.0 - (0.0 if np.isnan(val_metrics["export_auc"]) else val_metrics["export_auc"]))

        print(
            f"[epoch {ep:03d}] loss={loss:.4f} | "
            f"val vm_rmse={val_metrics['vm_rmse']:.4f} vm_mae={val_metrics['vm_mae']:.4f} | "
            f"vviol_auc={val_metrics['vviol_auc']:.4f} vviol_f1={val_metrics['vviol_f1']:.4f} (thr={val_metrics['thr_vviol']:.2f}) | "
            f"rpf_auc={val_metrics['rpf_auc']:.4f} rpf_f1={val_metrics['rpf_f1']:.4f} (thr={val_metrics['thr_rpf']:.2f}) | "
            f"export_auc={val_metrics['export_auc']:.4f} export_acc={val_metrics['export_acc']:.4f} (thr={val_metrics['thr_export']:.2f})"
        )

        if key < best_val:
            best_val = key
            best_thresholds = thresholds.copy()
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": vars(tcfg),
                    "args": vars(args),
                    "best_thresholds": best_thresholds,
                    "tasks": tasks,
                },
                best_path,
            )
            with open(best_thr_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "thresholds": best_thresholds,
                        "thr_mode": args.thr_mode,
                        "thr_fixed": args.thr_fixed,
                        "val_meta": thr_val,
                    },
                    f,
                    indent=2,
                )

    # test best
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    thresholds = ckpt.get("best_thresholds", best_thresholds)
    tasks = ckpt.get("tasks", tasks)

    out_test = collect_outputs(model, test_loader, device, use_edge_attr=use_edge_attr)
    test_metrics = evaluate_with_thresholds(out_test, thresholds=thresholds)

    print("\n=== Paper-ready Test Table ===")
    print(f"(tasks={tasks} backbone={args.backbone} edge_attr={use_edge_attr} posw={args.use_pos_weight} thr={args.thr_mode})")
    print(f"Node VM   : RMSE={test_metrics['vm_rmse']:.4f}, MAE={test_metrics['vm_mae']:.4f}")
    print(f"Node VVIOL: AUC ={test_metrics['vviol_auc']:.4f}, F1 ={test_metrics['vviol_f1']:.4f}  (thr={test_metrics['thr_vviol']:.2f})")
    print(f"Edge RPF  : AUC ={test_metrics['rpf_auc']:.4f}, F1 ={test_metrics['rpf_f1']:.4f}  (thr={test_metrics['thr_rpf']:.2f})")
    print(f"Export    : AUC ={test_metrics['export_auc']:.4f}, ACC={test_metrics['export_acc']:.4f} (thr={test_metrics['thr_export']:.2f})")
    print("saved best to:", best_path)
    print("saved thresholds to:", best_thr_path)

    # save run metrics for ablation aggregation
    run_metrics = {
        "data_root": args.data_root,
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "wd": args.wd,
        "hidden": args.hidden,
        "backbone": args.backbone,
        "use_edge_attr": int(use_edge_attr),
        "tasks": tasks,
        "use_pos_weight": int(args.use_pos_weight),
        "thr_mode": args.thr_mode,
        "thr_fixed": float(args.thr_fixed),
        "thresholds": thresholds,
        "test": test_metrics,
    }
    with open(best_metrics_path, "w", encoding="utf-8") as f:
        json.dump(run_metrics, f, indent=2)


if __name__ == "__main__":
    main()