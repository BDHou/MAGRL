# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Heads：把 backbone 的 embedding 变成具体任务输出

我们做 4 个监督任务（论文第一条结果很稳）：
  1) node vm 回归：y_node_vm
  2) node vviol 分类：y_node_vviol
  3) edge rpf 分类：y_line_rpf
  4) graph export 分类：y_export

edge rpf 用法：
  用 node embedding 取 (src, dst) 然后 concat，再 concat edge_attr
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HeadConfig:
    hidden_dim: int = 128
    edge_attr_dim: int = 4
    mlp_hidden: int = 128
    dropout: float = 0.1


class NodeVMRegressor(nn.Module):
    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # (N, H) -> (N,)
        return self.net(h).squeeze(-1)


class NodeVViolClassifier(nn.Module):
    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # logits: (N,)
        return self.net(h).squeeze(-1)


class EdgeRPFClassifier(nn.Module):
    def __init__(self, cfg: HeadConfig):
        super().__init__()
        in_dim = cfg.hidden_dim * 2 + cfg.edge_attr_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden, 1),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        h: (N, H)
        edge_index: (2, E)
        edge_attr: (E, Fe)
        """
        src = edge_index[0]
        dst = edge_index[1]
        hs = h[src]  # (E, H)
        hd = h[dst]  # (E, H)
        z = torch.cat([hs, hd, edge_attr], dim=-1)  # (E, 2H+Fe)
        return self.net(z).squeeze(-1)  # (E,)


class ExportClassifier(nn.Module):
    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden, 1),
        )

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        # logits: (B,)
        return self.net(g).squeeze(-1)