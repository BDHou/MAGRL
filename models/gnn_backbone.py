# # -*- coding: utf-8 -*-
# from __future__ import annotations

# """
# GNN backbone：把图编码成 node embedding / graph embedding
# 先用最稳的 GraphSAGE（对你这种电网图很常用，训练稳定）

# 输入：
#   x: (N, Fin)
#   edge_index: (2, E)
#   edge_attr: (E, Fe)   （backbone 先不强依赖 edge_attr，head 会用）

# 输出：
#   h: (N, H)  node embedding
#   g: (B, H)  graph embedding (global mean pool)
# """

# from dataclasses import dataclass
# from typing import Optional, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from torch_geometric.nn import SAGEConv, global_mean_pool


# @dataclass
# class GNNBackboneConfig:
#     in_dim: int = 8
#     hidden_dim: int = 128
#     num_layers: int = 3
#     dropout: float = 0.1


# class GraphSAGEBackbone(nn.Module):
#     def __init__(self, cfg: GNNBackboneConfig):
#         super().__init__()
#         self.cfg = cfg

#         self.convs = nn.ModuleList()
#         self.convs.append(SAGEConv(cfg.in_dim, cfg.hidden_dim))
#         for _ in range(cfg.num_layers - 1):
#             self.convs.append(SAGEConv(cfg.hidden_dim, cfg.hidden_dim))

#         self.dropout = float(cfg.dropout)

#     def forward(self, x, edge_index, batch=None) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         batch: (N,)  每个 node 属于哪个 graph（DataLoader 会给）
#         """
#         h = x
#         for i, conv in enumerate(self.convs):
#             h = conv(h, edge_index)
#             h = F.relu(h)
#             h = F.dropout(h, p=self.dropout, training=self.training)

#         if batch is None:
#             # 单图：假设都在 batch=0
#             batch = x.new_zeros(x.size(0), dtype=torch.long)

#         g = global_mean_pool(h, batch)  # (B, H)
#         return h, g

# models/gnn_backbone.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    SAGEConv,
    GCNConv,
    GATv2Conv,
    global_mean_pool,
)


@dataclass
class GNNBackboneConfig:
    in_dim: int = 8
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    backbone: str = "sage"  # sage|gcn|gat|mlp
    gat_heads: int = 4      # for gat only


class _BaseBackbone(nn.Module):
    def __init__(self, cfg: GNNBackboneConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, x, edge_index, batch=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return:
          node_emb: (N, H)
          graph_emb: (B, H)  via global_mean_pool
        """
        raise NotImplementedError


class MLPBackbone(_BaseBackbone):
    """Ablation: no graph structure"""
    def __init__(self, cfg: GNNBackboneConfig):
        super().__init__(cfg)
        layers = []
        in_d = cfg.in_dim
        for i in range(cfg.num_layers):
            layers.append(nn.Linear(in_d, cfg.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.dropout))
            in_d = cfg.hidden_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, edge_index=None, batch=None):
        h = self.mlp(x)
        if batch is None:
            # assume single graph
            g = h.mean(dim=0, keepdim=True)
        else:
            g = global_mean_pool(h, batch)
        return h, g


class GCNBackbone(_BaseBackbone):
    def __init__(self, cfg: GNNBackboneConfig):
        super().__init__(cfg)
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(cfg.in_dim, cfg.hidden_dim))
        for _ in range(cfg.num_layers - 1):
            self.convs.append(GCNConv(cfg.hidden_dim, cfg.hidden_dim))
        self.dropout = cfg.dropout

    def forward(self, x, edge_index, batch=None):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        g = global_mean_pool(h, batch) if batch is not None else h.mean(dim=0, keepdim=True)
        return h, g


class GATBackbone(_BaseBackbone):
    def __init__(self, cfg: GNNBackboneConfig):
        super().__init__(cfg)
        heads = cfg.gat_heads
        self.convs = nn.ModuleList()

        # first layer: in -> hidden (via heads)
        self.convs.append(GATv2Conv(cfg.in_dim, cfg.hidden_dim // heads, heads=heads, concat=True))
        for _ in range(cfg.num_layers - 1):
            self.convs.append(GATv2Conv(cfg.hidden_dim, cfg.hidden_dim // heads, heads=heads, concat=True))

        self.dropout = cfg.dropout

    def forward(self, x, edge_index, batch=None):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        g = global_mean_pool(h, batch) if batch is not None else h.mean(dim=0, keepdim=True)
        return h, g


class GraphSAGEBackbone(_BaseBackbone):
    def __init__(self, cfg: GNNBackboneConfig):
        super().__init__(cfg)
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(cfg.in_dim, cfg.hidden_dim))
        for _ in range(cfg.num_layers - 1):
            self.convs.append(SAGEConv(cfg.hidden_dim, cfg.hidden_dim))
        self.dropout = cfg.dropout

    def forward(self, x, edge_index, batch=None):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        g = global_mean_pool(h, batch) if batch is not None else h.mean(dim=0, keepdim=True)
        return h, g


def build_backbone(cfg: GNNBackboneConfig) -> nn.Module:
    name = (cfg.backbone or "sage").lower()
    if name == "mlp":
        return MLPBackbone(cfg)
    if name == "gcn":
        return GCNBackbone(cfg)
    if name == "gat":
        return GATBackbone(cfg)
    if name == "sage":
        return GraphSAGEBackbone(cfg)
    raise ValueError(f"Unknown backbone: {cfg.backbone} (expected: sage|gcn|gat|mlp)")