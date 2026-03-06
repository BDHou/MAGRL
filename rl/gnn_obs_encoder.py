# -*- coding: utf-8 -*-
"""
rl/gnn_obs_encoder.py

把预训练 GNN backbone 包装成 RLlib 可用的自定义 Model。

两种使用场景：
  1. 独立推理：GNNObsEncoder.encode(net, cfg, ...) -> flat obs vector (用于调试/eval)
  2. RLlib TorchModelV2：作为 PPO policy 的 feature extractor
     输入为来自 GridPVEnv 的 flat obs (n_bus*8 + n_pv*3)
     先 reshape 回节点特征矩阵 (n_bus, 8)，再过 GNN backbone，最后拼 PV 状态输出。

GNN 权重可以：
  - frozen=True: 冻结，只训练 RL head（快，稳定，推荐先跑）
  - frozen=False: 端到端微调（慢，可能更优，后期尝试）
"""
from __future__ import annotations

import os
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn

from models.gnn_backbone import GNNBackboneConfig, build_backbone


class GNNObsEncoder(nn.Module):
    """
    将 flat obs vector 从 GridPVEnv 解码后过 GNN backbone，
    输出用于 RL policy 的特征向量。

    参数
    ----
    n_bus : int
        节点数量
    node_feat_dim : int
        每个节点的特征维度（8）
    n_pv : int
        PV 数量（决定 PV 局部状态的维度 n_pv*3）
    hidden_dim : int
        GNN hidden dim（与预训练时保持一致）
    ckpt_path : str | None
        预训练 checkpoint 路径（best.pt）。None 则随机初始化（ablation 用）
    frozen : bool
        True: 冻结 GNN 权重（只训练 RL head）
    backbone : str
        backbone 类型，需与预训练时一致（默认 'sage'）
    """

    def __init__(
        self,
        n_bus: int,
        node_feat_dim: int = 8,
        n_pv: int = 3,
        hidden_dim: int = 128,
        ckpt_path: Optional[str] = None,
        frozen: bool = True,
        backbone: str = "sage",
    ):
        super().__init__()

        self.n_bus = n_bus
        self.node_feat_dim = node_feat_dim
        self.n_pv = n_pv
        self.hidden_dim = hidden_dim
        self.frozen = frozen

        # 构建 GNN backbone
        gnn_cfg = GNNBackboneConfig(
            in_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            backbone=backbone,
        )
        self.backbone = build_backbone(gnn_cfg)

        # 加载预训练权重（只取 backbone 部分）
        if ckpt_path is not None and os.path.isfile(ckpt_path):
            self._load_backbone_weights(ckpt_path)
            print(f"[GNNObsEncoder] Loaded backbone weights from: {ckpt_path}")
        else:
            if ckpt_path is not None:
                print(f"[GNNObsEncoder] Warning: ckpt not found at {ckpt_path}, using random init.")
            else:
                print("[GNNObsEncoder] No ckpt_path provided, using random init (ablation mode).")

        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            print("[GNNObsEncoder] Backbone frozen.")

        # 输出维度：graph embedding (hidden_dim) + PV 局部状态 (n_pv*3)
        self._out_dim = hidden_dim + n_pv * 3

    def _load_backbone_weights(self, ckpt_path: str):
        """从 supervised_pretrain 保存的 best.pt 里只取 backbone 部分。"""
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model", ckpt)  # best.pt 里 key 是 "model"

        # 只取 backbone.* 开头的 key
        backbone_state = {
            k.removeprefix("backbone."): v
            for k, v in state.items()
            if k.startswith("backbone.")
        }

        if not backbone_state:
            print("[GNNObsEncoder] Warning: no backbone.* keys found in ckpt, using random init.")
            return

        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        if missing:
            print(f"[GNNObsEncoder] Missing keys: {missing}")
        if unexpected:
            print(f"[GNNObsEncoder] Unexpected keys: {unexpected}")

    @property
    def out_dim(self) -> int:
        """输出特征维度，RL policy 的输入维度。"""
        return self._out_dim

    def forward(
        self,
        obs_flat: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数
        ----
        obs_flat : (B, n_bus*node_feat_dim + n_pv*3)
            来自 GridPVEnv 的 flat obs，Batch 可以是 1（单步推理）或 B（训练 batch）
        edge_index : (2, E)
            图的连接关系（固定不变，可提前缓存）

        返回
        ----
        feat : (B, hidden_dim + n_pv*3)
        """
        B = obs_flat.shape[0]

        # 切分节点部分和 PV 局部状态部分
        node_flat = obs_flat[:, : self.n_bus * self.node_feat_dim]  # (B, n_bus*8)
        pv_state  = obs_flat[:, self.n_bus * self.node_feat_dim :]  # (B, n_pv*3)

        # 将节点特征 reshape 回 (B*n_bus, node_feat_dim)，拼成一个大图 batch
        x = node_flat.reshape(B * self.n_bus, self.node_feat_dim)

        # 构造 batch 向量：第 i 个样本的所有节点 batch id = i
        batch_vec = torch.arange(B, device=obs_flat.device).repeat_interleave(self.n_bus)

        # edge_index 需要偏移到 batch 索引
        # edge_index shape: (2, E) → 复制 B 份并加偏移
        offsets = torch.arange(B, device=obs_flat.device) * self.n_bus  # (B,)
        edge_index_batch = torch.cat(
            [edge_index + off for off in offsets], dim=1
        )  # (2, B*E)

        # GNN forward
        _h, g = self.backbone(x, edge_index_batch, batch=batch_vec)  # g: (B, hidden_dim)

        # 拼接 graph embedding 和 PV 局部状态
        feat = torch.cat([g, pv_state], dim=-1)  # (B, hidden_dim + n_pv*3)
        return feat


def build_edge_index_for_env(env) -> torch.Tensor:
    """
    从 GridPVEnv 的 backend 里提取 edge_index，转成 tensor。
    在训练开始前调用一次，缓存到外部。
    """
    import numpy as np
    from scenario.dataset_graph import build_graph_tensors
    from scenario.base_scenario import ScenarioConfig

    # 用一个临时 obs 取 edge_index（不依赖 PF 结果）
    net = env._backend.net
    cfg = env.cfg
    dummy = build_graph_tensors(
        net=net,
        cfg=cfg,
        load_forecast_mult=1.0,
        pv_forecast_mult=0.5,
        last_vm_obs=None,
        rng=np.random.default_rng(0),
    )
    ei = torch.tensor(dummy["edge_index"], dtype=torch.long)
    return ei
