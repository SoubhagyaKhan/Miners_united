"""
models/gat.py  --  GAT model for Dataset A using PyG's GATConv

GATModel(x, edge_index) -> logits [N, num_classes]
Uses PyG's optimized GATConv with multi-head attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from config import GATConfig


class GATModel(nn.Module):
    """
    Multi-layer GAT using PyG's GATConv.

    Layer layout (hidden_dim=256, num_heads=8, out_per_head=32):
        - Intermediate: GATConv(in, 32, heads=8, concat=True)  -> [N, 256]
        - Last:         GATConv(in, 256, heads=8, concat=False) -> [N, 256]
    BN is always BatchNorm1d(256) = BatchNorm1d(hidden_dim).

    predict.py interface: model(x, edge_index) -> logits [N, num_classes]
    """

    def __init__(self, in_dim: int, num_classes: int, cfg: GATConfig):
        super().__init__()
        self.cfg = cfg

        heads        = cfg.num_heads
        out_per_head = cfg.hidden_dim // heads   # 256 // 8 = 32

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(cfg.num_layers):
            is_last = (i == cfg.num_layers - 1)
            in_ch   = in_dim if i == 0 else cfg.hidden_dim

            if is_last:
                # average heads → output is hidden_dim
                self.convs.append(GATConv(
                    in_ch, cfg.hidden_dim,
                    heads=heads,
                    concat=False,
                    dropout=cfg.attn_dropout,
                    add_self_loops=True,
                ))
            else:
                # concat heads → output is out_per_head * heads == hidden_dim
                self.convs.append(GATConv(
                    in_ch, out_per_head,
                    heads=heads,
                    concat=True,
                    dropout=cfg.attn_dropout,
                    add_self_loops=True,
                ))
            # BN dim is always hidden_dim (both branches produce hidden_dim)
            self.norms.append(nn.BatchNorm1d(cfg.hidden_dim))

        self.dropout    = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_in = x
            x    = conv(x, edge_index)   # always [N, hidden_dim]
            x    = norm(x)
            x    = F.elu(x)
            # residual on non-first layers (dims always match after layer 0)
            if i > 0:
                x = x + x_in
            x = self.dropout(x)

        return self.classifier(x)        # [N, num_classes]