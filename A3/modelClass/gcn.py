"""
models/gcn.py  --  GCN model for Dataset A using PyG's GCNConv

GCNModel(x, edge_index) -> logits [N, num_classes]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from config import GCNConfig


class GCNModel(nn.Module):
    """
    Multi-layer GCN using PyG's GCNConv with residual connections and BN.

    predict.py interface: model(x, edge_index) -> logits [N, num_classes]
    """

    def __init__(self, in_dim: int, num_classes: int, cfg: GCNConfig):
        super().__init__()
        self.cfg = cfg

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(cfg.num_layers):
            in_ch = in_dim if i == 0 else cfg.hidden_dim
            self.convs.append(GCNConv(in_ch, cfg.hidden_dim, cached=True))
            self.norms.append(nn.BatchNorm1d(cfg.hidden_dim))

        self.dropout    = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.hidden_dim, num_classes)

        # projection for first-layer residual (in_dim != hidden_dim)
        self.input_proj = nn.Linear(in_dim, cfg.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_in = x
            x    = conv(x, edge_index)
            x    = norm(x)
            x    = F.relu(x)
            # residual: project input on first layer
            if i == 0:
                x = x + self.input_proj(x_in)
            else:
                x = x + x_in
            x = self.dropout(x)

        return self.classifier(x)               # [N, num_classes]