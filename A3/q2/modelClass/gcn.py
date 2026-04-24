"""
models/gcn.py  --  GCN model for Dataset A using PyG's GCNConv

GCNModel(x, edge_index) -> logits [N, num_classes]

Matches GATModel structure:
    - LayerNorm instead of BatchNorm
    - ELU instead of ReLU
    - Residual on layers 1+ (layer 0 skipped, dims differ)
    - input_proj for layer-0 residual removed (matches GAT which also skips it)
    - No cached=True (safe for train/eval switching)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from config import GCNConfig


class GCNModel(nn.Module):
    """
    predict.py interface: model(x, edge_index) -> logits [N, num_classes]
    """

    def __init__(self, in_dim: int, num_classes: int, cfg: GCNConfig):
        super().__init__()
        self.cfg = cfg

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(cfg.num_layers):
            in_ch = in_dim if i == 0 else cfg.hidden_dim
            self.convs.append(GCNConv(in_ch, cfg.hidden_dim))
            self.norms.append(nn.LayerNorm(cfg.hidden_dim))

        self.dropout    = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_in = x
            x    = conv(x, edge_index)
            x    = norm(x)
            x    = F.elu(x)
            if i > 0:
                x = x + x_in          # residual: skip layer 0 (in_dim != hidden_dim)
            x = self.dropout(x)

        return self.classifier(x)