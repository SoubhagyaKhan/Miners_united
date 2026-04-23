"""
models/sage.py  --  GraphSAGE model for Dataset B

GraphSAGEModel(x, edge_index) -> logits [N, 2]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from config import GraphSAGEConfig


class GraphSAGEModel(nn.Module):
    """
    Multi-layer GraphSAGE using SAGEConv + BatchNorm + residual connections.

    predict.py interface: model(x, edge_index) -> logits [N, 2]
    """

    def __init__(self, in_dim: int, num_classes: int, cfg: GraphSAGEConfig):
        super().__init__()
        self.cfg = cfg

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        for i in range(cfg.num_layers):
            in_ch = in_dim if i == 0 else cfg.hidden_dim
            self.convs.append(SAGEConv(in_ch, cfg.hidden_dim))
            self.bns.append(nn.BatchNorm1d(cfg.hidden_dim))

        # input projection for residual on first layer
        self.input_proj = nn.Linear(in_dim, cfg.hidden_dim, bias=False)

        self.dropout    = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_in = x
            x    = conv(x, edge_index)
            x    = bn(x)
            x    = F.relu(x)
            # residual
            if i == 0:
                x = x + self.input_proj(x_in)
            else:
                x = x + x_in
            x = self.dropout(x)

        return self.classifier(x)               # [N, num_classes]