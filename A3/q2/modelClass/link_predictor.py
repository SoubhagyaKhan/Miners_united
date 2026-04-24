"""
models/link_predictor.py  --  Link predictor for Dataset C

Key insight: x is already precomputed GNN embeddings ("gnn_feature").
Running another GCN on top adds noise and causes overfit on 3199 nodes.

Two modes controlled by LinkPredConfig.use_encoder:
  False (default): MLP-only decoder on raw features  <- use this for C
  True:            GCN encoder + MLP decoder          <- legacy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from config import LinkPredConfig


class LinkPredictor(nn.Module):
    def __init__(self, in_dim: int, hidden_or_cfg=None, dropout: float = None):
        super().__init__()

        if isinstance(hidden_or_cfg, LinkPredConfig):
            cfg = hidden_or_cfg
        elif hidden_or_cfg is None:
            cfg = LinkPredConfig()
        else:
            cfg = LinkPredConfig(
                hidden_dim=int(hidden_or_cfg),
                dropout=dropout if dropout is not None else 0.3,
            )

        self.cfg        = cfg
        self.use_encoder = cfg.use_encoder

        if self.use_encoder:
            # GCN encoder path
            self.encoder_convs = nn.ModuleList()
            self.encoder_norms = nn.ModuleList()
            for i in range(cfg.num_layers):
                in_ch = in_dim if i == 0 else cfg.hidden_dim
                self.encoder_convs.append(GCNConv(in_ch, cfg.hidden_dim))
                self.encoder_norms.append(nn.BatchNorm1d(cfg.hidden_dim))
            self.input_proj      = nn.Linear(in_dim, cfg.hidden_dim, bias=False)
            self.encoder_dropout = nn.Dropout(cfg.dropout)
            feat_dim = cfg.hidden_dim
        else:
            # No encoder: use raw features directly
            # Project to hidden_dim first to reduce 3703-dim input
            self.feat_proj = nn.Sequential(
                nn.Linear(in_dim, cfg.hidden_dim),
                nn.BatchNorm1d(cfg.hidden_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            )
            feat_dim = cfg.hidden_dim

        # Decoder: [u, v, u*v, |u-v|] -> score
        # Input dim: feat_dim * 4
        dec_h = cfg.decoder_hidden
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim * 4, dec_h),
            nn.BatchNorm1d(dec_h),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(dec_h, dec_h // 2),
            nn.BatchNorm1d(dec_h // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(dec_h // 2, 1),
        )

    def encode(self, x, edge_index):
        if self.use_encoder:
            for i, (conv, norm) in enumerate(zip(self.encoder_convs, self.encoder_norms)):
                x_in = x
                x    = conv(x, edge_index)
                x    = norm(x)
                x    = F.relu(x)
                x    = x + (self.input_proj(x_in) if i == 0 else x_in)
                x    = self.encoder_dropout(x)
            return x
        else:
            return self.feat_proj(x)

    def decode(self, z, edge_pairs):
        src = edge_pairs[:, 0]
        dst = edge_pairs[:, 1]
        u   = z[src]
        v   = z[dst]
        h   = torch.cat([u, v, u * v, (u - v).abs()], dim=1)   # [u, v, hadamard, L1]
        return self.decoder(h).squeeze(1)

    def forward(self, x, edge_index, edge_pairs):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_pairs)