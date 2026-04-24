"""
models/link_predictor.py  --  GraphSAGE encoder + flexible MLP decoder

Architecture
------------
    Input projection : Linear(in_dim, proj_dim)   -- 3703 → 256
    Encoder          : 2-layer SAGEConv            -- 256  → 256
    Decoder          : Hadamard or Concat + MLP    -- scores [E]

predict.py interface
--------------------
    model(x, edge_index, edge_pairs) -> FloatTensor [E]
    where edge_pairs : [E, 2], each row is (u, v) node indices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from config import LinkPredConfig


# ─────────────────────────────────────────────────────────────────────────────
# MLP helper
# ─────────────────────────────────────────────────────────────────────────────

def build_mlp(in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> nn.Module:
    """
    Builds:  Linear → [LayerNorm → ReLU → Dropout → Linear] × (num_layers-1) → Linear(1)
    """
    assert num_layers >= 1
    layers = []
    cur_dim = in_dim
    for i in range(num_layers - 1):
        layers += [
            nn.Linear(cur_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        cur_dim = hidden_dim
    layers.append(nn.Linear(cur_dim, 1))   # final scalar output
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class LinkPredictor(nn.Module):

    def __init__(self, in_dim: int, cfg: LinkPredConfig):
        super().__init__()
        self.cfg = cfg

        # ── Input projection ─────────────────────────────────────────────────
        # Reduces 3703-dim pre-computed embeddings to proj_dim before GNN.
        # Acts as a learned dimensionality reduction + regulariser.
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, cfg.proj_dim),
            nn.LayerNorm(cfg.proj_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

        # ── Encoder: 2-layer GraphSAGE ───────────────────────────────────────
        # SAGEConv concatenates self + aggregated neighbour → good for sparse graphs.
        # Layer dims: proj_dim → hidden_dim → hidden_dim
        dims = [cfg.proj_dim] + [cfg.hidden_dim] * cfg.num_layers

        self.convs = nn.ModuleList([
            SAGEConv(dims[i], dims[i + 1], aggr=cfg.sage_aggr)
            for i in range(cfg.num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(cfg.hidden_dim)
            for _ in range(cfg.num_layers)
        ])

        self.encoder_dropout = nn.Dropout(cfg.dropout)

        # ── Decoder ──────────────────────────────────────────────────────────
        # Hadamard: pair_dim = hidden_dim   (element-wise product, symmetric)
        # Concat  : pair_dim = 2*hidden_dim (concatenation, asymmetric)
        assert cfg.decoder_type in ("hadamard", "concat"), \
            f"decoder_type must be 'hadamard' or 'concat', got '{cfg.decoder_type}'"

        pair_dim = cfg.hidden_dim if cfg.decoder_type == "hadamard" else cfg.hidden_dim * 2

        self.decoder = build_mlp(
            in_dim=pair_dim,
            hidden_dim=cfg.decoder_hidden,
            num_layers=cfg.decoder_layers,
            dropout=cfg.dropout,
        )

    # ── Forward passes ───────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x          : [N, in_dim]   (raw node features, N = x.shape[0] = 3327)
        edge_index : [2, E]
        returns      [N, hidden_dim]
        """
        h = self.input_proj(x)                         # [N, proj_dim]

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = h
            h = conv(h, edge_index)                    # [N, hidden_dim]
            h = norm(h)
            h = F.relu(h)
            if i > 0:                                  # skip layer-0 residual
                h = h + residual                       # (dims match from layer 1+)
            h = self.encoder_dropout(h)

        return h                                       # [N, hidden_dim]

    def decode(self, z: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        """
        z          : [N, hidden_dim]
        edge_pairs : [E, 2]
        returns      [E]   raw logits (no sigmoid — BCEWithLogitsLoss handles it)
        """
        u = edge_pairs[:, 0]                           # [E]
        v = edge_pairs[:, 1]                           # [E]

        z_u = z[u]                                     # [E, hidden_dim]
        z_v = z[v]                                     # [E, hidden_dim]

        if self.cfg.decoder_type == "hadamard":
            pair = z_u * z_v                           # [E, hidden_dim]
        else:
            pair = torch.cat([z_u, z_v], dim=1)        # [E, 2*hidden_dim]

        return self.decoder(pair).squeeze(1)           # [E]

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_pairs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward: encode all nodes, then decode requested pairs.
        Called by predict.py as: model(dataset.x, dataset.edge_index, edge_pairs)
        """
        z = self.encode(x, edge_index)
        return self.decode(z, edge_pairs)              # [E]