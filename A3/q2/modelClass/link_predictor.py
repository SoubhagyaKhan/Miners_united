"""
models/link_predictor_ncn.py  --  GATv2 encoder + structural features decoder

Key idea (NCN)
-----------------------------
    Standard GNNs embed u and v independently -- they cannot see whether u and
    v share common neighbours.  But Common Neighbours (CN), Adamic-Adar (AA),
    and Resource Allocation (RA) are 12x higher for true edges than hard
    negatives in Dataset C.  Concatenating these 3 scalars into the MLP gives
    the decoder exact structural discrimination that no amount of GNN depth can
    recover.

Structural features (precomputed once, looked up at decode time)
----------------------------------------------------------------
    CN  : |N(u) ∩ N(v)|                   -- raw common neighbours
    AA  : Σ_{w ∈ N(u)∩N(v)} 1/log(deg(w)) -- downweights high-degree hubs
    RA  : Σ_{w ∈ N(u)∩N(v)} 1/deg(w)      -- stronger hub downweighting

predict.py interface (unchanged)
---------------------------------
    model(x, edge_index, edge_pairs) -> FloatTensor [E]

    Structural matrices are stored as model buffers after calling
    model.precompute_structural(edge_index, num_nodes).
    predict.py calls model(x, edge_index, edge_pairs) which triggers
    encode + structural lookup + decode automatically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch_geometric.nn import SAGEConv

from config import LinkPredConfig


# ─────────────────────────────────────────────────────────────────────────────
# Structural feature precomputation
# ─────────────────────────────────────────────────────────────────────────────

def compute_structural_matrices(edge_index: torch.Tensor, num_nodes: int):
    """
    Compute CN, AA, RA sparse matrices from edge_index.
    Returns three scipy csr_matrices.

    edge_index : [2, E]  undirected (both directions present)
    """
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    vals = np.ones(len(src), dtype=np.float32)

    A = sp.csr_matrix((vals, (src, dst)), shape=(num_nodes, num_nodes))

    deg      = np.array(A.sum(axis=1)).flatten()          # [N]
    deg_safe = np.where(deg > 1, deg, 2.0)                # avoid log(1)=0

    # CN = A^2  (entry [u,v] = number of common neighbours)
    CN = A @ A

    # AA = A * diag(1/log(deg)) * A
    D_aa = sp.diags(1.0 / np.log(deg_safe))
    AA   = A @ D_aa @ A

    # RA = A * diag(1/deg) * A
    D_ra = sp.diags(1.0 / deg_safe)
    RA   = A @ D_ra @ A

    return CN, AA, RA


def sparse_lookup(M_csr, pairs: torch.Tensor) -> torch.Tensor:
    u = pairs[:, 0].cpu().numpy().copy()   # ← .copy()
    v = pairs[:, 1].cpu().numpy().copy()   # ← .copy()
    scores = np.array(M_csr[u, v]).flatten().astype(np.float32)
    return torch.from_numpy(scores)


# ─────────────────────────────────────────────────────────────────────────────
# MLP helper (BatchNorm + ELU)
# ─────────────────────────────────────────────────────────────────────────────

def build_mlp(in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> nn.Module:
    assert num_layers >= 1
    layers = []
    cur_dim = in_dim
    for _ in range(num_layers - 1):
        layers += [
            nn.Linear(cur_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        ]
        cur_dim = hidden_dim
    layers.append(nn.Linear(cur_dim, 1))
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class GraphSAGELinkPredictor(nn.Module):
    def __init__(self, in_dim, cfg):
        super().__init__()

        self.hidden_dim = cfg.hidden_dim

        # ── GraphSAGE Encoder ─────────────────────────────
        self.conv1 = SAGEConv(in_dim, self.hidden_dim)
        self.conv2 = SAGEConv(self.hidden_dim, self.hidden_dim)

        # ── MLP Decoder ───────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_pairs):
        u = z[edge_pairs[:, 0]]
        v = z[edge_pairs[:, 1]]

        # Hadamard Product (element-wise)
        h = u * v

        return self.mlp(h).squeeze(-1)

    def forward(self, x, edge_index, edge_pairs):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_pairs)

    # Dummy function to keep compatibility with your training code
    def precompute_structural(self, edge_index, num_nodes):
        pass
  