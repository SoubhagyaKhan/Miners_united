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
from torch_geometric.nn import GATv2Conv

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

class NCNLinkPredictor(nn.Module):
    """
    GATv2 encoder  +  symmetric concat decoder  +  CN/AA/RA structural features

    Decoder input per pair (u,v):
        [z_first(512) || z_second(512) || cn(1) || aa(1) || ra(1)]  =  1027-dim
    """

    def __init__(self, in_dim: int, cfg: LinkPredConfig):
        super().__init__()
        self.cfg = cfg

        heads    = cfg.num_heads
        head_dim = cfg.hidden_dim // heads
        assert cfg.hidden_dim % heads == 0

        # ── Input projection ─────────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, cfg.proj_dim),
            nn.LayerNorm(cfg.proj_dim),
            nn.ELU(),
            nn.Dropout(cfg.dropout),
        )

        # ── GATv2 Encoder ────────────────────────────────────────────────────
        self.conv1 = GATv2Conv(
            cfg.proj_dim, head_dim,
            heads=heads, concat=True,
            dropout=cfg.attn_dropout, add_self_loops=True,
        )
        self.norm1 = nn.BatchNorm1d(cfg.hidden_dim)

        self.conv2 = GATv2Conv(
            cfg.hidden_dim, cfg.hidden_dim,
            heads=heads, concat=False,
            dropout=cfg.attn_dropout, add_self_loops=True,
        )
        self.norm2 = nn.BatchNorm1d(cfg.hidden_dim)

        self.enc_drop = nn.Dropout(cfg.dropout)

        # ── Structural feature projection ─────────────────────────────────────
        # 3 raw scalars (CN, AA, RA) → struct_dim via small MLP
        # This lets the model learn non-linear combinations of the heuristics
        self.struct_proj = nn.Sequential(
            nn.Linear(3, cfg.struct_dim),
            nn.BatchNorm1d(cfg.struct_dim),
            nn.ELU(),
        )

        # ── Decoder MLP ───────────────────────────────────────────────────────
        # input: [z_first || z_second || struct]
        decoder_in = cfg.hidden_dim * 2 + cfg.struct_dim
        self.decoder = build_mlp(
            in_dim=decoder_in,
            hidden_dim=cfg.decoder_hidden,
            num_layers=cfg.decoder_layers,
            dropout=cfg.dropout,
        )

        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Structural matrices (set after construction via precompute_structural)
        self.CN = None
        self.AA = None
        self.RA = None

    # ── Structural precomputation ─────────────────────────────────────────────

    def precompute_structural(self, edge_index: torch.Tensor, num_nodes: int):
        """Call once after model init, before training."""
        print("Precomputing structural features (CN, AA, RA) ...")
        self.CN, self.AA, self.RA = compute_structural_matrices(edge_index, num_nodes)
        print(f"  Done. Matrices shape: {self.CN.shape}")

    def get_structural_features(
        self, edge_pairs: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Look up CN, AA, RA for each pair and project to struct_dim.
        edge_pairs : [E, 2]
        returns      [E, struct_dim]
        """
        cn = sparse_lookup(self.CN, edge_pairs).to(device)  # [E]
        aa = sparse_lookup(self.AA, edge_pairs).to(device)  # [E]
        ra = sparse_lookup(self.RA, edge_pairs).to(device)  # [E]

        raw = torch.stack([cn, aa, ra], dim=1)              # [E, 3]
        return self.struct_proj(raw)                         # [E, struct_dim]

    # ── Encoder ──────────────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)

        # Layer 0
        h = self.conv1(h, edge_index)
        h = self.norm1(h)
        h = F.elu(h)
        h = self.enc_drop(h)

        # Layer 1 + residual
        residual = h
        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        h = h + residual
        h = F.elu(h)
        h = self.enc_drop(h)

        return h   # [N, hidden_dim]

    # ── Decoder ──────────────────────────────────────────────────────────────

    def decode(
        self,
        z: torch.Tensor,
        edge_pairs: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        u = edge_pairs[:, 0]
        v = edge_pairs[:, 1]

        z_u = z[u]
        z_v = z[v]

        # Symmetric concat
        swap     = (u > v).unsqueeze(1)
        z_first  = torch.where(swap, z_v, z_u)
        z_second = torch.where(swap, z_u, z_v)

        # Structural features
        struct = self.get_structural_features(edge_pairs, device)  # [E, struct_dim]

        pair = torch.cat([z_first, z_second, struct], dim=1)       # [E, D]
        return self.decoder(pair).squeeze(1)                        # [E]

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_pairs: torch.Tensor,
    ) -> torch.Tensor:
        device = x.device
        z = self.encode(x, edge_index)
        return self.decode(z, edge_pairs, device)