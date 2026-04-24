"""
config.py  --  Q2 hyperparameter configuration
"""

from dataclasses import dataclass, field
from typing import List


# Dataset A  --  GAT / GCN, full-graph, 7-class node classification
@dataclass
class GATConfig:
    # Architecture
    hidden_dim:       int   = 256
    num_heads:        int   = 4          # multi-head attention
    num_layers:       int   = 4
    dropout:          float = 0.6
    attn_dropout:     float = 0.6
    leaky_slope:      float = 0.2

    # Training
    lr:               float = 5e-4
    weight_decay:     float = 5e-4
    epochs:           int   = 2000
    patience:         int   = 30         # in eval_every units, not raw epochs
    eval_every:       int   = 10
    eta_min:          float = 5e-6
    label_smoothing:  float = 0.0


@dataclass
class GCNConfig:
    # Architecture
    hidden_dim:       int   = 256
    num_layers:       int   = 3
    dropout:          float = 0.5

    # Training
    lr:               float = 5e-4
    weight_decay:     float = 5e-4
    epochs:           int   = 2000
    patience:         int   = 30         # in eval_every units
    eval_every:       int   = 10
    eta_min:          float = 5e-6
    label_smoothing:  float = 0.0


# Dataset B  --  GraphSAGE, mini-batch, 2-class node classification
@dataclass
class GraphSAGEConfig:
    # Architecture
    hidden_dim:       int         = 256
    num_layers:       int         = 3
    dropout:          float       = 0.5

    # Training
    lr:               float       = 0.001
    weight_decay:     float       = 1e-5
    epochs:           int         = 40
    patience:         int         = 8
    eta_min:          float       = 1e-6

    # Mini-batch sampling
    batch_size:       int         = 2048
    num_neighbors:    List[int]   = field(default_factory=lambda: [15, 10, 5])
    num_workers:      int         = 4
    val_batch_size:   int         = 4096


@dataclass
class LinkPredConfig:
    # ── Encoder ───────────────────────────────────────────────────────────────
    proj_dim:         int   = 256        # input projection: 3703 → proj_dim
    hidden_dim:       int   = 256        # SAGE layer output dim
    num_layers:       int   = 2          # SAGE layers (keep at 2 for sparse graph)
    sage_aggr:        str   = "mean"     # "mean" | "max"  (SAGEConv aggr)
    dropout:          float = 0.3
 
    # ── Decoder ───────────────────────────────────────────────────────────────
    decoder_type:     str   = "hadamard" # "hadamard" | "concat"
    decoder_hidden:   int   = 128        # MLP hidden dim
    decoder_layers:   int   = 2          # MLP depth
 
    # ── Loss ──────────────────────────────────────────────────────────────────
    bce_weight:       float = 1.0
    margin_weight:    float = 0.5
    margin:           float = 1.0
    neg_ratio:        int   = 3          # negatives per positive (oversampling)
 
    # ── Training ──────────────────────────────────────────────────────────────
    lr:               float = 1e-3
    weight_decay:     float = 1e-5
    epochs:           int   = 500
    patience:         int   = 50         # in eval_every units
    eval_every:       int   = 5
    eta_min:          float = 1e-6
 
    # ── Eval ──────────────────────────────────────────────────────────────────
    hits_k:           int   = 50

    fixed_neg_ratio:  float = 0.5    # fraction from train_neg; rest are random
    warmup_epochs:    int   = 20 
    ema_alpha:        float = 0.3    # weight on current eval (higher = less smooth)
    decoder_type:     str   = "concat"