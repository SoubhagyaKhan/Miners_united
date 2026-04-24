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


# Dataset C  --  link prediction
@dataclass
class LinkPredConfig:
    # Architecture
    # use_encoder=False: skip GCN, use raw precomputed embeddings directly
    # (x is already a GNN embedding -- adding another GCN hurts on 3199 nodes)
    use_encoder:      bool  = False
    hidden_dim:       int   = 256
    num_layers:       int   = 2          # only used when use_encoder=True
    dropout:          float = 0.3
    decoder_hidden:   int   = 128

    # Loss weights
    margin:           float = 0.5

    # Training
    lr:               float = 0.003
    weight_decay:     float = 1e-4
    epochs:           int   = 200
    patience:         int   = 20         # in eval_every units (= 100 epochs)
    eval_every:       int   = 5
    eta_min:          float = 1e-4

    # Hits@K evaluation
    hits_k:           int   = 50