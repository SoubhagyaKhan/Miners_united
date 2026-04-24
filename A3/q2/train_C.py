"""
train_C.py  --  Dataset C link prediction (Hits@50)

Strategy
--------
    Encoder  : input projection (3703→256) + 2-layer GraphSAGE
    Decoder  : Hadamard or Concat + MLP  (set via LinkPredConfig.decoder_type)
    Loss     : BCE + Margin Ranking, with negative oversampling
    Negatives: resample neg_ratio × |train_pos| pairs from train_neg each epoch

Usage
-----
    python train_C.py \
        --data_dir /absolute/path/to/public_datasets \
        --model_dir ./best_models \
        --kerberos YOUR_KERBEROS \
        [--decoder hadamard|concat]
"""

import argparse
import os
import sys
import time
import random

import numpy as np
import torch
import torch.nn.functional as F

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, ".."))

from config import LinkPredConfig
from modelClass.link_predictor import LinkPredictor
from load_dataset import load_dataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def sample_negatives(train_neg: torch.Tensor, n: int) -> torch.Tensor:
    """Resample n negatives from train_neg with replacement. [M,2] -> [n,2]"""
    idx = torch.randint(0, train_neg.shape[0], (n,))
    return train_neg[idx]


def hits_at_k(pos_scores: torch.Tensor, neg_scores: torch.Tensor, k: int = 50) -> float:
    """
    pos_scores : [P]
    neg_scores : [P, K]
    Returns fraction of positives ranking in top-k against their K hard negatives.
    """
    n_neg_higher = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)
    return (n_neg_higher < k).float().mean().item()


def train_epoch(model, dataset, optimizer, cfg, device):
    model.train()
    optimizer.zero_grad()

    x          = dataset.x
    edge_index = dataset.edge_index
    pos_edges  = dataset.train_pos.to(device)         # [P, 2]

    # oversample: neg_ratio negatives per positive
    n_neg     = cfg.neg_ratio * pos_edges.shape[0]
    neg_edges = sample_negatives(dataset.train_neg, n_neg).to(device)  # [n_neg, 2]

    pos_scores = model(x, edge_index, pos_edges)      # [P]
    neg_scores = model(x, edge_index, neg_edges)      # [n_neg]

    # BCE
    bce_loss = (
        F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
        + F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
    ) * cfg.bce_weight

    # Margin ranking: repeat each pos score neg_ratio times to pair with negatives
    # Enforces score(pos) > score(neg) + margin
    pos_rep     = pos_scores.repeat_interleave(cfg.neg_ratio)   # [n_neg]
    margin_loss = F.margin_ranking_loss(
        pos_rep,
        neg_scores,
        torch.ones(n_neg, device=device),
        margin=cfg.margin,
    ) * cfg.margin_weight

    loss = bce_loss + margin_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, dataset, cfg, device):
    model.eval()

    pos_edges = dataset.valid_pos.to(device)          # [P, 2]
    neg_edges = dataset.valid_neg.to(device)          # [P, 500, 2]
    P, K, _  = neg_edges.shape

    pos_scores = model(dataset.x, dataset.edge_index, pos_edges)          # [P]
    neg_scores = model(
        dataset.x, dataset.edge_index, neg_edges.view(P * K, 2)
    ).view(P, K)                                                           # [P, K]

    return hits_at_k(pos_scores, neg_scores, k=cfg.hits_k)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  required=True)
    p.add_argument("--model_dir", default="./best_models")
    p.add_argument("--kerberos",  required=True)
    p.add_argument("--decoder",   default="hadamard", choices=["hadamard", "concat"])
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg    = LinkPredConfig(decoder_type=args.decoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device  : {device}")
    print(f"Decoder : {cfg.decoder_type}")
    print(f"Config  : {cfg}")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\nLoading dataset C ...")
    dataset = load_dataset("C", args.data_dir)

    # CRITICAL: x.shape[0]=3327 is the true node count; dataset.num_nodes=3199 is wrong
    true_N = dataset.x.shape[0]
    in_dim = dataset.x.shape[1]
    print(f"  true nodes = {true_N}  (loader reports {dataset.num_nodes})")
    print(f"  in_dim     = {in_dim}")
    print(f"  train_pos  = {dataset.train_pos.shape[0]}")
    print(f"  train_neg  = {dataset.train_neg.shape[0]}")
    print(f"  valid_pos  = {dataset.valid_pos.shape[0]}")
    print(f"  valid_neg  = {dataset.valid_neg.shape[:2]}")

    # Move static tensors to device once
    dataset.x          = dataset.x.to(device)
    dataset.edge_index  = dataset.edge_index.to(device)

    # ── Model ────────────────────────────────────────────────────────────────
    model        = LinkPredictor(in_dim, cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nLinkPredictor params : {total_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min
    )

    # ── Training loop ────────────────────────────────────────────────────────
    os.makedirs(args.model_dir, exist_ok=True)
    model_path   = os.path.join(args.model_dir, f"{args.kerberos}_model_C.pt")
    best_hits    = 0.0
    patience_ctr = 0
    t0           = time.time()

    for epoch in range(1, cfg.epochs + 1):
        loss = train_epoch(model, dataset, optimizer, cfg, device)
        scheduler.step()

        if epoch % cfg.eval_every == 0 or epoch == 1:
            hits    = evaluate(model, dataset, cfg, device)
            elapsed = time.time() - t0
            print(f"Epoch {epoch:4d}/{cfg.epochs}  loss={loss:.4f}  "
                  f"Hits@{cfg.hits_k}={hits:.4f}  [{elapsed:.1f}s]")

            if hits > best_hits:
                best_hits    = hits
                patience_ctr = 0
                torch.save(model, model_path)
                print(f"  ↑ best Hits@{cfg.hits_k}={best_hits:.4f}  saved -> {model_path}")
            else:
                patience_ctr += 1
                if patience_ctr >= cfg.patience:
                    print(f"Early stopping at epoch {epoch} "
                          f"(no improvement for {cfg.patience} evals)")
                    break

    print(f"\nDone. Best Hits@{cfg.hits_k} = {best_hits:.4f}")


if __name__ == "__main__":
    main()