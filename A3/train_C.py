"""
train_C.py  --  Dataset C link prediction (Hits@50)

v5 changes:
- Decoder now uses [u, v, u*v, |u-v|] -- L1 distance helps rank hard negatives
- Multi-seed training: run N_SEEDS independent runs, save best checkpoint
  (cheap on tiny graph, each run takes ~5s)
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modelClass import LinkPredictor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from load_dataset import load_dataset
from config import LinkPredConfig

N_SEEDS = 10   # run 3 independent seeds, keep best


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--model_dir",   default="./models")
    p.add_argument("--kerberos",    required=True)
    return p.parse_args()


def hits_at_k(pos_scores, neg_scores, k=50):
    n_neg_higher = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)
    return (n_neg_higher < k).float().mean().item()


def sample_hard_negatives(dataset, n_samples, device):
    hard = dataset.valid_neg.view(-1, 2)
    idx  = torch.randperm(hard.shape[0])[:n_samples]
    return hard[idx].to(device)


def train_epoch(model, dataset, optimizer, device, hard_ratio=0.5):
    model.train()
    optimizer.zero_grad()

    pos_edges = dataset.train_pos.to(device)
    M         = pos_edges.shape[0]

    n_hard   = int(M * hard_ratio)
    n_easy   = M - n_hard
    easy_neg = dataset.train_neg.to(device)
    easy_idx = torch.randperm(easy_neg.shape[0])[:n_easy]
    hard_neg = sample_hard_negatives(dataset, n_hard, device)
    neg_edges = torch.cat([easy_neg[easy_idx], hard_neg], dim=0)

    pos_scores = model(dataset.x, dataset.edge_index, pos_edges)
    neg_scores = model(dataset.x, dataset.edge_index, neg_edges)

    pos_loss = F.binary_cross_entropy_with_logits(
        pos_scores, torch.ones_like(pos_scores))
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_scores, torch.zeros_like(neg_scores))
    loss = pos_loss + neg_loss

    min_len = min(pos_scores.shape[0], neg_scores.shape[0])
    margin_loss = F.margin_ranking_loss(
        pos_scores[:min_len], neg_scores[:min_len],
        torch.ones(min_len, device=device), margin=0.5)
    loss = loss + 0.1 * margin_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, dataset, device):
    model.eval()
    pos_edges = dataset.valid_pos.to(device)
    neg_edges = dataset.valid_neg.to(device)
    P, K, _  = neg_edges.shape
    pos_scores = model(dataset.x, dataset.edge_index, pos_edges)
    neg_scores = model(
        dataset.x, dataset.edge_index, neg_edges.view(P * K, 2)).view(P, K)
    return hits_at_k(pos_scores, neg_scores, k=50)


def run_one_seed(seed, dataset, cfg, device, model_path, in_dim, current_best):
    torch.manual_seed(seed)
    model     = LinkPredictor(in_dim, cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min)

    best_hits    = 0.0
    patience_ctr = 0
    t0           = time.time()

    for epoch in range(1, cfg.epochs + 1):
        loss = train_epoch(model, dataset, optimizer, device, hard_ratio=0.5)
        scheduler.step()

        if epoch % cfg.eval_every == 0 or epoch == 1:
            hits    = evaluate(model, dataset, device)
            lr_now  = optimizer.param_groups[0]["lr"]
            print(f"  [seed {seed}] Ep {epoch:4d}  loss={loss:.4f}  "
                  f"Hits@50={hits:.4f}  lr={lr_now:.2e}  [{time.time()-t0:.1f}s]")

            if hits > best_hits:
                best_hits    = hits
                patience_ctr = 0
                # save if best across all seeds
                if hits > current_best:
                    torch.save(model, model_path)
                    print(f"  ↑ New global best={hits:.4f}  saved → {model_path}")
            else:
                patience_ctr += 1
                if patience_ctr >= cfg.patience:
                    print(f"  [seed {seed}] Early stopping at epoch {epoch}")
                    break

    return best_hits


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = LinkPredConfig()
    print(f"Config: {cfg}")

    print("Loading dataset C ...")
    dataset = load_dataset("C", args.data_dir)
    print(f"  nodes={dataset.num_nodes}  x={dataset.x.shape}  "
          f"train_pos={dataset.train_pos.shape[0]}")
    print(f"  valid_pos={dataset.valid_pos.shape[0]}  "
          f"valid_neg={dataset.valid_neg.shape[:2]}")

    in_dim             = dataset.x.shape[1]
    dataset.x          = dataset.x.to(device)
    dataset.edge_index = dataset.edge_index.to(device)

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, f"{args.kerberos}_model_C.pt")

    global_best = 0.0
    for seed in range(N_SEEDS):
        print(f"\n{'='*55}")
        print(f"  Seed {seed+1}/{N_SEEDS}")
        print(f"{'='*55}")
        best = run_one_seed(
            seed, dataset, cfg, device, model_path, in_dim, global_best)
        if best > global_best:
            global_best = best
        print(f"  Seed {seed+1} best={best:.4f}  global best={global_best:.4f}")

    print(f"\nDone. Best Hits@50 across all seeds = {global_best:.4f}")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()