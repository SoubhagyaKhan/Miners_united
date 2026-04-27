"""
train_C_ncn.py  --  Dataset C link prediction with NCN structural features

Usage
-----
    python train_C_ncn.py \
        --data_dir /absolute/path/to/public_datasets \
        --model_dir ./best_models \
        --kerberos YOUR_KERBEROS \
        [--seed 42]
"""

import argparse, os, sys, time, random
import numpy as np
import torch
import torch.nn.functional as F

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, ".."))

from config import LinkPredConfig
from modelClass.link_predictor import GraphSAGELinkPredictor
from load_dataset import load_dataset


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_scheduler(optimizer, warmup_epochs, total_epochs, eta_min):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine   = 0.5 * (1.0 + np.cos(np.pi * progress))
        return cosine * (1.0 - eta_min) + eta_min
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_pos_set(train_pos):
    s = set()
    for u, v in train_pos.tolist():
        s.add((u, v)); s.add((v, u))
    return s


def sample_random_negatives(n, num_nodes, pos_set, device):
    pairs = []
    while len(pairs) < n:
        needed = (n - len(pairs)) * 2
        u = torch.randint(0, num_nodes, (needed,)).tolist()
        v = torch.randint(0, num_nodes, (needed,)).tolist()
        for ui, vi in zip(u, v):
            if ui != vi and (ui, vi) not in pos_set:
                pairs.append([ui, vi])
                if len(pairs) == n:
                    break
    return torch.tensor(pairs, dtype=torch.long, device=device)


def sample_mixed_negatives(train_neg, n_total, num_nodes, pos_set, device, fixed_ratio=0.5):
    n_fixed  = int(n_total * fixed_ratio)
    n_random = n_total - n_fixed
    idx      = torch.randint(0, train_neg.shape[0], (n_fixed,))
    fixed    = train_neg[idx].to(device)
    rnd      = sample_random_negatives(n_random, num_nodes, pos_set, device)
    return torch.cat([fixed, rnd], dim=0)


def hits_at_k(pos_scores, neg_scores, k=50):
    n_neg_higher = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)
    return (n_neg_higher < k).float().mean().item()


def train_epoch(model, dataset, optimizer, cfg, device, pos_set, num_nodes):
    model.train()
    optimizer.zero_grad()

    pos_edges = dataset.train_pos.to(device)
    n_neg     = cfg.neg_ratio * pos_edges.shape[0]
    neg_edges = sample_mixed_negatives(
        dataset.train_neg, n_neg, num_nodes, pos_set, device,
        fixed_ratio=cfg.fixed_neg_ratio,
    )

    pos_s = model(dataset.x, dataset.edge_index, pos_edges)
    neg_s = model(dataset.x, dataset.edge_index, neg_edges)

    bce = (
        F.binary_cross_entropy_with_logits(pos_s, torch.ones_like(pos_s))
        + F.binary_cross_entropy_with_logits(neg_s, torch.zeros_like(neg_s))
    ) * cfg.bce_weight

    pos_rep = pos_s.repeat_interleave(cfg.neg_ratio)
    margin  = F.margin_ranking_loss(
        pos_rep, neg_s,
        torch.ones(n_neg, device=device),
        margin=cfg.margin,
    ) * cfg.margin_weight

    loss = bce + margin
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, dataset, cfg, device):
    model.eval()
    pos_edges = dataset.valid_pos.to(device)
    neg_edges = dataset.valid_neg.to(device)
    P, K, _  = neg_edges.shape
    pos_s = model(dataset.x, dataset.edge_index, pos_edges)
    neg_s = model(
        dataset.x, dataset.edge_index, neg_edges.view(P*K, 2)
    ).view(P, K)
    return hits_at_k(pos_s, neg_s, k=cfg.hits_k)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  required=True)
    p.add_argument("--model_dir", default="./best_models")
    p.add_argument("--kerberos",  required=True)
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    cfg    = LinkPredConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device : {device}")
    print(f"Config : {cfg}")

    print("\nLoading dataset C ...")
    dataset   = load_dataset("C", args.data_dir)
    num_nodes = dataset.x.shape[0]   # 3327
    in_dim    = dataset.x.shape[1]
    pos_set   = build_pos_set(dataset.train_pos)
    print(f"  true nodes={num_nodes}  in_dim={in_dim}  pos_set={len(pos_set)}")

    dataset.x          = dataset.x.to(device)
    dataset.edge_index  = dataset.edge_index.to(device)

    # ── Model ────────────────────────────────────────────────────────────────
    model = GraphSAGELinkPredictor(in_dim, cfg).to(device)

    # Precompute structural features (CN, AA, RA) — done once on CPU
    model.precompute_structural(dataset.edge_index, num_nodes)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  params={total_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = get_scheduler(optimizer, cfg.warmup_epochs, cfg.epochs, cfg.eta_min)

    os.makedirs(args.model_dir, exist_ok=True)
    model_path   = os.path.join(args.model_dir, f"{args.kerberos}_model_C.pt")
    best_hits    = 0.0
    ema_hits     = 0.0
    patience_ctr = 0
    t0           = time.time()

    for epoch in range(1, cfg.epochs + 1):
        loss = train_epoch(model, dataset, optimizer, cfg, device, pos_set, num_nodes)
        scheduler.step()

        if epoch % cfg.eval_every == 0 or epoch == 1:
            hits     = evaluate(model, dataset, cfg, device)
            ema_hits = cfg.ema_alpha * hits + (1 - cfg.ema_alpha) * ema_hits
            elapsed  = time.time() - t0
            print(f"Epoch {epoch:4d}/{cfg.epochs}  loss={loss:.4f}  "
                  f"Hits@{cfg.hits_k}={hits:.4f}  EMA={ema_hits:.4f}  [{elapsed:.1f}s]")

            if hits > best_hits:
                best_hits    = hits
                patience_ctr = 0
                torch.save(model, model_path)
                print(f"  ↑ best Hits@{cfg.hits_k}={best_hits:.4f}  saved -> {model_path}")
            else:
                patience_ctr += 1
                if patience_ctr >= cfg.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    print(f"\nDone. Best Hits@{cfg.hits_k} = {best_hits:.4f}")


if __name__ == "__main__":
    main()