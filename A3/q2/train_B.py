"""
train_B.py  --  Dataset B node classification (2 classes, AUC-ROC)

Dataset B is ~2.89M nodes / ~24.7M edges -- too large for full-graph GPU training.
We use NeighborLoader (mini-batch) + GraphSAGE with BatchNorm.

Usage
-----
    python train_B.py \
        --data_dir /absolute/path/to/public_datasets \
        --model_dir ./models \
        --kerberos YOUR_KERBEROS
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import roc_auc_score
import random, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modelClass import GraphSAGEModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from load_dataset import load_dataset
from config import GraphSAGEConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--model_dir",   default="./models")
    p.add_argument("--kerberos",    required=True)
    return p.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # makes CUDA deterministic (slight slowdown)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, optimizer, device):
    """One pass over the mini-batches of training nodes."""
    model.train()
    total_loss     = 0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index)   # [batch_size + neighbors, 2]

        # Only use seed nodes (first batch.batch_size rows)
        seed_logits = logits[:batch.batch_size]      # [seed_nodes, 2]
        seed_labels = batch.y[:batch.batch_size]     # [seed_nodes]

        # Only compute loss on labeled train nodes (y != -1)
        mask = seed_labels >= 0
        if mask.sum() == 0:
            continue

        loss = F.cross_entropy(seed_logits[mask], seed_labels[mask].long())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss     += loss.item() * mask.sum().item()
        total_examples += mask.sum().item()

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate_val(model, data, labeled_nodes, val_mask, device, batch_size=4096, num_neighbors=None):
    """Compute AUC-ROC on validation nodes using NeighborLoader."""
    model.eval()

    if num_neighbors is None:
        num_neighbors = [15, 10, 5]

    val_node_idx = labeled_nodes[val_mask]

    val_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=val_node_idx,
        shuffle=False,
        num_workers=0,
    )

    all_scores = []
    all_labels = []

    for batch in val_loader:
        batch   = batch.to(device)
        logits  = model(batch.x, batch.edge_index)
        seed_l  = logits[:batch.batch_size]
        scores  = torch.softmax(seed_l, dim=1)[:, 1].cpu().numpy()
        labels  = batch.y[:batch.batch_size].cpu().numpy()

        # filter out unlabeled (y == -1)
        valid = labels >= 0
        all_scores.append(scores[valid])
        all_labels.append(labels[valid])

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    return roc_auc_score(all_labels, all_scores)


def main():
    args   = parse_args()
    set_seed(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = GraphSAGEConfig()

    print("Loading dataset B...")
    t0      = time.time()
    dataset = load_dataset("B", args.data_dir)
    data    = dataset[0]
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  nodes={data.num_nodes:,}  edges={data.num_edges:,}")

    labeled_nodes = data.labeled_nodes
    train_mask    = data.train_mask
    val_mask      = data.val_mask
    in_dim        = data.x.shape[1]
    num_classes   = dataset.num_classes

    # Build full [N] label tensor: -1 for unlabeled
    full_y = torch.full((data.num_nodes,), -1, dtype=torch.long)
    full_y[labeled_nodes] = data.y.long()
    data.y = full_y

    # Full [N] train mask
    full_train = torch.zeros(data.num_nodes, dtype=torch.bool)
    full_train[labeled_nodes[train_mask]] = True
    data.train_mask = full_train

    train_node_idx = labeled_nodes[train_mask]

    print(f"  train nodes={train_node_idx.shape[0]:,}  val nodes={val_mask.sum().item():,}")
    print(f"Config: {cfg}")

    train_loader = NeighborLoader(
        data,
        num_neighbors=cfg.num_neighbors,
        batch_size=cfg.batch_size,
        input_nodes=train_node_idx,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    model = GraphSAGEModel(in_dim, num_classes, cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"GraphSAGE  hidden={cfg.hidden_dim}  params={total_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min
    )

    os.makedirs(args.model_dir, exist_ok=True)
    model_path   = os.path.join(args.model_dir, f"{args.kerberos}_model_B.pt")

    best_auc     = 0.0
    patience_ctr = 0

    for epoch in range(1, cfg.epochs + 1):
        t_ep = time.time()
        loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        print(f"\nEpoch {epoch}/{cfg.epochs}  loss={loss:.4f}  [{time.time()-t_ep:.1f}s]")
        print("  Running val AUC-ROC ...")

        auc = evaluate_val(
            model, data, labeled_nodes, val_mask, device,
            batch_size=cfg.val_batch_size,
            num_neighbors=cfg.num_neighbors,
        )
        print(f"  Val AUC-ROC = {auc:.4f}")

        if auc > best_auc:
            best_auc     = auc
            patience_ctr = 0
            torch.save(model, model_path)
            print(f"  ↑ Best AUC={best_auc:.4f}  saved → {model_path}")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nDone. Best val AUC-ROC = {best_auc:.4f}")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()