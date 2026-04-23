"""
train_A.py  --  Dataset A node classification (7 classes, Accuracy)

Usage
-----
    python train_A.py \
        --data_dir /absolute/path/to/public_datasets \
        --model_dir ./models \
        --kerberos YOUR_KERBEROS \
        [--arch gat|gcn]

All hyperparameters live in config.py (GATConfig / GCNConfig).
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch_geometric.transforms import NormalizeFeatures

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, ".."))

from config import GATConfig, GCNConfig
from modelClass import GATModel, GCNModel
from load_dataset import load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  required=True)
    p.add_argument("--model_dir", default="./models")
    p.add_argument("--kerberos",  required=True)
    p.add_argument("--arch",      default="gat", choices=["gat", "gcn"])
    return p.parse_args()


def train_epoch(model, data, optimizer, labeled_nodes, train_mask, label_smoothing=0.0):
    model.train()
    optimizer.zero_grad()
    logits    = model(data.x, data.edge_index)       # [N, C]
    train_idx = labeled_nodes[train_mask]
    loss      = F.cross_entropy(
        logits[train_idx], data.y[train_mask],
        label_smoothing=label_smoothing,
    )
    loss.backward()
    # gradient clipping helps with GAT
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, labeled_nodes, mask):
    model.eval()
    logits   = model(data.x, data.edge_index)        # [N, C]
    node_idx = labeled_nodes[mask]
    preds    = logits[node_idx].argmax(dim=1)
    return (preds == data.y[mask]).float().mean().item()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}  |  arch : {args.arch.upper()}")

    dataset = load_dataset("A", args.data_dir)
    data    = NormalizeFeatures()(dataset[0]).to(device)

    labeled_nodes = data.labeled_nodes
    train_mask    = data.train_mask
    val_mask      = data.val_mask
    in_dim        = data.x.shape[1]
    num_classes   = dataset.num_classes

    print(f"Dataset A  |  nodes={data.num_nodes}  in_dim={in_dim}  classes={num_classes}")
    print(f"  train={train_mask.sum().item()}  val={val_mask.sum().item()}")

    if args.arch == "gat":
        cfg   = GATConfig()
        model = GATModel(in_dim, num_classes, cfg).to(device)
    else:
        cfg   = GCNConfig()
        model = GCNModel(in_dim, num_classes, cfg).to(device)

    print(f"Model params : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Config : {cfg}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min
    )

    os.makedirs(args.model_dir, exist_ok=True)
    model_path   = os.path.join(args.model_dir, f"{args.kerberos}_model_A.pt")
    best_val_acc = 0.0
    patience_ctr = 0   # counts eval checks with no improvement, NOT raw epochs
    t0           = time.time()

    for epoch in range(1, cfg.epochs + 1):
        loss = train_epoch(
            model, data, optimizer, labeled_nodes, train_mask,
            label_smoothing=cfg.label_smoothing,
        )
        scheduler.step()

        if epoch % cfg.eval_every == 0 or epoch == 1:
            val_acc   = evaluate(model, data, labeled_nodes, val_mask)
            train_acc = evaluate(model, data, labeled_nodes, train_mask)
            print(f"Epoch {epoch:4d}/{cfg.epochs}  loss={loss:.4f}  "
                  f"train={train_acc:.4f}  val={val_acc:.4f}  "
                  f"[{time.time()-t0:.1f}s]")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_ctr = 0
                torch.save(model, model_path)
                print(f"  ↑ best val={best_val_acc:.4f}  saved -> {model_path}")
            else:
                patience_ctr += 1
                # patience is in units of eval checks
                if patience_ctr >= cfg.patience:
                    print(f"Early stopping at epoch {epoch} "
                          f"(no improvement for {cfg.patience} evals)")
                    break

    print(f"\nDone. Best val accuracy = {best_val_acc:.4f}")


if __name__ == "__main__":
    main()