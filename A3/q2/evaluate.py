"""
evaluate.py  –  COL761 Assignment 3 evaluation script

Usage
--------
    python evaluate.py --dataset A|B|C --task node|link --data_dir \
        /absolute/path/to/datasets --output_dir /path/to/outputs --kerberos YOUR_KERBEROS

Metrics
-------
    A  →  Accuracy   (multi-class node classification)
    B  →  AUC-ROC    (binary node classification)
    C  →  Hits@50    (link prediction)
"""

import argparse
import os

import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from load_dataset import load_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def hits_at_k(pos_scores: torch.Tensor, neg_scores: torch.Tensor, k: int) -> float:
    """
    Fraction of positives that rank in the top-k against their hard negatives.

    pos_scores : FloatTensor [P]
    neg_scores : FloatTensor [P, K]
    """
    n_neg_higher = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)  # [P]
    return (n_neg_higher < k).float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset evaluators
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_A(pred_path: str, split: str, data_dir: str, gt_dir: str) -> float:
    """
    Accuracy for dataset A.
    Predictions must contain y_pred: LongTensor [N]
    """
    pred  = torch.load(pred_path, weights_only=False)
    y_pred = pred["y_pred"]

    if split == "val":
        data        = load_dataset("A", data_dir)[0]
        node_idx    = data.labeled_nodes[data.val_mask]
        true_labels = data.y[data.val_mask]
    else:
        gt          = torch.load(
            os.path.join(gt_dir, "A", "test.pt"), weights_only=False
        )
        node_idx    = gt["test_node_idx"]
        true_labels = gt["test_labels"]

    return accuracy_score(true_labels.numpy(), y_pred[node_idx].numpy())


def evaluate_B(pred_path: str, split: str, data_dir: str, gt_dir: str) -> float:
    """
    AUC-ROC for dataset B.
    Predictions must contain y_score: FloatTensor [N]
    """
    pred    = torch.load(pred_path, weights_only=False)
    y_score = pred["y_score"]

    if split == "val":
        data        = load_dataset("B", data_dir)[0]
        node_idx    = data.labeled_nodes[data.val_mask]
        true_labels = data.y[data.val_mask]
    else:
        gt          = torch.load(
            os.path.join(gt_dir, "B", "test.pt"), weights_only=False
        )
        node_idx    = gt["test_node_idx"]
        true_labels = gt["test_labels"]

    return roc_auc_score(true_labels.numpy(), y_score[node_idx].numpy())


def evaluate_C(pred_path: str, k: int = 50) -> tuple[float, str]:
    """
    Hits@k for dataset C.

    Predictions must contain:
        pos_scores : FloatTensor [P]
        neg_scores : FloatTensor [P, 500]
        split      : str
    """
    pred       = torch.load(pred_path, weights_only=False)
    pos_scores = pred["pos_scores"]
    neg_scores = pred["neg_scores"]
    split      = pred.get("split", "unknown")
    return hits_at_k(pos_scores, neg_scores, k=k), split


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate COL761 A3 predictions."
    )
    parser.add_argument("--dataset",    required=True, choices=["A", "B", "C"])
    parser.add_argument("--task",       required=True, choices=["node", "link"],
                        help="Task type: node classification (A/B) or link prediction (C)")
    parser.add_argument("--data_dir",   required=True,
                        help="Absolute path to the shared datasets directory")
    parser.add_argument("--output_dir", required=True,
                        help="Directory containing the predictions file")
    parser.add_argument("--kerberos",   required=True,
                        help="Your Kerberos ID (used to locate the predictions file)")
    # instructor-only args, hidden from student docs
    parser.add_argument("--split",   default="val", choices=["val", "test"],
                        help=argparse.SUPPRESS)
    parser.add_argument("--gt_dir",  default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    valid = {"node": ("A", "B"), "link": ("C",)}
    if args.dataset not in valid[args.task]:
        parser.error(
            f"--task {args.task} is not valid for --dataset {args.dataset}. "
            f"Expected dataset in {valid[args.task]}."
        )

    pred_file = os.path.join(args.output_dir, f"{args.kerberos}_predictions_{args.dataset}.pt")
    if not os.path.isfile(pred_file):
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")

    if args.dataset in ("A", "B"):
        if args.split == "test" and args.gt_dir is None:
            args.gt_dir = args.data_dir

    print(f"\nEvaluating dataset {args.dataset} ...")

    if args.dataset == "A":
        acc = evaluate_A(pred_file, args.split, args.data_dir, args.gt_dir)
        print(f"  Kerberos : {args.kerberos}")
        print(f"  Split    : {args.split}")
        print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")

    elif args.dataset == "B":
        auc = evaluate_B(pred_file, args.split, args.data_dir, args.gt_dir)
        print(f"  Kerberos : {args.kerberos}")
        print(f"  Split    : {args.split}")
        print(f"  AUC-ROC  : {auc:.4f}")

    elif args.dataset == "C":
        h50, split = evaluate_C(pred_file, k=50)
        print(f"  Kerberos : {args.kerberos}")
        print(f"  Split    : {split}")
        print(f"  Hits@50  : {h50:.4f}  ({h50*100:.2f}%)")

    print()


if __name__ == "__main__":
    main()
