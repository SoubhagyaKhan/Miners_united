"""
predict.py  –  COL761 Assignment 3 prediction script

Usage
-----
python predict.py --dataset A|B|C --task node|link --data_dir /absolute/path/to/data_dir \
--model_dir /path/to/models --output_dir /path/to/outputs --kerberos YOUR_KERBEROS

If you do not pass model_dir, the script will generate random predictions in the correct format. You can use this to test your evaluation setup before training a model.
"""

import argparse
import os

import numpy as np
import torch

from load_dataset import COL761NodeDataset, COL761LinkDataset, load_dataset, _load_edge_list


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str) -> torch.nn.Module:
    """Load a model saved with torch.save(model, path)."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = torch.load(model_path, weights_only=False, map_location="cpu")
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Random fallbacks (used when no --model_path is provided)
# ─────────────────────────────────────────────────────────────────────────────

def _random_A(dataset: COL761NodeDataset) -> torch.Tensor:
    return torch.randint(0, dataset.num_classes, (dataset[0].num_nodes,))

def _random_B(dataset: COL761NodeDataset) -> torch.Tensor:
    return torch.rand(dataset[0].num_nodes)

def _random_C(V: int, K: int) -> tuple:
    return torch.rand(V), torch.rand(V, K)


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset prediction functions
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_A(model: torch.nn.Module, dataset: COL761NodeDataset) -> torch.Tensor:
    """
    Returns predicted class index for every node → LongTensor [N].
    model(x, edge_index) must return logits [N, num_classes].
    """
    data   = dataset[0]

    ################################################
    """ Replace with your code to compute logits using model, if needed."""
    logits = model(data.x, data.edge_index)    # [N, num_classes]
    ################################################
    #Do not change return format — we will use automated scripts to evaluate your predictions.
    return logits.argmax(dim=1)                # [N]


@torch.no_grad()
def predict_B(model: torch.nn.Module, dataset: COL761NodeDataset) -> torch.Tensor:
    """
    Returns positive-class probability for every node → FloatTensor [N].
    model(x, edge_index) must return logits [N, 2] or [N, 1].
    """
    data   = dataset[0]
    ################################################
    """ Replace with your code to compute logits using model, if needed."""
    logits = model(data.x, data.edge_index)    # [N, 2] or [N, 1]
    ################################################

    #Do not change return format — we will use automated scripts to evaluate your predictions.
    if logits.shape[1] == 1:
        return torch.sigmoid(logits).squeeze(1)
    return torch.softmax(logits, dim=1)[:, 1]


@torch.no_grad()
def predict_C(
    model: torch.nn.Module,
    dataset: COL761LinkDataset,
    test_dir: str = None,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """
    Returns (pos_scores [P], neg_scores [P, K], split_name).

    test_dir=None  →  scores valid_pos / valid_neg   (student use)
    test_dir=path  →  scores test_pos  / test_neg    (instructor use)
    model(x, edge_index, edge_pairs) must return FloatTensor [E].
    """
    if test_dir is None:
        pos   = dataset.valid_pos                      # [P, 2]
        neg   = dataset.valid_neg                      # [P, K, 2]
        split = "valid"
    else:
        pos   = _load_edge_list(os.path.join(test_dir, "test_pos.txt"))
        npy   = os.path.join(test_dir, "test_neg_hard.npy")
        with open(npy, "rb") as f:
            neg = torch.from_numpy(np.load(f))         # [P, K, 2]
        split = "test"

    P, K, _ = neg.shape

    ################################################
    """Replace with your code to compute pos_scores and neg_scores using model, if needed """
    pos_scores = model(dataset.x, dataset.edge_index, pos)              # [P]
    neg_scores = model(
        dataset.x, dataset.edge_index, neg.view(P * K, 2)
    ).view(P, K)                                                         # [P, K]
    ################################################

    #Do not change return format — we will use automated scripts to evaluate your predictions.
    return pos_scores, neg_scores, split


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def predict_and_save(
    dataset_name: str,
    data_dir: str,
    model_path: str,
    out_dir: str,
    test_dir: str = None,
    kerberos: str = "student",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading dataset {dataset_name} ...")
    ds = load_dataset(dataset_name, data_dir)

    if model_path is not None:
        print(f"Loading model from {model_path} ...")
        model = load_model(model_path)
    else:
        print("No --model_path given — using random predictions.")
        model = None

    if dataset_name == "A":
        y_pred = predict_A(model, ds) if model else _random_A(ds)
        assert y_pred.shape == (ds[0].num_nodes,), \
            f"y_pred must be shape [N={ds[0].num_nodes}], got {y_pred.shape}"
        assert y_pred.dtype == torch.long, \
            f"y_pred must be LongTensor, got {y_pred.dtype}"

        out_path = os.path.join(out_dir, f"{kerberos}_predictions_A.pt")
        torch.save({"y_pred": y_pred}, out_path)
        print(f"Saved {out_path}  shape={y_pred.shape}")

    elif dataset_name == "B":
        y_score = predict_B(model, ds) if model else _random_B(ds)
        assert y_score.shape == (ds[0].num_nodes,), \
            f"y_score must be shape [N={ds[0].num_nodes}], got {y_score.shape}"
        assert y_score.is_floating_point(), \
            f"y_score must be float, got {y_score.dtype}"

        out_path = os.path.join(out_dir, f"{kerberos}_predictions_B.pt")
        torch.save({"y_score": y_score}, out_path)
        print(f"Saved {out_path}  shape={y_score.shape}")

    elif dataset_name == "C":
        if model:
            pos_scores, neg_scores, split = predict_C(model, ds, test_dir=test_dir)
        else:
            if test_dir or not hasattr(ds, "valid_pos"):
                pos   = ds.test_pos
                neg   = ds.test_neg
                split = "test"
            else:
                pos   = ds.valid_pos
                neg   = ds.valid_neg
                split = "valid"
            V, K = pos.shape[0], neg.shape[1]
            pos_scores, neg_scores = _random_C(V, K)

        out_path = os.path.join(out_dir, f"{kerberos}_predictions_C.pt")
        torch.save(
            {"pos_scores": pos_scores, "neg_scores": neg_scores, "split": split},
            out_path,
        )
        print(f"Saved {out_path}  split={split}")
        print(f"  pos_scores : {pos_scores.shape}")
        print(f"  neg_scores : {neg_scores.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions for COL761 A3 datasets."
    )
    parser.add_argument("--dataset",    required=True, choices=["A", "B", "C"])
    parser.add_argument("--task",       required=True, choices=["node", "link"],
                        help="Task type: node classification (A/B) or link prediction (C)")
    parser.add_argument("--data_dir",   required=True,
                        help="Absolute path to the shared datasets directory")
    parser.add_argument("--model_dir",  default=None,
                        help="Directory containing your saved model. "
                             "The script looks for <kerberos>_model_<dataset>.pt here.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory where predictions will be written")
    parser.add_argument("--kerberos",   required=True,
                        help="Your Kerberos ID (used to name the output file)")
    parser.add_argument("--test_dir",   default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # validate task ↔ dataset
    valid = {"node": ("A", "B"), "link": ("C",)}
    if args.dataset not in valid[args.task]:
        parser.error(
            f"--task {args.task} is not valid for --dataset {args.dataset}. "
            f"Expected dataset in {valid[args.task]}."
        )

    if not os.path.isabs(args.data_dir):
        parser.error("--data_dir must be an absolute path")

    model_path = None
    if args.model_dir is not None:
        model_path = os.path.join(
            args.model_dir, f"{args.kerberos}_model_{args.dataset}.pt"
        )

    predict_and_save(
        args.dataset, args.data_dir, model_path, args.output_dir,
        test_dir=args.test_dir,
        kerberos=args.kerberos,
    )


if __name__ == "__main__":
    main()
