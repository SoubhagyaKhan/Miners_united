"""
load_dataset.py  –  COL761 Assignment 3 dataset loader

Usage
-----
    python load_dataset.py --dataset A|B|C --data_dir /absolute/path/to/data_dir

Dataset A / B fields (data = dataset[0])
-----------------------------------------
    x             – node features for ALL nodes            [N, F]
    edge_index    – all edges (for GNN message passing)    [2, E]
    y             – labels for train+val nodes ONLY        [L]
    labeled_nodes – node indices that have a valid label   [L]
    train_mask    – boolean mask over labeled_nodes        [L]
    val_mask      – boolean mask over labeled_nodes        [L]
    num_classes   – number of distinct classes             int

Dataset C attributes
---------------------
    x             – precomputed node feature embeddings    [N, D]
    edge_index    – undirected training graph              [2, 2M]
    num_nodes     – total node count                       int
    train_pos     – positive training edges                [M, 2]
    train_neg     – negative training edges                [M, 2]
    valid_pos     – positive validation edges              [V, 2]
    valid_neg     – hard negative candidates per valid pos  [V, 500, 2]
"""

import argparse
import os

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset


# ─────────────────────────────────────────────────────────────────────────────
# Dataset A / B  –  node classification
# ─────────────────────────────────────────────────────────────────────────────

class COL761NodeDataset(InMemoryDataset):
    """
    PyG InMemoryDataset for COL761 node-classification tasks (datasets A and B).
    """

    def __init__(self, root: str, name: str, transform=None):
        assert name in ("A", "B"), f"name must be 'A' or 'B', got '{name}'"
        self.name = name
        super().__init__(root, transform)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, "_pyg")

    @property
    def raw_file_names(self):
        for fname in ("data.pt", "test.pt"):
            if os.path.isfile(os.path.join(self.raw_dir, fname)):
                return [fname]
        return ["data.pt"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass  # data.pt is already placed by prepare_assignment.py

    def process(self):
        for fname in ("data.pt", "test.pt"):
            candidate = os.path.join(self.raw_dir, fname)
            if os.path.isfile(candidate):
                raw = torch.load(candidate, weights_only=False)
                self.save([raw], self.processed_paths[0])
                return
        raise FileNotFoundError(f"Neither data.pt nor test.pt found in {self.raw_dir}")

    @property
    def num_classes(self) -> int:
        return int(self[0].y.max().item()) + 1

    def __repr__(self) -> str:
        return (f"COL761NodeDataset(name={self.name}, "
                f"nodes={self[0].num_nodes}, "
                f"edges={self[0].num_edges}, "
                f"classes={self.num_classes})")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset C  –  link prediction
# ─────────────────────────────────────────────────────────────────────────────

def _load_edge_list(path: str) -> torch.Tensor:
    """Read a tab-separated edge-list file → LongTensor [E, 2]."""
    edges = []
    with open(path) as f:
        for line in f:
            u, v = line.strip().split("\t")
            edges.append((int(u), int(v)))
    return torch.tensor(edges, dtype=torch.long)


class COL761LinkDataset:
    """
    Dataset wrapper for COL761 link-prediction task (dataset C).
    """

    def __init__(self, ds_dir: str):
        if not os.path.isdir(ds_dir):
            raise FileNotFoundError(f"Dataset directory not found: {ds_dir}")

        # --- edge lists
        for split in ("train", "valid", "test"):
            pos_path      = os.path.join(ds_dir, f"{split}_pos.txt")
            neg_hard_path = os.path.join(ds_dir, f"{split}_neg_hard.npy")
            neg_path      = os.path.join(ds_dir, f"{split}_neg.txt")

            if os.path.isfile(pos_path):
                setattr(self, f"{split}_pos", _load_edge_list(pos_path))

            if os.path.isfile(neg_hard_path):
                with open(neg_hard_path, "rb") as f:
                    setattr(self, f"{split}_neg", torch.from_numpy(np.load(f)))
            elif os.path.isfile(neg_path):
                setattr(self, f"{split}_neg", _load_edge_list(neg_path))

        # --- undirected graph for message passing
        # prefer train_pos; fall back to test_pos if only test data is present
        ref_pos = None
        for attr in ("train_pos", "test_pos", "valid_pos"):
            if hasattr(self, attr):
                ref_pos = getattr(self, attr)
                break

        if ref_pos is not None:
            ref_edge = ref_pos.t()                                   # [2, M]
            self.edge_index = torch.cat(
                [ref_edge, ref_edge[[1, 0]]], dim=1
            )                                                        # [2, 2M]
        else:
            self.edge_index = torch.zeros((2, 0), dtype=torch.long)

        # --- node count (union of all node ids seen across splits)
        node_set: set = set()
        for attr in ("train_pos", "valid_pos", "test_pos"):
            if hasattr(self, attr):
                node_set.update(getattr(self, attr).flatten().tolist())
        self.num_nodes: int = len(node_set)

        # --- precomputed GNN node features
        feat_path = os.path.join(ds_dir, "gnn_feature")
        if os.path.isfile(feat_path):
            feat = torch.load(feat_path, weights_only=False)
            self.x: torch.Tensor = feat["entity_embedding"]

    def __repr__(self) -> str:
        return (f"COL761LinkDataset("
                f"nodes={self.num_nodes}, "
                f"train_pos={self.train_pos.shape[0]}, "
                f"valid_pos={self.valid_pos.shape[0]})")


# ─────────────────────────────────────────────────────────────────────────────
# Public factory
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(dataset: str, data_dir: str):
    """
    Load dataset A, B, or C from *data_dir*.

    Parameters
    ----------
    dataset  : "A", "B", or "C"
    data_dir : absolute path to the directory containing A/, B/, C/ subfolders

    Returns
    -------
    COL761NodeDataset  for A or B
    COL761LinkDataset  for C
    """
    dataset = dataset.upper()
    if dataset not in ("A", "B", "C"):
        raise ValueError(f"dataset must be 'A', 'B', or 'C', got '{dataset}'")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    if dataset in ("A", "B"):
        return COL761NodeDataset(root=data_dir, name=dataset)
    else:
        return COL761LinkDataset(ds_dir=os.path.join(data_dir, "C"))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _print_stats(dataset: str, ds) -> None:
    print(f"\n{'='*52}")
    print(f"  Dataset {dataset}")
    print(f"{'='*52}")

    if isinstance(ds, COL761NodeDataset):
        data = ds[0]
        n_labeled   = data.labeled_nodes.shape[0]
        n_unlabeled = data.num_nodes - n_labeled
        print(f"  {ds}")
        print(f"  Node features   : {data.x.shape[1]}")
        print(f"  Classes         : {ds.num_classes}")
        print(f"  Labeled nodes   : {n_labeled:,}")
        print(f"  Train         : {data.train_mask.sum().item():,}")
        print(f"  Val           : {data.val_mask.sum().item():,}")
        print(f"  Unlabeled nodes : {n_unlabeled:,}")

    else:  # COL761LinkDataset
        print(f"  {ds}")
        for attr in ("train_pos", "train_neg", "valid_pos", "valid_neg",
                     "test_pos", "test_neg"):
            if hasattr(ds, attr):
                print(f"  {attr:<15} : {getattr(ds, attr).shape}")
        if hasattr(ds, "x"):
            print(f"  x (node feats)  : {ds.x.shape}")
        print(f"  edge_index      : {ds.edge_index.shape}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Load COL761 A3 assignment datasets A, B, or C."
    )
    parser.add_argument(
        "--dataset", required=True, choices=["A", "B", "C"],
        help="Which dataset to load: A, B, or C"
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Absolute path to the directory containing A/, B/, C/ subfolders"
    )
    args = parser.parse_args()

    if not os.path.isabs(args.data_dir):
        parser.error("--data_dir must be an absolute path")

    ds = load_dataset(args.dataset, args.data_dir)
    _print_stats(args.dataset, ds)


if __name__ == "__main__":
    main()
