#!/usr/bin/env python3
import sys
import os

# Usage:
# python3 dataset_converter.py <input_yeast> <output_dir>
# Creates:
#   yeast_fsg.dat
#   yeast_gspan.dat
#   yeast_gaston.dat

if len(sys.argv) != 3:
    print("Usage: python3 dataset_converter.py <input_yeast> <output_dir>")
    sys.exit(1)

INPUT_FILE = sys.argv[1]
OUT_DIR = sys.argv[2]
os.makedirs(OUT_DIR, exist_ok=True)

FSG_OUT = os.path.join(OUT_DIR, "yeast_fsg.dat")
GSPAN_OUT = os.path.join(OUT_DIR, "yeast_gspan.dat")
GASTON_OUT = os.path.join(OUT_DIR, "yeast_gaston.dat")

# --------------------------------------------------
# Global node-label mapping (S,O,N,C,H → 0,1,2...)
# --------------------------------------------------
node_label_map = {}
next_node_label = 0

def map_node_label(lbl):
    global next_node_label
    if lbl not in node_label_map:
        node_label_map[lbl] = next_node_label
        next_node_label += 1
    return node_label_map[lbl]

# --------------------------------------------------
# Read Yeast dataset
# --------------------------------------------------
def read_yeast(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]

    graphs = []
    i = 0

    while i < len(lines):
        if not lines[i].startswith("#"):
            i += 1
            continue

        gid = lines[i][1:].strip()
        i += 1

        n = int(lines[i]); i += 1
        nodes = []
        for _ in range(n):
            nodes.append(map_node_label(lines[i]))
            i += 1

        m = int(lines[i]); i += 1
        edges = []
        for _ in range(m):
            u, v, el = lines[i].split()
            edges.append((int(u), int(v), int(el)))  # edge labels untouched
            i += 1

        graphs.append((gid, nodes, edges))

    return graphs

graphs = read_yeast(INPUT_FILE)
print(f"[converter] Parsed {len(graphs)} graphs")
print(f"[converter] Node label map: {node_label_map}")

# --------------------------------------------------
# Helpers for FSG string labels
# --------------------------------------------------
def fsg_node_label(i):
    return f"V{i}"

def fsg_edge_label(i):
    return f"E{i}"

# --------------------------------------------------
# Write FSG (STRICT format)
# --------------------------------------------------
with open(FSG_OUT, "w") as f:
    for idx, (_, nodes, edges) in enumerate(graphs):
        f.write(f"t # {idx}\n")
        for i, lbl in enumerate(nodes):
            f.write(f"v {i} {fsg_node_label(lbl)}\n")
        for u, v, el in edges:
            a, b = min(u, v), max(u, v)
            f.write(f"u {a} {b} {fsg_edge_label(el)}\n")

print(f"✔ FSG written → {FSG_OUT}")

# --------------------------------------------------
# Write gSpan
# --------------------------------------------------
with open(GSPAN_OUT, "w") as f:
    for gid, nodes, edges in graphs:
        f.write(f"t # {gid}\n")
        for i, lbl in enumerate(nodes):
            f.write(f"v {i} {lbl}\n")
        for u, v, el in edges:
            f.write(f"e {u} {v} {el}\n")

print(f"✔ gSpan written → {GSPAN_OUT}")

# --------------------------------------------------
# Write Gaston (same as gSpan)
# --------------------------------------------------
with open(GASTON_OUT, "w") as f:
    for gid, nodes, edges in graphs:
        f.write(f"t # {gid}\n")
        for i, lbl in enumerate(nodes):
            f.write(f"v {i} {lbl}\n")
        for u, v, el in edges:
            f.write(f"e {u} {v} {el}\n")

print(f"✔ Gaston written → {GASTON_OUT}")
print("✔ Dataset conversion completed successfully")
