import sys
import os
import json
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# -----------------------------
# Argument Parsing
# -----------------------------

def parse_argument():

    if len(sys.argv) != 2:
        print("Usage: python3 Q1.py <1|2|dataset.npy>")
        sys.exit(1)

    arg = sys.argv[1]

    # dataset via API
    if arg in {"1", "2"}:
        return ("api", int(arg))

    # dataset via npy file
    if arg.endswith(".npy"):

        if not os.path.isfile(arg):
            print("Error: .npy file not found")
            sys.exit(1)

        return ("file", arg)

    print("Invalid argument. Use dataset 1, dataset 2, or a .npy file.")
    sys.exit(1)


# -----------------------------
# Dataset Loading
# -----------------------------

def load_from_api(dataset_num):

    url = f"http://hulk.cse.iitd.ac.in:3000/dataset?student_id=aib252566&dataset_num={dataset_num}"

    try:
        with urllib.request.urlopen(url) as response:
            raw = response.read().decode("utf-8")
            data = json.loads(raw)
            X = np.array(data["X"])
    except Exception as e:
        print("Error fetching dataset:", e)
        sys.exit(1)

    return X


def load_dataset():

    mode, value = parse_argument()

    if mode == "api":
        return load_from_api(value)

    else:
        try:
            return np.load(value)
        except Exception:
            print("Invalid numpy file")
            sys.exit(1)


# -----------------------------
# PCA for High Dimensional Data
# -----------------------------

def maybe_apply_pca(X):

    n_samples, n_features = X.shape

    # heuristic threshold
    if n_features > 50:

        # reduce while preserving 95% variance
        pca = PCA(n_components=0.95, random_state=0)

        X_reduced = pca.fit_transform(X)

        return X_reduced

    return X


# -----------------------------
# KMeans Objective Calculation
# -----------------------------

def compute_kmeans_objectives(X):

    ks = list(range(1, 16))
    objectives = []

    for k in ks:

        kmeans = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=0
        )

        kmeans.fit(X)

        objectives.append(kmeans.inertia_)

    return ks, np.array(objectives)


# -----------------------------
# Geometric Elbow Detection
# -----------------------------

def find_elbow(ks, objectives):

    ks = np.array(ks)
    objs = np.array(objectives)

    p1 = np.array([ks[0], objs[0]])
    p2 = np.array([ks[-1], objs[-1]])

    line_vec = p2 - p1
    norm_line = np.linalg.norm(line_vec)

    distances = []

    for k, obj in zip(ks, objs):

        p = np.array([k, obj])

        # cross product magnitude
        numerator = abs(
            line_vec[0]*(p1[1]-p[1]) -
            line_vec[1]*(p1[0]-p[0])
        )

        dist = numerator / norm_line

        distances.append(dist)

    elbow_index = np.argmax(distances)

    return ks[elbow_index]


# -----------------------------
# Plot Generation
# -----------------------------

def generate_plot(ks, objectives):

    plt.figure(figsize=(12,7), dpi=200)

    plt.plot(ks, objectives, marker="o")

    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("KMeans Objective (Inertia)")
    plt.title("Elbow Method for Optimal k")

    plt.grid(True)

    plt.savefig("plot.png")


# -----------------------------
# Main
# -----------------------------

def main():

    X = load_dataset()

    # ensure numpy array
    X = np.asarray(X)

    # handle high dimensions
    X = maybe_apply_pca(X)

    ks, objectives = compute_kmeans_objectives(X)

    optimal_k = find_elbow(ks, objectives)

    generate_plot(ks, objectives)

    # required stdout output
    print(optimal_k)


if __name__ == "__main__":
    main()