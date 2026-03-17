#!/usr/bin/env python3

import json
import sys
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


student_id = "aib252566"


# -----------------------------
# Dataset Loading
# -----------------------------

def load_from_api(dataset_num):

    url = f"http://hulk.cse.iitd.ac.in:3000/dataset?student_id={student_id}&dataset_num={dataset_num}"

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urllib.request.urlopen(req) as response:
        raw = response.read().decode("utf-8")
        data = json.loads(raw)
        X = np.array(data["X"])

    return X


def load_from_npy(path):
    return np.load(path)


# -----------------------------
# PCA
# -----------------------------

def maybe_apply_pca(X):

    if X.shape[1] > 50:
        pca = PCA(n_components=0.95, random_state=0)
        X = pca.fit_transform(X)

    return X


# -----------------------------
# KMeans Objective
# -----------------------------

def compute_kmeans_objectives(X):

    ks = list(range(1, 16))
    objectives = []

    for k in ks:

        kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
        kmeans.fit(X)

        objectives.append(kmeans.inertia_)

    return ks, np.array(objectives)


# -----------------------------
# Elbow Detection
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
        p = np.array([k, obj]) # cross product magnitude 
        numerator = abs( line_vec[0]*(p1[1]-p[1]) - line_vec[1]*(p1[0]-p[0]) ) 
        dist = numerator / norm_line 
        distances.append(dist) 

    elbow_index = np.argmax(distances) 
    return ks[elbow_index]

# -----------------------------
# Process Dataset
# -----------------------------

def process_dataset(X):

    X = maybe_apply_pca(X)

    ks, objectives = compute_kmeans_objectives(X)
    optimal_k = find_elbow(ks, objectives)

    return ks, objectives, optimal_k


# -----------------------------
# MAIN
# -----------------------------

if len(sys.argv) != 2:
    print("Usage: python rrunner.py <digit | dataset.npy>")
    sys.exit()

arg = sys.argv[1]


# ==========================
# MODE 1 : API (two datasets)
# ==========================

if arg.isdigit():

    X1 = load_from_api("1")
    X2 = load_from_api("2")

    ks1, obj1, k1 = process_dataset(X1)
    ks2, obj2, k2 = process_dataset(X2)

    print("k for Dataset 1:",k1)
    print("k for Dataset 2:",k2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Dataset 1
    axes[0].plot(ks1, obj1, marker="o")
    axes[0].scatter(k1, obj1[ks1.index(k1)], label=f"k={k1}")
    axes[0].set_title("Dataset 1")
    axes[0].set_xlabel("Number of Clusters")
    axes[0].set_ylabel("KMeans Objective")
    axes[0].legend()
    axes[0].grid(True)

    # Dataset 2
    axes[1].plot(ks2, obj2, marker="o")
    axes[1].scatter(k2, obj2[ks2.index(k2)], label=f"k={k2}")
    axes[1].set_title("Dataset 2")
    axes[1].set_xlabel("Number of Clusters")
    axes[1].set_ylabel("KMeans Objective")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle(f"KMeans Elbow Plot \n Student ID: {student_id}")

    plt.tight_layout()

    plt.savefig("plot.png")
    plt.close()


# ==========================
# MODE 2 : NPY (single plot)
# ==========================

elif arg.endswith(".npy"):

    X = load_from_npy(arg)

    ks, obj, k = process_dataset(X)

    print(k)
    plt.figure()

    plt.plot(ks, obj, marker="o")
    plt.scatter(k, obj[ks.index(k)], label=f"k={k}")

    plt.xlabel("Number of Clusters")
    plt.ylabel("KMeans Objective")
    plt.title(f"KMeans Elbow Plot \n {arg}")
    plt.legend()
    plt.grid(True)

    plt.savefig("plot.png")
    plt.close()

else:
    print("Invalid input")