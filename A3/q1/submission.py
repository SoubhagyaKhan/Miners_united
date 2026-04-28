import numpy as np
import faiss
import random
import os
import time

# ─────────────────────────────
# Global settings
# ─────────────────────────────
np.random.seed(42)
random.seed(42)
faiss.omp_set_num_threads(8)  # adjust based on your CPU

DEBUG = False  # ← TURN OFF for final submission

# Log file path (written alongside the script)
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission_log.txt")


# ─────────────────────────────────────────────
# Logging helper
# ─────────────────────────────────────────────
def _log(msg, log_path=LOG_FILE):
    """Append a timestamped message to the log file and optionally print."""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    if DEBUG:
        print(line)
    with open(log_path, "a") as f:
        f.write(line + "\n")


# ─────────────────────────────────────────────
# 1. Build Index (Dynamic)
# ─────────────────────────────────────────────
def build_index(base, value_range):
    N, d = base.shape

    # ─────────────────────────────
    # D2 → IVF (fast)
    # ─────────────────────────────
    if value_range > 100:
        quantizer = faiss.IndexFlatL2(d)

        nlist = 4000
        index = faiss.IndexIVFFlat(quantizer, d, nlist)

        # FIX 1: increase training size
        train_size = min(40 * nlist, N)
        index.train(base[:train_size])
        index.add(base)

        # FIX 2: increase nprobe
        index.nprobe = 20

        _log(f"Built IVFFlat index | nlist={nlist} nprobe={index.nprobe} N={N} d={d}")
        return index


# ─────────────────────────────
# D1 → IVF (balanced accuracy + speed)
# ─────────────────────────────
    else:
        quantizer = faiss.IndexFlatL2(d)

        nlist = 4000   # moderate partitioning
        index = faiss.IndexIVFFlat(quantizer, d, nlist)

        # more training for stability
        train_size = min(40 * nlist, N)
        index.train(base[:train_size])
        index.add(base)

        # higher probes → maintain accuracy
        index.nprobe = 75

        _log(f"Built IVFFlat (D1) | nlist={nlist} nprobe={index.nprobe} N={N} d={d}")
        return index


# ─────────────────────────────────────────────
# 2. Search (batched)
# ─────────────────────────────────────────────
def batched_search(index, queries, k):
    Q = queries.shape[0]

    nn = np.empty((Q, k), dtype=np.int64)
    dist = np.empty((Q, k), dtype=np.float32)

    batch_size = 20000

    for i in range(0, Q, batch_size):
        D, I = index.search(queries[i:i + batch_size], k)
        nn[i:i + len(I)] = I
        dist[i:i + len(I)] = D

    return nn, dist


# ─────────────────────────────────────────────
# 3. Aggregation
# ─────────────────────────────────────────────
def aggregate(nn_indices, distances, value_range, N, K):
    Q, k = nn_indices.shape

    flat_idx = nn_indices.flatten()

    # SAFETY (important for HNSW edge cases)
    flat_idx = flat_idx[flat_idx >= 0]

    # ─────────────────────────────
    # D1 → rank + distance weighted
    # ─────────────────────────────
    if value_range <= 100:
        rank_w = 1.0 / np.log2(np.arange(k) + 2)
        weights = rank_w[None, :] * (1.0 / (1.0 + distances))

        flat_w = weights.flatten()
        flat_w = flat_w[:len(flat_idx)]  # align sizes after filtering

        scores = np.bincount(flat_idx, weights=flat_w, minlength=N)

    # ─────────────────────────────
    # D2 → pure frequency
    # ─────────────────────────────
    else:
        scores = np.bincount(flat_idx, minlength=N)

    return np.argsort(-scores)[:K]


# ─────────────────────────────────────────────
# 4. Validate output
# ─────────────────────────────────────────────
def validate_output(result, N, K):
    """
    Checks the three hard requirements from the spec:
      1. Exactly K indices returned.
      2. All indices are valid (0 <= idx < N).
      3. No duplicate indices.
    Logs warnings for any violation; returns the (possibly repaired) array.
    """
    result = np.asarray(result, dtype=np.int64)

    # --- duplicates ---
    unique, counts = np.unique(result, return_counts=True)
    dupes = unique[counts > 1]
    if len(dupes):
        _log(f"WARNING: {len(dupes)} duplicate index/indices found: {dupes[:10]} — deduplicating.")
        seen = set()
        deduped = []
        for idx in result:
            if idx not in seen:
                seen.add(idx)
                deduped.append(idx)
        result = np.array(deduped, dtype=np.int64)

    # --- out-of-range ---
    bad_mask = (result < 0) | (result >= N)
    if bad_mask.any():
        _log(f"WARNING: {bad_mask.sum()} out-of-range indices found — removing.")
        result = result[~bad_mask]

    # --- wrong length ---
    if len(result) != K:
        _log(f"WARNING: result length is {len(result)}, expected {K}.")
        if len(result) > K:
            result = result[:K]
        else:
            # pad with lowest-index valid items not already in result
            existing = set(result.tolist())
            pad = [i for i in range(N) if i not in existing]
            needed = K - len(result)
            result = np.concatenate([result, np.array(pad[:needed], dtype=np.int64)])
            _log(f"Padded result to length {K} with {needed} fallback indices.")

    return result


# ─────────────────────────────────────────────
# 5. Save output to text log
# ─────────────────────────────────────────────
def save_output(result, log_path=LOG_FILE):
    """
    Appends the final ranked list to the log file in a human-readable block.
    Format:
        === OUTPUT (K=100) ===
        rank_1: <idx>
        rank_2: <idx>
        ...
    """
    _log(f"=== OUTPUT (K={len(result)}) ===")
    for rank, idx in enumerate(result, start=1):
        _log(f"  rank_{rank:04d}: {idx}")
    _log("=== END OUTPUT ===")


# ─────────────────────────────────────────────
# 6. Solve  (public interface)
# ─────────────────────────────────────────────
def solve(base_vectors, query_vectors, k, K, time_budget):
    """
    base_vectors : np.ndarray of shape (N, d)
    query_vectors: np.ndarray of shape (Q, d)
    k            : int, nearest neighbours per query
    K            : int, number of output items to return
    time_budget  : float, maximum allowed time in seconds

    Returns
    -------
    np.ndarray of shape (K,) — ranked list of base indices,
    most representative first, no duplicates, all valid.
    """
    t0 = time.time()

    # ── Convert to float32 ──────────────────────────────────────────────
    base    = np.ascontiguousarray(base_vectors.astype(np.float32))
    queries = np.ascontiguousarray(query_vectors.astype(np.float32))

    N = base.shape[0]
    value_range = float(np.max(base) - np.min(base))

    _log(f"solve() called | N={N} d={base.shape[1]} Q={queries.shape[0]} "
         f"k={k} K={K} budget={time_budget}s value_range={value_range:.4f}")

    if DEBUG:
        print("\n===== DEBUG START =====")
        print("Base shape :", base.shape)
        print("Query shape:", queries.shape)
        print("Value range:", value_range)

    # ── Build index ─────────────────────────────────────────────────────
    index = build_index(base, value_range)
    _log(f"Index built in {time.time()-t0:.2f}s | ntotal={index.ntotal}")

    # ── Search ──────────────────────────────────────────────────────────
    t1 = time.time()
    nn_indices, distances = batched_search(index, queries, k)
    _log(f"Search done in {time.time()-t1:.2f}s")

    if DEBUG:
        print("Sample neighbors:", nn_indices[0][:10])

    # ── Aggregate ───────────────────────────────────────────────────────
    t2 = time.time()
    result = aggregate(nn_indices, distances, value_range, N, K)
    _log(f"Aggregation done in {time.time()-t2:.2f}s")

    if DEBUG:
        print("Top-K preview:", result[:10])
        print("===== DEBUG END =====\n")

    # ── Validate ────────────────────────────────────────────────────────
    result = validate_output(result, N, K)

    elapsed = time.time() - t0
    _log(f"Total elapsed: {elapsed:.2f}s  (budget={time_budget}s)")

    # ── Save ranked list to log ─────────────────────────────────────────
    save_output(result)

    return result.astype(np.int64)
