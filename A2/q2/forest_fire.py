import sys
import random
import heapq
from collections import defaultdict, deque

# Load graph
def load_graph(path):
    graph = defaultdict(list)
    edges = []

    with open(path) as f:
        for line in f:
            u, v, p = line.split()
            u = int(u)
            v = int(v)
            p = float(p)

            graph[u].append((v, p))
            edges.append((u, v, p))
    print(f"Total edges on the graph : {len(edges)}")
    return graph, edges


# Load seed set
def load_seeds(path):
    seeds = []
    with open(path) as f:
        for line in f:
            seeds.append(int(line.strip()))
    return seeds

# Generate Monte Carlo realizations
def generate_realizations(edges, r):
    print(f"Generating {r} Monte Carlo episodes") 
    # Here, we are trying to generate r Monte carlo episodes 
    # Each episode considers the edges that were burnt on the flow 
    # Complexity : O(kE)
    realizations = [] 

    for _ in range(r):

        g = defaultdict(list)

        for u, v, p in edges:
            if random.random() <= p:
                g[u].append(v)

        realizations.append(g)

    return realizations

# BFS spread
def simulate_spread(graph, seeds, blocked, hops):
    # Complexity : O(E)
    visited = set(seeds) # Seeds are A_0 - the initial nodes on fire
    q = deque([(s, 0) for s in seeds])

    while q:
        u, depth = q.popleft()

        if hops != -1 and depth >= hops:
            continue

        for v in graph.get(u, []): # iterating over neighbors

            if (u, v) in blocked:
                continue

            if v not in visited:
                visited.add(v)
                q.append((v, depth + 1))

    return len(visited)

# Expected spread
def estimate_spread(realizations, seeds, blocked, hops):
    # Complexity : O(kE)
    total = 0
    # for all monte carlo episodes that we stimulated, we gonna get the total edges visited from the inital given node
    for g in realizations:
        total += simulate_spread(g, seeds, blocked, hops)

    return total / len(realizations)

# CELF Adaptive Greedy
def adaptive_greedy(edges, seeds, realizations, k, hops, output_file):

    blocked = set()

    base_spread = estimate_spread(realizations, seeds, blocked, hops)
    print(f"Base spread for {k} episodes : {base_spread}")
    print(f"Initial Nodes on fire {seeds}")

    heap = []

    # initial gains :: Complexity : O(kE*E)
    for u, v, p in edges: 
        # here, we are trying to evaluate for each edge, explicitly how much is its total spread

        blocked.add((u, v))
        new_spread = estimate_spread(realizations, seeds, blocked, hops)
        blocked.remove((u, v))

        gain = base_spread - new_spread

        heapq.heappush(heap, (-gain, (u, v), 0))

    # clear output file first
    open(output_file, "w").close()

    for i in range(k):

        while True:

            neg_gain, edge, last_iter = heapq.heappop(heap)
            u, v = edge

            if last_iter == i:
                break

            blocked.add((u, v))
            new_spread = estimate_spread(realizations, seeds, blocked, hops)
            blocked.remove((u, v))

            gain = base_spread - new_spread

            heapq.heappush(heap, (-gain, (u, v), i))

        blocked.add(edge) # blocking the optimal edge

        base_spread -= -neg_gain

        # write edge immediately
        with open(output_file, "a") as out:
            out.write(f"{u} {v}\n")
            out.flush()

        print(f"Selected edge {i+1}/{k}: {edge} :: spread over k episodes : {base_spread}")

# Main
def main():

    graph_path = sys.argv[1]
    seed_path = sys.argv[2]
    output_path = sys.argv[3]
    k = int(sys.argv[4])
    r = int(sys.argv[5])
    hops = int(sys.argv[6])

    random.seed(42)

    graph, edges = load_graph(graph_path)
    print(f"Loaded Graph...{graph_path}")

    seeds = load_seeds(seed_path)
    print(f"Loaded seeds...{seeds}")

    realizations = generate_realizations(edges, r)

    adaptive_greedy(edges, seeds, realizations, k, hops, output_path)


if __name__ == "__main__":
    main()