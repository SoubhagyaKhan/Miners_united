import sys
import random
import heapq
from collections import deque

# Load seed set
def load_seeds(path):
    seeds = []
    with open(path) as f:
        for line in f:
            seeds.append(int(line.strip()))
    return seeds

# Load graph and find max_node for array allocations
def load_graph(path):
    edges = []
    max_node = 0 # we will use it later for array allocations
    with open(path) as f:
        for line in f:
            u, v, p = line.split()
            u, v, p = int(u), int(v), float(p)
            max_node = max(max_node, u, v)
            edges.append((u, v, p))
            
    print(f"Total edges in the graph: {len(edges)}")
    return edges, max_node

# Generate Monte Carlo realizations using ALL edges to maintain RNG sync
def generate_realizations(edges, r, max_node):
    print(f"Generating {r} Monte Carlo episodes...") 
    realizations = [] 

    for _ in range(r):
        g = [[] for _ in range(max_node + 1)]
        
        for u, v, p in edges:
            if random.random() <= p:
                g[u].append(v)

        realizations.append(g)

    return realizations

# BFS spread with O(1) array lookups
def simulate_spread(graph, seeds, blocked, hops, max_node):
    visited = [False] * (max_node + 1) # O(1) direct memory access
    
    q = deque()
    for s in seeds:
        visited[s] = True
        q.append((s, 0))
        
    visited_count = len(seeds)

    while q:
        u, depth = q.popleft()

        if hops != -1 and depth >= hops:
            continue

        for v in graph[u]: 
            if (u, v) in blocked:
                continue

            if not visited[v]:
                visited[v] = True
                visited_count += 1
                q.append((v, depth + 1))

    return visited_count

# Expected spread
def estimate_spread(realizations, seeds, blocked, hops, max_node):
    total = 0
    # for all monte carlo episodes that we stimulated, we gonna get the total edges visited from the inital given node
    for g in realizations:
        total += simulate_spread(g, seeds, blocked, hops, max_node)
    return total / len(realizations)

# CELF Adaptive Greedy
def adaptive_greedy(candidate_edges, seeds, realizations, k, hops, output_file, max_node):

    blocked = set()

    base_spread = estimate_spread(realizations, seeds, blocked, hops, max_node)
    print(f"Base spread for {len(realizations)} episodes : {base_spread}")
    print(f"Initial Nodes on fire {seeds}")

    heap = []

    # Initial gains :: ONLY evaluate the reachable candidate edges
    for u, v in candidate_edges: 
        blocked.add((u, v))
        new_spread = estimate_spread(realizations, seeds, blocked, hops, max_node)
        blocked.remove((u, v))

        gain = base_spread - new_spread

        # Only push edges that actually provide a benefit
        if gain > 0:
            heapq.heappush(heap, (-gain, (u, v), 0))

    # Clear output file first
    open(output_file, "w").close()

    for i in range(k):
        if not heap:
            print("No remaining edges provide any positive gain. Stopping early.")
            break

        while True:
            neg_gain, edge, last_iter = heapq.heappop(heap)
            u, v = edge

            # Zero-Gain early stopping or up-to-date score
            if -neg_gain <= 0 or last_iter == i:
                break

            blocked.add((u, v))
            new_spread = estimate_spread(realizations, seeds, blocked, hops, max_node)
            blocked.remove((u, v))

            gain = base_spread - new_spread

            heapq.heappush(heap, (-gain, (u, v), i))

        blocked.add(edge) 
        base_spread -= -neg_gain

        # Write edge immediately
        with open(output_file, "a") as out:
            out.write(f"{u} {v}\n")
            out.flush()

        print(f"Selected edge {i+1}/{k}: {edge} :: spread over {len(realizations)} episodes : {base_spread}")


# Main
def main():
    graph_path = sys.argv[1]
    seed_path = sys.argv[2]
    output_path = sys.argv[3]
    k = int(sys.argv[4])
    r = int(sys.argv[5])
    hops = int(sys.argv[6])

    random.seed(42)

    seeds = load_seeds(seed_path)
    print(f"Loaded seeds...{seeds}")

    edges, max_node = load_graph(graph_path)
    print(f"Loaded Graph...{graph_path}")

    realizations = generate_realizations(edges, r, max_node)

    # --- THE CANDIDATE PRUNING PHASE ---
    # Find which edges are actually reachable in these specific realizations
    candidate_edges = set()
    for g in realizations:
        visited = [False] * (max_node + 1)
        q = deque()
        for s in seeds:
            visited[s] = True
            q.append((s, 0))
            
        while q:
            u, depth = q.popleft()
            if hops != -1 and depth >= hops: 
                continue
                
            for v in g[u]:
                candidate_edges.add((u, v)) # This edge is reachable!
                if not visited[v]:
                    visited[v] = True
                    q.append((v, depth + 1))

    print(f"Reduced CELF search space from {len(edges)} to {len(candidate_edges)} reachable candidate edges.")

    adaptive_greedy(candidate_edges, seeds, realizations, k, hops, output_path, max_node)

if __name__ == "__main__":
    main()