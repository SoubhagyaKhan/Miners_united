"""
Identify Discriminative Subgraphs using gSpan (Frequent Subgraph Mining)
Falls back to simple mining if gSpan is not available
"""

import sys
import pickle
import subprocess
import os
import tempfile
from collections import defaultdict
from graph_parser import parse_graph_dataset, remove_duplicates, Graph
from typing import List, Set

# HYPER PARAMETERS
GSPAN_SUPPORT_THRESHOLD = 5.0
K = 100
MINM_SUPPPORT_THRESHOLD = 0.05
MAXM_SUPPORT_THRESHOLD = 0.80
MIN_EDGES = 2

class SubgraphPattern:
    """Represents a mined subgraph pattern"""
    def __init__(self, pattern_id: int, graph: Graph, support: int, graph_ids: Set[int]):
        self.id = pattern_id
        self.graph = graph
        self.support = support  # Number of database graphs containing this pattern
        self.graph_ids = graph_ids  # IDs of graphs containing this pattern
        self.discriminative_score = 0.0
        
    def __repr__(self):
        return f"Pattern(id={self.id}, nodes={self.graph.num_nodes()}, edges={self.graph.num_edges()}, support={self.support})"


def convert_to_gspan_format(graphs: List[Graph], output_file: str):
    """
    Convert graphs to gSpan format
    
    gSpan format:
    t # graph_id
    v vertex_id vertex_label
    e vertex1 vertex2 edge_label
    """
    print(f"Converting {len(graphs)} graphs to gSpan format...")
    
    with open(output_file, 'w') as f:
        for graph in graphs:
            # Graph header
            f.write(f"t # {graph.id}\n")
            
            # Vertices (nodes)
            for node_id in sorted(graph.nodes.keys()):
                node_label = graph.nodes[node_id]
                f.write(f"v {node_id} {node_label}\n")
            
            # Edges
            for source, target, edge_label in graph.edges:
                f.write(f"e {source} {target} {edge_label}\n")
    
    print(f"Saved gSpan format to {output_file}")


def parse_gspan_output(gspan_output_file: str) -> List[SubgraphPattern]:
    """
    Parse gSpan output to extract frequent patterns
    
    gSpan output format:
    t # pattern_id * support
    v vertex_id vertex_label
    e vertex1 vertex2 edge_label
    x: graph_id1 graph_id2 ...  (optional, contains which graphs have this pattern)
    """
    patterns = []
    
    if not os.path.exists(gspan_output_file):
        print(f"WARNING: gSpan output file not found: {gspan_output_file}")
        return patterns
    
    print(f"Parsing gSpan output from {gspan_output_file}...")
    
    with open(gspan_output_file, 'r') as f:
        lines = f.readlines()
    
    current_pattern = None
    current_graph = None
    pattern_id = 0
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            i += 1
            continue
        
        parts = line.split()
        
        # Pattern header: t # <pattern_id> * <support>
        if parts[0] == 't' and len(parts) >= 3:
            # Save previous pattern if exists
            if current_pattern is not None and current_graph is not None:
                current_pattern.graph = current_graph
                patterns.append(current_pattern)
            
            # Extract support
            support = 0
            if '*' in parts:
                star_idx = parts.index('*')
                if star_idx + 1 < len(parts):
                    support = int(parts[star_idx + 1])
            
            # Create new pattern
            current_graph = Graph(pattern_id)
            current_pattern = SubgraphPattern(pattern_id, current_graph, support, set())
            pattern_id += 1
        
        # Vertex: v vertex_id vertex_label
        elif parts[0] == 'v' and current_graph is not None:
            vertex_id = int(parts[1])
            vertex_label = int(parts[2])
            current_graph.add_node(vertex_id, vertex_label)
        
        # Edge: e vertex1 vertex2 edge_label
        elif parts[0] == 'e' and current_graph is not None:
            vertex1 = int(parts[1])
            vertex2 = int(parts[2])
            edge_label = int(parts[3])
            current_graph.add_edge(vertex1, vertex2, edge_label)
        
        # Graph IDs: x: graph_id1 graph_id2 ...
        elif parts[0] == 'x:' and current_pattern is not None:
            graph_ids = [int(gid) for gid in parts[1:]]
            current_pattern.graph_ids = set(graph_ids)
        
        i += 1
    
    # Don't forget the last pattern
    if current_pattern is not None and current_graph is not None:
        current_pattern.graph = current_graph
        patterns.append(current_pattern)
    
    print(f"Parsed {len(patterns)} patterns from gSpan output")
    return patterns


def run_gspan(input_file: str, output_file: str, min_support_pct: float, gspan_path: str = "./gSpan") -> bool:
    """
    Run gSpan mining algorithm
    
    gSpan parameters:
    -f <input_file>  : input graph file
    -s <support>     : minimum support (percentage or absolute count)
    -o <output_file> : output file
    -i               : output graph IDs (which graphs contain each pattern)
    """
    try:
        # gSpan expects support as percentage (0-100) or absolute count
        # We'll use percentage
        cmd = [
            gspan_path,
            "-f", input_file,
            "-s", str(min_support_pct),
            "-o", output_file,
            "-i"  # Include graph IDs in output
        ]
        
        print(f"Running gSpan with min_support={min_support_pct}%...")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode != 0:
            print(f"gSpan failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
        
        print("✓ gSpan completed successfully")
        if result.stdout:
            print(f"gSpan output: {result.stdout[:500]}")
        
        return True
        
    except FileNotFoundError:
        print(f"gSpan executable not found at: {gspan_path}")
        return False
    except subprocess.TimeoutExpired:
        print("gSpan timed out (>10 minutes)")
        return False
    except Exception as e:
        print(f"Error running gSpan: {e}")
        return False


def calculate_discriminative_power(support: int, total_graphs: int) -> float:
    """
    Calculate discriminative power based on FG-Index approach
    Power = |D1| * |D2| where D1 = graphs with feature, D2 = graphs without
    Normalized by total^2 to get score between 0 and 0.25
    """
    d1 = support
    d2 = total_graphs - support
    if total_graphs == 0:
        return 0.0
    return (d1 * d2) / (total_graphs ** 2)


def mine_frequent_subgraphs_simple(graphs: List[Graph], min_support_pct: float = GSPAN_SUPPORT_THRESHOLD) -> List[SubgraphPattern]:
    """
    Simple frequent subgraph mining using pattern enumeration
    This is a simplified approach - in practice, you'd use FSG/gSpan
    
    For now, we'll mine frequent substructures like:
    - Single edges with labels
    - Small paths
    - Triangles
    - Stars
    """
    total_graphs = len(graphs)
    min_support_count = int(total_graphs * min_support_pct / 100)
    
    print(f"Mining subgraphs with min support = {min_support_pct}% ({min_support_count} graphs)")
    
    patterns = []
    pattern_id = 0
    
    # 1. Mine frequent single edges (edge labels)
    edge_label_support = defaultdict(set)  # edge_label -> set of graph IDs
    
    for graph in graphs:
        for source, target, label in graph.edges:
            edge_label_support[label].add(graph.id)
    
    for edge_label, graph_ids in edge_label_support.items():
        if len(graph_ids) >= min_support_count:
            # Create a simple graph with single edge
            g = Graph(pattern_id)
            g.add_node(0, 0)  # dummy node label
            g.add_node(1, 0)
            g.add_edge(0, 1, edge_label)
            
            pattern = SubgraphPattern(pattern_id, g, len(graph_ids), graph_ids)
            patterns.append(pattern)
            pattern_id += 1
    
    print(f"Mined {len(patterns)} single-edge patterns")
    
    # 2. Mine frequent node labels
    node_label_support = defaultdict(set)
    
    for graph in graphs:
        for node_id, label in graph.nodes.items():
            node_label_support[label].add(graph.id)
    
    for node_label, graph_ids in node_label_support.items():
        if len(graph_ids) >= min_support_count:
            g = Graph(pattern_id)
            g.add_node(0, node_label)
            
            pattern = SubgraphPattern(pattern_id, g, len(graph_ids), graph_ids)
            patterns.append(pattern)
            pattern_id += 1
    
    print(f"Total patterns after node labels: {len(patterns)}")
    
    # 2b. Mine node-edge pairs (node with specific outgoing edge)
    node_edge_support = defaultdict(set)
    
    for graph in graphs:
        for source, target, edge_label in graph.edges:
            source_label = graph.nodes[source]
            target_label = graph.nodes[target]
            # Store as (source_node_label, edge_label)
            node_edge_support[(source_label, edge_label)].add(graph.id)
            node_edge_support[(target_label, edge_label)].add(graph.id)
    
    for (node_label, edge_label), graph_ids in node_edge_support.items():
        if len(graph_ids) >= min_support_count:
            g = Graph(pattern_id)
            g.add_node(0, node_label)
            g.add_node(1, 0)  # dummy target
            g.add_edge(0, 1, edge_label)
            
            pattern = SubgraphPattern(pattern_id, g, len(graph_ids), graph_ids)
            patterns.append(pattern)
            pattern_id += 1
    
    print(f"Total patterns after node-edge pairs: {len(patterns)}")
    
    # 3. Mine frequent paths of length 2 (node-edge-node-edge-node)
    path2_support = defaultdict(set)  # (n1_label, e1_label, n2_label, e2_label, n3_label) -> graph_ids
    
    for graph in graphs:
        # Find all paths of length 2
        for node in graph.G.nodes():
            neighbors = list(graph.G.neighbors(node))
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i+1:]:
                    # Path: n1 - node - n2
                    n1_label = graph.nodes[n1]
                    node_label = graph.nodes[node]
                    n2_label = graph.nodes[n2]
                    e1_label = graph.G[n1][node]['label']
                    e2_label = graph.G[node][n2]['label']
                    
                    # Canonical form (sort to avoid duplicates)
                    if (n1_label, e1_label) > (n2_label, e2_label):
                        path_sig = (n2_label, e2_label, node_label, e1_label, n1_label)
                    else:
                        path_sig = (n1_label, e1_label, node_label, e2_label, n2_label)
                    
                    path2_support[path_sig].add(graph.id)
    
    for path_sig, graph_ids in path2_support.items():
        if len(graph_ids) >= min_support_count:
            n1_l, e1_l, n2_l, e2_l, n3_l = path_sig
            g = Graph(pattern_id)
            g.add_node(0, n1_l)
            g.add_node(1, n2_l)
            g.add_node(2, n3_l)
            g.add_edge(0, 1, e1_l)
            g.add_edge(1, 2, e2_l)
            
            pattern = SubgraphPattern(pattern_id, g, len(graph_ids), graph_ids)
            patterns.append(pattern)
            pattern_id += 1
    
    print(f"Total patterns after 2-paths: {len(patterns)}")
    
    # 4. Mine triangles
    triangle_support = defaultdict(set)
    
    for graph in graphs:
        # Find all triangles
        for node in graph.G.nodes():
            neighbors = list(graph.G.neighbors(node))
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i+1:]:
                    # Check if n1 and n2 are connected
                    if graph.G.has_edge(n1, n2):
                        # Triangle found: node - n1 - n2 - node
                        labels = sorted([
                            (graph.nodes[node], graph.nodes[n1], graph.nodes[n2]),
                            (graph.G[node][n1]['label'], graph.G[n1][n2]['label'], graph.G[n2][node]['label'])
                        ])
                        triangle_sig = tuple(labels)
                        triangle_support[triangle_sig].add(graph.id)
    
    for tri_sig, graph_ids in triangle_support.items():
        if len(graph_ids) >= min_support_count:
            # Create triangle pattern
            g = Graph(pattern_id)
            # Simplified - just record that it's a triangle
            g.add_node(0, 0)
            g.add_node(1, 0)
            g.add_node(2, 0)
            g.add_edge(0, 1, 0)
            g.add_edge(1, 2, 0)
            g.add_edge(2, 0, 0)
            
            pattern = SubgraphPattern(pattern_id, g, len(graph_ids), graph_ids)
            patterns.append(pattern)
            pattern_id += 1
    
    print(f"Total patterns after triangles: {len(patterns)}")
    
    # 5. Mine 3-paths (longer chains)
    path3_support = defaultdict(set)
    
    for graph in graphs:
        # Find all paths of length 3
        for node in graph.G.nodes():
            for n1 in graph.G.neighbors(node):
                for n2 in graph.G.neighbors(n1):
                    if n2 != node:
                        for n3 in graph.G.neighbors(n2):
                            if n3 != n1 and n3 != node:
                                # Path: node - n1 - n2 - n3
                                labels = (
                                    graph.nodes[node],
                                    graph.G[node][n1]['label'],
                                    graph.nodes[n1],
                                    graph.G[n1][n2]['label'],
                                    graph.nodes[n2],
                                    graph.G[n2][n3]['label'],
                                    graph.nodes[n3]
                                )
                                path3_support[labels].add(graph.id)
    
    for path_sig, graph_ids in path3_support.items():
        if len(graph_ids) >= min_support_count:
            n1_l, e1_l, n2_l, e2_l, n3_l, e3_l, n4_l = path_sig
            g = Graph(pattern_id)
            g.add_node(0, n1_l)
            g.add_node(1, n2_l)
            g.add_node(2, n3_l)
            g.add_node(3, n4_l)
            g.add_edge(0, 1, e1_l)
            g.add_edge(1, 2, e2_l)
            g.add_edge(2, 3, e3_l)
            
            pattern = SubgraphPattern(pattern_id, g, len(graph_ids), graph_ids)
            patterns.append(pattern)
            pattern_id += 1
    
    print(f"Total patterns after 3-paths: {len(patterns)}")
    
    # 6. Mine star patterns (node with 3+ neighbors)
    star_support = defaultdict(set)
    
    for graph in graphs:
        for node in graph.G.nodes():
            neighbors = list(graph.G.neighbors(node))
            if len(neighbors) >= 3:
                # Take first 3 neighbors to form a star
                neighbor_labels = sorted([
                    (graph.nodes[n], graph.G[node][n]['label']) 
                    for n in neighbors[:3]
                ])
                star_sig = (graph.nodes[node],) + tuple(neighbor_labels)
                star_support[star_sig].add(graph.id)
    
    for star_sig, graph_ids in star_support.items():
        if len(graph_ids) >= min_support_count:
            g = Graph(pattern_id)
            center_label = star_sig[0]
            g.add_node(0, center_label)
            for i, (n_label, e_label) in enumerate(star_sig[1:], 1):
                g.add_node(i, n_label)
                g.add_edge(0, i, e_label)
            
            pattern = SubgraphPattern(pattern_id, g, len(graph_ids), graph_ids)
            patterns.append(pattern)
            pattern_id += 1
    
    print(f"Total patterns after stars: {len(patterns)}")
    
    # 7. Mine 4-cycles (squares)
    square_support = defaultdict(set)
    
    for graph in graphs:
        # Find all 4-cycles
        for node in graph.G.nodes():
            neighbors = list(graph.G.neighbors(node))
            for i, n1 in enumerate(neighbors):
                n1_neighbors = [n for n in graph.G.neighbors(n1) if n != node]
                for n2 in n1_neighbors:
                    n2_neighbors = [n for n in graph.G.neighbors(n2) if n != n1]
                    for n3 in n2_neighbors:
                        if graph.G.has_edge(n3, node):
                            # Square found: node - n1 - n2 - n3 - node
                            square_support[('square',)].add(graph.id)
                            break
    
    for square_sig, graph_ids in square_support.items():
        if len(graph_ids) >= min_support_count:
            g = Graph(pattern_id)
            g.add_node(0, 0)
            g.add_node(1, 0)
            g.add_node(2, 0)
            g.add_node(3, 0)
            g.add_edge(0, 1, 0)
            g.add_edge(1, 2, 0)
            g.add_edge(2, 3, 0)
            g.add_edge(3, 0, 0)
            
            pattern = SubgraphPattern(pattern_id, g, len(graph_ids), graph_ids)
            patterns.append(pattern)
            pattern_id += 1
    
    print(f"Total patterns after 4-cycles: {len(patterns)}")
    
    return patterns


def select_discriminative_subgraphs(patterns: List[SubgraphPattern], 
                                   total_graphs: int, 
                                   k: int = K) -> List[SubgraphPattern]:
    """
    Select k most discriminative subgraphs using hybrid approach
    """
    print(f"\n=== Selecting {k} discriminative subgraphs from {len(patterns)} patterns ===")
    
    # Step 1: Filter by support range (gIndex approach)
    candidates = []

    for pattern in patterns:
        support_ratio = pattern.support / total_graphs
        num_edges = pattern.graph.num_edges()

        # Hard constraints
        if num_edges < MIN_EDGES:
            continue
        if support_ratio < MINM_SUPPPORT_THRESHOLD:
            continue
        if support_ratio > MAXM_SUPPORT_THRESHOLD:
            continue

        # Passed hard constraints → compute discriminative power
        pattern.discriminative_score = calculate_discriminative_power(
            pattern.support, total_graphs
        )
        candidates.append(pattern)

    print(f"After hard-constraint filtering: {len(candidates)} candidates")
    
    if len(candidates) == 0:
        print("WARNING: No patterns in support range, using all patterns")
        candidates = patterns
        for pattern in candidates:
            pattern.discriminative_score = calculate_discriminative_power(pattern.support, total_graphs)
    
    # Step 2: Sort by discriminative score and size
    # Prefer patterns with high discriminative power and larger size
    candidates.sort(key=lambda p: (p.discriminative_score, p.graph.num_edges()), reverse=True)
    
    selected = candidates[:k]
    
    print(f"\n=== Selected {len(selected)} discriminative subgraphs ===")
    print("Distribution:")
    
    return selected


def save_discriminative_subgraphs(patterns: List[SubgraphPattern], output_path: str):
    """Save the discriminative subgraphs to file"""
    import os
    
    # If output_path is a directory, create a default filename
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, 'patterns.pkl')
    
    # Ensure parent directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save as pickle for easy loading
    with open(output_path, 'wb') as f:
        pickle.dump(patterns, f)
    
    # Also save human-readable version
    text_path = output_path + ".txt"
    with open(text_path, 'w') as f:
        f.write(f"# {len(patterns)} Discriminative Subgraphs\n\n")
        for i, pattern in enumerate(patterns):
            f.write(f"Pattern {i}:\n")
            f.write(f"  Nodes: {pattern.graph.num_nodes()}, Edges: {pattern.graph.num_edges()}\n")
            f.write(f"  Support: {pattern.support}\n")
            f.write(f"  Discriminative Score: {pattern.discriminative_score:.4f}\n")
            f.write(f"  Node labels: {list(pattern.graph.nodes.values())}\n")
            f.write(f"  Edges: {pattern.graph.edges}\n\n")
    
    print(f"Saved discriminative subgraphs to {output_path}")
    print(f"Saved human-readable version to {text_path}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python identify_discriminative.py <graph_dataset> <output_path> [gspan_path]")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    output_path = sys.argv[2]
    gspan_path = sys.argv[3] if len(sys.argv) > 3 else "./gSpan"
    
    print("=== Step 1: Loading and preprocessing graphs ===")
    graphs = parse_graph_dataset(dataset_path)
    print(f"Loaded {len(graphs)} graphs")
    
    # Remove duplicates
    graphs = remove_duplicates(graphs)
    print(f"After removing duplicates: {len(graphs)} graphs")
    total_graphs = len(graphs)
    
    print("\n=== Step 2: Mining frequent subgraphs ===")
    patterns = []
    
    # Try to use gSpan first
    use_gspan = os.path.exists(gspan_path) or os.path.exists(gspan_path + ".exe")
    
    if use_gspan:
        print("Using gSpan for frequent subgraph mining...")
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.gspan', delete=False) as f:
            gspan_input = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.out', delete=False) as f:
            gspan_output = f.name
        
        try:
            # Convert to gSpan format
            convert_to_gspan_format(graphs, gspan_input)
            
            # Run gSpan with 1% minimum support
            min_support_pct = GSPAN_SUPPORT_THRESHOLD
            success = run_gspan(gspan_input, gspan_output, min_support_pct, gspan_path)
            
            if success:
                # Parse gSpan output
                patterns = parse_gspan_output(gspan_output)
                
                if len(patterns) == 0:
                    print("WARNING: gSpan returned no patterns, trying higher support...")
                    # Try with higher support
                    success = run_gspan(gspan_input, gspan_output, GSPAN_SUPPORT_THRESHOLD, gspan_path)
                    if success:
                        patterns = parse_gspan_output(gspan_output)
            
            # Cleanup temp files
            if os.path.exists(gspan_input):
                os.remove(gspan_input)
            if os.path.exists(gspan_output):
                os.remove(gspan_output)
                
        except Exception as e:
            print(f"Error with gSpan: {e}")
            success = False
    else:
        print(f"gSpan not found at: {gspan_path}")
        success = False
    
    # Fallback to simple mining if gSpan failed
    if not success or len(patterns) == 0:
        print("\nFalling back to simple pattern mining...")
        patterns = mine_frequent_subgraphs_simple(graphs, min_support_pct=GSPAN_SUPPORT_THRESHOLD)
    else:
        print(f"\n✓ Successfully mined {len(patterns)} patterns using gSpan")
    
    print("\n=== Step 3: Selecting discriminative subgraphs ===")
    selected = select_discriminative_subgraphs(patterns, total_graphs, k=K)
    
    print("\n=== Step 4: Saving discriminative subgraphs ===")
    save_discriminative_subgraphs(selected, output_path)
    
    print("\n=== Done! ===")


if __name__ == "__main__":
    main()