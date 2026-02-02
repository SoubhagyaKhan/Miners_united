"""
Convert graphs to binary feature vectors using subgraph isomorphism
"""

import sys
import pickle
import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
from graph_parser import parse_graph_dataset, Graph
from identify_discriminative import SubgraphPattern
from typing import List

def graph_to_networkx(graph: Graph) -> nx.Graph:
    """Convert our Graph object to NetworkX graph for isomorphism testing"""
    G = nx.Graph()
    
    for node_id, label in graph.nodes.items():
        G.add_node(node_id, label=label)
    
    for source, target, label in graph.edges:
        G.add_edge(source, target, label=label)
    
    return G


def is_subgraph_isomorphic(pattern_graph: Graph, target_graph: Graph) -> bool:
    """
    Check if pattern is subgraph isomorphic to target
    Uses NetworkX's subgraph isomorphism with node and edge label matching
    """
    pattern_nx = graph_to_networkx(pattern_graph)
    target_nx = graph_to_networkx(target_graph)
    
    # Node label matching
    def node_match(n1, n2):
        return n1.get('label') == n2.get('label')
    
    # Edge label matching
    def edge_match(e1, e2):
        return e1.get('label') == e2.get('label')
    
    # Create graph matcher
    GM = isomorphism.GraphMatcher(target_nx, pattern_nx, 
                                   node_match=node_match,
                                   edge_match=edge_match)
    
    return GM.subgraph_is_isomorphic()


def convert_graphs_to_features(graphs: List[Graph], 
                               patterns: List[SubgraphPattern]) -> np.ndarray:
    """
    Convert graphs to binary feature vectors
    
    Args:
        graphs: List of graphs to convert
        patterns: List of discriminative subgraph patterns
        
    Returns:
        2D numpy array of shape (n_graphs, n_patterns) with binary values
    """
    n_graphs = len(graphs)
    n_patterns = len(patterns)
    
    print(f"Converting {n_graphs} graphs using {n_patterns} patterns...")
    
    # Initialize feature matrix
    features = np.zeros((n_graphs, n_patterns), dtype=np.int8)
    
    # For each graph
    for i, graph in enumerate(graphs):
        if i % 100 == 0:
            print(f"  Processing graph {i}/{n_graphs}...")
        
        # Check each pattern
        for j, pattern in enumerate(patterns):
            # Check if pattern is subgraph isomorphic to graph
            if is_subgraph_isomorphic(pattern.graph, graph):
                features[i, j] = 1
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Sparsity: {(features == 0).sum() / features.size * 100:.2f}% zeros")
    
    return features


def main():
    if len(sys.argv) != 4:
        print("Usage: python convert_to_features.py <graphs_path> <patterns_path> <output_path>")
        sys.exit(1)
    
    graphs_path = sys.argv[1]
    patterns_path = sys.argv[2]
    output_path = sys.argv[3]
    
    print("=== Step 1: Loading graphs ===")
    graphs = parse_graph_dataset(graphs_path)
    print(f"Loaded {len(graphs)} graphs")
    
    print("\n=== Step 2: Loading discriminative patterns ===")
    with open(patterns_path, 'rb') as f:
        patterns = pickle.load(f)
    print(f"Loaded {len(patterns)} discriminative patterns")
    
    print("\n=== Step 3: Converting to feature vectors ===")
    features = convert_graphs_to_features(graphs, patterns)
    
    print("\n=== Step 4: Saving feature vectors ===")
    np.save(output_path, features)
    print(f"Saved feature vectors to {output_path}")
    
    # Also save as .npy if not already
    if not output_path.endswith('.npy'):
        np.save(output_path + '.npy', features)
        print(f"Also saved as {output_path}.npy")
    
    print("\n=== Done! ===")


if __name__ == "__main__":
    main()