"""
Graph Parser for Assignment Dataset
Reads graphs in the format:
# (new graph)
v node_id node_label
e source_id target_id edge_label
"""

import networkx as nx
from typing import List, Tuple, Dict

class Graph:
    """Simple graph class to store labeled graphs"""
    def __init__(self, graph_id: int):
        self.id = graph_id
        self.nodes = {}  # node_id -> label
        self.edges = []  # list of (source, target, label)
        self.G = nx.Graph()  # NetworkX graph for algorithms
        
    def add_node(self, node_id: int, label: int):
        self.nodes[node_id] = label
        self.G.add_node(node_id, label=label)
        
    def add_edge(self, source: int, target: int, label: int):
        self.edges.append((source, target, label))
        self.G.add_edge(source, target, label=label)
        
    def num_nodes(self):
        return len(self.nodes)
    
    def num_edges(self):
        return len(self.edges)
    
    def __repr__(self):
        return f"Graph(id={self.id}, nodes={self.num_nodes()}, edges={self.num_edges()})"


def parse_graph_dataset(filepath: str) -> List[Graph]:
    """
    Parse graph dataset from file
    
    Args:
        filepath: path to dataset file
        
    Returns:
        List of Graph objects
    """
    graphs = []
    current_graph = None
    graph_id = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # New graph marker
            if line.startswith('#'):
                if current_graph is not None:
                    graphs.append(current_graph)
                current_graph = Graph(graph_id)
                graph_id += 1
                continue
            
            parts = line.split()
            
            # Node line: v node_id node_label
            if parts[0] == 'v':
                node_id = int(parts[1])
                node_label = int(parts[2])
                current_graph.add_node(node_id, node_label)
                
            # Edge line: e source target edge_label
            elif parts[0] == 'e':
                source = int(parts[1])
                target = int(parts[2])
                edge_label = int(parts[3])
                current_graph.add_edge(source, target, edge_label)
    
    # Don't forget the last graph
    if current_graph is not None:
        graphs.append(current_graph)
    
    return graphs


def remove_duplicates(graphs: List[Graph]) -> List[Graph]:
    """
    Remove duplicate graphs while preserving order
    Two graphs are duplicates if they have same structure and labels
    """
    seen = set()
    unique_graphs = []
    
    for graph in graphs:
        # Create a canonical representation
        # Sort nodes and edges to get consistent hash
        nodes_str = str(sorted(graph.nodes.items()))
        edges_str = str(sorted(graph.edges))
        signature = nodes_str + edges_str
        
        if signature not in seen:
            seen.add(signature)
            # Reassign ID to maintain sequential ordering
            graph.id = len(unique_graphs)
            unique_graphs.append(graph)
    
    return unique_graphs


def write_graph_dataset(graphs: List[Graph], filepath: str):
    """Write graphs back to file in the same format"""
    with open(filepath, 'w') as f:
        for graph in graphs:
            f.write("#\n")
            for node_id, label in sorted(graph.nodes.items()):
                f.write(f"v {node_id} {label}\n")
            for source, target, label in graph.edges:
                f.write(f"e {source} {target} {label}\n")


if __name__ == "__main__":
    # Test the parser
    import sys
    if len(sys.argv) > 1:
        graphs = parse_graph_dataset(sys.argv[1])
        print(f"Parsed {len(graphs)} graphs")
        print(f"First graph: {graphs[0]}")
        print(f"Last graph: {graphs[-1]}")
        
        # Test duplicate removal
        unique = remove_duplicates(graphs)
        print(f"After removing duplicates: {len(unique)} unique graphs")