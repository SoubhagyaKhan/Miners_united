"""
Convert graph dataset to FSG format

FSG format:
#graphID
number of nodes
node_label
node_label
...
number of edges
source_id, target_id, edge_label
source_id, target_id, edge_label
...
"""

import sys
from graph_parser import parse_graph_dataset, remove_duplicates

def convert_to_fsg_format(graphs, output_file):
    """Convert graphs to FSG format"""
    with open(output_file, 'w') as f:
        for graph in graphs:
            # Graph ID
            f.write(f"#{graph.id}\n")
            
            # Number of nodes
            f.write(f"{graph.num_nodes()}\n")
            
            # Node labels (in order of node IDs)
            for node_id in sorted(graph.nodes.keys()):
                f.write(f"{graph.nodes[node_id]}\n")
            
            # Number of edges
            f.write(f"{graph.num_edges()}\n")
            
            # Edges
            for source, target, label in graph.edges:
                f.write(f"{source}, {target}, {label}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_fsg.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"Reading graphs from {input_file}...")
    graphs = parse_graph_dataset(input_file)
    print(f"Parsed {len(graphs)} graphs")
    
    # Remove duplicates
    graphs = remove_duplicates(graphs)
    print(f"After removing duplicates: {len(graphs)} graphs")
    
    print(f"Converting to FSG format...")
    convert_to_fsg_format(graphs, output_file)
    print(f"Wrote FSG format to {output_file}")