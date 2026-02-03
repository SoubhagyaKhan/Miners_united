"""
Generate candidate sets for query graphs based on feature vectors

For each query graph q with feature vector vq,
find all database graphs gi with feature vector vi where vq <= vi (component-wise)

This is based on the principle: if q ⊆ gi, then every feature in q must be in gi
"""

import sys
import numpy as np

def generate_candidates(db_features: np.ndarray, query_features: np.ndarray) -> list:
    """
    Generate candidate sets for each query graph
    
    Args:
        db_features: 2D array (n_db_graphs, n_features)
        query_features: 2D array (n_query_graphs, n_features)
        
    Returns:
        List of lists, where result[i] contains candidate graph IDs for query i
    """
    n_queries = query_features.shape[0]
    n_db_graphs = db_features.shape[0]
    
    print(f"Generating candidates for {n_queries} queries from {n_db_graphs} database graphs...")
    
    candidates = []
    
    for i in range(n_queries):
        query_vec = query_features[i]
        
        # Find all database graphs where query_vec <= db_vec (component-wise)
        # This means: for all j, if query_vec[j] == 1, then db_vec[j] must be 1
        
        query_candidates = []
        
        for j in range(n_db_graphs):
            db_vec = db_features[j]
            
            # Check if query_vec <= db_vec component-wise
            # Equivalent to: (query_vec AND NOT db_vec) == 0
            # Or: all positions where query has 1, db must also have 1
            
            is_candidate = True
            for k in range(len(query_vec)):
                if query_vec[k] == 1 and db_vec[k] == 0:
                    is_candidate = False
                    break
            
            if is_candidate:
                query_candidates.append(j)
        
        candidates.append(query_candidates)
        
        if (i + 1) % 10 == 0 or (i + 1) == n_queries:
            print(f"  Query {i+1}/{n_queries}: {len(query_candidates)} candidates")
    
    return candidates


def write_candidates_file(candidates: list, output_path: str):
    """
    Write candidates to file in the required format:
    
    q # 0
    c # 1 2 3 4 5
    q # 1
    c # 4 5 12 45
    ...
    """
    with open(output_path, 'w') as f:
        for i, candidate_list in enumerate(candidates):
            # Query number (0-indexed in output)
            f.write(f"q # {i}\n")
            
            # Candidate graph numbers  (0-indexed in output)
            candidate_str = " ".join(str(c) for c in candidate_list)
            f.write(f"c # {candidate_str}\n")
    
    print(f"Wrote candidates to {output_path}")


def main():
    if len(sys.argv) != 4:
        print("Usage: python generate_candidates.py <db_features> <query_features> <output_file>")
        sys.exit(1)
    
    db_features_path = sys.argv[1]
    query_features_path = sys.argv[2]
    output_path = sys.argv[3]
    
    print("=== Step 1: Loading feature vectors ===")
    
    # Load numpy arrays
    # Handle both .npy and without extension
    if db_features_path.endswith('.npy'):
        db_features = np.load(db_features_path)
    else:
        try:
            db_features = np.load(db_features_path)
        except:
            db_features = np.load(db_features_path + '.npy')
    
    if query_features_path.endswith('.npy'):
        query_features = np.load(query_features_path)
    else:
        try:
            query_features = np.load(query_features_path)
        except:
            query_features = np.load(query_features_path + '.npy')
    
    print(f"Database features shape: {db_features.shape}")
    print(f"Query features shape: {query_features.shape}")
    
    print("\n=== Step 2: Generating candidates ===")
    candidates = generate_candidates(db_features, query_features)
    
    print("\n=== Step 3: Writing candidates file ===")
    write_candidates_file(candidates, output_path)
    
    # Print statistics
    total_candidates = sum(len(c) for c in candidates)
    avg_candidates = total_candidates / len(candidates) if candidates else 0
    print(f"\nStatistics:")
    print(f"  Total queries: {len(candidates)}")
    print(f"  Total candidates: {total_candidates}")
    print(f"  Average candidates per query: {avg_candidates:.2f}")
    print(f"  Min candidates: {min(len(c) for c in candidates) if candidates else 0}")
    print(f"  Max candidates: {max(len(c) for c in candidates) if candidates else 0}")
    
    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
