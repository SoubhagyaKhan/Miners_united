#!/bin/bash

# generate_candidates.sh - Generate candidate sets for query graphs
# Usage: bash generate_candidates.sh <path_database_graph_features> <path_query_graph_features> <path_out_file>

if [ "$#" -ne 3 ]; then
    echo "Usage: bash generate_candidates.sh <path_database_graph_features> <path_query_graph_features> <path_out_file>"
    exit 1
fi

DB_FEATURES=$1
QUERY_FEATURES=$2
OUTPUT_FILE=$3

echo "=== Generating Candidate Sets ==="
echo "Database features: $DB_FEATURES"
echo "Query features: $QUERY_FEATURES"
echo "Output file: $OUTPUT_FILE"

python3 generate_candidates.py "$DB_FEATURES" "$QUERY_FEATURES" "$OUTPUT_FILE"

echo "=== Candidates generated ==="