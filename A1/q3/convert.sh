#!/bin/bash

# convert.sh - Convert graphs to feature vectors
# Usage: bash convert.sh <path_graphs> <path_discriminative_subgraphs> <path_features>

if [ "$#" -ne 3 ]; then
    echo "Usage: bash convert.sh <path_graphs> <path_discriminative_subgraphs> <path_features>"
    exit 1
fi

GRAPHS_PATH=$1
PATTERNS_PATH=$2
OUTPUT_PATH=$3

echo "=== Converting Graphs to Feature Vectors ==="
echo "Input graphs: $GRAPHS_PATH"
echo "Discriminative patterns: $PATTERNS_PATH"
echo "Output features: $OUTPUT_PATH"

python3 convert_to_features.py "$GRAPHS_PATH" "$PATTERNS_PATH" "$OUTPUT_PATH"

echo "=== Feature vectors created ==="