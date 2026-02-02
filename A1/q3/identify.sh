#!/bin/bash

# identify.sh - Identify discriminative subgraphs using gSpan (with fallback)
# Usage: bash identify.sh <path_graph_dataset> <path_discriminative_subgraphs> [gspan_path]

if [ "$#" -lt 2 ]; then
    echo "Usage: bash identify.sh <path_graph_dataset> <path_discriminative_subgraphs> [gspan_path]"
    exit 1
fi

GRAPH_DATASET=$1
OUTPUT_PATH=$2
GSPAN_PATH=${3:-"./gSpan-64"}  # Default to ./gSpan in current directory

echo "=== Identifying Discriminative Subgraphs ==="
echo "Input dataset: $GRAPH_DATASET"
echo "Output path: $OUTPUT_PATH"
echo "gSpan path: $GSPAN_PATH"

# Validate input file exists
if [ ! -f "$GRAPH_DATASET" ]; then
    echo "ERROR: Input dataset not found: $GRAPH_DATASET"
    exit 1
fi

# If output path is a directory, create a default filename
if [ -d "$OUTPUT_PATH" ]; then
    echo "WARNING: Output path is a directory, using default filename"
    OUTPUT_PATH="$OUTPUT_PATH/patterns.pkl"
    echo "New output path: $OUTPUT_PATH"
fi

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Run pattern mining (will use gSpan if available, otherwise fallback)
python3 identify_discriminative.py "$GRAPH_DATASET" "$OUTPUT_PATH" "$GSPAN_PATH"

# Verify output was created
if [ ! -f "$OUTPUT_PATH" ]; then
    echo "ERROR: Pattern file was not created: $OUTPUT_PATH"
    exit 1
fi

echo "=== Discriminative subgraphs identified and saved ==="
echo "Pattern file: $OUTPUT_PATH"