#!/bin/bash

# Environment setup for graph indexing
echo "Setting up environment for graph indexing..."

# Create conda environment if needed
# conda create -n graph_index python=3.10 -y
# conda activate graph_index

# Install required packages
pip install numpy scipy networkx

echo "Environment setup complete!"