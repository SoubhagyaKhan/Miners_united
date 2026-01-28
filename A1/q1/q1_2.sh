#!/bin/bash

# q1_2.sh - Generate dataset for Task 2
# Usage: bash q1_2.sh <num_items> <num_transactions>

NUM_ITEMS=$1
NUM_TRANSACTIONS=$2

echo "Dataset Generation"
echo "Number of items: $NUM_ITEMS (1 to $NUM_ITEMS)"
echo "Number of transactions: $NUM_TRANSACTIONS"
echo ""

# Generate comma-separated itemset from 1 to NUM_ITEMS
echo "Generating universal itemset..."
UNIVERSAL_ITEMSET=$(seq -s, 1 $NUM_ITEMS)

# Run the dataset generation script
python3 generate_dataset.py "$UNIVERSAL_ITEMSET" "$NUM_TRANSACTIONS"