#!/bin/bash
# bash q1_2.sh <itemset | num_items> <num_transactions>

ITEMSET_OR_NUM=$1
NUM_TRANSACTIONS=$2

echo "Dataset Generation"
echo "Transactions: $NUM_TRANSACTIONS"
echo ""

# Check if first argument is a number
if [[ "$ITEMSET_OR_NUM" =~ ^[0-9]+$ ]]; then
    echo "Generating itemset from 1 to $ITEMSET_OR_NUM"
    UNIVERSAL_ITEMSET=$(seq -s, 1 "$ITEMSET_OR_NUM")
else
    echo "Using provided itemset"
    UNIVERSAL_ITEMSET="$ITEMSET_OR_NUM"
fi

echo "Itemset: $UNIVERSAL_ITEMSET"
echo ""

# Run dataset generator
python3 generate_dataset.py "$UNIVERSAL_ITEMSET" "$NUM_TRANSACTIONS"
