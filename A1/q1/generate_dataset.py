#!/usr/bin/env python3
"""
Dataset Generator for Apriori vs FP-Growth runtime behavior

Design:
- 20% of items are CORE
- ~80% of transactions contain most CORE items
- Remaining transactions contain tail / mixed items
- No duplicated transactions
"""

import sys
import random

def parse_itemset(itemset_str):
    # itemset_str is comma-separated
    items = [int(x.strip()) for x in itemset_str.split(",") if x.strip()]
    return items

def generate_dataset(universal_items, num_transactions):
    num_items = len(universal_items)

    # Split items
    num_core = max(1, int(0.05 * num_items))
    core_items = universal_items[:num_core]
    tail_items = universal_items[num_core:]

    transactions = []
    seen = set()

    for _ in range(num_transactions):
        r = random.random()

        # Case 1: Core-dominated transaction (80%)
        if r < 0.95:
            txn = set()
            # Include core items with high probability
            for item in core_items:
                if random.random() < 0.95:
                    txn.add(item)

            # Add a little noise from tail
            noise_k = random.randint(0, 2)
            if tail_items and noise_k > 0:
                txn.update(random.sample(tail_items, min(noise_k, len(tail_items))))

        # Case 2: Mixed transaction (10%)
        else:
            txn = set()
            core_k = random.randint(1, max(1, len(core_items)//3))
            tail_k = random.randint(1, min(5, len(tail_items)))

            txn.update(random.sample(core_items, core_k))
            if tail_items:
                txn.update(random.sample(tail_items, tail_k))

        # Avoid empty transactions
        if not txn:
            txn.add(random.choice(universal_items))

        # Ensure no exact duplicates
        txn_tuple = tuple(sorted(txn))
        if txn_tuple not in seen:
            seen.add(txn_tuple)
            transactions.append(txn_tuple)
        else:
            # regenerate one transaction
            continue

    return transactions

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 generate_dataset.py <itemset> <num_transactions>")
        sys.exit(1)

    itemset_str = sys.argv[1]
    num_transactions = int(sys.argv[2])

    universal_items = parse_itemset(itemset_str)

    transactions = generate_dataset(universal_items, num_transactions)

    # Output to stdout (redirect in bash if needed)
    output_file = 'generate_datasets.dat'
    
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w') as f:
        for transaction in transactions:
            f.write(' '.join(map(str, transaction)) + '\n')

if __name__ == "__main__":
    main()
