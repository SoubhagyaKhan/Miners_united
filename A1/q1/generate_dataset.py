"""
Dataset Generator for Task 2
Creates a transactional dataset that produces the runtime behavior.
"""

import sys
import numpy as np
from collections import defaultdict, Counter

def parse_itemset(itemset_str):
    # Remove any quotes that might be present
    itemset_str = itemset_str.strip().strip('"').strip("'")
    
    # comma-separated
    items = [int(x.strip()) for x in itemset_str.split(',') if x.strip()]
    
    return items

def generate_dataset(universal_itemset, num_transactions, output_file='generated_transactions.dat'):

    np.random.seed(42)  # For reproducibility
    
    items = list(universal_itemset)
    n_items = len(items)
    
    print(f"Generating {num_transactions} transactions from {n_items} items...")
    
    # Partition items into frequency groups
    # Group 1: Super frequent (>90% support) - very few items
    super_frequent_size = max(3, n_items // 100) # FIRST 10 ITEMS
    super_frequent = items[:super_frequent_size]
    
    # Group 2: Frequent (60-80% support)
    start = super_frequent_size
    frequent_size = max(5, n_items // 50) # FIRST 20 ITEMS
    frequent = items[start:start + frequent_size]
    
    # Group 3: Semi-frequent (35-55% support)
    start = start + frequent_size
    semi_frequent_size = max(10, n_items // 30) # FIRST 33 ITEMS
    semi_frequent = items[start:start + semi_frequent_size]
    
    # Group 4: Medium (15-30% support)
    start = start + semi_frequent_size
    medium_size = max(20, n_items // 20) # FIRST 50 ITEMS
    medium = items[start:start + medium_size]
    
    # Group 5: Low-medium (6-14% support) - for Apriori slowdown
    start = start + medium_size
    low_medium_size = max(50, n_items // 10) # FIRST 100 ITEMS
    low_medium = items[start:start + low_medium_size]
    
    # Group 6: Low (2-6% support)
    start = start + low_medium_size # REMAINING 787 ITEMS
    low = items[start:]
    
    print(f"\nItem frequency distribution:")
    print(f"  Super frequent (>90%):    {len(super_frequent):4d} items")
    print(f"  Frequent (60-80%):        {len(frequent):4d} items")
    print(f"  Semi-frequent (35-55%):   {len(semi_frequent):4d} items")
    print(f"  Medium (15-30%):          {len(medium):4d} items")
    print(f"  Low-medium (6-14%):       {len(low_medium):4d} items")
    print(f"  Low (2-6%):               {len(low):4d} items")
    
    transactions = []
    
    print(f"\nGenerating transactions...")
    for i in range(num_transactions):
        transaction = set()
        
        # Add super frequent items (>90% support)
        for item in super_frequent:
            if np.random.rand() < 0.93:
                transaction.add(item)
        
        # Add frequent items (60-80% support)
        for item in frequent:
            if np.random.rand() < np.random.uniform(0.6, 0.8):
                transaction.add(item)
        
        # Add semi-frequent items (35-55% support)
        if len(semi_frequent) > 0:
            n_sample = max(1, len(semi_frequent) // 2)
            sampled = np.random.choice(semi_frequent, size=min(n_sample, len(semi_frequent)), replace=False)
            for item in sampled:
                if np.random.rand() < np.random.uniform(0.35, 0.55):
                    transaction.add(item)
        
        # Add medium frequency items (15-30% support)
        if len(medium) > 0:
            n_sample = max(1, len(medium) // 2)
            sampled = np.random.choice(medium, size=min(n_sample, len(medium)), replace=False)
            for item in sampled:
                if np.random.rand() < np.random.uniform(0.15, 0.3):
                    transaction.add(item)
        
        # Add low-medium frequency items (6-14% support) - KEY GROUP
        if len(low_medium) > 0:
            n_sample = max(5, len(low_medium) // 2)
            sampled = np.random.choice(low_medium, size=min(n_sample, len(low_medium)), replace=False)
            for item in sampled:
                if np.random.rand() < np.random.uniform(0.06, 0.14):
                    transaction.add(item)
        
        # Add low frequency items (2-6% support)
        if len(low) > 0:
            n_sample = max(10, len(low) // 3)
            sampled = np.random.choice(low, size=min(n_sample, len(low)), replace=False)
            for item in sampled:
                if np.random.rand() < np.random.uniform(0.02, 0.06):
                    transaction.add(item)
        
        # Ensure minimum transaction length
        min_length = 25
        if len(transaction) < min_length:
            remaining = list(set(items) - transaction)
            if len(remaining) > 0:
                add_count = min(min_length - len(transaction), len(remaining))
                additional = np.random.choice(remaining, size=add_count, replace=False)
                transaction.update(additional)
        
        transactions.append(sorted(transaction))
        
        if (i + 1) % 3000 == 0:
            print(f"  Progress: {i + 1}/{num_transactions} transactions generated")
    
    # Write to file
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w') as f:
        for transaction in transactions:
            f.write(' '.join(map(str, transaction)) + '\n')
    
    
    # Calculate actual item supports
    item_counts = Counter()
    for txn in transactions:
        item_counts.update(txn)
    
    support_ranges = {
        '>90%': 0,
        '50-90%': 0,
        '25-50%': 0,
        '10-25%': 0,
        '5-10%': 0,
        '<5%': 0
    }
    
    for item, count in item_counts.items():
        support = count / len(transactions)
        if support > 0.9:
            support_ranges['>90%'] += 1
        elif support > 0.5:
            support_ranges['50-90%'] += 1
        elif support > 0.25:
            support_ranges['25-50%'] += 1
        elif support > 0.1:
            support_ranges['10-25%'] += 1
        elif support > 0.05:
            support_ranges['5-10%'] += 1
        else:
            support_ranges['<5%'] += 1
    
    print(f"\nActual item support distribution:")
    for range_label in ['>90%', '50-90%', '25-50%', '10-25%', '5-10%', '<5%']:
        count = support_ranges[range_label]
        print(f"  {range_label:8s}:            {count:4d} items")
    
    return transactions

def main():
    # Parse arguments
    items = parse_itemset(sys.argv[1])
    
    num_transactions = int(sys.argv[2])
    # Generate dataset
    generate_dataset(items, num_transactions)

if __name__ == "__main__":
    main()

# Actual item support distribution:
#   >90%    :              10 items
#   50-90%  :              20 items
#   25-50%  :               0 items
#   10-25%  :              83 items
#   5-10%   :              54 items
#   <5%     :             833 items