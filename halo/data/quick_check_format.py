#!/usr/bin/env python3
"""Quick check of OpenThoughts3 conversation format."""
from datasets import load_dataset

dataset = load_dataset('open-thoughts/OpenThoughts3-1.2M', split='train', streaming=True)

print('Checking first 5 samples for "from" field values...\n')
from_values = set()

for idx, item in enumerate(dataset):
    if idx >= 5:
        break
    conversations = item.get('conversations', [])
    print(f'Sample {idx}: {len(conversations)} turns')
    for turn in conversations:
        from_val = turn.get('from')
        from_values.add(from_val)
        print(f'  from: "{from_val}"')
    print()

print(f'\nUnique "from" values found: {from_values}')
