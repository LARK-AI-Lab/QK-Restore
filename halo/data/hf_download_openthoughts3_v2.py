#!/usr/bin/env python3
"""
Download a sample of OpenThoughts3-1.2M dataset from HuggingFace.
Improved version with better progress tracking and early stopping.
"""
from datasets import load_dataset, Dataset
import time

MAX_SAMPLES = 200000
SAVE_PATH = "/path/to/data/openthoughts3_sft_200K"

print(f"Loading OpenThoughts3-1.2M dataset (streaming mode)...")
dataset = load_dataset(
    "open-thoughts/OpenThoughts3-1.2M",
    split="train",
    streaming=True,
)

data = []
skipped = 0
start_time = time.time()

print(f"Collecting up to {MAX_SAMPLES:,} valid samples...")
print(f"Progress updates every 10,000 valid samples collected.\n")

for idx, item in enumerate(dataset):
    # OpenThoughts3 format: conversations list with "from" and "value" keys
    conversations = item.get("conversations", [])

    # Expect at least 2 turns: human + assistant
    if len(conversations) < 2:
        skipped += 1
        continue

    # Extract first user turn and last assistant turn
    human_turns = [c["value"] for c in conversations if c["from"] == "human"]
    assistant_turns = [c["value"] for c in conversations if c["from"] == "assistant"]

    if not human_turns or not assistant_turns:
        skipped += 1
        continue

    # Store the full conversations structure
    data.append({
        "conversations": conversations,
        "problem": human_turns[0],
        "answer": assistant_turns[-1],
    })

    # Progress update every 10K valid samples
    if len(data) % 10000 == 0 and len(data) > 0:
        elapsed = time.time() - start_time
        rate = len(data) / elapsed
        eta = (MAX_SAMPLES - len(data)) / rate if rate > 0 else 0
        print(f"  ✓ Collected {len(data):,} valid samples "
              f"(processed {idx + 1:,} total, skipped {skipped:,}) "
              f"[{rate:.1f} samples/sec, ETA {eta/60:.1f} min]")

    # Stop when we reach the target
    if len(data) >= MAX_SAMPLES:
        elapsed = time.time() - start_time
        print(f"\n✓ Target reached!")
        print(f"  Collected: {len(data):,} valid samples")
        print(f"  Processed: {idx + 1:,} total samples")
        print(f"  Skipped: {skipped:,} invalid samples")
        print(f"  Time: {elapsed/60:.1f} minutes")
        break

print(f"\nFinal count: {len(data):,} samples")

# Convert to Dataset and save
print(f"Converting to Dataset format...")
dataset_filtered = Dataset.from_list(data)

print(f"Saving to {SAVE_PATH}...")
dataset_filtered.save_to_disk(SAVE_PATH)

print("\n✓ Done!")
print(f"\nSample structure:")
print(f"  conversations: list of {{'from': 'human'|'assistant', 'value': str}}")
print(f"  problem: first user turn")
print(f"  answer: last assistant turn")
